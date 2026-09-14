"""2D fully-coupled Newton solve using quasi-Fermi potentials (psi, phin,
phip) over a point-cloud/box-FV mesh (mesh2d/mesh2d.py) - the point-cloud
generalization of core/newton_solver_qf.py. Lives in solver2d/ (not
mesh2d/) since this is solver/physics code, not mesh geometry - mesh2d/
builds the Mesh2D object this module consumes, the same way core/mesh.py
builds the grid core/newton_solver_qf.py consumes for 1D. See that
module's docstring for the full physics derivation (plain-gradient
current, no Scharfetter-Gummel) and why this formulation was chosen: it is
the one that resolved the 1D extreme-doping convergence failure, and per
the project's mesh-robustness principle, a 2D solver especially cannot
depend on the user hand-resolving the Debye length everywhere the way
fine 1D meshing could.

The only real generalization is the discretization geometry: 1D's two
fixed neighbors (i-1, i+1) become an arbitrary-degree graph edge list
(mesh2d/fvgeometry.py's Voronoi-box edges). This module loops (vectorized,
no per-node Python loop) over that edge list and lets scipy.sparse's COO
format sum duplicate (row, col) entries automatically on conversion to
CSC - the same accumulation 1D achieves implicitly via its two explicitly
named e_lo/e_hi edges, just generalized to however many edges a point
happens to have.

Boundary conditions need no special-casing beyond that: every point gets
the SAME generic Poisson/continuity row, contact points included, and is
then overwritten with a Dirichlet row afterward (contact points are
excluded from the generic row assembly via a boolean mask - see
`active_i`/`active_j` below - so the Dirichlet overwrite is a clean replacement,
not a sum). A `symmetry` or `free_surface` point never gets a special
row at all: its control volume simply has fewer incident edges than an
interior point's, which is the entire implementation of "default
insulating" (see mesh2d/boundary.py and mesh2d/fvgeometry.py's module
docstrings).

Materials: this module takes a plain scalar Material (today's homogeneous-
silicon first example) rather than a MaterialField - MaterialField's
eps_edge/mu_n_edge/etc. arrays are sized len(x)-1, a 1D-topology-specific
convention that doesn't generalize to an arbitrary edge count without
being rebuilt anyway, so heterojunction-in-2D support is left as future
work rather than force-fitting MaterialField here.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from core.params import Q, Material
from core.jacobian_scaling import equilibrated_spsolve
from core.physics import equilibrium_bulk_potential_arr
from core.solver import contact_values

MAX_QF_STEP = 5.0

# Regularization field scale (V/cm) for the Caughey-Thomas |E| -> smooth
# even function of E: E_reg = sqrt(E^2 + E_SMOOTH^2). This is what makes
# mu(E) (and hence d(mu)/dE) a well-defined, smooth function of E right at
# E=0 - the literal |E|/E^beta forms have a kink or (for beta<1) a
# diverging derivative there, which would poison the Jacobian at exactly
# the equilibrium/near-flatband bias points this project starts every
# Newton continuation from. 1 V/cm is ~4-7 orders of magnitude below any
# physically meaningful field in this device (channel fields are
# ~1e3-1e6 V/cm), so it is a pure numerical regularization, not a physics
# approximation.
E_SMOOTH = 1.0


def unpack_qf(U, N):
    return U[:N], U[N:2 * N], U[2 * N:3 * N]


def mobility_doping(N_abs, mu_max, mu_min, N_ref, alpha):
    """Caughey-Thomas DOPING-dependence low-field mobility (see
    core/params.py's Material.mu_n_max/etc. docstring for the formula's
    citation/parameter source):

        mu(N) = mu_min + (mu_max - mu_min) / (1 + (N/N_ref)^alpha)

    N_abs is the LOCAL TOTAL (ionized) doping magnitude, cm^-3 - the usual
    convention for this model (not net doping, since even a compensated
    region's mobility is degraded by both dopant species' scattering).
    Depends ONLY on the fixed mesh doping array, never on the solved
    psi/phin/phip, so (unlike the old field-dependent mobility_field())
    this needs no Jacobian term of its own - it is exactly as simple as a
    scalar constant mobility, just spatially varying."""
    return mu_min + (mu_max - mu_min) / (1.0 + (N_abs / N_ref) ** alpha)


def mobility_field(E, mu0, vsat, beta):
    """Caughey-Thomas velocity-saturation mobility and its derivative
    w.r.t. the (signed) driving field E:

        mu(E) = mu0 / (1 + (mu0*|E|/vsat)^beta)^(1/beta)

    |E| is replaced by the smooth even regularization sqrt(E^2+E_SMOOTH^2)
    (see E_SMOOTH's docstring above) so mu and dmu_dE are both finite,
    smooth functions of E everywhere including E=0 - required since E=0
    is exactly where every Newton continuation in this project starts
    (equilibrium/flatband). Returns (mu, dmu_dE), both arrays shaped like E.

    Derived by direct differentiation of mu(E) treating x = mu0*Ereg/vsat:
    mu = mu0*(1+x^beta)^(-1/beta), so
    dmu/dEreg = -mu0*(mu0/vsat)*x^(beta-1)*(1+x^beta)^(-1/beta-1),
    and dmu/dE = dmu/dEreg * dEreg/dE = dmu/dEreg * (E/Ereg).
    Note beta_n=2 and beta_p=1 (this project's defaults, core/params.py)
    both give x^(beta-1) with a non-negative integer exponent (x^1 or x^0),
    so this derivative is itself finite and smooth at x=0 (E=0) even
    without the E_SMOOTH regularization - E_SMOOTH is still kept for
    robustness against any future non-default beta<1 choice.
    """
    Ereg = np.sqrt(E ** 2 + E_SMOOTH ** 2)
    x = mu0 * Ereg / vsat
    base = 1.0 + x ** beta
    mu = mu0 * base ** (-1.0 / beta)
    dmu_dEreg = -mu0 * (mu0 / vsat) * x ** (beta - 1.0) * base ** (-1.0 / beta - 1.0)
    dmu_dE = dmu_dEreg * (E / Ereg)
    return mu, dmu_dE


def _densities(psi, phin, phip, ni_arr, Vt):
    n = ni_arr * np.exp((psi - phin) / Vt)
    p = ni_arr * np.exp((phip - psi) / Vt)
    return n, p


def _mesh_mobility_nodal(mesh, mat):
    """Static (solve-independent) per-NODE doping-dependent mobility
    arrays, following the same precomputed-array pattern as
    _mesh_ni_edge_g's ni_arr/edge_g below (mesh.Cdop is always populated,
    homogeneous mesh or not, so - unlike ni_arr - this needs no
    homogeneous-mesh fallback branch)."""
    N_abs = np.abs(mesh.Cdop)
    mu_n_node = mobility_doping(N_abs, mat.mu_n_max, mat.mu_n_min, mat.N_ref_n, mat.alpha_n)
    mu_p_node = mobility_doping(N_abs, mat.mu_p_max, mat.mu_p_min, mat.N_ref_p, mat.alpha_p)
    return mu_n_node, mu_p_node


def _semiconductor_edge_mask(mesh):
    """Boolean (E,) mask, True on edges where BOTH endpoints are real
    semiconductor (not insulator/oxide) - the physically correct no-flux
    BC at a semiconductor/insulator interface (Jn.n_hat = Jp.n_hat = 0
    there, current is confined to the semiconductor). Found (2026-09-13,
    jointly - see this module's mobility-doping-model commit and the
    matching root-cause writeup) to be a genuine, previously-undiagnosed
    discretization bug when missing: an edge with one endpoint a real
    semiconductor node (nonzero n or p) and the other an is_oxide_free
    insulator node (pinned phin=phip=0 - an ARBITRARY placeholder, not a
    real potential) computes a nonzero Jn_e/Jp_e driven by that arbitrary
    pinned value, using n_avg=0.5*(n_real+0) (still nonzero since only the
    OXIDE side's density is zero). The is_oxide_free row-pinning logic
    below only stops that node's OWN row from being assembled via the
    normal continuity equation - it does nothing to stop this edge's
    current from still being added into the ADJACENT semiconductor (or
    even CONTACT) node's row, so it silently leaks a spurious current with
    no governing conservation law. Confirmed empirically at a real MOSFET
    off-state bias point: 99.995% of the entire (spurious) drain current
    came from a single such edge at the corner where the drain contact
    meets the gate-oxide/mesa wall, tracking the DRAIN's own fixed
    Dirichlet BC (hence flat vs. Vgs) and completely independent of
    substrate doping (hence flat vs. a 1e16->1e18 doping sweep) - exactly
    the two symptoms flagged (but never root-caused) in earlier sessions.
    Returns an all-True mask (no insulator anywhere) for a homogeneous
    mesh (mesh.is_insulator is None, e.g. the plain diode)."""
    if mesh.is_insulator is None:
        return np.ones(len(mesh.edges), dtype=bool)
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    return ~(mesh.is_insulator[ii] | mesh.is_insulator[jj])


def _mesh_ni_edge_g(mesh, mat):
    """A homogeneous-silicon mesh (the diode) has mesh.ni_arr/edge_g unset
    (mesh2d.build_mesh2d only populates them when built with mat= given -
    see that module's docstring); a heterogeneous mesh (a MOSFET or MOS
    capacitor, with an oxide region) has them populated. Falling back to
    mat.ni (scalar broadcast) / mat.eps*mesh.edge_weight here means this
    solver works unchanged for both kinds of mesh."""
    ni_arr = mesh.ni_arr if mesh.ni_arr is not None else np.full(len(mesh.points), mat.ni)
    g_e = mesh.edge_g if mesh.edge_g is not None else mat.eps * mesh.edge_weight
    return ni_arr, g_e


def poisson_row_scale(mat: Material, h_typ: float) -> float:
    return mat.eps * mat.Vt / h_typ ** 2


def continuity_row_scale(mat: Material, h_typ: float) -> float:
    """Q*Dn*ni/h_typ**2 - one power of h_typ MORE than
    core/newton_solver_qf.py's 1D formula (Q*Dn*ni/h_typ). 1D's continuity
    row divides a flux-density difference by cvol_i, a control-volume
    LENGTH (~h_typ); this module's 2D row instead divides by cv_area, a
    control-volume AREA (~h_typ**2) - so matching 1D's row-scale formula
    verbatim under-compensates by one power of h_typ, leaving every row
    roughly 1/h_typ too large (caught via a finite-difference Jacobian
    check whose typical, non-pathological row magnitude came out ~1e9-1e10
    instead of O(1) even at a physically sane near-equilibrium test state -
    not a Jacobian correctness bug, since FD and analytic agreed once eps
    was chosen to resolve derivatives at that same huge scale, but a
    genuine row-conditioning bug all the same, worth fixing before trusting
    Newton's convergence on it). Matches poisson_row_scale's own h_typ**2
    convention, which was already correct for the same reason."""
    return Q * mat.Dn * mat.ni / h_typ ** 2


def _residual(U, mesh, mat, is_contact, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale):
    N = len(mesh.points)
    psi, phin, phip = unpack_qf(U, N)
    ni_arr, g_e = _mesh_ni_edge_g(mesh, mat)
    n, p = _densities(psi, phin, phip, ni_arr, mat.Vt)

    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    edge_len = np.linalg.norm(mesh.points[jj] - mesh.points[ii], axis=1)

    n_avg = 0.5 * (n[ii] + n[jj])
    p_avg = 0.5 * (p[ii] + p[jj])
    # Doping-dependent (NOT field-dependent) mobility: static per-node
    # arrays averaged to an edge value, same convention as n_avg/p_avg
    # above - see mobility_doping()'s docstring. The field-dependent
    # (velocity-saturation) Caughey-Thomas model that used to sit here
    # (mobility_field(), still defined above for future use) is
    # deliberately NOT used in this active solve path for now (2026-09-13:
    # it was making the MOSFET Ids-Vgs transfer curve worse, not better -
    # collapsing at high Vgs instead of gracefully rolling off - so it was
    # backed out pending its own separate debugging pass).
    mu_n_node, mu_p_node = _mesh_mobility_nodal(mesh, mat)
    mu_n_e = 0.5 * (mu_n_node[ii] + mu_n_node[jj])
    mu_p_e = 0.5 * (mu_p_node[ii] + mu_p_node[jj])
    Jn_e = -Q * mu_n_e * n_avg * (phin[jj] - phin[ii]) / edge_len
    Jp_e = -Q * mu_p_e * p_avg * (phip[jj] - phip[ii]) / edge_len
    In_e = Jn_e * mesh.facet_length
    Ip_e = Jp_e * mesh.facet_length
    # 2026-09-13: the semiconductor/insulator no-flux edge mask
    # (_semiconductor_edge_mask(), still defined above for a later pass) is
    # deliberately NOT applied here for now - it is physically correct (see
    # its own docstring) but made near-threshold/off-state Newton
    # continuation dramatically slower and less robust across the whole
    # sweep, not just the corner it targeted. Backed out so the on-state
    # curve stays fast and clean; the known consequence is the flat
    # off-state "leakage" floor from the contact/oxide corner edge is back
    # (tracked as a known, deferred issue - to be solved in the 1D diode
    # first per the user's own direction, then reapplied here).

    div_psi = np.zeros(N)
    np.add.at(div_psi, ii, g_e * (psi[jj] - psi[ii]))
    np.add.at(div_psi, jj, g_e * (psi[ii] - psi[jj]))

    div_n = np.zeros(N)
    np.add.at(div_n, ii, In_e)
    np.add.at(div_n, jj, -In_e)

    div_p = np.zeros(N)
    np.add.at(div_p, ii, Ip_e)
    np.add.at(div_p, jj, -Ip_e)

    # SRH denom is exactly 0 at an insulator node (n=p=ni_arr=0 there,
    # since ni_arr=0 makes _densities return 0 regardless of psi/phin/phip)
    # - guard the divide rather than let it raise/NaN; R is physically
    # moot there anyway (no recombination in an insulator).
    denom = mat.tau_p * (n + ni_arr) + mat.tau_n * (p + ni_arr)
    denom_safe = np.where(denom > 0.0, denom, 1.0)
    R = np.where(denom > 0.0, (n * p - ni_arr ** 2) / denom_safe, 0.0)

    Rpsi = (div_psi / mesh.cv_area - Q * (n - p - mesh.Cdop)) / poisson_scale
    Rn = (div_n / mesh.cv_area - Q * R) / cont_scale
    Rp = (div_p / mesh.cv_area + Q * R) / cont_scale

    # phin/phip have NO governing equation at a non-contact insulator node
    # (n=p=0 there regardless of their value, since ni_arr=0 - there is no
    # current, no recombination, nothing for a continuity equation to
    # balance). Left alone, that makes the corresponding Jacobian row
    # either identically zero (an interior oxide node with only other
    # oxide neighbors - an exactly singular row) or driven by a spurious
    # "current" from a real semiconductor neighbor that has no physical
    # meaning (a node one layer into the oxide from the interface). Pin
    # both to 0 (arbitrary but harmless, exactly mirroring how a Dirichlet
    # contact row is overwritten below) at every such node instead.
    is_oxide_free = mesh.is_insulator & ~is_contact if mesh.is_insulator is not None \
        else np.zeros(N, dtype=bool)
    Rn = np.where(is_oxide_free, phin, Rn)
    Rp = np.where(is_oxide_free, phip, Rp)

    Rpsi[is_contact] = psi[is_contact] - psi_bc
    Rn[is_contact] = phin[is_contact] - phin_bc
    Rp[is_contact] = phip[is_contact] - phip_bc

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, mesh, mat, is_contact, psi_bc, phin_bc, phip_bc,
                            poisson_scale, cont_scale):
    N = len(mesh.points)
    psi, phin, phip = unpack_qf(U, N)
    ni_arr, g_e = _mesh_ni_edge_g(mesh, mat)
    Vt = mat.Vt
    n, p = _densities(psi, phin, phip, ni_arr, Vt)

    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    edge_len = np.linalg.norm(mesh.points[jj] - mesh.points[ii], axis=1)

    n_avg = 0.5 * (n[ii] + n[jj])
    p_avg = 0.5 * (p[ii] + p[jj])
    dphin_e = phin[jj] - phin[ii]
    dphip_e = phip[jj] - phip[ii]

    # Doping-dependent (NOT field-dependent) mobility - see the matching
    # comment in _residual() above for why the field-dependent
    # (Caughey-Thomas velocity-saturation) model was backed out of this
    # active path. Depends only on the fixed mesh.Cdop array, so it needs
    # NO Jacobian term of its own (no dmu/dpsi contribution below) -
    # mu_n_e/mu_p_e here are plain constants as far as Newton's linearization
    # is concerned, exactly like the original scalar mat.mu_n/mat.mu_p case.
    mu_n_node, mu_p_node = _mesh_mobility_nodal(mesh, mat)
    mu_n_e = 0.5 * (mu_n_node[ii] + mu_n_node[jj])
    mu_p_e = 0.5 * (mu_p_node[ii] + mu_p_node[jj])
    # 2026-09-13: semiconductor/insulator no-flux edge masking backed out
    # here too - see the matching comment in _residual() above.

    Jn_e = -Q * mu_n_e * n_avg * dphin_e / edge_len
    Jp_e = -Q * mu_p_e * p_avg * dphip_e / edge_len
    In_e = Jn_e * mesh.facet_length
    Ip_e = Jp_e * mesh.facet_length

    div_psi = np.zeros(N)
    np.add.at(div_psi, ii, g_e * (psi[jj] - psi[ii]))
    np.add.at(div_psi, jj, g_e * (psi[ii] - psi[jj]))
    div_n = np.zeros(N)
    np.add.at(div_n, ii, In_e)
    np.add.at(div_n, jj, -In_e)
    div_p = np.zeros(N)
    np.add.at(div_p, ii, Ip_e)
    np.add.at(div_p, jj, -Ip_e)

    # SRH denom is exactly 0 at an insulator node (n=p=ni_arr=0 there) -
    # guard the divide; R and its derivatives are physically moot there
    # (no recombination in an insulator) and must not propagate a NaN into
    # the rest of the (otherwise perfectly well-posed) linear system.
    denom = mat.tau_p * (n + ni_arr) + mat.tau_n * (p + ni_arr)
    denom_safe = np.where(denom > 0.0, denom, 1.0)
    num = n * p - ni_arr ** 2
    R = np.where(denom > 0.0, num / denom_safe, 0.0)

    Rpsi = (div_psi / mesh.cv_area - Q * (n - p - mesh.Cdop)) / poisson_scale
    Rn = (div_n / mesh.cv_area - Q * R) / cont_scale
    Rp = (div_p / mesh.cv_area + Q * R) / cont_scale

    # See _residual()'s matching comment: phin/phip are pinned to 0 (not
    # solved via the normal continuity equation) at every non-contact
    # insulator node - otherwise either an exactly-singular all-zero
    # Jacobian row (an interior oxide node with only other oxide
    # neighbors) or a spurious coupling to a real semiconductor neighbor's
    # phin/phip with no physical basis.
    is_oxide_free = mesh.is_insulator & ~is_contact if mesh.is_insulator is not None \
        else np.zeros(N, dtype=bool)
    Rn = np.where(is_oxide_free, phin, Rn)
    Rp = np.where(is_oxide_free, phip, Rp)

    Rpsi[is_contact] = psi[is_contact] - psi_bc
    Rn[is_contact] = phin[is_contact] - phin_bc
    Rp[is_contact] = phip[is_contact] - phip_bc
    F = np.concatenate([Rpsi, Rn, Rp])

    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt
    dR_dn = np.where(denom > 0.0, (p * denom - num * mat.tau_p) / denom_safe ** 2, 0.0)
    dR_dp = np.where(denom > 0.0, (n * denom - num * mat.tau_n) / denom_safe ** 2, 0.0)

    # base_{n,p}_e are the mobility-FREE prefactors (Q*facet_length/edge_len);
    # cn_e/cp_e fold in the (now doping-dependent, psi-INDEPENDENT) mu_n_e/
    # mu_p_e per edge.
    base_n_e = Q * mesh.facet_length / edge_len
    base_p_e = Q * mesh.facet_length / edge_len
    cn_e = mu_n_e * base_n_e
    cp_e = mu_p_e * base_p_e

    # dphin/dphip derivatives - identical in form to the original constant-
    # mobility case (mu depends only on fixed doping, not phin/phip).
    dIn_dphin_i = -cn_e * ((dn_dphin[ii] / 2.0) * dphin_e - n_avg)
    dIn_dphin_j = -cn_e * ((dn_dphin[jj] / 2.0) * dphin_e + n_avg)
    dIp_dphip_i = -cp_e * ((dp_dphip[ii] / 2.0) * dphip_e - p_avg)
    dIp_dphip_j = -cp_e * ((dp_dphip[jj] / 2.0) * dphip_e + p_avg)

    # dpsi derivatives: back to the SINGLE contribution from n_avg/p_avg's
    # own psi dependence (mu is fixed w.r.t. psi now, so there is no
    # product-rule mobility term here anymore - see this module's Task-1
    # backout comment near mu_n_e/mu_p_e's definition above).
    dIn_dpsi_i = -cn_e * (dn_dpsi[ii] / 2.0) * dphin_e
    dIn_dpsi_j = -cn_e * (dn_dpsi[jj] / 2.0) * dphin_e
    dIp_dpsi_i = -cp_e * (dp_dpsi[ii] / 2.0) * dphip_e
    dIp_dpsi_j = -cp_e * (dp_dpsi[jj] / 2.0) * dphip_e

    rows_list, cols_list, data_list = [], [], []

    def add(rows, cols, vals):
        rows_list.append(rows)
        cols_list.append(cols)
        data_list.append(vals)

    # --- Poisson: per-edge Laplacian coupling, row masked to non-contact ---
    active_i = ~is_contact[ii]
    active_j = ~is_contact[jj]
    cv_i, cv_j = mesh.cv_area[ii], mesh.cv_area[jj]

    # Electron/hole continuity rows additionally exclude is_oxide_free
    # nodes (pinned to phin=0/phip=0 above, not assembled normally here).
    is_np_fixed = is_contact | is_oxide_free
    cont_active_i = ~is_np_fixed[ii]
    cont_active_j = ~is_np_fixed[jj]

    add(ii[active_i], ii[active_i], (-g_e / cv_i / poisson_scale)[active_i])
    add(ii[active_i], jj[active_i], (g_e / cv_i / poisson_scale)[active_i])
    add(jj[active_j], jj[active_j], (-g_e / cv_j / poisson_scale)[active_j])
    add(jj[active_j], ii[active_j], (g_e / cv_j / poisson_scale)[active_j])

    # --- Poisson: per-node local charge derivative, non-contact nodes only ---
    node = np.arange(N)
    free = node[~is_contact]
    add(free, free,
        (-Q * (dn_dpsi[free] - dp_dpsi[free]) / poisson_scale))
    add(free, N + free, (-Q * dn_dphin[free] / poisson_scale))
    add(free, 2 * N + free, (Q * dp_dphip[free] / poisson_scale))

    # --- Electron continuity: per-edge current derivative ---
    r_n_i, r_n_j = N + ii, N + jj
    add(r_n_i[cont_active_i], ii[cont_active_i], (dIn_dpsi_i / cv_i / cont_scale)[cont_active_i])
    add(r_n_i[cont_active_i], jj[cont_active_i], (dIn_dpsi_j / cv_i / cont_scale)[cont_active_i])
    add(r_n_i[cont_active_i], N + ii[cont_active_i], (dIn_dphin_i / cv_i / cont_scale)[cont_active_i])
    add(r_n_i[cont_active_i], N + jj[cont_active_i], (dIn_dphin_j / cv_i / cont_scale)[cont_active_i])

    add(r_n_j[cont_active_j], ii[cont_active_j], (-dIn_dpsi_i / cv_j / cont_scale)[cont_active_j])
    add(r_n_j[cont_active_j], jj[cont_active_j], (-dIn_dpsi_j / cv_j / cont_scale)[cont_active_j])
    add(r_n_j[cont_active_j], N + ii[cont_active_j], (-dIn_dphin_i / cv_j / cont_scale)[cont_active_j])
    add(r_n_j[cont_active_j], N + jj[cont_active_j], (-dIn_dphin_j / cv_j / cont_scale)[cont_active_j])

    # --- Electron continuity: per-node recombination derivative ---
    free_np = node[~is_np_fixed]
    r_n_free = N + free_np
    add(r_n_free, free_np, -Q * (dR_dn[free_np] * dn_dpsi[free_np] + dR_dp[free_np] * dp_dpsi[free_np]) / cont_scale)
    add(r_n_free, N + free_np, -Q * dR_dn[free_np] * dn_dphin[free_np] / cont_scale)
    add(r_n_free, 2 * N + free_np, -Q * dR_dp[free_np] * dp_dphip[free_np] / cont_scale)

    # --- Hole continuity: per-edge current derivative ---
    r_p_i, r_p_j = 2 * N + ii, 2 * N + jj
    add(r_p_i[cont_active_i], ii[cont_active_i], (dIp_dpsi_i / cv_i / cont_scale)[cont_active_i])
    add(r_p_i[cont_active_i], jj[cont_active_i], (dIp_dpsi_j / cv_i / cont_scale)[cont_active_i])
    add(r_p_i[cont_active_i], 2 * N + ii[cont_active_i], (dIp_dphip_i / cv_i / cont_scale)[cont_active_i])
    add(r_p_i[cont_active_i], 2 * N + jj[cont_active_i], (dIp_dphip_j / cv_i / cont_scale)[cont_active_i])

    add(r_p_j[cont_active_j], ii[cont_active_j], (-dIp_dpsi_i / cv_j / cont_scale)[cont_active_j])
    add(r_p_j[cont_active_j], jj[cont_active_j], (-dIp_dpsi_j / cv_j / cont_scale)[cont_active_j])
    add(r_p_j[cont_active_j], 2 * N + ii[cont_active_j], (-dIp_dphip_i / cv_j / cont_scale)[cont_active_j])
    add(r_p_j[cont_active_j], 2 * N + jj[cont_active_j], (-dIp_dphip_j / cv_j / cont_scale)[cont_active_j])

    # --- Hole continuity: per-node recombination derivative ---
    r_p_free = 2 * N + free_np
    add(r_p_free, free_np, Q * (dR_dn[free_np] * dn_dpsi[free_np] + dR_dp[free_np] * dp_dpsi[free_np]) / cont_scale)
    add(r_p_free, N + free_np, Q * dR_dn[free_np] * dn_dphin[free_np] / cont_scale)
    add(r_p_free, 2 * N + free_np, Q * dR_dp[free_np] * dp_dphip[free_np] / cont_scale)

    # --- phin/phip identity-pinning rows at non-contact insulator nodes
    # (see the matching Rn/Rp override above) - these rows have NO other
    # contribution (excluded from every block above via is_np_fixed), so a
    # plain diagonal 1 is exact, not just a pivoting safety net. ---
    oxide_free_idx = node[is_oxide_free]
    add(N + oxide_free_idx, N + oxide_free_idx, np.ones(len(oxide_free_idx)))
    add(2 * N + oxide_free_idx, 2 * N + oxide_free_idx, np.ones(len(oxide_free_idx)))

    interior_rows = np.concatenate(rows_list)
    interior_cols = np.concatenate(cols_list)
    interior_data = np.concatenate(data_list)

    # --- Dirichlet rows: scaled-identity, same pivoting-safety trick as
    # core/newton_solver_qf.py (diag = max(1.0, largest existing entry in
    # that column) so equilibration/pivoting never dilutes it). ---
    contact_idx = node[is_contact]
    dirichlet_idx = np.concatenate([contact_idx, N + contact_idx, 2 * N + contact_idx])
    dirichlet_rows, dirichlet_cols, dirichlet_data = [], [], []
    for i in dirichlet_idx:
        col_mask = interior_cols == i
        local_max = np.max(np.abs(interior_data[col_mask])) if np.any(col_mask) else 0.0
        dirichlet_rows.append(i)
        dirichlet_cols.append(i)
        dirichlet_data.append(max(1.0, local_max))

    rows = np.concatenate([interior_rows, dirichlet_rows])
    cols = np.concatenate([interior_cols, dirichlet_cols])
    data = np.concatenate([interior_data, dirichlet_data])
    J = sp.coo_matrix((data, (rows, cols)), shape=(3 * N, 3 * N)).tocsc()
    return F, J


def newton_solve_2d(mesh, mat: Material, bias_by_contact, f_tol=1e-9, maxiter=50, verbose=False,
                     psi_init=None, phin_init=None, phip_init=None,
                     psi_bc_override=None, phin_bc_override=None, phip_bc_override=None):
    """Solve the 2D QF system with each contact held at the voltage given
    in `bias_by_contact` ({contact_name: volts} - any contact not listed
    defaults to 0V). Returns a dict matching core/newton_solver_qf.py's
    shape (psi, n, p, phin, phip, iters).

    This assumes every contact is an ideal OHMIC contact on real
    semiconductor (mass-action/charge-neutrality BC via contact_values) by
    default - correct for a diode's anode/cathode or a MOSFET's source/
    drain/body, but WRONG for an ideal-metal gate sitting on an insulator
    (see solver2d/poisson2d_mos.py's own gate BC: psi_bulk + (VG-V_FB), not
    an ohmic relation, which is meaningless where ni=0 anyway). For any
    such contact, pass its already-computed Dirichlet value(s) via
    `psi_bc_override`/`phin_bc_override`/`phip_bc_override`
    ({contact_name: value}) to bypass the ohmic-BC computation entirely for
    that contact - the driver (e.g. main2d_mosfet_sweep.py) is expected to
    compute the gate's psi_bc itself (mos.mos_analytic.flatband_voltage,
    same as the MOS capacitor) and pass it through here.

    psi_init/phin_init/phip_init, if given, warm-start Newton from a
    previous bias point's converged solution (see main2d_sweep.py) instead
    of the fresh charge-neutral-plus-tiny-perturbation guess - the same
    idea 1D's own voltage_sweep uses, needed here because a handful of
    reverse-bias points failed to converge from a cold start every time
    (e.g. -0.9V, -0.7V while -0.8V/-1.0V converged fine) even though this
    is otherwise the mild, first-example doping case. If a warm start
    still fails to converge, retry once from the fresh cold start (the one
    piece of core/newton_solver_qf.py's fuller Gummel-restart robustness
    layer ported here - a full 2D Gummel solver is still deferred)."""
    N = len(mesh.points)
    Vt = mat.Vt
    ni_arr, _ = _mesh_ni_edge_g(mesh, mat)
    psi_bc_override = psi_bc_override or {}
    phin_bc_override = phin_bc_override or {}
    phip_bc_override = phip_bc_override or {}

    boundary_bc_type = np.array(mesh.boundary_bc_type)
    is_contact_boundary = np.array([bc.startswith("contact:") for bc in boundary_bc_type])
    contact_point_idx = mesh.boundary_point_index[is_contact_boundary]
    contact_names = [bc.split(":", 1)[1] for bc in boundary_bc_type[is_contact_boundary]]

    is_contact_full = np.zeros(N, dtype=bool)
    is_contact_full[contact_point_idx] = True

    # ni_arr is 0 at an insulator node (mesh.is_insulator), which would
    # divide-by-zero into a NaN "equilibrium potential" there (moot anyway,
    # since that node is never a free/non-contact unknown in the physics
    # sense - it's either a Dirichlet override, e.g. a gate on oxide, or an
    # interior insulator node with no equilibrium relation to speak of).
    # Substitute a safe placeholder ni there so this stays finite and
    # doesn't poison the cold-start guess with a NaN.
    is_insulator = mesh.is_insulator if mesh.is_insulator is not None else np.zeros(N, dtype=bool)
    ni_safe_eq = np.where(is_insulator, 1.0, ni_arr)
    psi_eq = np.where(is_insulator, 0.0, equilibrium_bulk_potential_arr(Vt, ni_safe_eq, mesh.Cdop))

    psi_bc = np.empty(len(contact_point_idx))
    phin_bc = np.empty(len(contact_point_idx))
    phip_bc = np.empty(len(contact_point_idx))
    for k, (pt, name) in enumerate(zip(contact_point_idx, contact_names)):
        v_applied = bias_by_contact.get(name, 0.0)
        if name in psi_bc_override:
            psi_bc[k] = psi_bc_override[name]
            phin_bc[k] = phin_bc_override[name]
            phip_bc[k] = phip_bc_override[name]
        else:
            n_bc, p_bc = contact_values(mat, mesh.Cdop[pt], ni=ni_arr[pt])
            psi_bc[k] = psi_eq[pt] + v_applied
            phin_bc[k] = psi_bc[k] - Vt * np.log(n_bc / ni_arr[pt])
            phip_bc[k] = psi_bc[k] + Vt * np.log(p_bc / ni_arr[pt])

    def _cold_start():
        # phin0=phip0=0 EXACTLY everywhere (the natural equilibrium guess)
        # makes every edge's dphin_e/dphip_e term vanish identically, which
        # singles out an actually-singular direction in the Jacobian - the
        # same "flat interior at Va=0" critical point core/newton_solver_qf.py's
        # own comments describe (there recovered via a Gummel-restart;
        # confirmed here directly via spsolve raising MatrixRankWarning:
        # Matrix is exactly singular at the very first Newton step). Break
        # the EXACT flat degeneracy with a tiny, deterministic, physically
        # negligible perturbation (~1e-6*Vt, six orders of magnitude below
        # any physically meaningful potential) so Newton has a well-posed
        # direction to move in from the first step.
        rng = np.random.default_rng(0)
        psi0 = psi_eq.copy()
        psi0[contact_point_idx] = psi_bc
        phin0 = 1e-6 * Vt * rng.standard_normal(N)
        phin0[contact_point_idx] = phin_bc
        phip0 = 1e-6 * Vt * rng.standard_normal(N)
        phip0[contact_point_idx] = phip_bc
        return psi0, phin0, phip0

    # Use the mesh generator's own intended nominal minimum spacing, NOT
    # np.min(actual edge lengths) - mesh2d/pointcloud.py's jitter (needed to
    # break Delaunay-of-a-grid degeneracy, see its own docstring) can by
    # design push an isolated edge a little below h_min_cm, and since this
    # single global h_typ sets EVERY row's normalization for the whole
    # system, one such outlier edge anywhere throws off poisson_scale/
    # cont_scale for the entire solve, not just its own two nodes (caught
    # via a finite-difference Jacobian check whose residual blew up to
    # ~1e11-1e12 well away from any Dirichlet or degenerate-mesh row before
    # this fix).
    h_typ = mesh.h_min_cm
    poisson_scale = poisson_row_scale(mat, h_typ)
    cont_scale = continuity_row_scale(mat, h_typ)

    def _run_newton(psi0, phin0, phip0):
        psi0 = psi0.copy(); phin0 = phin0.copy(); phip0 = phip0.copy()
        psi0[contact_point_idx] = psi_bc
        phin0[contact_point_idx] = phin_bc
        phip0[contact_point_idx] = phip_bc
        U = np.concatenate([psi0, phin0, phip0])

        F, J = _residual_and_jacobian(U, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                       poisson_scale, cont_scale)
        res_norm = np.max(np.abs(F))

        it = 0
        tiny_step_streak = 0
        for it in range(1, maxiter + 1):
            if res_norm < f_tol:
                break
            delta = equilibrated_spsolve(J, -F)
            delta[N:3 * N] = np.clip(delta[N:3 * N], -MAX_QF_STEP, MAX_QF_STEP)

            step = 1.0
            for _ in range(20):
                U_try = U + step * delta
                F_try = _residual(U_try, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                   poisson_scale, cont_scale)
                res_try = np.max(np.abs(F_try))
                if np.isfinite(res_try) and res_try < res_norm * (1 - 1e-4 * step):
                    break
                step *= 0.5
            else:
                U_try, res_try = U, res_norm

            U = U_try
            F, J = _residual_and_jacobian(U, mesh, mat, is_contact_full, psi_bc, phin_bc, phip_bc,
                                           poisson_scale, cont_scale)
            res_norm = np.max(np.abs(F))
            if verbose:
                print(f"  Newton(QF,2D) it {it}: |F|_inf={res_norm:.3e}  step={step:.3g}")

            tiny_step_streak = tiny_step_streak + 1 if step < 1e-4 else 0
            if tiny_step_streak >= 2:
                break
        return U, res_norm, it

    if psi_init is None:
        U, res_norm, it = _run_newton(*_cold_start())
    else:
        U, res_norm, it = _run_newton(psi_init, phin_init, phip_init)
        if res_norm > 1.0:
            # A warm start can inherit a bad basin from its source point
            # (mirroring core/newton_solver_qf.py's own Gummel-restart
            # rationale) - retry from the fresh cold start rather than
            # accepting a stuck, wrong answer.
            U_retry, res_retry, it_retry = _run_newton(*_cold_start())
            if res_retry < res_norm:
                U, res_norm, it = U_retry, res_retry, it_retry

    if res_norm > 1.0:
        warnings.warn(
            f"Newton(QF,2D) solve did not converge at bias_by_contact={bias_by_contact} "
            f"(|F|_inf={res_norm:.3e} at iteration {it}) even after a cold-start retry.")

    psi, phin, phip = unpack_qf(U, N)
    n, p = _densities(psi, phin, phip, ni_arr, Vt)
    return {"psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
            "iters": it, "res_norm": res_norm}
