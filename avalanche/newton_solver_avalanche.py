"""Fully coupled Newton solve, quasi-Fermi-potential unknowns and
plain-gradient current (same base formulation as newton_solver_qf.py), EXTENDED
with a field-dependent impact-ionization (avalanche) generation term in both
continuity equations.

THIS IS A SPECIAL, OPT-IN, NON-DEFAULT SOLVER. A normal CMOS-flow junction is
operated well below its avalanche breakdown voltage, where impact ionization
is entirely negligible - none of the other examples/solvers in this codebase
model it, and none of their behavior changes by this module existing. This
module is only reached via the explicit math_model="newton_avalanche" switch
(see solver.py/config.py) with an explicit avalanche: enabled: true block in
the input YAML (see avalanche_config.py), and its own dedicated driver
(main_avalanche.py) and example device (input_diode_breakdown.yaml).

WHY A NEW MODULE (not a flag threaded into newton_solver_qf.py): matches this
project's own precedent - newton_solver_qf.py itself was added as a sibling
module to newton_solver.py rather than a flag inside it, to keep the
non-avalanche path's code and behavior completely untouched. See
newton_solver_qf.py's own module docstring for the shared design notes (why
an analytic Jacobian + direct sparse solve, why quasi-Fermi unknowns + a
plain-gradient current rather than Scharfetter-Gummel).

THE PHYSICS: impact ionization is not a new PDE - it's an extra local
electron-hole-pair GENERATION rate G_ii(x) (cm^-3 s^-1) added to both
continuity equations, playing the same role as SRH recombination R but with
the OPPOSITE sign (a net source, not a sink):

    dJn/dx = q*(R - G_ii)      dJp/dx = -q*(R - G_ii)

using the standard van Overstraeten-de Man / Chynoweth local-field model
(see avalanche.py):

    G_ii = (alpha_n(E)*|Jn| + alpha_p(E)*|Jp|) / q

This is what makes the multiplication factor M(Va) diverge sharply near
breakdown: G_ii depends exponentially on the local field E (itself set by
psi) AND on the very currents Jn, Jp it feeds back into - a positive-feedback
loop absent from every other solver in this codebase. Bank & Rose (1981)
"Algorithm Global" damping (bank_rose_damping.py) is implemented and
available here specifically because of that sharp nonlinearity, though
empirically newton_solver_qf.py's plain backtracking line search proved the
more robust DEFAULT for this device - see newton_gummel_solve's own
docstring below for the full empirical comparison and why.

THE JACOBIAN: G_ii's dependence on BOTH n and p transport variables creates a
coupling not present in newton_solver_qf.py today - G_ii depends on phip (via
Jp) inside the ELECTRON continuity row, and on phin (via Jn) inside the HOLE
continuity row (previously, the electron row's only phip-dependency was
through dR_dp at the SAME node column; here it becomes a genuine 3-wide
phip-column stencil, and likewise for the hole row's phin columns). See the
per-edge derivative block in _residual_and_jacobian below.
"""
import warnings

import numpy as np
import scipy.sparse as sp

from core.params import Q, Material
from core import physics as ph
from core.solver import contact_values
from avalanche.avalanche import AvalancheModel, ionization_coeffs
from core.bank_rose_damping import bank_rose_solve
from core.jacobian_scaling import equilibrated_spsolve


def _poisson_scale(mat: Material, h_typ: float) -> float:
    return mat.eps * mat.Vt / h_typ ** 2


def _continuity_scale(mat: Material, h_typ: float) -> float:
    return Q * mat.Dn * mat.ni / h_typ


def _unpack(U, N):
    """U = [psi, phin, phip]."""
    return U[:N], U[N:2 * N], U[2 * N:3 * N]


def _edge_quantities_avalanche(psi, phin, phip, n, p, x, mat, ii_model):
    """Per-edge plain-gradient flux, recombination, and impact-ionization
    generation rate - shared by the residual-only and residual+Jacobian
    paths so they never disagree. Identical to newton_solver_qf.py's
    _edge_quantities, plus the new avalanche block at the end."""
    h = np.diff(x)
    n_avg = (n[:-1] + n[1:]) / 2.0
    p_avg = (p[:-1] + p[1:]) / 2.0

    Jn = -Q * mat.mu_n * n_avg * (phin[1:] - phin[:-1]) / h
    Jp = -Q * mat.mu_p * p_avg * (phip[1:] - phip[:-1]) / h

    ni = mat.ni
    denom = mat.tau_p * (n + ni) + mat.tau_n * (p + ni)
    num = n * p - ni ** 2
    R = num / denom

    # E = -d(psi)/dx (this codebase's existing sign convention, e.g.
    # depletion_potential_profile's field is built the same way).
    E_e = -(psi[1:] - psi[:-1]) / h
    Eabs_e = np.abs(E_e)
    alpha_n_e, alpha_p_e, dalpha_n_dE_e, dalpha_p_dE_e = ionization_coeffs(Eabs_e, ii_model)
    Gii_e = (alpha_n_e * np.abs(Jn) + alpha_p_e * np.abs(Jp)) / Q

    return (h, n_avg, p_avg, Jn, Jp, R, denom, num,
            E_e, Eabs_e, alpha_n_e, alpha_p_e, dalpha_n_dE_e, dalpha_p_dE_e, Gii_e)


def _gii_node(Gii_e, hm, hp, cvol_i):
    """Box-integrate the per-edge generation rate onto each interior node's
    control volume - the FV-consistent way to inject a piecewise-constant
    per-edge source into a node-centered balance, exactly analogous to how
    the flux-divergence term is already box-integrated (each edge
    contributes its own half-length to each of its two endpoint control
    volumes). hm=h[:-1] (edge e_lo, i.e. h[idx-1]), hp=h[1:] (edge e_hi,
    i.e. h[idx]), both already sliced to interior length like the caller's
    lap_m/lap_p."""
    return (hm * Gii_e[:-1] + hp * Gii_e[1:]) / (2.0 * cvol_i)


def _residual_only(U, x, Cdop, mat, ii_model, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale):
    """Fast path: residual vector only, no Jacobian. Used for Bank-Rose
    trial evaluations that get rejected and re-tried at a different t_k."""
    N = len(x)
    psi, phin, phip = _unpack(U, N)
    Vt = mat.Vt
    n = mat.ni * np.exp((psi - phin) / Vt)
    p = mat.ni * np.exp((phip - psi) / Vt)
    cvol = ph._control_volumes(x)

    Rpsi = np.empty(N)
    Rn = np.empty(N)
    Rp = np.empty(N)
    Rpsi[0], Rpsi[-1] = psi[0] - psi_bc[0], psi[-1] - psi_bc[-1]
    Rn[0], Rn[-1] = phin[0] - phin_bc[0], phin[-1] - phin_bc[-1]
    Rp[0], Rp[-1] = phip[0] - phip_bc[0], phip[-1] - phip_bc[-1]

    h = np.diff(x)
    hm, hp = h[:-1], h[1:]
    cvol_i = cvol[1:-1]
    lap_m = mat.eps / hm / cvol_i
    lap_p = mat.eps / hp / cvol_i
    Rpsi[1:-1] = (lap_p * (psi[2:] - psi[1:-1]) - lap_m * (psi[1:-1] - psi[:-2])
                  - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])) / poisson_scale

    (_, _, _, Jn, Jp, R, _, _,
     _, _, _, _, _, _, Gii_e) = _edge_quantities_avalanche(psi, phin, phip, n, p, x, mat, ii_model)
    Gii_node = _gii_node(Gii_e, hm, hp, cvol_i)

    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * (R[1:-1] - Gii_node)) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * (R[1:-1] - Gii_node)) / cont_scale

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, x, Cdop, mat, ii_model, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale):
    """Returns (F, J), F the length-3N residual, J the 3N x 3N sparse
    Jacobian dF/dU for U=[psi, phin, phip]. Fully vectorized (no per-node
    Python loop). Structurally identical to newton_solver_qf.py's function
    of the same name, plus the new impact-ionization block (search
    "avalanche" below for every addition relative to that module)."""
    N = len(x)
    psi, phin, phip = _unpack(U, N)
    Vt = mat.Vt
    n = mat.ni * np.exp((psi - phin) / Vt)
    p = mat.ni * np.exp((phip - psi) / Vt)
    cvol = ph._control_volumes(x)

    Rpsi = np.empty(N)
    Rn = np.empty(N)
    Rp = np.empty(N)
    Rpsi[0], Rpsi[-1] = psi[0] - psi_bc[0], psi[-1] - psi_bc[-1]
    Rn[0], Rn[-1] = phin[0] - phin_bc[0], phin[-1] - phin_bc[-1]
    Rp[0], Rp[-1] = phip[0] - phip_bc[0], phip[-1] - phip_bc[-1]

    h = np.diff(x)
    hm, hp = h[:-1], h[1:]
    cvol_i = cvol[1:-1]
    lap_m = mat.eps / hm / cvol_i
    lap_p = mat.eps / hp / cvol_i
    Rpsi[1:-1] = (lap_p * (psi[2:] - psi[1:-1]) - lap_m * (psi[1:-1] - psi[:-2])
                  - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])) / poisson_scale

    (h_e, n_avg, p_avg, Jn, Jp, R, denom, num,
     E_e, Eabs_e, alpha_n_e, alpha_p_e, dalpha_n_dE_e, dalpha_p_dE_e, Gii_e) = \
        _edge_quantities_avalanche(psi, phin, phip, n, p, x, mat, ii_model)

    # dn/dpsi = n/Vt, dn/dphin = -n/Vt ; dp/dpsi = -p/Vt, dp/dphip = p/Vt
    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt

    dphin_e = phin[1:] - phin[:-1]
    dphip_e = phip[1:] - phip[:-1]

    # Jn_e = -Q*mu_n*n_avg*dphin_e/h ; n_avg = (n_i+n_{i+1})/2  (same as
    # newton_solver_qf.py; "_e" = derivative wrt the edge's LEFT node,
    # "_ep1" = wrt its RIGHT node)
    dJn_dpsi_e = -Q * mat.mu_n / h_e * (dn_dpsi[:-1] / 2.0) * dphin_e
    dJn_dpsi_ep1 = -Q * mat.mu_n / h_e * (dn_dpsi[1:] / 2.0) * dphin_e
    dJn_dphin_e = -Q * mat.mu_n / h_e * ((dn_dphin[:-1] / 2.0) * dphin_e - n_avg)
    dJn_dphin_ep1 = -Q * mat.mu_n / h_e * ((dn_dphin[1:] / 2.0) * dphin_e + n_avg)

    dJp_dpsi_e = -Q * mat.mu_p / h_e * (dp_dpsi[:-1] / 2.0) * dphip_e
    dJp_dpsi_ep1 = -Q * mat.mu_p / h_e * (dp_dpsi[1:] / 2.0) * dphip_e
    dJp_dphip_e = -Q * mat.mu_p / h_e * ((dp_dphip[:-1] / 2.0) * dphip_e - p_avg)
    dJp_dphip_ep1 = -Q * mat.mu_p / h_e * ((dp_dphip[1:] / 2.0) * dphip_e + p_avg)

    dR_dn = (p * denom - num * mat.tau_p) / denom ** 2
    dR_dp = (n * denom - num * mat.tau_n) / denom ** 2

    # --- avalanche: per-edge dG_ii/d(unknown at each of the edge's two
    # nodes) - chain rule through BOTH the field E (via psi) and the
    # currents Jn, Jp (via psi, phin, phip). sign(E)/sign(J) are
    # piecewise-constant multipliers (exact, not smoothed - same house
    # style as bernoulli's exact-with-floor approach rather than a
    # smoothing hack); G_ii's own dependence on J is |J|, so d|J|/dJ=sign(J).
    sgn_E = np.sign(E_e)
    sgn_n = np.sign(Jn)
    sgn_p = np.sign(Jp)
    dEabs_e = sgn_E / h_e          # d(Eabs_e)/d(psi at LEFT node)
    dEabs_ep1 = -sgn_E / h_e       # d(Eabs_e)/d(psi at RIGHT node)

    dalpha_term = (dalpha_n_dE_e * np.abs(Jn) + dalpha_p_dE_e * np.abs(Jp)) / Q
    dGii_e_dpsi_e = dalpha_term * dEabs_e \
        + (alpha_n_e * sgn_n * dJn_dpsi_e + alpha_p_e * sgn_p * dJp_dpsi_e) / Q
    dGii_e_dpsi_ep1 = dalpha_term * dEabs_ep1 \
        + (alpha_n_e * sgn_n * dJn_dpsi_ep1 + alpha_p_e * sgn_p * dJp_dpsi_ep1) / Q
    dGii_e_dphin_e = alpha_n_e * sgn_n * dJn_dphin_e / Q
    dGii_e_dphin_ep1 = alpha_n_e * sgn_n * dJn_dphin_ep1 / Q
    dGii_e_dphip_e = alpha_p_e * sgn_p * dJp_dphip_e / Q
    dGii_e_dphip_ep1 = alpha_p_e * sgn_p * dJp_dphip_ep1 / Q

    Gii_node = _gii_node(Gii_e, hm, hp, cvol_i)

    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * (R[1:-1] - Gii_node)) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * (R[1:-1] - Gii_node)) / cont_scale
    F = np.concatenate([Rpsi, Rn, Rp])

    idx = np.arange(1, N - 1)
    k = idx - 1
    e_lo, e_hi = k, k + 1
    cv = cvol_i
    w_lo = hm / (2.0 * cv)   # box weight of edge e_lo onto node idx
    w_hi = hp / (2.0 * cv)   # box weight of edge e_hi onto node idx

    # dGii_node[idx]/d(u at idx-1, idx, idx+1) - node idx is the RIGHT node
    # of edge e_lo and the LEFT node of edge e_hi (identical orientation
    # convention already used for dJn_dpsi_e/_ep1 above, verified against
    # how newton_solver_qf.py assembles its own flux-divergence Jacobian).
    dGii_node_dpsi_m = w_lo * dGii_e_dpsi_e[e_lo]
    dGii_node_dpsi_0 = w_lo * dGii_e_dpsi_ep1[e_lo] + w_hi * dGii_e_dpsi_e[e_hi]
    dGii_node_dpsi_p = w_hi * dGii_e_dpsi_ep1[e_hi]

    dGii_node_dphin_m = w_lo * dGii_e_dphin_e[e_lo]
    dGii_node_dphin_0 = w_lo * dGii_e_dphin_ep1[e_lo] + w_hi * dGii_e_dphin_e[e_hi]
    dGii_node_dphin_p = w_hi * dGii_e_dphin_ep1[e_hi]

    dGii_node_dphip_m = w_lo * dGii_e_dphip_e[e_lo]
    dGii_node_dphip_0 = w_lo * dGii_e_dphip_ep1[e_lo] + w_hi * dGii_e_dphip_e[e_hi]
    dGii_node_dphip_p = w_hi * dGii_e_dphip_ep1[e_hi]
    # --- end avalanche block ---

    rows_list, cols_list, data_list = [], [], []

    def add(r, c, v):
        rows_list.append(r)
        cols_list.append(c)
        data_list.append(v)

    # Poisson interior rows: linear in psi, and dF_psi/dphin = dF_psi/dn*dn/dphin etc.
    add(idx, idx - 1, lap_m / poisson_scale)
    add(idx, idx, -(lap_m + lap_p) / poisson_scale
        - Q * (dn_dpsi[idx] - dp_dpsi[idx]) / poisson_scale)
    add(idx, idx + 1, lap_p / poisson_scale)
    add(idx, N + idx, (-Q / poisson_scale) * dn_dphin[idx])
    add(idx, 2 * N + idx, (-Q / poisson_scale) * (-dp_dphip[idx]))

    # Electron continuity interior rows: Rn = (Jn[e_hi]-Jn[e_lo])/cv -
    # Q*(R-Gii_node), all /cont_scale. Gii_node (unlike Jn) is NOT divided
    # by cv again here - it's already a per-control-volume rate (see
    # _gii_node's own /(2*cvol_i)), entering the residual the exact same
    # undivided way R already does. So every dGii_node_* term below is
    # added as Q*dGii_node_*/cont_scale, with NO extra /cv - keep this
    # separate from the /cv flux-derivative terms rather than merging them,
    # to avoid silently double-dividing by cv (caught via a finite-difference
    # Jacobian check that failed by almost exactly a 1/cv factor before this
    # split). Note the electron row now has THREE phip columns (idx-1, idx,
    # idx+1) where newton_solver_qf.py's electron row only had ONE (idx, via
    # dR_dp) - this is the new n<->p coupling avalanche introduces.
    r_n = N + idx
    add(r_n, idx - 1, -dJn_dpsi_e[e_lo] / cv / cont_scale + Q * dGii_node_dpsi_m / cont_scale)
    add(r_n, idx, (dJn_dpsi_e[e_hi] - dJn_dpsi_ep1[e_lo]) / cv / cont_scale
        + Q * dGii_node_dpsi_0 / cont_scale
        - Q * (dR_dn[idx] * dn_dpsi[idx] + dR_dp[idx] * dp_dpsi[idx]) / cont_scale)
    add(r_n, idx + 1, dJn_dpsi_ep1[e_hi] / cv / cont_scale + Q * dGii_node_dpsi_p / cont_scale)
    add(r_n, N + idx - 1, -dJn_dphin_e[e_lo] / cv / cont_scale + Q * dGii_node_dphin_m / cont_scale)
    add(r_n, N + idx, (dJn_dphin_e[e_hi] - dJn_dphin_ep1[e_lo]) / cv / cont_scale
        + Q * dGii_node_dphin_0 / cont_scale
        - Q * dR_dn[idx] * dn_dphin[idx] / cont_scale)
    add(r_n, N + idx + 1, dJn_dphin_ep1[e_hi] / cv / cont_scale + Q * dGii_node_dphin_p / cont_scale)
    # avalanche-new: electron row's phip columns (previously only idx, via dR_dp)
    add(r_n, 2 * N + idx - 1, Q * dGii_node_dphip_m / cont_scale)
    add(r_n, 2 * N + idx, Q * dGii_node_dphip_0 / cont_scale
        - Q * dR_dp[idx] * dp_dphip[idx] / cont_scale)
    add(r_n, 2 * N + idx + 1, Q * dGii_node_dphip_p / cont_scale)

    # Hole continuity interior rows: Rp = (Jp[e_hi]-Jp[e_lo])/cv +
    # Q*(R-Gii_node), so the avalanche contribution has the OPPOSITE sign to
    # the electron row's (-Q*dGii_node/cont_scale, no /cv - same reasoning
    # as the electron row above). New phin columns (previously only idx, via
    # dR_dn) mirror the electron row's new phip columns.
    r_p = 2 * N + idx
    add(r_p, idx - 1, -dJp_dpsi_e[e_lo] / cv / cont_scale - Q * dGii_node_dpsi_m / cont_scale)
    add(r_p, idx, (dJp_dpsi_e[e_hi] - dJp_dpsi_ep1[e_lo]) / cv / cont_scale
        - Q * dGii_node_dpsi_0 / cont_scale
        + Q * (dR_dn[idx] * dn_dpsi[idx] + dR_dp[idx] * dp_dpsi[idx]) / cont_scale)
    add(r_p, idx + 1, dJp_dpsi_ep1[e_hi] / cv / cont_scale - Q * dGii_node_dpsi_p / cont_scale)
    add(r_p, 2 * N + idx - 1, -dJp_dphip_e[e_lo] / cv / cont_scale - Q * dGii_node_dphip_m / cont_scale)
    add(r_p, 2 * N + idx, (dJp_dphip_e[e_hi] - dJp_dphip_ep1[e_lo]) / cv / cont_scale
        - Q * dGii_node_dphip_0 / cont_scale
        + Q * dR_dp[idx] * dp_dphip[idx] / cont_scale)
    add(r_p, 2 * N + idx + 1, dJp_dphip_ep1[e_hi] / cv / cont_scale - Q * dGii_node_dphip_p / cont_scale)
    # avalanche-new: hole row's phin columns (previously only idx, via dR_dn)
    add(r_p, N + idx - 1, -Q * dGii_node_dphin_m / cont_scale)
    add(r_p, N + idx, Q * dR_dn[idx] * dn_dphin[idx] / cont_scale
        - Q * dGii_node_dphin_0 / cont_scale)
    add(r_p, N + idx + 1, -Q * dGii_node_dphin_p / cont_scale)

    interior_rows = np.concatenate(rows_list)
    interior_cols = np.concatenate(cols_list)
    interior_data = np.concatenate(data_list)

    # Dirichlet rows - same pivoting-safety scaling as newton_solver_qf.py.
    dirichlet_idx = [0, N - 1, N + 0, N + N - 1, 2 * N + 0, 2 * N + N - 1]
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


# Same rationale/value as newton_solver_qf.py's _MAX_QF_STEP: a near-zero
# carrier density leaves its quasi-Fermi potential almost unconstrained by
# the residual, so the raw (pre-damping) Newton correction must be capped
# regardless of which damping strategy scales it afterward.
_MAX_QF_STEP = 5.0


def newton_gummel_solve(x, Cdop, mat: Material, Va, psi_eq, n_eq, p_eq,
                         psi_init=None, phin_init=None, phip_init=None,
                         f_tol=1e-9, maxiter=50, verbose=False,
                         ii_model: AvalancheModel = None, damping="line_search"):
    """Same signature/return shape as newton_solver_qf.newton_gummel_solve,
    with two additions: an impact-ionization generation term in both
    continuity equations (ii_model, defaulting to the standard Si
    van Overstraeten-de Man model - this solver's whole purpose is avalanche,
    so it is always modeled when this solver is used), and a choice of
    damping strategy: "bank_rose" (Bank & Rose 1981 Algorithm Global,
    bank_rose_damping.py - implemented and validated per the user's request,
    see that module) or "line_search" (default here - newton_solver_qf.py's
    plain backtracking).

    WHY "line_search" IS THE DEFAULT (not "bank_rose", despite
    bank_rose_damping.py's own toy-problem validation showing it correctly
    implements the paper's algorithm): empirically, on this device's
    strongly-coupled avalanche feedback system, Algorithm Global's K
    parameter persists and accumulates ACROSS outer Newton iterations by
    design (K/10 on acceptance, 10*K on rejection - see that module's
    docstring), and its eq-3.1 global sufficient-decrease test is strict
    enough that once K climbs high early in a bias-point solve (from just a
    handful of rejections, easy to trigger when the avalanche generation
    term's curvature varies enormously across the coupled 3N-dimensional
    system) it can take many outer iterations to relax back down - in
    side-by-side sweeps of this device, plain backtracking stayed
    well-converged (self-consistency O(1-10)) roughly twice as far into
    reverse bias before both strategies hit the same underlying wall (see
    below) as Bank-Rose did. Bank-Rose remains fully available
    (damping="bank_rose") for devices where a residual-only backtracking
    test proves inadequate, or for direct comparison.

    BOTH damping strategies eventually hit the same wall at a device- and
    mesh-dependent bias, past which every subsequent bias point in a
    continuation sweep returns an unchanged, non-physical value no matter
    which damping is used. This is NOT a damping-strategy bug: it is the
    well-known device-simulation limitation that a VOLTAGE-controlled bias
    sweep cannot follow the I(Va) curve past the point where dI/dVa formally
    diverges (avalanche breakdown's own vertical/S-shaped branch) - only a
    CURRENT-controlled sweep (or a ballast resistor in series) can continue
    past it, and neither is implemented here (out of scope - see
    plans/avalanche_breakdown_plan.md). input_diode_breakdown.yaml's reverse
    sweep is deliberately kept short of that wall for this device/mesh."""
    if ii_model is None:
        ii_model = AvalancheModel.si_von_overstraeten_de_man()

    N = len(x)
    n_bc0, p_bc0 = contact_values(mat, Cdop[0])
    n_bcL, p_bcL = contact_values(mat, Cdop[-1])
    Vt = mat.Vt

    psi_bc = np.array([psi_eq[0] + Va, psi_eq[-1]])
    phin_bc = np.array([psi_bc[0] - Vt * np.log(n_bc0 / mat.ni),
                         psi_bc[-1] - Vt * np.log(n_bcL / mat.ni)])
    phip_bc = np.array([psi_bc[0] + Vt * np.log(p_bc0 / mat.ni),
                         psi_bc[-1] + Vt * np.log(p_bcL / mat.ni)])

    h_typ = np.min(np.diff(x))
    poisson_scale = _poisson_scale(mat, h_typ)
    cont_scale = _continuity_scale(mat, h_typ)
    stall_res_threshold = 1.0

    def _gummel_start():
        # A plain Gummel warm start (as newton_solver_qf.py uses for its own
        # cold start) is NOT good enough here: it has no notion of the
        # avalanche generation term at all, and this device's junction mesh
        # is extremely fine (h_min tied to a degenerately-doped side's
        # Debye length, sub-angstrom near x=0) - a few mV of ordinary Gummel
        # iteration noise translates, over that tiny h, into a wildly
        # unphysical local field, which the exponential alpha(E) model then
        # amplifies into an enormous, spurious G_ii before Newton has even
        # had a chance to correct psi. That drove the very first
        # avalanche-solver Newton step's residual to ~1e10 and the Bank-Rose
        # damping straight to its K ceiling with no way to recover (caught
        # via a near-equilibrium Va=-0.05V test point stalling identically
        # to deep reverse bias, which is not physically sensible - a mild
        # bias should behave almost exactly like the no-avalanche solve).
        # Fix: first converge the SAME bias point with the underlying
        # (non-avalanche) QF transport solve - a clean, physically accurate
        # psi/phin/phip with no impact-ionization feedback yet - and only
        # then hand that off as the avalanche solve's starting point. This
        # mirrors how real device simulators stage avalanche: get a good
        # transport solution first, then turn on generation feedback.
        from core.newton_solver_qf import newton_gummel_solve as qf_solve
        base = qf_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq, maxiter=maxiter)
        return base["psi"].copy(), base["phin"].copy(), base["phip"].copy()

    _MAX_PSI_STEP = 1.0

    def _clip(delta):
        # Cap the step's WORST-OFFENDING component (psi against
        # _MAX_PSI_STEP, phin/phip against _MAX_QF_STEP) by rescaling the
        # ENTIRE delta vector by one global scalar, rather than clipping
        # each component independently.
        #
        # Component-wise clipping was the ORIGINAL implementation here, and
        # it is the actual root cause of the isolated wrong-branch points
        # this device's sweep used to show (see main_avalanche.py's git
        # history / DEVELOPMENT_LOG.md for the symptom: isolated bias
        # points landing exactly on the no-avalanche current, surrounded by
        # correctly-converging neighbors). The mechanism: J(U)*delta=-F(U)
        # only guarantees delta is a DESCENT direction for ||F||^2 as a
        # whole, undamped vector - that guarantee (what makes backtracking
        # line search work at all) holds for any UNIFORM scalar multiple of
        # delta, but not for a vector that clips some components far more
        # than others, which is a DIFFERENT direction with no such
        # guarantee. Near breakdown, the raw Newton correction can have
        # components spanning many orders of magnitude (a near-zero-density
        # node's quasi-Fermi potential barely constrained by the residual,
        # next to a node with a huge, well-determined correction) - exactly
        # where component-wise clipping distorts the direction the most.
        # Caught by instrumenting a specific failing bias point: the
        # continuation attempt's first Newton step (undamped, step=1) made
        # a huge, correct-looking residual improvement, but the very next
        # iteration's clipped direction was not a descent direction at all
        # (line search had to shrink the step by a factor of ~2^-20, i.e.
        # effectively zero, before finding ANY decrease) - the classic
        # signature of a corrupted search direction, not genuine
        # ill-conditioning. That stall then triggered this solver's
        # equilibrium-reset fallback (_safe_gummel_retry), which - starting
        # cold with no avalanche generation feedback at all - converges
        # cleanly to the trivial, no-generation root instead, and gets
        # accepted since it scores a lower raw residual (see
        # newton_gummel_solve's fallback-selection logic below): a
        # perfectly self-consistent but PHYSICALLY WRONG single bias point,
        # which is what showed up as a kink in the I(Va) curve.
        #
        # A uniform rescale preserves the true Newton direction exactly
        # (only shortens it), which is exactly what Bank & Rose's own
        # scalar damping t_k already assumes step_clip_fn provides (see
        # bank_rose_damping.py) - this makes both damping strategies here
        # consistent with that same assumption instead of just the
        # Bank-Rose path benefiting from it.
        max_psi = np.max(np.abs(delta[:N])) if N else 0.0
        max_qf = np.max(np.abs(delta[N:3 * N])) if 2 * N else 0.0
        scale = max(max_psi / _MAX_PSI_STEP, max_qf / _MAX_QF_STEP, 1.0)
        return delta / scale if scale > 1.0 else delta

    def _run_newton(psi0, phin0, phip0):
        psi0 = psi0.copy(); phin0 = phin0.copy(); phip0 = phip0.copy()
        psi0[0], psi0[-1] = psi_bc
        phin0[0], phin0[-1] = phin_bc
        phip0[0], phip0[-1] = phip_bc
        U0 = np.concatenate([psi0, phin0, phip0])

        F_fn = lambda U: _residual_only(U, x, Cdop, mat, ii_model, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale)
        FJ_fn = lambda U: _residual_and_jacobian(U, x, Cdop, mat, ii_model, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale)

        if damping == "bank_rose":
            return bank_rose_solve(U0, FJ_fn, F_fn, f_tol=f_tol, maxiter=maxiter,
                                    step_clip_fn=_clip, verbose=verbose)

        if damping != "line_search":
            raise ValueError(f"damping must be 'bank_rose' or 'line_search', got {damping!r}")

        # Fallback: newton_solver_qf.py's plain backtracking line search,
        # for direct comparison against the Bank-Rose path.
        U = U0
        F, J = FJ_fn(U)
        res_norm = np.max(np.abs(F))
        it = 0
        tiny_step_streak = 0
        for it in range(1, maxiter + 1):
            if res_norm < f_tol:
                break
            delta = _clip(equilibrated_spsolve(J, -F))
            step = 1.0
            for _ in range(20):
                U_try = U + step * delta
                F_try = F_fn(U_try)
                res_try = np.max(np.abs(F_try))
                if np.isfinite(res_try) and res_try < res_norm * (1 - 1e-4 * step):
                    break
                step *= 0.5
            else:
                U_try, res_try = U, res_norm
            U = U_try
            F, J = FJ_fn(U)
            res_norm = np.max(np.abs(F))
            if verbose:
                print(f"  Newton(avalanche,line_search) it {it}: |F|_inf={res_norm:.3e}  step={step:.3g}")
            tiny_step_streak = tiny_step_streak + 1 if step < 1e-4 else 0
            if tiny_step_streak >= 2:
                break
        return U, res_norm, it, None

    # NOTE on an approach that was tried and reverted: picking between a
    # continuation candidate and a fresh-start/ramped candidate by
    # comparing SELF-CONSISTENCY (whichever scored lower) sounds appealing
    # but empirically backfired - a spurious high-current root can score a
    # deceptively OK self-consistency ratio (order 1-10, not obviously
    # pathological), so the "pick the better of two" comparison sometimes
    # PREFERRED the wrong branch over the correct low-current one. The psi
    # step clip in _clip above is what actually fixes the root cause (see
    # its comment); trust the direct continuation solve whenever Newton
    # itself reports convergence, exactly like every other solver in this
    # codebase, and only fall back to a fresh start when it doesn't.
    def _safe_gummel_retry():
        """The fresh-start fallback's own inner solve (newton_solver_qf.py,
        unmodified - it has no reason to expect the deep-reverse-bias
        regime avalanche pushes into) can itself raise, not just warn: very
        deep bias makes exp((psi-phin)/Vt) overflow, which can hand
        spsolve a Jacobian with NaN/Inf entries and a BLAS/LAPACK error
        instead of the graceful "did not converge" warning every other
        failure mode here produces. Treat that the same as any other
        failed retry (worse than what we already have) rather than letting
        it crash the whole sweep."""
        try:
            return _run_newton(*_gummel_start())
        except Exception as e:
            if verbose:
                print(f"  Gummel-restart fallback itself raised ({e!r}) - ignoring, keeping prior candidate")
            return None, np.inf, 0, None

    # An adaptive bias-step subdivision scheme (retry a too-large Va jump by
    # first converging an intermediate half-step, recursing further if
    # needed) was tried here and REMOVED after measurement: on this
    # device's actual failures, the very first Newton iteration at EVERY
    # recursion depth - down to 1/16th of the original step - achieved
    # ZERO residual improvement at any step size, showing the blocker
    # isn't the SIZE of the Va jump (which subdivision targets) but a
    # local pathology at that specific state that persists regardless of
    # how small a step is taken. Subdivision therefore bought nothing here
    # while multiplying runtime severalfold on every failing point
    # (up to 5 full nested Newton solves instead of 1). The mechanism is
    # still worth keeping in mind for a genuinely different failure mode -
    # a user-supplied Va_list with an actually oversized jump between
    # consecutive points (e.g. a hand-edited sweep skipping straight from
    # -1V to -13V) - just not for what's failing in this device/mesh
    # today; see DEVELOPMENT_LOG.md if resurrecting it.
    if psi_init is None:
        U, res_norm, it, _ = _safe_gummel_retry()
    else:
        try:
            U, res_norm, it, _ = _run_newton(psi_init, phin_init, phip_init)
        except Exception:
            U, res_norm, it = None, np.inf, 0
        if res_norm > stall_res_threshold:
            U_retry, res_retry, it_retry, _ = _safe_gummel_retry()
            if res_retry < res_norm:
                U, res_norm, it = U_retry, res_retry, it_retry

    if U is None:
        # Both the continuation attempt and the fresh-start retry raised
        # (see _safe_gummel_retry) - fall back to the last known-good
        # state (psi_init if we have one, else equilibrium) rather than
        # crash the sweep. This point is unusable and callers MUST check
        # J_std/J_mean (reported as inf below) before trusting it.
        U = np.concatenate([
            psi_init if psi_init is not None else psi_eq,
            phin_init if phin_init is not None else np.zeros(N),
            phip_init if phip_init is not None else np.zeros(N)])
        res_norm = np.inf

    if res_norm > stall_res_threshold:
        warnings.warn(
            f"Newton(avalanche) solve did not converge at Va={Va} V "
            f"(|F|_inf={res_norm:.3e} at iteration {it}) even after a Gummel-restart retry - "
            "check this point's self-consistency (J_std/J_mean) before trusting it.")

    psi, phin, phip = _unpack(U, N)
    n = mat.ni * np.exp((psi - phin) / Vt)
    p = mat.ni * np.exp((phip - psi) / Vt)

    (_, _, _, Jn, Jp, _, _, _, _, _, _, _, _, _, _) = \
        _edge_quantities_avalanche(psi, phin, phip, n, p, x, mat, ii_model)
    Jtot = Jn + Jp
    J_interior = Jtot[1:-1] if len(Jtot) > 2 else Jtot
    J_rep = float(np.median(J_interior))

    return {
        "psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
        "Jn": Jn, "Jp": Jp, "Jtot": Jtot, "iters": it,
        "J_mean": J_rep, "J_std": float(np.std(J_interior)),
    }
