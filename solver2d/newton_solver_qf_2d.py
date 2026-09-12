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


def unpack_qf(U, N):
    return U[:N], U[N:2 * N], U[2 * N:3 * N]


def _densities(psi, phin, phip, mat: Material):
    Vt = mat.Vt
    n = mat.ni * np.exp((psi - phin) / Vt)
    p = mat.ni * np.exp((phip - psi) / Vt)
    return n, p


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
    n, p = _densities(psi, phin, phip, mat)
    Vt, ni = mat.Vt, mat.ni

    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    edge_len = np.linalg.norm(mesh.points[jj] - mesh.points[ii], axis=1)
    g_e = mat.eps * mesh.edge_weight

    n_avg = 0.5 * (n[ii] + n[jj])
    p_avg = 0.5 * (p[ii] + p[jj])
    Jn_e = -Q * mat.mu_n * n_avg * (phin[jj] - phin[ii]) / edge_len
    Jp_e = -Q * mat.mu_p * p_avg * (phip[jj] - phip[ii]) / edge_len
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

    denom = mat.tau_p * (n + ni) + mat.tau_n * (p + ni)
    R = (n * p - ni ** 2) / denom

    Rpsi = (div_psi / mesh.cv_area - Q * (n - p - mesh.Cdop)) / poisson_scale
    Rn = (div_n / mesh.cv_area - Q * R) / cont_scale
    Rp = (div_p / mesh.cv_area + Q * R) / cont_scale

    Rpsi[is_contact] = psi[is_contact] - psi_bc
    Rn[is_contact] = phin[is_contact] - phin_bc
    Rp[is_contact] = phip[is_contact] - phip_bc

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, mesh, mat, is_contact, psi_bc, phin_bc, phip_bc,
                            poisson_scale, cont_scale):
    N = len(mesh.points)
    psi, phin, phip = unpack_qf(U, N)
    n, p = _densities(psi, phin, phip, mat)
    Vt, ni = mat.Vt, mat.ni

    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    edge_len = np.linalg.norm(mesh.points[jj] - mesh.points[ii], axis=1)
    g_e = mat.eps * mesh.edge_weight

    n_avg = 0.5 * (n[ii] + n[jj])
    p_avg = 0.5 * (p[ii] + p[jj])
    dphin_e = phin[jj] - phin[ii]
    dphip_e = phip[jj] - phip[ii]
    Jn_e = -Q * mat.mu_n * n_avg * dphin_e / edge_len
    Jp_e = -Q * mat.mu_p * p_avg * dphip_e / edge_len
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

    denom = mat.tau_p * (n + ni) + mat.tau_n * (p + ni)
    num = n * p - ni ** 2
    R = num / denom

    Rpsi = (div_psi / mesh.cv_area - Q * (n - p - mesh.Cdop)) / poisson_scale
    Rn = (div_n / mesh.cv_area - Q * R) / cont_scale
    Rp = (div_p / mesh.cv_area + Q * R) / cont_scale
    Rpsi[is_contact] = psi[is_contact] - psi_bc
    Rn[is_contact] = phin[is_contact] - phin_bc
    Rp[is_contact] = phip[is_contact] - phip_bc
    F = np.concatenate([Rpsi, Rn, Rp])

    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt
    dR_dn = (p * denom - num * mat.tau_p) / denom ** 2
    dR_dp = (n * denom - num * mat.tau_n) / denom ** 2

    cn_e = Q * mat.mu_n * mesh.facet_length / edge_len
    cp_e = Q * mat.mu_p * mesh.facet_length / edge_len

    dIn_dpsi_i = -cn_e * (dn_dpsi[ii] / 2.0) * dphin_e
    dIn_dpsi_j = -cn_e * (dn_dpsi[jj] / 2.0) * dphin_e
    dIn_dphin_i = -cn_e * ((dn_dphin[ii] / 2.0) * dphin_e - n_avg)
    dIn_dphin_j = -cn_e * ((dn_dphin[jj] / 2.0) * dphin_e + n_avg)

    dIp_dpsi_i = -cp_e * (dp_dpsi[ii] / 2.0) * dphip_e
    dIp_dpsi_j = -cp_e * (dp_dpsi[jj] / 2.0) * dphip_e
    dIp_dphip_i = -cp_e * ((dp_dphip[ii] / 2.0) * dphip_e - p_avg)
    dIp_dphip_j = -cp_e * ((dp_dphip[jj] / 2.0) * dphip_e + p_avg)

    rows_list, cols_list, data_list = [], [], []

    def add(rows, cols, vals):
        rows_list.append(rows)
        cols_list.append(cols)
        data_list.append(vals)

    # --- Poisson: per-edge Laplacian coupling, row masked to non-contact ---
    active_i = ~is_contact[ii]
    active_j = ~is_contact[jj]
    cv_i, cv_j = mesh.cv_area[ii], mesh.cv_area[jj]

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
    add(r_n_i[active_i], ii[active_i], (dIn_dpsi_i / cv_i / cont_scale)[active_i])
    add(r_n_i[active_i], jj[active_i], (dIn_dpsi_j / cv_i / cont_scale)[active_i])
    add(r_n_i[active_i], N + ii[active_i], (dIn_dphin_i / cv_i / cont_scale)[active_i])
    add(r_n_i[active_i], N + jj[active_i], (dIn_dphin_j / cv_i / cont_scale)[active_i])

    add(r_n_j[active_j], ii[active_j], (-dIn_dpsi_i / cv_j / cont_scale)[active_j])
    add(r_n_j[active_j], jj[active_j], (-dIn_dpsi_j / cv_j / cont_scale)[active_j])
    add(r_n_j[active_j], N + ii[active_j], (-dIn_dphin_i / cv_j / cont_scale)[active_j])
    add(r_n_j[active_j], N + jj[active_j], (-dIn_dphin_j / cv_j / cont_scale)[active_j])

    # --- Electron continuity: per-node recombination derivative ---
    r_n_free = N + free
    add(r_n_free, free, -Q * (dR_dn[free] * dn_dpsi[free] + dR_dp[free] * dp_dpsi[free]) / cont_scale)
    add(r_n_free, N + free, -Q * dR_dn[free] * dn_dphin[free] / cont_scale)
    add(r_n_free, 2 * N + free, -Q * dR_dp[free] * dp_dphip[free] / cont_scale)

    # --- Hole continuity: per-edge current derivative ---
    r_p_i, r_p_j = 2 * N + ii, 2 * N + jj
    add(r_p_i[active_i], ii[active_i], (dIp_dpsi_i / cv_i / cont_scale)[active_i])
    add(r_p_i[active_i], jj[active_i], (dIp_dpsi_j / cv_i / cont_scale)[active_i])
    add(r_p_i[active_i], 2 * N + ii[active_i], (dIp_dphip_i / cv_i / cont_scale)[active_i])
    add(r_p_i[active_i], 2 * N + jj[active_i], (dIp_dphip_j / cv_i / cont_scale)[active_i])

    add(r_p_j[active_j], ii[active_j], (-dIp_dpsi_i / cv_j / cont_scale)[active_j])
    add(r_p_j[active_j], jj[active_j], (-dIp_dpsi_j / cv_j / cont_scale)[active_j])
    add(r_p_j[active_j], 2 * N + ii[active_j], (-dIp_dphip_i / cv_j / cont_scale)[active_j])
    add(r_p_j[active_j], 2 * N + jj[active_j], (-dIp_dphip_j / cv_j / cont_scale)[active_j])

    # --- Hole continuity: per-node recombination derivative ---
    r_p_free = 2 * N + free
    add(r_p_free, free, Q * (dR_dn[free] * dn_dpsi[free] + dR_dp[free] * dp_dpsi[free]) / cont_scale)
    add(r_p_free, N + free, Q * dR_dn[free] * dn_dphin[free] / cont_scale)
    add(r_p_free, 2 * N + free, Q * dR_dp[free] * dp_dphip[free] / cont_scale)

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


def newton_solve_2d(mesh, mat: Material, Va, contact_bias_role, f_tol=1e-9, maxiter=50, verbose=False,
                     psi_init=None, phin_init=None, phip_init=None):
    """Solve the 2D QF system at applied bias Va (V), applied to whichever
    contact(s) have bias_role="anode" (bias_role="cathode" contacts stay at
    0V - see mesh2d/geometry2d.py::Contact). Returns a dict matching
    core/newton_solver_qf.py's shape (psi, n, p, phin, phip, iters).

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
    Vt, ni = mat.Vt, mat.ni

    boundary_bc_type = np.array(mesh.boundary_bc_type)
    is_contact_boundary = np.array([bc.startswith("contact:") for bc in boundary_bc_type])
    contact_point_idx = mesh.boundary_point_index[is_contact_boundary]
    contact_names = [bc.split(":", 1)[1] for bc in boundary_bc_type[is_contact_boundary]]

    is_contact_full = np.zeros(N, dtype=bool)
    is_contact_full[contact_point_idx] = True

    psi_eq = equilibrium_bulk_potential_arr(Vt, np.full(N, ni), mesh.Cdop)

    bias_of = {c.name: (Va if c.bias_role == contact_bias_role else 0.0) for c in mesh.domain.contacts}

    psi_bc = np.empty(len(contact_point_idx))
    phin_bc = np.empty(len(contact_point_idx))
    phip_bc = np.empty(len(contact_point_idx))
    for k, (pt, name) in enumerate(zip(contact_point_idx, contact_names)):
        n_bc, p_bc = contact_values(mat, mesh.Cdop[pt], ni=ni)
        v_applied = bias_of[name]
        psi_bc[k] = psi_eq[pt] + v_applied
        phin_bc[k] = psi_bc[k] - Vt * np.log(n_bc / ni)
        phip_bc[k] = psi_bc[k] + Vt * np.log(p_bc / ni)

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
            f"Newton(QF,2D) solve did not converge at Va={Va} V "
            f"(|F|_inf={res_norm:.3e} at iteration {it}) even after a cold-start retry.")

    psi, phin, phip = unpack_qf(U, N)
    n, p = _densities(psi, phin, phip, mat)
    return {"psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
            "iters": it, "res_norm": res_norm}
