"""Fully coupled Newton solve (analytic sparse Jacobian, direct sparse LU per
step), as a faster alternative to the decoupled Gummel iteration in
solver.py.

Gummel iteration alternates a linear continuity solve (n, then p, given psi)
with a nonlinear Poisson re-solve (psi, given n and p) and only mixes the two
through a lagged, under-relaxed update. That decoupling is what makes it
robust, but convergence is only linear (roughly constant-factor error
reduction per outer iteration), which is why it can take dozens to hundreds
of iterations at higher bias.

Here, psi, n, and p are unknowns of a single nonlinear system solved
together with Newton's method, which converges quadratically once close to
the solution — the number of outer iterations should drop sharply — using
the exact (hand-derived, finite-difference-validated) sparse Jacobian and a
direct sparse solve per step, globalized with a simple backtracking line
search so it is robust from a cold start, not just from a good initial
guess.

An earlier attempt used SciPy's Jacobian-free Newton-Krylov
(`newton_krylov`) instead of an analytic Jacobian, hoping to avoid deriving
one by hand. It did not converge at all (residual norm flat over 60
iterations): the discretization is intrinsically very stiff (the diffusion
term's effective coefficient scales as D/h^2, and h spans about two orders
of magnitude from the fine mesh at the junction to the coarse bulk), and
plain GMRES without a physics-based preconditioner cannot make progress on
that conditioning. An analytic Jacobian with a direct sparse solve sidesteps
the issue entirely (a direct solve doesn't care about the condition number
the way an unpreconditioned iterative Krylov solve does), which is why
that's what's implemented below instead.

Poisson's row is written directly in terms of (psi, n, p) rather than via
Boltzmann quasi-Fermi levels, so it is linear in all three unknowns (no
exponential term) - only the continuity rows are nonlinear, through the
Scharfetter-Gummel Bernoulli-function flux (nonlinear in psi) and SRH
recombination (nonlinear in n, p). The three equation blocks have very
different natural magnitudes, so each block is row-scaled to O(1) before
assembly.
"""
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from params import Q, Material
import physics as ph
from solver import contact_values


def _poisson_scale(mat: Material, h_typ: float) -> float:
    return mat.eps * mat.Vt / h_typ ** 2


def _continuity_scale(mat: Material, h_typ: float) -> float:
    return Q * mat.Dn * mat.ni / h_typ


def _unpack(U, N):
    return U[:N], U[N:2 * N], U[2 * N:3 * N]


def _edge_quantities(psi, n, p, x, mat):
    """Per-edge SG flux, recombination, and their derivatives - shared by the
    residual-only and residual+Jacobian paths so they never disagree."""
    h = np.diff(x)
    Vt = mat.Vt
    u = psi / Vt

    du = u[1:] - u[:-1]                    # u_{e+1} - u_e
    Bp = ph.bernoulli(du)                  # B(u_{e+1}-u_e)
    Bm = ph.bernoulli(-du)                 # B(u_e-u_{e+1})

    coef_n = Q * mat.Dn / h
    coef_p = Q * mat.Dp / h
    Jn = coef_n * (n[1:] * Bp - n[:-1] * Bm)
    Jp = coef_p * (p[:-1] * Bp - p[1:] * Bm)

    ni = mat.ni
    denom = mat.tau_p * (n + ni) + mat.tau_n * (p + ni)
    num = n * p - ni ** 2
    R = num / denom
    return h, Bp, Bm, coef_n, coef_p, Jn, Jp, R, denom, num


def _residual_only(U, x, Cdop, mat, psi_bc, n_bc, p_bc, poisson_scale, cont_scale):
    """Fast path: residual vector only, no Jacobian. Used for line-search
    trial evaluations, which don't need a new Jacobian until a step is
    accepted."""
    N = len(x)
    psi, n, p = _unpack(U, N)
    n = np.clip(n, 1.0, None)
    p = np.clip(p, 1.0, None)
    cvol = ph._control_volumes(x)

    Rpsi = np.empty(N)
    Rn = np.empty(N)
    Rp = np.empty(N)
    Rpsi[0], Rpsi[-1] = psi[0] - psi_bc[0], psi[-1] - psi_bc[-1]
    Rn[0], Rn[-1] = n[0] - n_bc[0], n[-1] - n_bc[-1]
    Rp[0], Rp[-1] = p[0] - p_bc[0], p[-1] - p_bc[-1]

    h = np.diff(x)
    hm, hp = h[:-1], h[1:]
    cvol_i = cvol[1:-1]
    lap_m = mat.eps / hm / cvol_i
    lap_p = mat.eps / hp / cvol_i
    Rpsi[1:-1] = (lap_p * (psi[2:] - psi[1:-1]) - lap_m * (psi[1:-1] - psi[:-2])
                  - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])) / poisson_scale

    _, _, _, _, _, Jn, Jp, R, _, _ = _edge_quantities(psi, n, p, x, mat)
    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * R[1:-1]) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * R[1:-1]) / cont_scale

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, x, Cdop, mat, psi_bc, n_bc, p_bc, poisson_scale, cont_scale):
    """Returns (F, J) where F is the length-3N residual vector and J is the
    3N x 3N sparse Jacobian dF/dU, for unknowns U=[psi, n, p]. Fully
    vectorized (no per-node Python loop) - assembly cost matters here since
    it happens every accepted Newton step."""
    N = len(x)
    psi, n, p = _unpack(U, N)
    n = np.clip(n, 1.0, None)
    p = np.clip(p, 1.0, None)
    cvol = ph._control_volumes(x)
    Vt = mat.Vt

    Rpsi = np.empty(N)
    Rn = np.empty(N)
    Rp = np.empty(N)
    Rpsi[0], Rpsi[-1] = psi[0] - psi_bc[0], psi[-1] - psi_bc[-1]
    Rn[0], Rn[-1] = n[0] - n_bc[0], n[-1] - n_bc[-1]
    Rp[0], Rp[-1] = p[0] - p_bc[0], p[-1] - p_bc[-1]

    h = np.diff(x)
    hm, hp = h[:-1], h[1:]
    cvol_i = cvol[1:-1]
    lap_m = mat.eps / hm / cvol_i
    lap_p = mat.eps / hp / cvol_i
    Rpsi[1:-1] = (lap_p * (psi[2:] - psi[1:-1]) - lap_m * (psi[1:-1] - psi[:-2])
                  - Q * (n[1:-1] - p[1:-1] - Cdop[1:-1])) / poisson_scale

    u = psi / Vt
    du = u[1:] - u[:-1]
    dBp = ph.bernoulli_deriv(du)
    dBm = ph.bernoulli_deriv(-du)
    h_e, Bp, Bm, coef_n, coef_p, Jn, Jp, R, denom, num = _edge_quantities(psi, n, p, x, mat)

    dJn_dpsi_e = coef_n * (n[1:] * (-dBp / Vt) - n[:-1] * (dBm / Vt))
    dJn_dpsi_ep1 = coef_n * (n[1:] * (dBp / Vt) - n[:-1] * (-dBm / Vt))
    dJn_dn_e = -coef_n * Bm
    dJn_dn_ep1 = coef_n * Bp

    dJp_dpsi_e = coef_p * (p[:-1] * (-dBp / Vt) - p[1:] * (dBm / Vt))
    dJp_dpsi_ep1 = coef_p * (p[:-1] * (dBp / Vt) - p[1:] * (-dBm / Vt))
    dJp_dp_e = coef_p * Bp
    dJp_dp_ep1 = -coef_p * Bm

    dR_dn = (p * denom - num * mat.tau_p) / denom ** 2
    dR_dp = (n * denom - num * mat.tau_n) / denom ** 2

    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * R[1:-1]) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * R[1:-1]) / cont_scale
    F = np.concatenate([Rpsi, Rn, Rp])

    # --- Jacobian, assembled as COO triplets built with vectorized numpy
    # indexing (no Python-level per-node loop) ---
    idx = np.arange(1, N - 1)          # interior node indices
    k = idx - 1                        # e_lo = idx-1, e_hi = idx = k+1, both index into the length-(N-1) edge arrays via k, k+1
    e_lo, e_hi = k, k + 1
    cv = cvol_i

    rows_list, cols_list, data_list = [], [], []

    def add(r, c, v):
        rows_list.append(r)
        cols_list.append(c)
        data_list.append(v)

    # Poisson interior rows (linear)
    add(idx, idx - 1, lap_m / poisson_scale)
    add(idx, idx, -(lap_m + lap_p) / poisson_scale)
    add(idx, idx + 1, lap_p / poisson_scale)
    add(idx, N + idx, np.full(len(idx), -Q / poisson_scale))
    add(idx, 2 * N + idx, np.full(len(idx), Q / poisson_scale))

    # Electron continuity interior rows
    r_n = N + idx
    add(r_n, idx - 1, -dJn_dpsi_e[e_lo] / cv / cont_scale)
    add(r_n, idx, (dJn_dpsi_e[e_hi] - dJn_dpsi_ep1[e_lo]) / cv / cont_scale)
    add(r_n, idx + 1, dJn_dpsi_ep1[e_hi] / cv / cont_scale)
    add(r_n, N + idx - 1, -dJn_dn_e[e_lo] / cv / cont_scale)
    add(r_n, N + idx, (dJn_dn_e[e_hi] - dJn_dn_ep1[e_lo]) / cv / cont_scale - Q * dR_dn[idx] / cont_scale)
    add(r_n, N + idx + 1, dJn_dn_ep1[e_hi] / cv / cont_scale)
    add(r_n, 2 * N + idx, -Q * dR_dp[idx] / cont_scale)

    # Hole continuity interior rows
    r_p = 2 * N + idx
    add(r_p, idx - 1, -dJp_dpsi_e[e_lo] / cv / cont_scale)
    add(r_p, idx, (dJp_dpsi_e[e_hi] - dJp_dpsi_ep1[e_lo]) / cv / cont_scale)
    add(r_p, idx + 1, dJp_dpsi_ep1[e_hi] / cv / cont_scale)
    add(r_p, 2 * N + idx - 1, -dJp_dp_e[e_lo] / cv / cont_scale)
    add(r_p, 2 * N + idx, (dJp_dp_e[e_hi] - dJp_dp_ep1[e_lo]) / cv / cont_scale + Q * dR_dp[idx] / cont_scale)
    add(r_p, 2 * N + idx + 1, dJp_dp_ep1[e_hi] / cv / cont_scale)
    add(r_p, N + idx, Q * dR_dn[idx] / cont_scale)

    interior_rows = np.concatenate(rows_list)
    interior_cols = np.concatenate(cols_list)
    interior_data = np.concatenate(data_list)

    # Dirichlet contact rows: bare diag=1.0 relies on this row's own
    # equation (1*delta_i = -F_i) staying the pivot spsolve's partial
    # pivoting picks for that column. That holds when nearby coefficients
    # are O(1)-ish, but a stiff/degenerate case (e.g. asymmetric doping
    # with one side near/above the effective density of states) can give
    # an interior row's coupling entry into this same column - dJn_dpsi_e,
    # dJn_dn_e etc. above, all evaluated at the node NEXT TO this contact -
    # a magnitude that dwarfs a bare 1.0, so the solver pivots onto that
    # row instead and corrupts what must be an exact Dirichlet constraint.
    # Same failure mode, and same fix, as physics.solve_poisson's Dirichlet
    # rows (see that function's comment) - scale the diagonal to at least
    # the largest actual coupling entry already assembled in that column,
    # so it can never lose the pivot regardless of how extreme the
    # continuity/Poisson coefficients elsewhere get.
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


def newton_gummel_solve(x, Cdop, mat: Material, Va, psi_eq, n_eq, p_eq,
                         psi_init=None, phin_init=None, phip_init=None,
                         f_tol=1e-9, maxiter=50, verbose=False):
    """Same signature/return shape as solver.gummel_solve, but solves the
    coupled Poisson + electron continuity + hole continuity system as one
    nonlinear system via full Newton (analytic Jacobian, direct sparse
    solve, backtracking line search) instead of decoupled Gummel
    iteration."""
    N = len(x)
    n_bc0, p_bc0 = contact_values(mat, Cdop[0])
    n_bcL, p_bcL = contact_values(mat, Cdop[-1])

    psi_bc = np.array([psi_eq[0] + Va, psi_eq[-1]])
    # _residual_and_jacobian clips its solution's n/p to a 1.0 cm^-3 floor
    # (avoids log/exp underflow issues elsewhere) - for a strongly
    # asymmetric/degenerate junction, the MINORITY carrier's own
    # equilibrium contact value (ni^2/N) can legitimately fall below that
    # floor (e.g. ni^2/Nd < 1 cm^-3 for Nd=1e21), and comparing the
    # solution's floor-clipped density against an unclipped boundary
    # target makes that boundary residual permanently unreachable (stuck
    # at exactly floor-target, never zero, however well everything else
    # converges) - clip the targets the same way so the comparison is
    # consistent and the residual can actually reach zero there.
    n_bc = np.clip(np.array([n_bc0, n_bcL]), 1.0, None)
    p_bc = np.clip(np.array([p_bc0, p_bcL]), 1.0, None)

    if psi_init is None:
        # Cold start (no previous bias point to warm-start from, i.e. the
        # first point of a sweep): a naive linear-ramp guess isn't always in
        # Newton's basin of quadratic convergence (this showed up as a
        # genuine stall - line search collapsing at a large, unconverged
        # residual - at the first point of a reverse-bias-heavy sweep). A
        # standard, cheap fix is to spend a handful of (robust, if slow)
        # Gummel iterations first to reach a good starting point, then
        # switch to full Newton for fast final convergence. This only runs
        # once per sweep, for the coldest-start point.
        from solver import gummel_solve
        warm = gummel_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq, max_gummel=15)
        psi0, n0, p0 = warm["psi"], warm["n"], warm["p"]
    else:
        psi0 = psi_init.copy()
        n0 = mat.ni * np.exp((psi_init - phin_init) / mat.Vt)
        p0 = mat.ni * np.exp((phip_init - psi_init) / mat.Vt)

    psi0[0], psi0[-1] = psi_bc
    n0 = np.clip(n0, 1.0, None)
    p0 = np.clip(p0, 1.0, None)
    n0[0], n0[-1] = n_bc
    p0[0], p0[-1] = p_bc

    h_typ = np.min(np.diff(x))
    poisson_scale = _poisson_scale(mat, h_typ)
    cont_scale = _continuity_scale(mat, h_typ)

    U = np.concatenate([psi0, n0, p0])
    F, J = _residual_and_jacobian(U, x, Cdop, mat, psi_bc, n_bc, p_bc, poisson_scale, cont_scale)
    res_norm = np.max(np.abs(F))

    it = 0
    tiny_step_streak = 0
    stall_res_threshold = 1.0
    for it in range(1, maxiter + 1):
        if res_norm < f_tol:
            break
        delta = spla.spsolve(J, -F)

        # Backtracking line search: halve the step until the residual norm
        # actually decreases (plain Newton steps can overshoot badly this
        # far from the solution on a cold start).
        step = 1.0
        for _ in range(20):
            U_try = U + step * delta
            F_try = _residual_only(U_try, x, Cdop, mat, psi_bc, n_bc, p_bc, poisson_scale, cont_scale)
            res_try = np.max(np.abs(F_try))
            if np.isfinite(res_try) and res_try < res_norm * (1 - 1e-4 * step):
                break
            step *= 0.5
        else:
            U_try, res_try = U, res_norm  # give up shrinking further this step

        U = U_try
        # Jacobian only needs rebuilding for the accepted step (not every
        # line-search trial), since it's the expensive part of each iteration.
        F, J = _residual_and_jacobian(U, x, Cdop, mat, psi_bc, n_bc, p_bc, poisson_scale, cont_scale)
        res_norm = np.max(np.abs(F))
        if verbose:
            print(f"  Newton it {it}: |F|_inf={res_norm:.3e}  step={step:.3g}")

        # Once the line search can no longer find a meaningfully better step,
        # further iterations don't help - this happens once the residual
        # hits its natural floor (density clipping near zero makes the
        # residual function non-smooth right at that floor, so exact
        # convergence to f_tol isn't reachable there even though the
        # solution is already effectively converged). Only treat a stall as
        # "done" if the residual is actually small when it happens - a
        # stall at a large residual is a real failure to converge, not a
        # precision floor, and should be reported as one rather than
        # silently accepted. A stall at a large residual is a genuine
        # failure to converge, not a precision floor - it's reported via a
        # warning (and the caller's own J_std/J_mean self-consistency check
        # on the returned solution, same diagnostic used everywhere else in
        # this codebase) rather than aborting the whole sweep over one
        # difficult bias point.
        tiny_step_streak = tiny_step_streak + 1 if step < 1e-4 else 0
        if tiny_step_streak >= 2:
            if res_norm > stall_res_threshold:
                warnings.warn(
                    f"Newton solve stalled at Va={Va} V without fully converging "
                    f"(|F|_inf={res_norm:.3e} at iteration {it}, line search collapsed) - "
                    "check this point's self-consistency (J_std/J_mean) before trusting it.")
            break
    else:
        if res_norm > stall_res_threshold:
            warnings.warn(
                f"Newton solve did not converge at Va={Va} V within {maxiter} iterations "
                f"(|F|_inf={res_norm:.3e}).")

    psi, n, p = _unpack(U, N)
    n = np.clip(n, 1.0, None)
    p = np.clip(p, 1.0, None)
    phin = psi - mat.Vt * np.log(n / mat.ni)
    phip = psi + mat.Vt * np.log(p / mat.ni)

    Jn, Jp, Jtot = ph.edge_currents(x, psi, n, p, mat)
    J_interior = Jtot[1:-1] if len(Jtot) > 2 else Jtot
    J_rep = float(np.median(J_interior))

    return {
        "psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
        "Jn": Jn, "Jp": Jp, "Jtot": Jtot, "iters": it,
        "J_mean": J_rep, "J_std": float(np.std(J_interior)),
    }
