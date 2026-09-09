"""Fully coupled Newton solve using QUASI-FERMI POTENTIALS (phin, phip) as
the transport unknowns, instead of raw carrier densities (newton_solver.py)
or log-densities (the abandoned log-density-formulation branch).

Same overall structure as newton_solver.py (analytic sparse Jacobian, direct
sparse LU per step, backtracking line search, Gummel cold-start) - only the
CONTINUITY discretization and the choice of unknowns differ. See that
module's docstring for the shared design notes (why an analytic Jacobian +
direct solve, not Newton-Krylov).

WHY THIS EXISTS: newton_solver.py (raw n, p + Scharfetter-Gummel) fails to
converge robustly across reverse bias for a strongly asymmetric, degenerately
doped junction (p-side 1e17 / n-side 1e20-1e21 cm^-3 - see the pinned
"tcad1d-extreme-doping-convergence-limit" note). Two log-density
reformulations were tried (on the now-abandoned log-density-formulation
branch) and both failed - they kept Scharfetter-Gummel's own exponential
(Bernoulli-function) flux fitting, just reparametrized the unknowns, so the
Jacobian still carried TWO compounding exponential nonlinearities (SG's own,
plus the density-vs-potential relation).

This formulation, modeled directly on the open-source FLOOXS device
simulator's "QF" model (TclLib/Device/floods/Silicon/Equations.tcl,
Generic/Transport/continuity.tcl - FLOOXS source available locally at
~/Desktop/github_flooxs/flooxs), removes SG entirely. The unknowns are the
electron/hole quasi-Fermi potentials phin, phip - already a quantity this
codebase understands (physics.py's solve_poisson already takes phin/phip as
FIXED inputs for the MOS quasi-small-signal trick; the raw-density Newton
solver already DERIVES phin/phip post-solve, phin = psi - Vt*ln(n/ni)) - so
this promotes an existing derived quantity to a primary unknown rather than
importing a foreign concept. The current is a PLAIN gradient of the
quasi-Fermi potential (no Bernoulli function at all):

    Jn = -q * mu_n * n * grad(phin)     Jp = -q * mu_p * p * grad(phip)

(standard textbook drift-diffusion current in quasi-Fermi form - derivable
directly from Jn = q*Dn*grad(n) - q*mu_n*n*grad(psi) via the Einstein
relation and n = ni*exp((psi-phin)/Vt): grad(n) = (n/Vt)*(grad(psi)-grad(phin)),
so Jn = q*mu_n*n*(grad(psi)-grad(phin)) - q*mu_n*n*grad(psi) = -q*mu_n*n*grad(phin)).
Discretized here with the edge's carrier density taken as the arithmetic
mean of its two nodal values (the simplest standard central finite-volume
choice - if validation ever shows oscillation on a coarse mesh, a geometric
mean is the documented fallback, but this project's meshes are already fine
enough at every junction to resolve exponential variation directly, which is
the whole reason SG's exponential fitting is not needed here).

Positivity is structural (n, p = ni*exp(...) can never go negative), so
there is no density-floor clip anywhere in this module.
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
    """U = [psi, phin, phip]."""
    return U[:N], U[N:2 * N], U[2 * N:3 * N]


def _edge_quantities(psi, phin, phip, n, p, x, mat):
    """Per-edge plain-gradient flux and recombination - shared by the
    residual-only and residual+Jacobian paths so they never disagree."""
    h = np.diff(x)
    n_avg = (n[:-1] + n[1:]) / 2.0
    p_avg = (p[:-1] + p[1:]) / 2.0

    Jn = -Q * mat.mu_n * n_avg * (phin[1:] - phin[:-1]) / h
    Jp = -Q * mat.mu_p * p_avg * (phip[1:] - phip[:-1]) / h

    ni = mat.ni
    denom = mat.tau_p * (n + ni) + mat.tau_n * (p + ni)
    num = n * p - ni ** 2
    R = num / denom
    return h, n_avg, p_avg, Jn, Jp, R, denom, num


def _residual_only(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale):
    """Fast path: residual vector only, no Jacobian. Used for line-search
    trial evaluations, which don't need a new Jacobian until a step is
    accepted."""
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

    _, _, _, Jn, Jp, R, _, _ = _edge_quantities(psi, phin, phip, n, p, x, mat)
    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * R[1:-1]) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * R[1:-1]) / cont_scale

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale):
    """Returns (F, J) where F is the length-3N residual vector and J is the
    3N x 3N sparse Jacobian dF/dU, for unknowns U=[psi, phin, phip]. Fully
    vectorized (no per-node Python loop)."""
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

    h_e, n_avg, p_avg, Jn, Jp, R, denom, num = _edge_quantities(psi, phin, phip, n, p, x, mat)

    # dn/dpsi = n/Vt, dn/dphin = -n/Vt ; dp/dpsi = -p/Vt, dp/dphip = p/Vt
    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt

    dphin_e = phin[1:] - phin[:-1]
    dphip_e = phip[1:] - phip[:-1]

    # Jn_e = -Q*mu_n*n_avg*dphin_e/h ; n_avg = (n_i+n_{i+1})/2
    dJn_dpsi_e = -Q * mat.mu_n / h_e * (dn_dpsi[:-1] / 2.0) * dphin_e
    dJn_dpsi_ep1 = -Q * mat.mu_n / h_e * (dn_dpsi[1:] / 2.0) * dphin_e
    dJn_dphin_e = -Q * mat.mu_n / h_e * ((dn_dphin[:-1] / 2.0) * dphin_e - n_avg)
    dJn_dphin_ep1 = -Q * mat.mu_n / h_e * ((dn_dphin[1:] / 2.0) * dphin_e + n_avg)

    # Jp_e = -Q*mu_p*p_avg*dphip_e/h
    dJp_dpsi_e = -Q * mat.mu_p / h_e * (dp_dpsi[:-1] / 2.0) * dphip_e
    dJp_dpsi_ep1 = -Q * mat.mu_p / h_e * (dp_dpsi[1:] / 2.0) * dphip_e
    dJp_dphip_e = -Q * mat.mu_p / h_e * ((dp_dphip[:-1] / 2.0) * dphip_e - p_avg)
    dJp_dphip_ep1 = -Q * mat.mu_p / h_e * ((dp_dphip[1:] / 2.0) * dphip_e + p_avg)

    dR_dn = (p * denom - num * mat.tau_p) / denom ** 2
    dR_dp = (n * denom - num * mat.tau_n) / denom ** 2

    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * R[1:-1]) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * R[1:-1]) / cont_scale
    F = np.concatenate([Rpsi, Rn, Rp])

    idx = np.arange(1, N - 1)
    k = idx - 1
    e_lo, e_hi = k, k + 1
    cv = cvol_i

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

    # Electron continuity interior rows. Row i = (Jn[e_hi] - Jn[e_lo])/cvol -
    # Q*R, all over cont_scale; e_lo connects (i-1,i), e_hi connects (i,i+1).
    # dJn_*_e is d(Jn at that edge)/d(unknown at the edge's LEFT node), so
    # e.g. column i-1 only sees Jn[e_lo] through its own left-node
    # derivative, with the row's own leading minus sign on Jn[e_lo].
    r_n = N + idx
    add(r_n, idx - 1, -dJn_dpsi_e[e_lo] / cv / cont_scale)
    add(r_n, idx, (dJn_dpsi_e[e_hi] - dJn_dpsi_ep1[e_lo]) / cv / cont_scale
        - Q * (dR_dn[idx] * dn_dpsi[idx] + dR_dp[idx] * dp_dpsi[idx]) / cont_scale)
    add(r_n, idx + 1, dJn_dpsi_ep1[e_hi] / cv / cont_scale)
    add(r_n, N + idx - 1, -dJn_dphin_e[e_lo] / cv / cont_scale)
    add(r_n, N + idx, (dJn_dphin_e[e_hi] - dJn_dphin_ep1[e_lo]) / cv / cont_scale
        - Q * dR_dn[idx] * dn_dphin[idx] / cont_scale)
    add(r_n, N + idx + 1, dJn_dphin_ep1[e_hi] / cv / cont_scale)
    add(r_n, 2 * N + idx, -Q * dR_dp[idx] * dp_dphip[idx] / cont_scale)

    # Hole continuity interior rows - same pattern, +Q*R sign (opposite
    # electron's -Q*R, same convention as newton_solver.py).
    r_p = 2 * N + idx
    add(r_p, idx - 1, -dJp_dpsi_e[e_lo] / cv / cont_scale)
    add(r_p, idx, (dJp_dpsi_e[e_hi] - dJp_dpsi_ep1[e_lo]) / cv / cont_scale
        + Q * (dR_dn[idx] * dn_dpsi[idx] + dR_dp[idx] * dp_dpsi[idx]) / cont_scale)
    add(r_p, idx + 1, dJp_dpsi_ep1[e_hi] / cv / cont_scale)
    add(r_p, 2 * N + idx - 1, -dJp_dphip_e[e_lo] / cv / cont_scale)
    add(r_p, 2 * N + idx, (dJp_dphip_e[e_hi] - dJp_dphip_ep1[e_lo]) / cv / cont_scale
        + Q * dR_dp[idx] * dp_dphip[idx] / cont_scale)
    add(r_p, 2 * N + idx + 1, dJp_dphip_ep1[e_hi] / cv / cont_scale)
    add(r_p, N + idx, Q * dR_dn[idx] * dn_dphin[idx] / cont_scale)

    interior_rows = np.concatenate(rows_list)
    interior_cols = np.concatenate(cols_list)
    interior_data = np.concatenate(data_list)

    # Dirichlet rows - same pivoting-safety scaling as newton_solver.py (see
    # that module's comment for why a bare diag=1.0 isn't always safe).
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


# Cap on a single Newton step's raw |delta_phin|, |delta_phip| (volts) - see
# the comment at its use site for why this is needed (a near-zero carrier
# density leaves its quasi-Fermi potential almost unconstrained by the
# residual, letting a raw step send it hundreds of volts off in one shot).
_MAX_QF_STEP = 5.0


def newton_gummel_solve(x, Cdop, mat: Material, Va, psi_eq, n_eq, p_eq,
                         psi_init=None, phin_init=None, phip_init=None,
                         f_tol=1e-9, maxiter=50, verbose=False):
    """Same signature/return shape as solver.gummel_solve and
    newton_solver.newton_gummel_solve, but solves for quasi-Fermi potentials
    (phin, phip) instead of raw densities - see module docstring."""
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
        from solver import gummel_solve
        warm = gummel_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq, max_gummel=15)
        return (warm["psi"].copy(),
                warm["psi"] - Vt * np.log(warm["n"] / mat.ni),
                warm["psi"] + Vt * np.log(warm["p"] / mat.ni))

    def _run_newton(psi0, phin0, phip0):
        """One full Newton attempt from a given starting point. Returns
        (U, res_norm, it)."""
        psi0 = psi0.copy(); phin0 = phin0.copy(); phip0 = phip0.copy()
        psi0[0], psi0[-1] = psi_bc
        phin0[0], phin0[-1] = phin_bc
        phip0[0], phip0[-1] = phip_bc

        U = np.concatenate([psi0, phin0, phip0])
        F, J = _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale)
        res_norm = np.max(np.abs(F))

        it = 0
        tiny_step_streak = 0
        for it in range(1, maxiter + 1):
            if res_norm < f_tol:
                break
            delta = spla.spsolve(J, -F)
            # Cap the raw phin/phip step: wherever a carrier's density is
            # near zero (deep in the bulk on the wrong side of the
            # junction, e.g. electrons on the p-side), that quasi-Fermi
            # potential barely affects the residual (dn/dphin = n/Vt approx
            # 0 there) and is nearly unconstrained by the physics - Newton
            # can then send it hundreds of volts off in one step without
            # the residual objecting (caught via a warm-start-from-
            # equilibrium case where phip swung to -612V for exactly this
            # reason, well past where a physically sane potential should
            # ever be at room temperature). A few volts is already a very
            # generous cap for this project's bias ranges.
            delta[N:3 * N] = np.clip(delta[N:3 * N], -_MAX_QF_STEP, _MAX_QF_STEP)

            step = 1.0
            for _ in range(20):
                U_try = U + step * delta
                F_try = _residual_only(U_try, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale)
                res_try = np.max(np.abs(F_try))
                if np.isfinite(res_try) and res_try < res_norm * (1 - 1e-4 * step):
                    break
                step *= 0.5
            else:
                U_try, res_try = U, res_norm

            U = U_try
            F, J = _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale)
            res_norm = np.max(np.abs(F))
            if verbose:
                print(f"  Newton(QF) it {it}: |F|_inf={res_norm:.3e}  step={step:.3g}")

            tiny_step_streak = tiny_step_streak + 1 if step < 1e-4 else 0
            if tiny_step_streak >= 2:
                break
        return U, res_norm, it

    if psi_init is None:
        U, res_norm, it = _run_newton(*_gummel_start())
    else:
        # phin_init/phip_init are already this solver's own native unknowns
        # (or, if warm-starting from the raw-density Newton solver, are
        # already provided in exactly this phin/phip convention too - both
        # solvers derive/use the identical phin = psi - Vt*ln(n/ni) relation).
        U, res_norm, it = _run_newton(psi_init, phin_init, phip_init)
        if res_norm > stall_res_threshold:
            # A warm start can inherit a structurally degenerate Jacobian
            # from its source point - most notably right after equilibrium
            # (Va=0), where phin=phip=psi_eq is exactly FLAT across the
            # whole interior, making every edge's flux-vs-psi Jacobian
            # coupling (proportional to that edge's own delta-phin, which is
            # then exactly zero) vanish identically, not just become small.
            # Warm-starting the next bias point straight from that untouched
            # flat interior inherits the same near-singular structure and
            # the line search can get stuck unable to escape it (caught via
            # exactly this failure at the first forward point after Va=0 in
            # a 1e21-doping sweep). Retry from a fresh Gummel-derived start
            # (the same recovery already used for the sweep's very first,
            # cold-start point) rather than accepting a stuck, wrong answer -
            # Gummel's own decoupled iteration doesn't share this failure
            # mode, so it reliably breaks the exact symmetry.
            U_retry, res_retry, it_retry = _run_newton(*_gummel_start())
            if res_retry < res_norm:
                U, res_norm, it = U_retry, res_retry, it_retry

    if res_norm > stall_res_threshold:
        warnings.warn(
            f"Newton(QF) solve did not converge at Va={Va} V "
            f"(|F|_inf={res_norm:.3e} at iteration {it}) even after a Gummel-restart retry - "
            "check this point's self-consistency (J_std/J_mean) before trusting it.")

    psi, phin, phip = _unpack(U, N)
    n = mat.ni * np.exp((psi - phin) / Vt)
    p = mat.ni * np.exp((phip - psi) / Vt)

    # Report currents from THIS solver's own plain-gradient flux (not
    # physics.edge_currents' Scharfetter-Gummel formula) - the two
    # discretizations agree closely once converged, but using SG here would
    # silently mix formulations in the reported self-consistency check.
    _, _, _, Jn, Jp, _, _, _ = _edge_quantities(psi, phin, phip, n, p, x, mat)
    Jtot = Jn + Jp
    J_interior = Jtot[1:-1] if len(Jtot) > 2 else Jtot
    J_rep = float(np.median(J_interior))

    return {
        "psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
        "Jn": Jn, "Jp": Jp, "Jtot": Jtot, "iters": it,
        "J_mean": J_rep, "J_std": float(np.std(J_interior)),
    }
