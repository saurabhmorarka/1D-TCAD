"""Fully coupled QF Newton solve (core.newton_solver_qf.py's formulation)
extended with reverse-bias junction leakage: Kane band-to-band tunneling
(purely additive generation) and Hurkx trap-assisted tunneling (a
field-enhancement of the SRH recombination/generation term this project
already implements). See tat/tat.py's module docstring for the physics and
plans/tat_btbt_plan.md for the full design rationale.

New module rather than a flag threaded into newton_solver_qf.py - matches
this project's own precedent (newton_solver_qf.py itself was added
alongside newton_solver.py, avalanche/newton_solver_avalanche.py alongside
that) - keeps the plain QF solve's code and behavior byte-for-byte
untouched.

UNLIKE avalanche/newton_solver_avalanche.py, this solver does NOT need
Bank-Rose damping. Avalanche's G_ii depends on |Jn|, |Jp| - the very
quantities the continuity equations solve for - creating direct positive
feedback (more current -> more generation -> more current) responsible for
the S-curve/fold-point pathology documented in DEVELOPMENT_LOG.md session
16. Both G_btbt and G_tat here depend only on the local field and local
densities n, p - never on Jn/Jp - so there is no such self-reinforcing
loop. Plain backtracking Newton (this module reuses
newton_solver_qf.py's own line-search loop, unmodified) is used instead;
this is checked, not merely assumed (see the finite-difference Jacobian
check and the example sweep's own self-consistency in main_tat.py).

Both generation terms plug in like a REPLACEMENT of the existing R term
in newton_solver_qf.py's continuity rows, not an addition alongside it:
the trap-assisted term R_trap(n, p, F) - Hurkx (default) or Schenk, see
tat/tat.py, selected via `trap_generation_fn`/`trap_model` - IS the
(field-enhanced) SRH-like term, collapsing exactly (Hurkx) or closely
(Schenk) onto ordinary SRH behavior at F=0, so it takes over that term's
role rather than sitting beside it. Kane's G_btbt is a genuinely separate
mechanism (no SRH trap at all - direct electron-hole-pair creation) and is
purely additive, subtracted from the effective recombination rate the
same way avalanche/avalanche.py's G_ii is:

    Reff(n, p, psi) = R_trap(n, p, F_node(psi)) - G_btbt(F_node(psi))
    Rn[1:-1] = (dJn/cvol - Q*Reff) / cont_scale   (replaces plain R)
    Rp[1:-1] = (dJp/cvol + Q*Reff) / cont_scale
"""
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.params import Q, Material
from core import physics as ph
from core.jacobian_scaling import equilibrated_spsolve
from core.solver import contact_values
from core.newton_solver_qf import poisson_row_scale, continuity_row_scale, unpack_qf, MAX_QF_STEP
from tat.tat import KaneBTBTModel, HurkxTATModel, btbt_generation, hurkx_tat_generation

# Default trap-assisted generation function - swappable per plans/tat_btbt_plan.md's
# "Hurkx first, Schenk second, same call signature" design (see tat.tat's
# hurkx_tat_generation/schenk_tat_generation docstrings). Passed as
# `trap_generation_fn` through _reff_and_derivs -> _residual_only /
# _residual_and_jacobian -> newton_gummel_solve, paired with whichever
# model object (HurkxTATModel or SchenkTATModel) matches it.
_DEFAULT_TRAP_GENERATION_FN = hurkx_tat_generation


def _node_field(psi, x):
    """Edge field E_e = -(psi[1:]-psi[:-1])/h (this project's existing sign
    convention, matches avalanche/avalanche's own doc), and a length-
    weighted average of the two edges flanking each interior node onto that
    node - same box-averaging style as avalanche's Gii_node, but averaging
    the field value itself rather than integrating a flux-divergence-like
    quantity, since G_btbt/G_tat depend on nodal n, p directly (like
    srh_recombination already does), not on edge-averaged currents.

    Returns (E_e, Eabs_e, F_node, dF_node_dpsi_im1, dF_node_dpsi_i,
    dF_node_dpsi_ip1) - the last three each length N-2 (interior nodes
    1..N-2), giving d(F_node[i])/d(psi at node i-1, i, i+1) respectively.
    """
    h = np.diff(x)
    E_e = -(psi[1:] - psi[:-1]) / h
    Eabs_e = np.abs(E_e)
    sgn_e = np.sign(E_e)

    hm, hp = h[:-1], h[1:]           # h[i-1], h[i] for interior node i
    e_lo, e_hi = slice(0, -1), slice(1, None)  # edge indices (i-1,i) and (i,i+1)
    Eabs_lo, Eabs_hi = Eabs_e[e_lo], Eabs_e[e_hi]
    sgn_lo, sgn_hi = sgn_e[e_lo], sgn_e[e_hi]

    denom = hm + hp
    F_node = (hm * Eabs_lo + hp * Eabs_hi) / denom

    dF_dpsi_im1 = sgn_lo / denom
    dF_dpsi_i = (sgn_hi - sgn_lo) / denom
    dF_dpsi_ip1 = -sgn_hi / denom

    return F_node, dF_dpsi_im1, dF_dpsi_i, dF_dpsi_ip1


def _reff_and_derivs(n, p, psi, x, mat, kane_model, trap_model, trap_generation_fn):
    """Interior-node (length N-2) effective recombination rate
    Reff = R_trap - G_btbt and its partial derivatives w.r.t. n[i], p[i]
    (local, diagonal) and psi[i-1], psi[i], psi[i+1] (via the field).

    trap_generation_fn is either hurkx_tat_generation or
    schenk_tat_generation (tat.tat) - both share the same
    (n, p, F_abs, mat, model) -> (G, dG_dn, dG_dp, dG_dF) signature, paired
    with trap_model being the matching HurkxTATModel or SchenkTATModel."""
    F_node, dF_im1, dF_i, dF_ip1 = _node_field(psi, x)
    n_i, p_i = n[1:-1], p[1:-1]

    G_tat, dGtat_dn, dGtat_dp, dGtat_dF = trap_generation_fn(n_i, p_i, F_node, mat, trap_model)
    G_btbt, dGbtbt_dF = btbt_generation(F_node, kane_model)

    R_trap = -G_tat
    dRh_dn, dRh_dp, dRh_dF = -dGtat_dn, -dGtat_dp, -dGtat_dF

    Reff = R_trap - G_btbt
    dReff_dn = dRh_dn
    dReff_dp = dRh_dp
    dReff_dF = dRh_dF - dGbtbt_dF

    dReff_dpsi_im1 = dReff_dF * dF_im1
    dReff_dpsi_i = dReff_dF * dF_i
    dReff_dpsi_ip1 = dReff_dF * dF_ip1

    return Reff, dReff_dn, dReff_dp, dReff_dpsi_im1, dReff_dpsi_i, dReff_dpsi_ip1


def _edge_quantities(psi, phin, phip, n, p, x, mat):
    """Plain-gradient flux only (no R here - Reff is computed separately
    above, unlike newton_solver_qf.py's _edge_quantities which bundles
    both)."""
    h = np.diff(x)
    n_avg = (n[:-1] + n[1:]) / 2.0
    p_avg = (p[:-1] + p[1:]) / 2.0
    Jn = -Q * mat.mu_n * n_avg * (phin[1:] - phin[:-1]) / h
    Jp = -Q * mat.mu_p * p_avg * (phip[1:] - phip[:-1]) / h
    return h, n_avg, p_avg, Jn, Jp


def _residual_only(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale,
                    kane_model, trap_model, trap_generation_fn=_DEFAULT_TRAP_GENERATION_FN):
    N = len(x)
    psi, phin, phip = unpack_qf(U, N)
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

    _, _, _, Jn, Jp = _edge_quantities(psi, phin, phip, n, p, x, mat)
    Reff, *_ = _reff_and_derivs(n, p, psi, x, mat, kane_model, trap_model, trap_generation_fn)
    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * Reff) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * Reff) / cont_scale

    return np.concatenate([Rpsi, Rn, Rp])


def _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc, poisson_scale, cont_scale,
                            kane_model, trap_model, trap_generation_fn=_DEFAULT_TRAP_GENERATION_FN):
    N = len(x)
    psi, phin, phip = unpack_qf(U, N)
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

    h_e, n_avg, p_avg, Jn, Jp = _edge_quantities(psi, phin, phip, n, p, x, mat)

    dn_dpsi = n / Vt
    dn_dphin = -n / Vt
    dp_dpsi = -p / Vt
    dp_dphip = p / Vt

    dphin_e = phin[1:] - phin[:-1]
    dphip_e = phip[1:] - phip[:-1]

    dJn_dpsi_e = -Q * mat.mu_n / h_e * (dn_dpsi[:-1] / 2.0) * dphin_e
    dJn_dpsi_ep1 = -Q * mat.mu_n / h_e * (dn_dpsi[1:] / 2.0) * dphin_e
    dJn_dphin_e = -Q * mat.mu_n / h_e * ((dn_dphin[:-1] / 2.0) * dphin_e - n_avg)
    dJn_dphin_ep1 = -Q * mat.mu_n / h_e * ((dn_dphin[1:] / 2.0) * dphin_e + n_avg)

    dJp_dpsi_e = -Q * mat.mu_p / h_e * (dp_dpsi[:-1] / 2.0) * dphip_e
    dJp_dpsi_ep1 = -Q * mat.mu_p / h_e * (dp_dpsi[1:] / 2.0) * dphip_e
    dJp_dphip_e = -Q * mat.mu_p / h_e * ((dp_dphip[:-1] / 2.0) * dphip_e - p_avg)
    dJp_dphip_ep1 = -Q * mat.mu_p / h_e * ((dp_dphip[1:] / 2.0) * dphip_e + p_avg)

    (Reff, dReff_dn, dReff_dp,
     dReff_dpsi_im1, dReff_dpsi_i, dReff_dpsi_ip1) = _reff_and_derivs(
        n, p, psi, x, mat, kane_model, trap_model, trap_generation_fn)

    Rn[1:-1] = ((Jn[1:] - Jn[:-1]) / cvol_i - Q * Reff) / cont_scale
    Rp[1:-1] = ((Jp[1:] - Jp[:-1]) / cvol_i + Q * Reff) / cont_scale
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

    # Poisson interior rows - identical to newton_solver_qf.py (Reff/G_btbt
    # don't touch Poisson's own charge term).
    add(idx, idx - 1, lap_m / poisson_scale)
    add(idx, idx, -(lap_m + lap_p) / poisson_scale
        - Q * (dn_dpsi[idx] - dp_dpsi[idx]) / poisson_scale)
    add(idx, idx + 1, lap_p / poisson_scale)
    add(idx, N + idx, (-Q / poisson_scale) * dn_dphin[idx])
    add(idx, 2 * N + idx, (-Q / poisson_scale) * (-dp_dphip[idx]))

    # Electron continuity interior rows.
    r_n = N + idx
    add(r_n, idx - 1, -dJn_dpsi_e[e_lo] / cv / cont_scale
        - Q * dReff_dpsi_im1 / cont_scale)
    add(r_n, idx, (dJn_dpsi_e[e_hi] - dJn_dpsi_ep1[e_lo]) / cv / cont_scale
        - Q * (dReff_dn * dn_dpsi[idx] + dReff_dp * dp_dpsi[idx] + dReff_dpsi_i) / cont_scale)
    add(r_n, idx + 1, dJn_dpsi_ep1[e_hi] / cv / cont_scale
        - Q * dReff_dpsi_ip1 / cont_scale)
    add(r_n, N + idx - 1, -dJn_dphin_e[e_lo] / cv / cont_scale)
    add(r_n, N + idx, (dJn_dphin_e[e_hi] - dJn_dphin_ep1[e_lo]) / cv / cont_scale
        - Q * dReff_dn * dn_dphin[idx] / cont_scale)
    add(r_n, N + idx + 1, dJn_dphin_ep1[e_hi] / cv / cont_scale)
    add(r_n, 2 * N + idx, -Q * dReff_dp * dp_dphip[idx] / cont_scale)

    # Hole continuity interior rows - same pattern, +Q*Reff sign.
    r_p = 2 * N + idx
    add(r_p, idx - 1, -dJp_dpsi_e[e_lo] / cv / cont_scale
        + Q * dReff_dpsi_im1 / cont_scale)
    add(r_p, idx, (dJp_dpsi_e[e_hi] - dJp_dpsi_ep1[e_lo]) / cv / cont_scale
        + Q * (dReff_dn * dn_dpsi[idx] + dReff_dp * dp_dpsi[idx] + dReff_dpsi_i) / cont_scale)
    add(r_p, idx + 1, dJp_dpsi_ep1[e_hi] / cv / cont_scale
        + Q * dReff_dpsi_ip1 / cont_scale)
    add(r_p, 2 * N + idx - 1, -dJp_dphip_e[e_lo] / cv / cont_scale)
    add(r_p, 2 * N + idx, (dJp_dphip_e[e_hi] - dJp_dphip_ep1[e_lo]) / cv / cont_scale
        + Q * dReff_dp * dp_dphip[idx] / cont_scale)
    add(r_p, 2 * N + idx + 1, dJp_dphip_ep1[e_hi] / cv / cont_scale)
    add(r_p, N + idx, Q * dReff_dn * dn_dphin[idx] / cont_scale)

    interior_rows = np.concatenate(rows_list)
    interior_cols = np.concatenate(cols_list)
    interior_data = np.concatenate(data_list)

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
                         kane_model=None, trap_model=None,
                         trap_generation_fn=_DEFAULT_TRAP_GENERATION_FN,
                         f_tol=1e-9, maxiter=50, verbose=False):
    """Same signature/return shape as newton_solver_qf.newton_gummel_solve,
    plus optional kane_model/trap_model (default to the standard Si
    constructors in tat/tat.py if not given) and trap_generation_fn -
    hurkx_tat_generation (default) or schenk_tat_generation from tat.tat,
    paired with a matching HurkxTATModel or SchenkTATModel `trap_model`
    (see plans/tat_btbt_plan.md's Hurkx-then-Schenk design)."""
    kane_model = kane_model or KaneBTBTModel.si_kane_quadratic()
    if trap_model is None:
        trap_model = HurkxTATModel() if trap_generation_fn is _DEFAULT_TRAP_GENERATION_FN else trap_model
    if trap_model is None:
        raise ValueError("trap_model must be given when trap_generation_fn is overridden "
                          "(e.g. schenk_tat_generation needs a matching SchenkTATModel)")

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
    poisson_scale = poisson_row_scale(mat, h_typ)
    cont_scale = continuity_row_scale(mat, h_typ)
    stall_res_threshold = 1.0

    def _gummel_start():
        from core.solver import gummel_solve
        warm = gummel_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq, max_gummel=15)
        return (warm["psi"].copy(),
                warm["psi"] - Vt * np.log(warm["n"] / mat.ni),
                warm["psi"] + Vt * np.log(warm["p"] / mat.ni))

    def _run_newton(psi0, phin0, phip0):
        psi0 = psi0.copy(); phin0 = phin0.copy(); phip0 = phip0.copy()
        psi0[0], psi0[-1] = psi_bc
        phin0[0], phin0[-1] = phin_bc
        phip0[0], phip0[-1] = phip_bc

        U = np.concatenate([psi0, phin0, phip0])
        F, J = _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc,
                                       poisson_scale, cont_scale, kane_model, trap_model, trap_generation_fn)
        res_norm = np.max(np.abs(F))

        it = 0
        tiny_step_streak = 0
        for it in range(1, maxiter + 1):
            if res_norm < f_tol:
                break
            # Ruiz row/column equilibration before the sparse solve (same
            # module avalanche/newton_solver_avalanche.py already reuses
            # for its own extreme-magnitude generation terms) -
            # mathematically exact/recoverable, just better-conditioned
            # arithmetic. Hurkx's own bounded-by-1/tau generation doesn't
            # need it (plain spsolve already converges cleanly), but
            # Schenk's much larger, more rapidly field-varying generation
            # (see tat.py's SchenkTATModel docstring) creates the same
            # class of severe ill-conditioning avalanche's Jacobian hit -
            # confirmed directly: an un-equilibrated Schenk solve stalled
            # at |F|~5e8 from a 1e11 cold start, the identical Newton
            # sequence with equilibration alone reached |F|~2e-5 in 12
            # clean, full-step iterations. Applying it unconditionally
            # (not just for Schenk) keeps one solve path for both models.
            delta = equilibrated_spsolve(J, -F)
            delta[N:3 * N] = np.clip(delta[N:3 * N], -MAX_QF_STEP, MAX_QF_STEP)

            step = 1.0
            for _ in range(20):
                U_try = U + step * delta
                F_try = _residual_only(U_try, x, Cdop, mat, psi_bc, phin_bc, phip_bc,
                                        poisson_scale, cont_scale, kane_model, trap_model, trap_generation_fn)
                res_try = np.max(np.abs(F_try))
                if np.isfinite(res_try) and res_try < res_norm * (1 - 1e-4 * step):
                    break
                step *= 0.5
            else:
                U_try, res_try = U, res_norm

            U = U_try
            F, J = _residual_and_jacobian(U, x, Cdop, mat, psi_bc, phin_bc, phip_bc,
                                           poisson_scale, cont_scale, kane_model, trap_model, trap_generation_fn)
            res_norm = np.max(np.abs(F))
            if verbose:
                print(f"  Newton(TAT) it {it}: |F|_inf={res_norm:.3e}  step={step:.3g}")

            tiny_step_streak = tiny_step_streak + 1 if step < 1e-4 else 0
            if tiny_step_streak >= 2:
                break
        return U, res_norm, it

    if psi_init is None:
        U, res_norm, it = _run_newton(*_gummel_start())
    else:
        U, res_norm, it = _run_newton(psi_init, phin_init, phip_init)
        if res_norm > stall_res_threshold:
            U_retry, res_retry, it_retry = _run_newton(*_gummel_start())
            if res_retry < res_norm:
                U, res_norm, it = U_retry, res_retry, it_retry

    if res_norm > stall_res_threshold:
        warnings.warn(
            f"Newton(TAT) solve did not converge at Va={Va} V "
            f"(|F|_inf={res_norm:.3e} at iteration {it}) even after a Gummel-restart retry - "
            "check this point's self-consistency (J_std/J_mean) before trusting it.")

    psi, phin, phip = unpack_qf(U, N)
    n = mat.ni * np.exp((psi - phin) / Vt)
    p = mat.ni * np.exp((phip - psi) / Vt)

    _, _, _, Jn, Jp = _edge_quantities(psi, phin, phip, n, p, x, mat)
    Jtot = Jn + Jp
    J_interior = Jtot[1:-1] if len(Jtot) > 2 else Jtot
    J_rep = float(np.median(J_interior))

    return {
        "psi": psi, "n": n, "p": p, "phin": phin, "phip": phip,
        "Jn": Jn, "Jp": Jp, "Jtot": Jtot, "iters": it,
        "J_mean": J_rep, "J_std": float(np.std(J_interior)),
    }
