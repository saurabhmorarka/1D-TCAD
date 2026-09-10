"""Bank & Rose (1981) "Global Approximate Newton Methods" (Numer. Math. 37,
279-295) - Algorithm Global from Sect. 3 (p.287-288), a globally and
quadratically convergent damped-Newton scheme with an adaptively tuned
damping parameter t_k = 1/(1+K*||g_k||).

Solver-agnostic: operates on any residual_and_jacobian_fn(U)->(F,J),
residual_only_fn(U)->F pair, not tied to any particular PDE discretization.

WHY THIS EXISTS (vs. the plain backtracking line search used elsewhere in
this codebase's Newton solvers): backtracking only tests whether the
RESIDUAL got smaller at a trial point. Algorithm Global instead tests
whether the trial point's OWN Newton correction g_{k+1} is enough smaller
than g_k's (eq 3.1 in the paper) - i.e. it checks that the trial step
actually made progress toward a point where Newton's method itself would
take small next steps, not just a point with smaller ||F||. This "global"
acceptance test is what the paper proves gives quadratic convergence from
ANY starting point (not just locally, once you're already close), which
matters for the avalanche impact-ionization solver: the generation term's
exponential field dependence creates a sharp positive-feedback nonlinearity
near breakdown where a residual-only test can accept a step that "looks"
better while leaving the next Newton correction just as large (an oscillation
plain backtracking can't detect but this test does).

Algorithm Global (paper's steps 1-9, K written here as a plain float that
persists across outer iterations - not reset each step - matching the
paper's own K_k/10-decay-on-acceptance / 10*K_k-growth-on-rejection policy,
which is what lets t_k -> 1 (undamped, quadratic Newton) once near the root
while staying globally robust far from it):

  1. at U_k: solve J(U_k) g_k = -F(U_k)
  2. if ||g_k|| < f_tol: converged, stop
  3. t_k = 1/(1+K*||g_k||)
  4. trial U_try = U_k + t_k * clip(g_k)
  5. evaluate F(U_try), J(U_try); solve for U_try's own correction g_try
  6. accept iff (1 - ||g_try||/||g_k||)/t_k >= delta   (paper's eq 3.1)
  7. reject: K <- 1 if K==0 else 10*K; go to step 3 (same g_k, new trial only)
  8. accept: U_{k+1} <- U_try, K <- K/10, g_k <- g_try (already computed, no
     redundant solve); k <- k+1, go to step 2
"""
import warnings

import numpy as np

from jacobian_scaling import equilibrated_spsolve


def bank_rose_solve(U0, residual_and_jacobian_fn, residual_only_fn,
                     f_tol=1e-9, maxiter=50, K0=0.0, delta=0.5,
                     step_clip_fn=None, max_reject_retries=30, K_max=1e12,
                     verbose=False):
    """Damped Newton iteration via Bank & Rose Algorithm Global.

    residual_and_jacobian_fn(U) -> (F, J), J sparse (any scipy.sparse format
        spsolve accepts).
    residual_only_fn(U) -> F. Fast path (no Jacobian), used for trial
        evaluations that get rejected and re-tried at a different t_k
        without needing a fresh Jacobian solve.
    step_clip_fn(delta) -> delta, optional. Applied to the RAW Newton
        correction before it is scaled by t_k (same rationale as this
        project's other QF-unknown Newton solves: an unconstrained
        quasi-Fermi-potential correction can be numerically huge at a
        near-zero-density node even though the damped step norm the
        algorithm's own K-adaptation targets stays sane).

    Returns (U, res_norm, it, K).

    Every linear solve here goes through equilibrated_spsolve
    (jacobian_scaling.py) rather than a plain spsolve - a numerically exact
    preconditioning step (see that module's docstring) needed because this
    solver's caller (newton_solver_avalanche.py) can hand it Jacobians
    spanning ~29 orders of magnitude between entries, which silently
    degrades a plain direct solve's accuracy on the small-magnitude,
    physically meaningful part of the correction.
    """
    U = U0.copy()
    F, J = residual_and_jacobian_fn(U)
    res_norm = float(np.max(np.abs(F)))
    K = K0
    it = 0

    if res_norm < f_tol:
        return U, res_norm, it, K

    g = equilibrated_spsolve(J, -F)
    if step_clip_fn is not None:
        g = step_clip_fn(g)
    g_norm = float(np.max(np.abs(g)))

    give_up_streak = 0
    for it in range(1, maxiter + 1):
        if g_norm < f_tol:
            F = residual_only_fn(U)
            res_norm = float(np.max(np.abs(F)))
            break

        passed_test = False
        for _retry in range(max_reject_retries):
            t = 1.0 / (1.0 + K * g_norm)
            U_try = U + t * g
            F_try, J_try = residual_and_jacobian_fn(U_try)
            res_try = float(np.max(np.abs(F_try)))
            if not np.isfinite(res_try):
                K = min(10.0 * K, K_max) if K > 0.0 else 1.0
                if K >= K_max:
                    break
                continue

            g_try = equilibrated_spsolve(J_try, -F_try)
            if step_clip_fn is not None:
                g_try = step_clip_fn(g_try)
            g_try_norm = float(np.max(np.abs(g_try)))

            test_lhs = (1.0 - g_try_norm / max(g_norm, 1e-300)) / t
            if test_lhs >= delta:
                passed_test = True
                break
            if K >= K_max:
                # Already at the damping ceiling and still failing the
                # sufficient-decrease test - further growth only pushes t
                # toward the point where a step underflows to no change at
                # all (U_try == U in floating point), which would make the
                # test's denominator go to exactly zero. Stop retrying and
                # fall through to the give-up path below with whatever
                # (however marginal) trial this last attempt produced.
                break
            K = min(10.0 * K, K_max) if K > 0.0 else 1.0
        else:
            # Retry budget for this outer iteration exhausted without
            # passing the test - accept the last trial anyway rather than
            # spin forever (same spirit as this project's other Newton
            # solvers' bounded line-search retry loops), and let the
            # stall-detection below report it if the residual is still large.
            pass

        if verbose:
            print(f"  Bank-Rose it {it}: |F|_inf={res_try:.3e}  t={t:.3g}  K={K:.3g}"
                  f"  passed_test={passed_test}")

        U = U_try
        F, J = F_try, J_try
        res_norm = res_try
        g, g_norm = g_try, g_try_norm
        if passed_test:
            K = K / 10.0
            give_up_streak = 0
        else:
            # K is left as-is (already at its maximum tried value from the
            # retry loop) rather than relaxed, since this trial did NOT
            # pass the sufficient-decrease test.
            give_up_streak += 1
            if give_up_streak >= 3 and res_norm > 1.0:
                warnings.warn(
                    "Bank-Rose damped Newton stalled (3 consecutive "
                    f"sufficient-decrease-test failures at K={K:.3g}, "
                    f"|F|_inf={res_norm:.3e}) - check this solve's "
                    "self-consistency before trusting it.")
                break

    else:
        if res_norm > 1.0:
            warnings.warn(
                f"Bank-Rose damped Newton did not converge within {maxiter} "
                f"iterations (|F|_inf={res_norm:.3e}).")

    return U, res_norm, it, K
