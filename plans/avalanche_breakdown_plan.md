# Avalanche / Impact-Ionization Breakdown Mode for tcad1d

## Context

The tcad1d 1D drift-diffusion solver currently models ordinary CMOS-flow p-n
junction and MOS-cap behavior (Gummel iteration, and two fully-coupled Newton
formulations — raw-density+Scharfetter-Gummel in `newton_solver.py`, and
quasi-Fermi-potential+plain-gradient in `newton_solver_qf.py`, the latter
built specifically to handle extreme/degenerate doping robustly). None of
these solve for impact ionization / avalanche multiplication, which is
correct for a normal CMOS flow (junctions are operated well below their
breakdown voltage) but means the solver cannot model or validate high-reverse-
-bias breakdown behavior at all today.

This plan adds avalanche breakdown as a new, strictly opt-in capability: a
new physics module (field-dependent impact-ionization generation), a new
solver built on `newton_solver_qf.py`'s robust QF formulation, a new damping
strategy (Bank & Rose 1981 "Global Approximate Newton Methods", Algorithm
Global) needed because the ionization generation term's exponential field
dependence creates a sharp positive-feedback nonlinearity that plain
backtracking line search may struggle with near breakdown, an optional mesh
refinement knob, a new example device with a checkable analytic breakdown
voltage, and a closed-form analytic comparison (matching this project's
existing house pattern of validating every new numeric capability against a
textbook closed form, e.g. `shockley_current`/`cv_curve_analytic` in
`analytic.py`). All work happens on a new branch; nothing about the default
CMOS-flow solve path, existing YAMLs, or existing golden tests changes.

## 1. Physics: impact-ionization generation term

Not a new PDE — an extra local electron-hole-pair GENERATION rate `G_ii(x)`
(cm^-3 s^-1) added symmetrically to both continuity equations, same role as
SRH `R` but opposite sign (a net source, not a sink):

```
dJn/dx = q*(R - G_ii)      dJp/dx = -q*(R - G_ii)
```

Van Overstraeten-de Man / Chynoweth local field model (standard silicon
TCAD choice):

```
G_ii = (alpha_n(E)*|Jn| + alpha_p(E)*|Jp|) / q
alpha_n(E) = a_n * exp(-b_n/E)      (single field region)
alpha_p(E) = a_p * exp(-b_p/E)      (two field regions, split ~4e5 V/cm)
```

Coefficients (Si, 300K, van Overstraeten & de Man 1970 — cross-check the 4
constants against Sze or the Sentaurus reference table while implementing
`avalanche.py`, since they're being transcribed from memory):

| carrier | field range (V/cm) | a (cm^-1) | b (V/cm) |
|---|---|---|---|
| electron | 1.75e5 – 6e5 | 7.03e5 | 1.231e6 |
| hole | 1.75e5 – 4e5 | 1.582e6 | 2.036e6 |
| hole | 4e5 – 6e5 | 6.71e5 | 1.693e6 |

Below `E < 1.75e5 V/cm` (the model's own validity floor): `alpha_n=alpha_p=0`
exactly (hard floor, not smoothed — consistent with this codebase's existing
style of exact-floor rather than smoothing, e.g. `bernoulli`'s Taylor-series
floor). This also makes `exp(-b/E)` never see `E->0`.

New file `avalanche.py` (physics.py-style, pure functions):
- `AvalancheModel` dataclass (the six constants above + `E_floor_V_cm`) with
  a `si_von_overstraeten_de_man()` constructor.
- `ionization_coeffs(E_abs, model)` -> `(alpha_n, alpha_p, dalpha_n_dE, dalpha_p_dE)`,
  vectorized `np.where`-based region selection, all same shape as `E_abs`.

## 2. Solver: `newton_solver_avalanche.py` (new module, built on `newton_solver_qf.py`)

New module rather than a boolean flag threaded into `newton_solver_qf.py` —
matches this project's own precedent (`newton_solver_qf.py` itself was added
as a new module alongside `newton_solver.py`, not a flag inside it), keeps
the non-avalanche path's code and behavior byte-for-byte untouched.

Per-edge (in a new `_edge_quantities_avalanche`, otherwise identical to
`newton_solver_qf.py`'s `_edge_quantities`):

```
E_e    = -(psi[1:] - psi[:-1]) / h        # existing sign convention
Eabs_e = |E_e|
alpha_n_e, alpha_p_e, dalpha_n_dE_e, dalpha_p_dE_e = ionization_coeffs(Eabs_e, ii_model)
Gii_e  = (alpha_n_e*|Jn_e| + alpha_p_e*|Jp_e|) / Q
```

Box-integrated onto node `i` the same way flux divergence already is
(length-weighted average of the two flanking edges, `hm=h[i-1]`, `hp=h[i]`,
`cvol_i` as already computed by `ph._control_volumes`):

```
Gii_node[i] = (hm*Gii_e[e_lo] + hp*Gii_e[e_hi]) / (2*cvol_i)
```

Residual: replace `R[1:-1]` with `(R[1:-1] - Gii_node[idx])` in both the
electron and hole continuity rows of `_residual_only`/`_residual_and_jacobian`
(one-line change per row relative to `newton_solver_qf.py`).

Jacobian — new nonlinear coupling not present in `newton_solver_qf.py`
today: `G_ii` depends on `phip` (via `Jp`) inside the *electron* row, and on
`phin` (via `Jn`) inside the *hole* row. Per edge, using
`sgn_n=sign(Jn_e)`, `sgn_p=sign(Jp_e)`, and reusing the 8 flux-derivative
quantities `newton_solver_qf.py` already computes
(`dJn_dpsi_e/ep1`, `dJn_dphin_e/ep1`, `dJp_dpsi_e/ep1`, `dJp_dphip_e/ep1`):

```
dEabs_e/dpsi[i]    =  sign(E_e)/h_e     dEabs_e/dpsi[i+1] = -sign(E_e)/h_e

dGii_e/dpsi[i(+1)]  = (dalpha_n_dE_e*|Jn_e| + dalpha_p_dE_e*|Jp_e|)/Q * dEabs_e/dpsi[i(+1)]
                       + (alpha_n_e*sgn_n*dJn_dpsi_e/ep1 + alpha_p_e*sgn_p*dJp_dpsi_e/ep1) / Q
dGii_e/dphin[i(+1)] = alpha_n_e*sgn_n*dJn_dphin_e/ep1 / Q
dGii_e/dphip[i(+1)] = alpha_p_e*sgn_p*dJp_dphip_e/ep1 / Q
```

then chained through the edge->node weighting exactly as `Gii_node` itself
is (node i's dependence on node i-1 only via edge e_lo's `_ep1` derivative,
on node i+1 only via edge e_hi's `_e` derivative, on node i via both edges'
near-node derivatives) and added via `Q*dGii_node/(...)/cont_scale` into the
COO-triplet `add(...)` calls: the existing `idx-1/idx/idx+1` (psi) and
same-carrier-phi columns get an additional term summed in, and the electron
row additionally gains 3 brand-new `phip` columns
(`2*N+idx-1, 2*N+idx, 2*N+idx+1`) while the hole row gains 3 brand-new
`phin` columns — 6 new nonzero entries per interior row, vectorized over
`idx`, no per-node Python loop, matching the existing assembly style exactly.

`sign(E)`/`sign(J)` are piecewise-constant, non-differentiable exactly at
zero — a standard, accepted simplification (not smoothed; consistent with
`bernoulli`'s house style of exact closed forms with documented floors).
`G_ii >= 0` structurally (like `n`,`p` in the QF formulation) — no clip
needed.

Validate the hand-derived Jacobian with a one-off finite-difference check
(same practice used previously for `newton_solver_qf.py`, see
`DEVELOPMENT_LOG.md` session 9) before trusting it — a throwaway script, not
shipped.

## 3. Damping: Bank & Rose (1981) "Algorithm Global"

New, reusable, solver-agnostic module `bank_rose_damping.py`. Implements the
paper's exact Algorithm Global (Sect. 3, p.287-288), which I've read in full
from the attached PDF:

```
bank_rose_solve(U0, residual_and_jacobian_fn, residual_only_fn,
                 f_tol=1e-9, maxiter=50, K0=0.0, delta=0.5,
                 step_clip_fn=None, verbose=False) -> (U, res_norm, it, K_final)
```

Loop (K tracked as a running damping parameter, not reset every outer step —
matches the paper's own K_k/10-decay-on-success / 10*K_k-growth-on-failure
policy, which is what gives near-quadratic convergence once close to the
root while staying globally robust far from it):

1. At `U_k`: solve `J(U_k) g_k = -F(U_k)` (the plain Newton correction).
2. If `||g_k||_inf < f_tol` (or `||F(U_k)||_inf`, whichever this project's
   existing stall/convergence convention uses — match `newton_solver_qf.py`'s
   own `res_norm = max(|F|)` test for consistency): converged, stop.
3. `t_k = 1/(1+K*||g_k||)`.
4. Trial `U_try = U_k + t_k * clip(g_k)` (optional `step_clip_fn`, used the
   same way `newton_solver_qf.py`'s `_MAX_QF_STEP` clip is used today —
   applied to the raw correction before scaling by `t_k`, since a
   near-zero-density node's quasi-Fermi potential can be almost
   unconstrained by the residual regardless of damping).
5. Evaluate `F(U_try)`, `J(U_try)`, solve for `U_try`'s own Newton
   correction `g_try` (this "global" full re-solve at the trial point,
   not just a residual check, is what makes step acceptance test the
   actual next Newton correction's size — more expensive per trial than
   plain backtracking, but the point of the exercise: a residual-only test
   can accept a step that looks better while leaving the next correction
   just as large, which is exactly the failure mode a sharp
   ionization-onset nonlinearity can trigger).
6. Accept iff `(1 - ||g_try||/||g_k||) / t_k >= delta` (paper's inequality
   3.1, `delta` in `(0, 1-alpha_0)`, a fixed sufficient-decrease constant —
   default `delta=0.5`, tunable).
7. FAIL: `K <- 1 if K==0 else 10*K`; go back to step 3 (same `g_k`, only a
   new trial+re-evaluation, no new Jacobian at `U_k`). Cap retries (e.g. 30)
   with a `warnings.warn` give-up fallback, same spirit as the existing
   line search's bounded retry loop.
8. ACCEPT: `U_{k+1} <- U_try`; `K <- K/10`; carry `g_try` forward as next
   iteration's `g_k` (already computed in step 5, no redundant solve).
9. `k <- k+1`, back to step 1 (well: step 2's convergence check, then step 1
   only needs a fresh solve when `g_k` wasn't already carried from step 8).

Integration in `newton_solver_avalanche.py`: `newton_gummel_solve` gets a
`damping="bank_rose"` argument (default, for this solver only — every other
solver's existing backtracking line search is untouched); `_run_newton`
dispatches to `bank_rose_solve` with `F_fn`/`FJ_fn` closures over
`_residual_only`/`_residual_and_jacobian` and the same `_MAX_QF_STEP`-style
`step_clip_fn`. Everything else in `newton_gummel_solve` (Gummel warm-start,
the flat-interior stall-retry logic, BC construction, current reporting) is
carried over from `newton_solver_qf.py` unchanged.

## 4. Mesh: opt-in extra refinement

`build_diode_grid`'s existing junction-tied `h_min` (Debye-length-based) is
often already fine enough, but the ionization mean free path `1/alpha(E)`
at breakdown fields is a genuinely different physical length scale that can
be shorter for a lightly-doped, high-BV design. Add a targeted, **off by
default** kwarg — every existing call site (and its golden `n_mesh_points`)
is unaffected:

```python
def build_diode_grid(mat, dev, growth=1.06, bulk_spacing_debye_factor=5.0,
                      junction_spacing_debye_factor=0.05,
                      avalanche_ii_refine: dict = None):
    ...
    h_min = junction_spacing_debye_factor * L_D_min
    if avalanche_ii_refine is not None:
        alpha_n, alpha_p, _, _ = ionization_coeffs(
            np.array([avalanche_ii_refine["E_crit_V_cm"]]), avalanche_ii_refine["ii_model"])
        ii_mfp = 1.0 / max(float(alpha_n[0]), float(alpha_p[0]), 1e-30)
        h_min = min(h_min, ii_mfp / avalanche_ii_refine.get("cells_per_mfp", 8.0))
```

`E_crit_V_cm` is a user-supplied a-priori estimate of the peak field near
breakdown (not solved for — just sizes the mesh conservatively); `h_max_*`
(bulk spacing) is untouched since avalanche is junction-local.

## 5. Analytic comparison (new, in `analytic.py`) — matches existing pattern

This project already validates every new numeric capability against a
closed form in `analytic.py` (`shockley_current` vs. the Gummel/Newton I-V,
`cv_curve_analytic` vs. the C-V extraction). Add the avalanche equivalent,
same file, same style:

- `breakdown_voltage_sze(mat, dev)`: Sze's empirical one-sided-abrupt-junction
  formula, `BV ~= 60*(Eg/1.1)^1.5 * (N_B/1e16)^-0.75` volts, `N_B` = the
  lighter side's doping (`min(dev.Na, dev.Nd)`) — a pure closed-form number,
  no simulation dependency, exactly reproducible.
- `ionization_integral(mat, dev, Va, ii_model)`: uses the ALREADY-EXISTING
  `depletion_potential_profile`/`depletion_widths` analytic triangular-field
  depletion approximation (no new field model) to get a closed-form `E(x)`
  across the depletion region at bias `Va`, then numerically quadratures
  `integral of alpha_eff(E(x)) dx` (via `avalanche.ionization_coeffs`) across
  `[-xp, xn]` — the classic Selberherr breakdown criterion (`integral -> 1`
  at breakdown), independent of the PDE solve entirely.
- `multiplication_factor_miller(Va, BV, n=3.0)`: the standard empirical
  Miller formula `M = 1/(1-(|Va|/BV)^n)` (n~3 for p+n-, ~4-6 for n+p- —
  use n=3 for this device's one-sided p+/n- structure), a second independent
  closed-form curve to overlay against the numeric `M(Va) = I_avalanche(Va)/I_no_avalanche(Va)`.

`main_avalanche.py`'s validation plot overlays: numeric I(Va) (avalanche
solver) vs. numeric I(Va) (plain `newton_qf`, no avalanche — same device) vs.
`multiplication_factor_miller` applied to the no-avalanche curve, plus a
vertical marker at `breakdown_voltage_sze`, plus `ionization_integral(Va)`
on a secondary axis/panel crossing 1 near the same `Va`. Three independent
signals (numeric divergence, Miller's closed form, the ionization-integral
closed form) should agree to within the loose tolerance this kind of
empirical cross-check always carries (~10-30%, consistent with this
project's own precedent for approximate closed-form comparisons, e.g. the
C-V curve's documented 2-8% forward-bias / worse-near-Vbi gap).

## 6. New example device, YAML, files

One-sided step junction: heavy side `p_side = 1e19 cm^-3`, light side
`n_side = 1e16 cm^-3` (sets `BV_sze ~= 60V`). Domain sized explicitly
(`Wp_um`/`Wn_um` in the new YAML) to comfortably exceed the depletion width
at `Va ~= -65V` via `analytic.depletion_widths` — the auto-sizing in
`mesh.build_diode_grid` has no notion of "grow for eventual breakdown
depletion width," so this must be explicit (documented in a YAML comment,
matching `input_diode_asymmetric.yaml`'s own precedent for such comments).

**New files:**
- `avalanche.py` — `AvalancheModel`, `ionization_coeffs`
- `bank_rose_damping.py` — `bank_rose_solve`
- `newton_solver_avalanche.py` — avalanche-aware QF Newton solve (Sections 2-3)
- `avalanche_config.py` — `parse_avalanche_config(cfg) -> dict` (reads the
  YAML's `avalanche:` block: `enabled`, `E_crit_V_cm`, `cells_per_mfp`,
  `ii_model` defaulting to `AvalancheModel.si_von_overstraeten_de_man()`)
- `main_avalanche.py` — new driver mirroring `main.py`'s structure (loads
  `input_diode_breakdown.yaml` via `config.load_config`/`build_from_config`
  + the new `avalanche_config.parse_avalanche_config`; builds the mesh with
  `avalanche_ii_refine` set; runs `solver.voltage_sweep(..., method="newton_avalanche")`
  and a second no-avalanche `newton_qf` sweep on the same device for
  comparison; produces the Section 5 validation plot; reuses
  `structure_io.py`/`plot.py` for the rest, same as `main.py`)
- `input_diode_breakdown.yaml` — new example config
- `testsuite/golden/diode_breakdown.json` — captured last, after manual
  verification of the physics

**Modified files (both purely additive allow-list widenings):**
- `config.py`: `math_model` allow-list gains `"newton_avalanche"`.
- `solver.py`: `voltage_sweep`'s dispatch gains
  `elif method == "newton_avalanche": from newton_solver_avalanche import newton_gummel_solve`.
- `mesh.py`: `build_diode_grid` gains the optional `avalanche_ii_refine=None` kwarg.
- `analytic.py`: gains `breakdown_voltage_sze`, `ionization_integral`,
  `multiplication_factor_miller` (Section 5) — pure additions.
- `testsuite/common.py`: new `run_diode_breakdown()` + one new `EXAMPLES`
  entry (regression scalars sampled well BELOW the runaway region — e.g.
  `n_mesh_points`, `BV_sze_estimate_V`, `I_numeric_A` at -10V/-30V where
  `M~=1`, `ionization_integral` at one fixed pre-breakdown `Va` like -48V —
  never inside the sharp divergence itself, so the golden test stays stable
  under normal `RTOL`). `test_examples.py`/`capture_golden.py` need zero
  changes (both already iterate the `EXAMPLES` dict generically).
- `DEVELOPMENT_LOG.md`: new session entry at the end, per established convention.

**Untouched:** `main.py`, `newton_solver.py`, `newton_solver_qf.py`,
`physics.py`, `params.py`, `input_diode.yaml`, `input_diode_asymmetric.yaml`,
`input_mos*.yaml`, existing golden files.

## 7. Implementation order

1. New branch `avalanche-impact-ionization` off `main`.
2. `avalanche.py` + manual sanity probe (e.g. `alpha_n(4e5)` order-of-magnitude check).
3. `analytic.py` additions (Section 5) — pure closed-form, no solver dependency, fast to validate standalone (e.g. `breakdown_voltage_sze` should print ~60V for the planned device).
4. `bank_rose_damping.py` + standalone smoke test against a toy nonlinear system before touching the PDE solver.
5. `newton_solver_avalanche.py` (Sections 2-3) + one-off finite-difference Jacobian check.
6. `mesh.py`'s `avalanche_ii_refine` kwarg; rerun existing testsuite to confirm zero diff.
7. `config.py`/`solver.py` allow-list widenings.
8. `avalanche_config.py`, `input_diode_breakdown.yaml` (compute `Wp_um`/`Wn_um` via a throwaway `analytic.depletion_widths` call first).
9. `main_avalanche.py`; run it, inspect the I-V/Miller/ionization-integral validation plot against the ~60V Sze target.
10. `testsuite/common.py`'s `run_diode_breakdown` + `EXAMPLES` entry; run full suite to confirm zero regression on existing 4 examples; capture the new golden file.
11. `DEVELOPMENT_LOG.md` entry.

## Verification

- `python3 testsuite/test_examples.py` passes for all existing examples
  unchanged, plus the new `diode_breakdown` example once golden-captured.
- `main_avalanche.py`'s printed self-consistency (`J_std/J_mean`) is small
  at every bias point, including near breakdown (checks the Bank-Rose
  damping actually converges the sharp nonlinearity, not just that it runs).
- The three-way analytic cross-check (Section 5) agrees within ~10-30%.
- A one-off finite-difference Jacobian check on `newton_solver_avalanche.py`
  passes at near-floating-point precision (same bar used for `newton_solver_qf.py`).
