# Trap-Assisted Band-to-Band Tunneling (TAT) Leakage for tcad1d

## Context

The drain-to-substrate junction of a real MOSFET is heavily asymmetric and
degenerately doped on the drain side (drain ~2e20 cm^-3, substrate ~1e17
cm^-3 of opposite type — the same doping regime `newton_solver_qf.py` was
built to handle robustly, see `DEVELOPMENT_LOG.md` session 9). At the
reverse biases this junction sees in normal circuit operation, the dominant
leakage mechanism is **not** avalanche multiplication (that requires fields
high enough to trigger carrier-multiplication feedback, well beyond normal
operating bias) but **band-to-band tunneling assisted by mid-gap trap
states** — carriers tunnel through the trap rather than needing the full
direct/indirect band-to-band gap, which turns on at much lower field and
produces the classic "soft," less sharply-onsetting exponential leakage
curve seen in real drain-substrate junction leakage and GIDL-adjacent
literature, well before any avalanche signature would appear. This is a
distinct, separately opt-in capability from `avalanche/` — same house
pattern (new physics module, new solver module built on
`newton_solver_qf.py`, new example device, new golden test), added without
touching any existing file's behavior.

**Scope for this phase: silicon-silicon junction only.** SiGe (reduced
bandgap on the drain side, e.g. Si0.6Ge0.4, which makes this mechanism
worse) is an explicit follow-on once the pure-Si model is validated — the
alloy-mixing machinery for `Eg`/`chi`/`eps_r` already exists
(`core/materials.py`'s `AlloyMaterial.resolve`), so that phase should mostly
be "hand the SiGe-side Material's `Eg_eV`/`chi_eV` into the same tunneling
formulas" rather than new physics, but is deliberately deferred.

**Longer-term roadmap note (not this phase, not the SiGe phase either):**
every model implemented here (Kane, Hurkx, Schenk) is a *local-field*
tunneling model — the tunneling rate at a point depends only on the field
value AT that point, not on the actual band-edge profile along a real
tunneling path. That's an accepted simplification for a 1D solver, but the
long-term intent (explicitly requested, not this task's scope) is a genuine
**nonlocal tunneling-path search** once this project extends to 2D/3D
device geometry — tracing the real band profile between two points
connected by tunneling, the way Sentaurus's "dynamic nonlocal BTBT" model
does, since local-field models are known to break down when the field
varies significantly over the tunneling distance. Keep `tat/tat.py`
structured so a future nonlocal module can sit alongside it as a new,
separate module later (this project's own established pattern), not
require rewriting the local models.

**Why Hurkx first, Schenk second (not a coin flip):** there is no target
leakage number to calibrate against yet, so the deciding factor is how many
free/unanchored parameters each model asks for, not raw fidelity. Hurkx
degrades EXACTLY onto this project's already-validated
`physics.py:srh_recombination` at zero field (same `tau_n`/`tau_p` this
project's examples already specify), adding only one new, loosely-anchored
knob (`m_t`, tunneling effective mass). Schenk is more microscopic
(phonon-assisted, physically the more correct lens for silicon's indirect
gap) and its constants are fixed materials parameters rather than per-trap
fit knobs — genuinely fewer things to guess in one sense — but it's a more
complex functional form with a thinner cross-validation trail (only found
in FLOOXS + TCAD-insider papers vs. Hurkx's broader citation base). Given
the uncertainty, implement Hurkx first as the lower-risk option, then port
Schenk in as a second, swappable model on the SAME device as a cross-check
— the two should agree in qualitative shape even where magnitudes differ,
and Schenk becomes directly relevant again for the SiGe follow-on (the
indirect-to-direct bandgap crossover as Ge fraction rises matters a lot
there, and Schenk is the model built around exactly that physics).

## 0. Sources actually consulted (not reconstructed from memory)

- **FLOOXS** (open-source TCAD, local clone at `~/Desktop/github_flooxs/flooxs`,
  already used as ground-truth for `newton_solver_qf.py`'s QF formulation):
  `TclLib/Device/floods/Generic/B2BTunnel/simple.tcl` implements the
  standard local **Kane model** for pure band-to-band (Zener) tunneling,
  `G_BTBT = A * F^P * exp(-B/F)`, with three fitted variants for Si (P=1,
  1.5, 2) and literal, in-source-comment-cited coefficients:

  | P | A | B (V/cm) |
  |---|---|---|
  | 1 | 1.1e27 cm^-2 s^-1 V^-1 | 21.3e6 |
  | 1.5 | 1.9e24 cm^-3/2 s^-1 V^-3/2 | 21.9e6 |
  | 2 | 3.4e21 cm^-1 s^-1 V^-2 | 21.6e6 |

  `schenk.tcl` in the same directory implements Schenk's more elaborate
  phonon-assisted model that explicitly couples an SRH-like expression to
  the tunneling prefactor — i.e. FLOOXS's own authors already treat
  "SRH-coupled" and "pure Kane" band-to-band tunneling as two distinct,
  separately-selectable models, which is exactly the Kane-vs-Hurkx/Schenk
  split this plan makes. `schenk.tcl`'s formula (transcribed directly,
  Si-only per its own source comment):
  ```
  n_eff = n * (ni/Nc)^(...Emfn/F correction...)     p_eff analogous
  SchenkSRH = (n_eff*p_eff - ni^2) / ((n_eff+ni)*(p_eff+ni))
  A = 8.977e20  # V/(cm*eV^1.5)
  B = 2.14667e7 # 1/(cm*s*V^2)
  hw = 0.0186   # hbar*omega, eV (phonon energy)
  Fc_mp = B*(Eg -+ hw)^1.5   # phonon absorption/emission branches
  Fc_pm = B*(Eg +- hw)^1.5
  G_schenk = -SchenkSRH * A * F^3.5 * (
        Fc_mp^-1.5 * exp(-Fc_mp/F) / (exp(hw/Vt)-1)
      + Fc_pm^-1.5 * exp(-Fc_pm/F) / (1-exp(-hw/Vt)) )
  ```
  This is directly portable (real, working reference code, not a
  from-memory reconstruction) — planned as the Section 1 second model
  (`SchenkTATModel`), reusing this project's own `Eg_eV`, `ni`, `Vt`
  Material fields for `Eg`/`ni`/`Vt` rather than FLOOXS's own separate
  lookups. FLOOXS's quasi-Fermi-gradient-based density correction
  (`Emfn`/`Emfp`) is the "field-driving-force" refinement noted in its own
  usage docstring as optional (defaults `F=E`) — the simpler `F=E`
  (local field, no quasi-Fermi-gradient correction) default is what this
  plan implements first, consistent with treating Schenk here as another
  LOCAL-field model, same tier as Kane and Hurkx (the QF-gradient
  refinement is closer in spirit to the nonlocal path-search roadmap item
  above than to this phase's scope).
- **Hurkx, Klaassen & Knuvers, "A New Recombination Model for Device
  Simulation Including Tunneling," IEEE Trans. Electron Devices 39(2),
  1992** — not read directly, but its exact resulting formula was recovered
  from a secondary primary source: M. S. Carroll et al. (Sandia National
  Laboratories, SAND2007-1497C conference paper, fetched and read in full),
  which reproduces Hurkx's own figure and states explicitly "expression for
  Hurkx tunneling model, which modifies the standard Shockley-Hall-Read
  generation term":
  ```
  R_trap = (p*n - ni^2) / ( [tau_p/(1+Gamma_p)]*(n + ni*exp(E/kT))
                           + [tau_n/(1+Gamma_n)]*(p + ni*exp(-E/kT)) )
  ```
  (`E` there is the trap level relative to intrinsic level, `Et - Ei`).
  This is literally the physics the task described: SRH recombination
  (this project already implements the `Et=Ei` special case of it,
  `physics.py:srh_recombination`) with the capture/emission process
  enhanced — field-dependent tunneling through the trap makes the
  effective lifetime shorter, `tau -> tau/(1+Gamma(F))`. In reverse bias
  (`n,p << ni`), `R_trap` goes strongly negative — a **generation** term,
  the same sign convention already used for avalanche's `G_ii`.
- The field-enhancement factor `Gamma(F)` itself is the standard
  closed form reported consistently across the TCAD/device-physics
  literature (it is what every commercial tool calls "the Hurkx model"):
  ```
  Gamma(F) = Delta * exp(Delta) * E1(Delta),   Delta = (F / F_Gamma)^2
  F_Gamma  = sqrt(24 * m_t * (kB*T)^3) / (q * hbar)
  ```
  `E1` is the exponential integral (`scipy.special.exp1`), `m_t` a
  tunneling effective mass (Si literature default `0.25*m0`, the same
  value independently confirmed by a second source during this session's
  research). This part is flagged at lower sourcing-confidence than the
  `R_trap`/SRH-modification structure above (recovered from broad
  literature convergence, not one primary PDF) — the finite-difference
  Jacobian check in Section 3 is the actual correctness gate, not the
  citation.

## 1. Physics: two additive generation terms, one new module `tat.py`

Mirrors `avalanche/avalanche.py`'s house style (pure functions, no
solver/mesh dependency, standalone-sanity-checkable).

```python
@dataclass
class KaneBTBTModel:
    A: float = 3.4e21     # P=2 default (indirect-gap Si; matches Kane's
    B: float = 21.6e6     #   phonon-assisted derivation used for indirect
    P: float = 2.0        #   semiconductors, of which Si is one)

def btbt_generation(F_abs, model) -> G_btbt (cm^-3 s^-1):
    return model.A * F_abs**model.P * np.exp(-model.B / F_abs)
    # no field floor needed: F**P -> 0 as F -> 0, exp(-B/F) -> 0 faster;
    # unlike avalanche's alpha(E), there's no historical fit validity
    # floor documented for the Kane form, so this one relies on the
    # functional form's own vanishing at low field, and the sweep's
    # bias range should stay well inside "reverse junction leakage,"
    # never approaching the Kane model's own high-field breakdown of
    # validity (that regime is pure BTBT territory the avalanche model
    # doesn't cover either, and both models sharing the same solver bias
    # range keeps this consistent with the existing breakdown example).

@dataclass
class HurkxTATModel:
    m_t_over_m0: float = 0.25   # tunneling effective mass / free electron mass
    Et_minus_Ei_eV: float = 0.0 # trap level; 0 = midgap, matches this
                                 # project's existing srh_recombination
                                 # convention AND the literature's own
                                 # standard worst-case/typical assumption
                                 # for maximal SRH-generation traps

def hurkx_gamma(F_abs, T, model) -> (Gamma, dGamma_dF):
    # F_Gamma, Delta, Gamma = Delta*exp(Delta)*scipy.special.exp1(Delta)
    # dGamma/dDelta = (1+Delta)*exp(Delta)*E1(Delta) - 1   (closed form,
    #   derived from d/dx[x*e^x*E1(x)], verified against the FD check)
    # dGamma/dF = dGamma/dDelta * dDelta/dF,  dDelta/dF = 2*Delta/F
    ...

def hurkx_tat_generation(n, p, F_abs, mat, model) -> (G_tat, dG_dn, dG_dp, dG_dF):
    # Gamma_n = Gamma_p = hurkx_gamma(...) (single m_t for both carriers,
    # matching the single-mass convention most Hurkx implementations use
    # absent per-carrier tunneling-mass data)
    # n1 = ni*exp(Et_minus_Ei/Vt), p1 = ni*exp(-Et_minus_Ei/Vt)  (=ni at Et=Ei)
    # R = (n*p - ni^2) / (tau_p/(1+Gamma)*(n+n1) + tau_n/(1+Gamma)*(p+p1))
    # G_tat = -R   (only meaningful/large in reverse bias where R<0;
    #   forward bias correctly reduces to ordinary SRH recombination as
    #   Gamma->0 at low field, so this is a strict generalization, not a
    #   reverse-bias-only hack)

@dataclass
class SchenkTATModel:
    A: float = 8.977e20    # V/(cm*eV^1.5), FLOOXS schenk.tcl, Section 0
    B: float = 2.14667e7   # 1/(cm*s*V^2)
    hbar_omega_eV: float = 0.0186   # phonon energy, Si TO-phonon-like value

def schenk_tat_generation(n, p, F_abs, mat, model) -> (G_tat, dG_dn, dG_dp, dG_dF):
    # Direct port of schenk.tcl (Section 0), F=E (local field, no
    # quasi-Fermi-gradient density correction - see Section 0's note).
    # SchenkSRH = (n*p - ni^2) / ((n+ni)*(p+ni))   [F=E special case
    #   collapses FLOOXS's n_eff/p_eff density correction to n,p directly]
    # Fc_mp/Fc_pm from mat.Eg_eV, hbar_omega_eV, model.B (Section 0)
    # G = -SchenkSRH * A * F^3.5 * ( Fc_mp^-1.5*exp(-Fc_mp/F)/(exp(hw/Vt)-1)
    #                               + Fc_pm^-1.5*exp(-Fc_pm/F)/(1-exp(-hw/Vt)) )
```

Both `hurkx_tat_generation` and `schenk_tat_generation` share the same
call signature (`n, p, F_abs, mat, model -> G, dG_dn, dG_dp, dG_dF`) so
`newton_solver_tat.py` (Section 2) can treat "which trap-assisted model"
as a single swappable argument, the same way `AvalancheModel` is swapped
into `newton_solver_avalanche.py` — implement and validate Hurkx first
(Section 5's implementation order), then add Schenk as a second, opt-in
model on the same solver/device once Hurkx's finite-difference check and
sanity probe both pass, per the Section 0 rationale for that ordering.

Sanity probes before wiring into any solver (mirrors avalanche's Step 2):
`btbt_generation` at a representative peak junction field for this device
(~1e6 V/cm, computed via `analytic.depletion_potential_profile` at a
representative reverse bias) should land in a physically reasonable
cm^-3 s^-1 range; `hurkx_gamma` should be ~0 at low field, `O(1)-O(10)` at
the field where TAT-driven leakage becomes visible, growing sharply beyond
that but never developing the pure-Kane term's dominance until much higher
field (this ordering — TAT turns on first, softer; Kane/BTBT dominates only
at much higher field — is the "starts earlier, less sharp" behavior the
task asked for, and should be checked explicitly, not just asserted).

## 2. Solver: `newton_solver_tat.py` (new module, built on `newton_solver_qf.py`)

Structurally simpler than `newton_solver_avalanche.py` — **no Bank-Rose
damping needed**. Avalanche's `G_ii` depends on `|Jn|,|Jp|` (the currents
the continuity equations themselves solve for), creating direct positive
feedback (more current -> more generation -> more current) responsible for
the S-curve/fold-point pathology `DEVELOPMENT_LOG.md` session 16 spent so
long on. Both `G_btbt` and `G_tat` here depend only on local field `F` and
local densities `n,p` — no dependence on `Jn`/`Jp` at all — so there is no
such self-reinforcing loop; the existing plain-backtracking-line-search
Newton machinery in `newton_solver_qf.py` should suffice. This gets
verified, not assumed (Section 4).

Per-node (not per-edge like avalanche's flux-derived `G_ii` — this
generation depends on nodal `n,p` and nodal-averaged field, so it plugs in
more like `srh_recombination` already does):

```
F_node[i] = average of the two flanking edge fields (same |E| already
            computed for other purposes) - or reuse the edge field
            directly analogous to Gii_node's box-integration, whichever
            keeps consistency with the existing box-integrated R[1:-1]
            term newton_solver_qf.py already computes per node.
G_total[i] = btbt_generation(F_node[i], kane_model) \
           + tat_generation(n[i], p[i], F_node[i], mat, hurkx_model)[0]
```

Residual: same one-line change per continuity row as avalanche's plan —
`R[1:-1] -> R[1:-1] - G_total[idx]` in both electron and hole rows.

Jacobian: new couplings are actually simpler than avalanche's — `dG/dn`,
`dG/dp` (both from the Hurkx SRH-like term, same functional shape as the
existing `dR_dn`/`dR_dp` in `newton_solver_qf.py:172-173`, just with
`tau_n`,`tau_p` replaced by their `Gamma`-enhanced effective values) go
into the SAME diagonal blocks the existing SRH term already populates (no
brand-new off-diagonal carrier-cross-coupling the way avalanche's
`phip`-in-the-electron-row coupling was); `dG/dpsi` (via `dF/dpsi`, chained
through both `dG_btbt/dF` and `dG_tat/dF`) goes into the existing
`idx-1/idx/idx+1` psi columns, same pattern as `dR_ii_node/dpsi` in the
avalanche plan.

## 3. Validate the Jacobian before trusting it

One-off finite-difference check (same practice as every prior solver
module in this codebase) on `newton_solver_tat.py`'s assembled Jacobian,
including specifically the `dGamma/dDelta` closed form above (compare
against a numerical derivative of `scipy.special.exp1` directly, since
that piece carries the lower sourcing confidence noted in Section 0) —
must pass at near-floating-point precision before the module is trusted,
per this project's own established bar.

## 4. New example device, YAML, files

Device: exactly the doping the task specified — drain side `Nd = 2e20
cm^-3`, substrate side `Na = 1e17 cm^-3` (opposite type), a one-sided
step junction sized (via `analytic.depletion_widths`, same approach as
the avalanche example's domain sizing) to comfortably hold the depletion
region at whatever reverse bias range this leakage becomes visible over
— informally 0 to -5V given how much earlier than avalanche this should
turn on (to be confirmed empirically once `tat.py`'s sanity probe runs,
not assumed up front).

**New files:**
- `tat/tat.py` — `KaneBTBTModel`, `HurkxTATModel`, `SchenkTATModel`,
  `btbt_generation`, `hurkx_gamma`, `hurkx_tat_generation`,
  `schenk_tat_generation` (Section 1)
- `tat/newton_solver_tat.py` — TAT+BTBT-aware QF Newton solve (Section 2),
  taking the trap-assisted generation function as a swappable argument
  (Hurkx first, Schenk added once Hurkx is validated)
- `tat/tat_config.py` — parses an optional `tat:` YAML block, including a
  `trap_model: hurkx|schenk` switch (mirrors
  `avalanche/avalanche_config.py`'s pattern exactly)
- `tat/main_tat.py` — driver mirroring `avalanche/main_avalanche.py`:
  runs the TAT+BTBT sweep and a plain `newton_qf` (no leakage) sweep on
  the same device, plots I(Va) on log scale showing where the leakage
  knee appears relative to a no-tunneling baseline, plus `Gamma(F)` and
  the two generation terms individually vs. position at a representative
  bias so the "TAT turns on before BTBT dominates" claim is visible
  directly, not just asserted.
- `configs/input_diode_drain_substrate.yaml` — new example config
- `testsuite/golden/diode_tat.json` — captured last, after manual
  verification of the physics, same convention as every other example

**Modified files (purely additive, same allow-list-widening pattern as
the avalanche plan's Section 6):**
- `core/config.py`: `math_model` allow-list gains `"newton_tat"`.
- `core/solver.py`: `voltage_sweep` dispatch gains the new method.
- `testsuite/common.py`: new `run_diode_tat()` + one new `EXAMPLES` entry
  (regression scalars at a couple of fixed, moderate reverse biases —
  same "never inside the steepest part of the curve" convention the
  avalanche golden test already follows, for the same stability reason).
- `DEVELOPMENT_LOG.md`: new session entry at the end.

**Untouched:** everything under `avalanche/`, `core/newton_solver.py`,
`core/newton_solver_qf.py`, `core/physics.py`, existing YAMLs, existing
golden files.

## 5. Implementation order

1. New branch `tat-btbt-leakage` off `main`.
2. `tat/tat.py` + manual sanity probes (Section 1's ordering check —
   TAT visible before BTBT dominates — run standalone before any solver
   work, since if this check fails the whole premise needs revisiting).
3. `tat/newton_solver_tat.py` (Section 2), Hurkx model only at first, +
   finite-difference Jacobian check (Section 3) — try plain backtracking
   first; only reach for `bank_rose_damping.py` (already exists, reusable)
   if plain Newton genuinely struggles, which the no-feedback argument
   above suggests it won't.
4. `core/config.py`/`core/solver.py` allow-list widenings.
5. `tat/tat_config.py` (Hurkx-only for now), `configs/input_diode_drain_substrate.yaml`.
6. `tat/main_tat.py`; run it, inspect the leakage-onset plot against the
   no-tunneling baseline.
7. `testsuite/common.py`'s `run_diode_tat` + `EXAMPLES` entry (Hurkx
   model); run full suite to confirm zero regression on existing
   examples; capture the new golden file.
8. **Add Schenk as the second trap-assisted model**, once 1-7 above are
   validated: `schenk_tat_generation` in `tat/tat.py`, wire into
   `newton_solver_tat.py`'s swappable trap-model argument, its own
   finite-difference Jacobian check, `trap_model: schenk` YAML switch in
   `tat_config.py`, and a second golden-test entry (or a second
   `EXAMPLES` variant of the same device) so Hurkx and Schenk stay
   independently regression-checked side by side.
9. `DEVELOPMENT_LOG.md` entry (covering both models).
10. **Only after Si is validated end-to-end for both models**: revisit
   for the SiGe drain-side follow-on (separate future plan, not this
   one) — swap the drain-side Material's `Eg_eV`/`chi_eV` for a resolved
   `AlloyMaterial("SixGe1-x", ...)` composition, since `KaneBTBTModel`'s
   `B` coefficient is Eg-dependent in the underlying Kane derivation
   (narrower gap -> smaller effective `B` -> stronger tunneling at the
   same field, which is exactly the "SiGe makes it worse" mechanism the
   task described), `HurkxTATModel`'s trap-emission terms depend on `ni`
   (already `Eg`-sensitive via `materials.py:ni_at_T`), and
   `SchenkTATModel`'s phonon-assisted branches depend on `Eg` directly —
   Schenk is expected to matter most here given the indirect-to-direct
   crossover as Ge fraction rises (Section 0's rationale).
11. **Not this phase, not the SiGe phase**: nonlocal tunneling-path
   search, once 2D/3D geometry exists (see the standing roadmap note
   above) — a new module alongside `tat/tat.py`, not a rewrite of it.

## Verification

- `python3 testsuite/test_examples.py` passes for all existing examples
  unchanged, plus the new `diode_tat` example once golden-captured.
- Finite-difference Jacobian check on `newton_solver_tat.py` passes at
  near-floating-point precision.
- `main_tat.py`'s plot shows the TAT-driven leakage knee appearing at
  lower |Va| than the avalanche example's breakdown onset, with a visibly
  softer (less sharp) exponential rise, matching the task's own
  description of the expected qualitative behavior.
