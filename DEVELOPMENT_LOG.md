# Development Log: 1D Diode TCAD Solver

This is a narrative record of how this simulator was designed, built, and
debugged, session by session, so that anyone (human or AI) can understand
*why* the code looks the way it does and can recreate or extend it from
scratch. It is written from the actual build process, including the bugs
that were hit and how they were found — those are as instructive as the
final working code.

## 1. The request

The goal: build a TCAD (technology computer-aided design) partial
differential equation solver, starting with a 1D simulation of a diode.
Requirements:

- Vary doping in the n-type and p-type regions (constant within each region
  for this first version).
- Sweep the voltage on one side of the diode with the other side grounded.
- Compare against closed-form (analytic) expressions where they exist.
- Create a mesh/grid appropriate for the physics (i.e. resolve the junction
  properly, not just a uniform coarse grid).

## 2. Scoping the model before writing code

Rather than jumping straight to code, the physics model was laid out in
plain language first and checked with the user before implementation, since
there are many reasonable modeling choices for a first TCAD-style solver.
The scope settled on:

- **Grid**: 1D nonuniform mesh spanning a p-region and an n-region with a
  step junction in the middle. Fine spacing (sub-Debye-length, nm-scale)
  near the junction, geometrically expanding to a coarser spacing in the
  bulk. Each quasi-neutral region extends about 5x the minority-carrier
  diffusion length beyond the junction, so the ohmic contacts sit far from
  the depletion region — this is what makes the "long-base" analytic diode
  formula a fair comparison.
- **Equilibrium solve**: the *full* nonlinear Poisson equation (not just the
  depletion approximation) via Newton's method, using Boltzmann statistics
  `n = ni*exp(psi/Vt)`, `p = ni*exp(-psi/Vt)`. This gives the true
  self-consistent built-in potential and space-charge profile, which is then
  compared against the idealized depletion approximation rather than being
  replaced by it.
- **Biased solve**: a Gummel-iteration drift-diffusion solve (the standard,
  more numerically robust alternative to a full 3-way coupled Newton solve
  for a first implementation) — decoupled continuity solves (Scharfetter-
  Gummel discretization, exponentially-fitted flux) alternating with a
  nonlinear Poisson re-solve, with SRH recombination.
- **Closed-form comparisons**: built-in potential `Vbi = Vt*ln(Na*Nd/ni^2)`,
  depletion widths from the depletion approximation, and the Shockley
  long-base ideal diode law `I = I0*(exp(Va/Vt)-1)` with
  `I0 = q*A*ni^2*(Dp/(Lp*Nd) + Dn/(Ln*Na))`.
- **Default parameters**: silicon at 300 K, `ni=1e10 cm^-3`, constant
  mobility (`mu_n=1350`, `mu_p=480 cm^2/Vs`), SRH lifetime `tau_n=tau_p=1 ns`
  (chosen short enough to keep diffusion lengths, and therefore the mesh, a
  manageable size), `Na=1e17 cm^-3` (p-side), `Nd=1e16 cm^-3` (n-side) — all
  exposed as parameters in `params.py`, not hardcoded.

This scoping step mattered: it's what made the later debugging tractable,
because every equation and boundary condition had an explicit, written-down
justification to check bugs against.

## 3. Architecture

| File | Purpose |
|---|---|
| `params.py` | Physical constants and material/device parameters (`Material`, `Device` dataclasses) |
| `mesh.py` | Nonuniform grid generator: fine spacing at the junction, geometric growth to a bulk spacing |
| `physics.py` | Bernoulli function, nonlinear Poisson (Newton, tridiagonal), Scharfetter-Gummel continuity solves, edge-current extraction |
| `analytic.py` | Closed-form comparisons: built-in potential, depletion approximation, Shockley ideal diode law |
| `solver.py` | Equilibrium solve + Gummel-iteration bias sweep with solution continuation across voltage points |
| `main.py` | Driver: builds everything, runs the sweep, generates plots and `iv_sweep.csv` |

All quantities are physical units (cm, s, V, C) throughout — no artificial
Debye-length/ni rescaling was needed, since `psi/Vt` stays O(1-40), well
within double-precision range.

## 4. Building and debugging, step by step

### 4.1 Equilibrium solve — worked on the first try

The nonlinear Poisson Newton solve for the equilibrium (zero-bias) potential
was implemented and tested in isolation first. It matched the analytic
built-in potential `Vt*ln(Na*Nd/ni^2)` to 8 significant figures, and the
bulk carrier concentrations matched the expected doping-set values. This
gave confidence the Poisson solver itself (assembly, Jacobian, tridiagonal
solve) was correct before building anything on top of it.

### 4.2 Bug #1 — Scharfetter-Gummel sign error, found via an equilibrium fixed-point test

Before running the full coupled Gummel loop, the continuity solvers were
tested against a *known exact solution*: feed the exact equilibrium
Boltzmann carrier profile (`n = ni*exp(psi/Vt)` using the converged
equilibrium `psi`) into the continuity solver with zero recombination and
zero net current expected. The Scharfetter-Gummel scheme is specifically
constructed so this equilibrium profile is an *exact* fixed point (zero
current identically) — a textbook property of the scheme.

The first attempt failed this test badly (relative error ~10^13). Tracing
it down to a tiny 6-point uniform-grid test made the bug obvious: the
Bernoulli-function arguments in the flux formula were in the wrong order.
The fix was derived by hand from the detailed-balance identity
`B(a)/B(-a) = e^{-a}` (where `B(x) = x/(exp(x)-1)`) rather than
re-guessing a remembered formula, and applied consistently to both the
electron and hole flux formulas and the terminal-current extraction. After
the fix, both carrier continuity solves reproduced the equilibrium profile
to machine precision (relative error ~1e-16) and gave zero current, as
required.

**Lesson**: test a discretization against a known analytic fixed point on
a tiny grid before ever running it inside the full nonlinear iteration —
it turns an opaque NaN-producing failure into a two-line, hand-checkable
bug.

### 4.3 Bug #2 — undamped Gummel iteration diverging to negative densities

With the flux formula fixed, the full bias sweep still failed: some
voltage points diverged to NaN, and even points that "converged" (in the
sense of hitting the iteration cap) showed a self-consistency residual
(the spread of total current across mesh edges, which must be ~0 in true
steady state) of tens of percent. Digging in with `verbose=True` logging
per Gummel iteration revealed the real problem: the potential update
(`d_psi`) looked like it had converged tightly, while a density-based
convergence metric stayed enormous — and printing the actual continuity-
solve output showed it was producing outright *negative* electron
concentrations by iteration 2-3. The plain (undamped) fixed-point Gummel
map was overshooting and diverging, not oscillating around the right
answer.

The fix: add under-relaxation on the quasi-Fermi-level update (blend only
~35% of each new `phin`/`phip` into the running solution rather than
accepting it outright), clip densities to a small positive floor, tighten
the Newton damping cap inside the Poisson solve, and — critically — switch
the convergence check from a log-ratio of densities (which is numerically
meaningless near the density floor: two physically negligible values like
1e-25 and 1e-5 both round to "not converged" even though neither matters)
to the absolute change in `phin`/`phip`, which are potentials in volts and
therefore well-scaled regardless of how many orders of magnitude the
carrier density spans. After this, the Gummel loop converged monotonically
in ~35-60 iterations with the current self-consistency residual down to
~1e-5 (0.001%) in the well-converged mid-forward-bias regime.

**Lesson**: track convergence in a variable that stays well-scaled across
the whole physical range (here, quasi-Fermi levels in volts), not a ratio
of a raw quantity that spans tens of orders of magnitude — the metric
itself can hide real divergence or manufacture false non-convergence.

### 4.4 Bug #3 — boundary-edge current artifact

Even after fixing the Gummel instability, the reported terminal current
(mean of the current density over *all* mesh edges) showed a larger-than-
expected spread. Printing the per-edge current array showed the bulk was
flat to ~1e-5 relative precision, except the very first edge (touching the
Dirichlet-pinned contact node), which was a clear outlier. This makes
sense: a Dirichlet boundary condition pins the density directly rather than
enforcing a discrete continuity equation at that node, so the flux
computed right at that edge isn't forced to match the interior value even
at full convergence. The fix was to report the terminal current as the
median over interior edges (robust to the 1-2 boundary outliers), and to
use the spread over interior edges as the actual self-consistency
diagnostic.

### 4.5 Bug #4 — inverted ideality-factor formula, caught by picking a better control case

After the physics started producing sensible-looking I-V curves, an
ideality-factor extraction (`n` in `I = I0*exp(V/(n*Vt))`) was added to
visualize the expected transition from recombination-dominated (`n~2`) to
diffusion-dominated (`n~1`) current. The first implementation used
`n = Vt * d(ln I)/dV` and got values clipped between 0.5 and 1 — never
above 1, contradicting the textbook 1-2 range. The bug was a straight
inversion of the defining relation: differentiating `ln I = ln I0 + V/(n Vt)`
correctly gives `n = 1 / (Vt * d(ln I)/dV)`, not `Vt * d(ln I)/dV`.

What made this bug sneaky is that a first sanity check — a control run
with recombination turned off (`tau -> infinity`), where the diode should
be ideal with `n=1` — passed with the *wrong* formula too, because
`1/x = x` at `x=1`; the two formulas only disagree once a second physical
regime (the `n=2` recombination current) is actually present. Only after
fixing the formula did the extracted ideality factor show the expected
rise toward `n~1.85` near the recombination-current peak and relaxation
toward `n~1` at higher forward bias.

**Lesson**: when validating a derived formula with a control case, pick a
control where the correct and incorrect versions give *different* answers
— a control that happens to make them coincide (like `n=1` here) can pass
while hiding an inverted or otherwise wrong formula.

### 4.6 Bug #5 — auto-sized mesh domain blowing up in an edge-case control run

While setting up the no-recombination control run above
(`tau_n=tau_p=1e6 s`), the process hung. The cause: the mesh's automatic
domain sizing (each quasi-neutral region set to ~5x the minority-carrier
diffusion length `L=sqrt(D*tau)`) scales with `sqrt(tau)`, so pushing `tau`
to an extreme deliberately for a control experiment made the requested
domain size explode to tens of thousands of centimeters. The fix for that
one-off test was to pass explicit `Wp`/`Wn` device lengths, decoupling
domain size from the physics parameter being varied. (This is noted in the
`tcad-numerics` agent as a general lesson: don't let a physical parameter a
caller might reasonably push to an extreme silently drive automatic
geometry sizing to something absurd.)

## 5. Final validated results

- Numeric equilibrium built-in potential matches
  `Vt*ln(Na*Nd/ni^2)` to 8 significant figures.
- Equilibrium potential profile matches the depletion approximation, with
  the expected physical difference: the numeric profile is smoothed over a
  Debye length at the depletion edges, where the depletion approximation
  idealizes an abrupt transition.
- Carrier profiles under forward bias show the expected exponential
  minority-carrier injection decaying into each quasi-neutral bulk region.
- Forward I-V current-density is self-consistent (spatially constant) to
  better than 0.01% in the well-converged mid-bias regime.
- Extracted ideality factor rises to `n~1.85` near the SRH recombination-
  current peak (low-to-mid forward bias) and relaxes toward `n~1` at higher
  forward bias where bulk diffusion current dominates — the textbook
  two-regime diode curve — with a further uptick above ~0.6 V consistent
  with the onset of high-level injection.
- Reverse leakage current sits orders of magnitude above the ideal
  Shockley `I0`, which is physically correct: it's dominated by SRH
  generation current in the depletion region, a mechanism the simple
  long-base ideal-diode formula doesn't include.

## 6. Repository and follow-on setup

- A local git repository was initialized in this directory, with `out/`
  (generated plots and `iv_sweep.csv`) and source files committed, and
  `__pycache__`/`*.pyc` excluded via `.gitignore`.
- `gh` (GitHub CLI) was installed via Homebrew; the user authenticated
  interactively (`gh auth login`) since that step can't be done
  non-interactively.
- The repository was created on GitHub as `saurabhmorarka/1D-diode`
  (private) and the local `main` branch pushed and set to track
  `origin/main`.
- Collaborator `marklaw59` was invited with write (push) access via the
  GitHub API (`gh api repos/.../collaborators/... -X PUT -f permission=push`).
- A reusable Claude Code subagent, `tcad-numerics` (stored globally at
  `~/.claude/agents/tcad-numerics.md`, not inside this repo, since it's
  meant to carry forward to future/bigger projects), was created to encode
  the debugging methodology and specific bugs from sections 4.2-4.5 above,
  so a future, larger simulator (e.g. 2D/3D device simulation, additional
  physics like doping-dependent mobility or velocity saturation) can reuse
  the lessons instead of re-discovering them.

## 7. How to recreate or extend this

To rebuild from scratch, follow sections 2-3 above for the model scope and
architecture, then section 4 in order — each subsection's "lesson" is a
test worth writing *before* the corresponding piece of physics, not after:

1. Implement and unit-test the equilibrium nonlinear Poisson solve against
   the analytic built-in potential.
2. Implement the Scharfetter-Gummel continuity solve and test it against
   the equilibrium Boltzmann profile as an exact fixed point (zero current)
   before wiring it into any outer loop.
3. Implement Gummel iteration with under-relaxation from the start, and
   track convergence via quasi-Fermi-level change, not raw density ratios.
4. Report terminal current from interior mesh edges (median), and treat
   the spread across interior edges as a first-class self-consistency
   diagnostic on every solve, not just a debugging aid.
5. Validate every derived/extracted quantity (like the ideality factor)
   against a control case chosen so that a plausible-looking wrong formula
   would give a visibly different answer, not one where it happens to
   coincide with the right one.
6. Keep automatic mesh/domain sizing decoupled from physics parameters that
   might reasonably be swept to an extreme value during testing.

To run the simulator as-is: `python3 main.py` (requires `numpy`, `scipy`,
`matplotlib`); see `README.md` for details.

## 8. Session 2: input file, and a faster coupled Newton solver

Two follow-on requests: (1) move all simulation parameters into a separate,
user-editable input file rather than hardcoded Python, so anyone can rerun
the tool with different doping/geometry/voltage sweep without touching code;
(2) the Gummel iteration felt slow to converge - investigate faster
alternatives and report a runtime comparison.

### 8.1 Input file (`input.yaml` + `config.py`)

`input.yaml` now holds doping (`Na_cm3`, `Nd_cm3`), device thickness
(`Wp_um`, `Wn_um`, or `null` to keep mesh.py's auto-sizing), the voltage
sweep range, and which solver to use (`solver.math_model: gummel | newton`
- see below). `config.py` loads it and overrides the `Material`/`Device`
defaults from `params.py`; `main.py` reads its sweep and solver choice from
there instead of hardcoding them. YAML (via `pyyaml`, added as a
dependency) was chosen over JSON specifically so the file can carry inline
comments explaining each field - it's meant to be hand-edited.

### 8.2 A faster solver: fully coupled Newton instead of Gummel iteration

Gummel iteration is a *decoupled* fixed-point map (solve continuity for n,
then p, given psi; re-solve Poisson given n and p; repeat) and only
converges linearly - each outer iteration reduces the error by roughly a
constant factor, which is why it needed 30-300 iterations per bias point in
session 1. The standard fix is to instead solve Poisson and both continuity
equations as **one coupled nonlinear system** with Newton's method, which
converges *quadratically* near the solution (each iteration roughly squares
the error), needing far fewer outer iterations - at the cost of each
iteration being more expensive (a 3N-unknown linear solve instead of two
N-unknown ones). This is genuinely new work, not a tuning tweak, and it took
three attempts to get right - each attempt's failure was diagnostic, in
keeping with the project's running debugging discipline.

**Attempt 1: Jacobian-free Newton-Krylov (`scipy.optimize.newton_krylov`).**
The appeal was avoiding hand-deriving a Jacobian - just write the residual
and let Krylov (GMRES) iterations approximate the Newton step. It did not
converge at all: the residual norm stayed flat (~1e10) over 60 iterations
regardless of the starting guess. The root cause, found by inspecting which
mesh node dominated the residual, was **the mesh itself**: `mesh.py`'s
grid generator could leave a tiny leftover sliver at a domain boundary
(discovered value: 1.9e-7 cm next to a 6.5e-6 cm neighbor, a >30x jump) when
the geometric spacing sequence didn't evenly divide the requested region
length. That's a real, general mesh-quality bug - the tiny cell gives a
Scharfetter-Gummel flux coefficient `q*D/h` that's enormous, which Gummel's
tridiagonal linear solves tolerated silently (same underlying issue as the
"boundary-edge current artifact" from session 1) but which is exactly the
kind of severe multi-scale ill-conditioning that plain, unpreconditioned
Krylov iteration cannot handle. Fixed by regrading `mesh._one_sided_nodes`
so it never creates a segment smaller than half a step - if the remaining
distance to the boundary is under `0.5*h`, it's absorbed into landing
exactly on the boundary instead of becoming its own sliver segment.

Fixing the mesh did **not** fix Newton-Krylov, though: the residual moved
to a different node (now the finest cell at the junction, where the
diffusion coefficient `D/h^2` is inherently large by construction, not a
bug) and still didn't decrease. This confirmed the deeper issue: the
discretization is intrinsically stiff (h spans ~2 orders of magnitude
between the junction and the bulk), and plain GMRES without a
physics-based preconditioner cannot make progress on that conditioning.
Building a proper preconditioner was judged not worth the complexity when
a more standard alternative was available (see attempt 2).

**Attempt 2: analytic Jacobian, direct sparse solve.** Rather than fight
Krylov conditioning, hand-derive the exact Jacobian (Poisson's row is
linear in `(psi, n, p)` by construction when those are the unknowns
directly, rather than via Boltzmann quasi-Fermi levels - only the
continuity rows are nonlinear, through the Bernoulli-function SG flux and
SRH recombination) and solve each Newton step with a direct sparse LU
(`scipy.sparse.linalg.spsolve`), which doesn't care about the conditioning
the way an unpreconditioned iterative solve does. This needed a new
`bernoulli_deriv` (dB/dx for the Bernoulli function), validated against a
finite-difference derivative before use (matched to ~1e-9 - the same
"validate the building block first" discipline as session 1's SG flux
fix). Deriving the ~20-term Jacobian by hand for the coupled 3-equation
system was error-prone: a first attempt had several row/column
index-mapping mistakes (mixing up which of an edge's two nodes a given
partial derivative belonged to). These were **not** caught by inspection -
they were caught by a random-direction finite-difference check
(`J @ v` vs. `(F(U+eps*v)-F(U-eps*v))/(2*eps)` for random `v`), which is
now the standard way any Jacobian in this codebase should be checked before
it's trusted in a solver loop. After that check passed (relative error
~1e-6, consistent with finite-difference truncation), Newton converged in
the expected 5-15 iterations with genuinely quadratic behavior visible in
the residual trace (e.g. 5.9e9 -> 2.5e8 -> 1.3e6 -> 3.5 -> 5.5e-3 in five
steps) - but wall-clock time was *slower* than Gummel, because the
Jacobian assembly used a Python-level `for` loop over every interior node
and the line search rebuilt the full Jacobian on every trial step, not
just the accepted one.

**Attempt 3 (final): vectorize, and stop rebuilding the Jacobian during
line search.** The assembly loop was rewritten with numpy array indexing
(same style as the vectorized Poisson/continuity assembly from session 1)
instead of a per-node Python loop, and a separate `_residual_only` fast
path was added for line-search trial evaluations, so the (expensive)
Jacobian is only rebuilt once a step is actually accepted. This flipped the
result: Newton became consistently **~2.6x-5x faster in wall-clock time**
than Gummel across the bias sweep, using roughly 5x fewer outer iterations,
while matching Gummel's current to 4+ significant figures.

Two more robustness issues turned up wiring this into the full voltage
sweep (both diagnosed with the same "find the specific failing point,
don't guess" approach as session 1's bugs):

- **Stall detection needed to check the residual size, not just that the
  line search collapsed.** A backtracking line search that can no longer
  find a better step usually means convergence (once the step floor is hit
  right at the solution), but it can also mean a genuine failure to
  converge from a bad starting point, and the original stall check
  couldn't tell these apart - it accepted both. Fixed by only treating a
  stall as "done" when the residual is also below a sanity threshold;
  above it, the solve reports a warning (via `warnings.warn`, matching the
  self-consistency-check style the rest of the codebase uses to surface
  problems, rather than either silently returning a wrong answer or
  aborting an entire multi-point sweep over one difficult bias point).
- **The voltage sweep's own initial-guess handling was undermining
  Newton's cold-start fallback.** `newton_gummel_solve` has a fallback for
  when it has no previous-point solution to warm-start from (run a handful
  of cheap Gummel iterations first to reach a good starting point, then
  switch to Newton) - but `solver.voltage_sweep` was always passing an
  explicit initial guess (the equilibrium solution, unchanged) even for the
  very first sweep point, so that fallback's "no previous solution"
  check (`psi_init is None`) never actually triggered, and the guess it
  passed instead turned out to be worse than either solver's own default.
  Fixed by having `voltage_sweep` pass `None` for the first point so each
  solver falls back to its own appropriate from-scratch guess, and only
  pass the real previous-point solution once one exists.

`solver.py`'s `voltage_sweep` now takes a `method="gummel"|"newton"`
argument and shares its continuation/bookkeeping logic between both
solvers, so they're directly comparable point-by-point; `main.py` runs both
across the full sweep specifically to produce that comparison
(`out/05_solver_benchmark.png`, `out/solver_benchmark.csv`) regardless of
which one `input.yaml` selects for the "primary" results.

**Lesson**, consistent with session 1's: a faster/more sophisticated
numerical method is not a drop-in swap. Each of the three attempts above
failed for a specific, diagnosable reason (a mesh defect, then intrinsic
stiffness defeating an unpreconditioned iterative method, then a
performance bug in an otherwise-correct implementation, then two
robustness gaps in the surrounding sweep logic) - and each was found by
building a small, targeted check (which mesh edge, which Jacobian entry,
which residual node) rather than by tuning parameters and hoping.

### 8.3 Stress-testing the comparison: forward bias up to 1.2 V (high injection)

`input.yaml`'s `voltage_sweep.forward_stop_V` was pushed from 0.65 V to
1.2 V - well past the built-in potential (0.7738 V) and into a regime the
model isn't really designed for (the ohmic contacts are pinned to their
equilibrium majority-carrier concentration with no series resistance, so
there's nothing in the physics to stop the exponential once the junction
approaches flat-band other than the numerics themselves), specifically to
put real stress on the Gummel-vs-Newton comparison rather than only testing
it in the well-behaved low-to-mid-bias range.

The result was the clearest evidence yet for the coupled Newton solver:
in the 0.80-0.95 V band, **Gummel iteration hit its 300-iteration cap
without fully converging** (self-consistency, the J_std/J_mean spread
across interior mesh edges that should be ~0 in true steady state, degraded
to 0.5-7.2% there - visibly worse than the <0.01% it achieves everywhere
else), while **Newton converged cleanly in 7-10 iterations with
self-consistency at essentially machine precision (0.0000%) throughout the
same band**. Total sweep time: Gummel 4.39 s, Newton 0.44 s - a 9.9x
speedup (up from ~4.25x over the milder 0.65 V sweep in section 8.2),
because Gummel's iteration count itself roughly doubled in the hard region
(up to ~90 iterations/point average, spiking to the 300 cap) while
Newton's stayed flat at 5-15 iterations throughout the entire sweep,
reverse bias included. See `out/05_solver_benchmark.png` for the iteration
count and per-point solve time visibly spiking for Gummel and staying flat
for Newton, and `out/gummel_vs_newton_comparison.csv` for the full
point-by-point comparison (current, iteration count, solve time, and
self-consistency for both solvers side by side).

Where Gummel didn't fully converge, its current disagreed with Newton's by
up to ~1.6% (e.g. 0.039359 A vs 0.038830 A at 0.85 V) - given Newton's
self-consistency is essentially exact there and Gummel's is not, Newton's
answer is the more trustworthy one in that band, not just the faster one.

Physically, the sweep also shows why the Shockley ideal-diode law is only
a low-injection approximation: the numeric current visibly saturates
above ~0.8 V (reaching only ~0.21 A at 1.2 V) as the finite doping and
lack of series resistance in this model limit how much current the
junction can actually pass, while the naive exponential extrapolation of
the ideal diode law diverges to a physically absurd ~3e6 A at 1.2 V (see
`out/03_iv_curve.png`) - and the extracted ideality factor climbs well
past n=2, up to about 12 at 1.2 V (`out/04_ideality_factor.png`), which is
the expected signature of high-level injection rather than a numerical
artifact.

## 9. Session 3: extending to a MOS capacitor C-V simulator

New request: build a second device simulator, a 1D MOS capacitor (metal
gate - thin oxide - uniform substrate, no source/drain), reusing the
diode's structure, and compute its C-V curve compared against closed-form
theory. Before writing code, the physics was scoped out loud first (per the
standing preference recorded in this session's memory), specifically to
settle one open question: does a MOS C-V curve need a genuine small-signal
(AC) solve, or can it be done "quasi"?

### 9.1 The key scoping insight: no current path changes everything

A MOS capacitor has an insulating gate, so unlike the diode there is no
steady-state current path at all - the whole structure sits at a single,
uniform Fermi level at every DC gate voltage (exactly the diode's
equilibrium case, `phin=phip=0` everywhere), *provided enough time has
passed for generation-recombination to populate any inversion layer*. That
turns out to settle the "small-signal or not" question cleanly:

- **Low-frequency (quasi-static) C-V** needs no continuity equations, no
  G-R kinetics, and no AC analysis at all - just a sequence of equilibrium
  nonlinear-Poisson solves (`physics.solve_poisson`, reused as-is with an
  oxide/semiconductor permittivity and intrinsic-concentration profile),
  one per gate voltage, with `C(V_G) = dQ/dV_G` from numerically
  differentiating the swept charge. This reuses the diode's equilibrium
  solver almost unchanged.
- **High-frequency C-V** (minority/inversion carriers can't follow a fast
  probe signal) doesn't need a literal frequency-domain solve either - it's
  a **quasi-small-signal** calculation: take the low-frequency solution's
  minority-carrier density at a DC bias, freeze it, perturb V_G by a small
  amount, and let only the majority carrier and potential respond. This is
  the precise, per-point version of the textbook "high-frequency C-V"
  definition (a real AC solve at intermediate frequencies, or the
  frequency-dependent deep-depletion transient behavior from a fast sweep
  with no S/D to supply carriers, would need real G-R kinetics and either
  time-stepping or a complex-linear frequency-domain solve - deliberately
  out of scope for this version).

This meant most of the new work was building the right *structure*
(mesh, permittivity, doping, boundary conditions) rather than a new solver
algorithm - `physics.solve_poisson` needed generalizing (see 9.2) but not
replacing.

### 9.2 Generalizing `physics.solve_poisson` for a layered structure

Three extensions to the diode's Poisson solver, all backward-compatible
(the diode's calls, which pass scalars, are unaffected):

- **`eps` as a per-edge array**, not just `mat.eps`: an oxide/semiconductor
  stack needs a permittivity that's discontinuous at the interface, and
  assigning it per mesh *edge* (not per node) is what makes the
  finite-volume flux automatically enforce D-field continuity there, with
  no special-cased interface treatment needed.
- **`ni` as a per-node array**, not just `mat.ni`: setting `ni=0` in the
  oxide makes `n=p=0` there identically (correct - an insulator has no
  mobile carriers), including a correctly-zeroed Jacobian contribution,
  with no separate "is this an oxide node" branching needed anywhere else
  in the solver.
- **`n_frozen`/`p_frozen`**: override one carrier's density to a fixed
  array at selected nodes instead of the Boltzmann relation, for the
  high-frequency trick above. Symmetric support for freezing either
  carrier was added specifically because the user asked, mid-task, "what
  if it were n-sub and you needed frozen-p mode... should be general
  enough that nmos or pmos could be simulated" - the initial
  implementation only had `n_frozen` (built with the p-substrate example in
  mind), and generalizing it to accept either was a small, clean addition
  once asked for, validated by testing an n-substrate case afterward and
  confirming the threshold voltage and C-V curve come out as an exact
  mirror image of the p-substrate case.

Each new mechanism was checked in isolation before trusting it in the
larger MOS-cap solve: freezing `n` at its equilibrium value and perturbing
a boundary condition confirmed `n` stayed exactly frozen (0.0 relative
difference) while `p` and `psi` responded, before any MOS-specific code
used it.

### 9.3 Three bugs, found the same way as session 1/2: build a small check, don't guess

- **Gate boundary condition was missing the substrate's own reference
  potential.** The first attempt set `psi(gate) = V_G - V_FB` directly.
  At `V_G=0, V_FB=0` this gave 0.35 V of *spurious* band-bending, because
  the substrate's own equilibrium potential (`psi_bulk`, referenced to the
  intrinsic level) isn't zero - it's `-phi_F` for a p-substrate. Applied
  gate voltage is relative to the substrate contact's own Fermi level (the
  external "ground"), not the absolute intrinsic-level reference psi is
  expressed in, so the correct BC is `psi(gate) = psi_bulk + (V_G - V_FB)`.
  Caught by explicitly checking for flat bands at `V_G=0` with the ideal
  `V_FB=0` assumption - it wasn't flat until the offset was added, and was
  flat (surface potential ~1e-15) once it was.
- **Semiconductor charge came out with the wrong sign.** Charge was
  extracted from Gauss's law across the (charge-free, uniform-field) oxide,
  but the first sign choice gave *negative* charge in accumulation, where
  piled-up majority carriers (holes, for a p-substrate) must be net
  *positive*. Caught the same way: compute a known case (strong
  accumulation) and check the sign matches the obvious physical
  expectation, rather than trusting the algebra. A second, related sign
  bug followed immediately: capacitance is `-dQ_semiconductor/dV_G`, not
  `+dQ_semiconductor/dV_G` - the semiconductor charge decreases
  monotonically as `V_G` increases (accumulation to inversion), but
  capacitance must be positive, since it's the *gate* charge
  (`Q_gate=-Q_semiconductor`) that increases with `V_G` in the
  conventional definition.
- **Quasi-Fermi potentials showed a nonsensical +-18V spike right at the
  oxide.** Requested mid-task ("plot the quasi fermi potentials in
  non-equilibrium conditions for both diode and MOS-capacitor"), this
  surfaced immediately on the first MOS plot: `phin`/`phip` are undefined
  in an insulator (n=p=0 there identically, not some small-but-nonzero
  value), but the formula divided by a dummy placeholder value there
  instead of excluding those nodes, giving `Vt*ln(tiny/1.0) ~ -690*Vt`. Fixed
  by masking `phin`/`phip` to NaN wherever `ni_arr==0` (oxide), so they
  simply aren't plotted there - matplotlib skips NaN automatically. This is
  the same class of mistake as session 1's ideality-factor formula bug: a
  quantity that's mathematically well-defined everywhere the formula is
  evaluated can still be *physically meaningless* in part of the domain,
  and needs an explicit mask rather than relying on the numbers looking
  reasonable.

### 9.4 A real physical effect the mesh needed to be pushed to resolve

The numeric low-frequency C-V matched the analytic depletion-approximation
curve well in depletion, but sat visibly below the idealized `C=C_ox` in
accumulation (0.86 x C_ox at strong accumulation with the initial mesh).
Rather than assume this was mesh error to eliminate, it was checked with a
convergence study: refining the near-interface spacing from ~0.82 nm down
to ~0.004 nm converged the result smoothly to a stable ~0.843 x C_ox - not
drifting further as the mesh refined, confirming it's a real effect (finite
accumulation-layer screening length in series with C_ox, not the depletion
approximation's idealized "perfect majority-carrier screening" assumption)
that the *default* mesh simply hadn't been fine enough to resolve. The
default `interface_spacing_debye_factor` for the MOS mesh was tightened
from 0.02 to 0.001 as a result - the accumulation/inversion layers here can
be much thinner than the bulk-doping Debye length that sizes the mesh,
unlike the diode's depletion region.

### 9.5 Gate work function made explicit, not an arbitrary default

Prompted mid-task ("gate work function should be clearly defined... so
[the C-V curve behaves correctly for a p-substrate]"): `V_FB` was
initially just a bare configurable number defaulting to 0 with no stated
justification. It's now computed from an explicit
`gate.workfunction_eV` in `input_mos.yaml` (`null` = the "ideal MOS"
assumption, metal Fermi level aligned with the substrate's own equilibrium
Fermi level, i.e. `V_FB=0` by construction) via the standard
`V_FB = phi_M - phi_S` work-function-difference formula, with
`phi_S = chi_Si + Eg/2 +/- phi_F` computed from the actual substrate doping
- so the flat-band and threshold voltages are always traceable to a stated
physical assumption (or a named real gate material) rather than a silent
default, and the printed summary (`Cox`, `phi_F`, `V_FB`, `V_T`, `W_max`)
makes it easy to check where accumulation/depletion/inversion actually
fall for a given voltage sweep before running it.

### 9.6 Final validated results

- Reproduces the textbook MOS C-V shape exactly, including the low-
  frequency/high-frequency split in inversion: low-frequency capacitance
  rises back toward `C_ox` as the inversion layer forms and responds;
  high-frequency capacitance stays pinned near its value at threshold,
  since the frozen inversion charge can't. See `out/01_cv_curve.png`.
- Depletion-region capacitance matches the analytic depletion
  approximation closely (agreement within a few percent) across the whole
  depletion range on both sides of flat-band.
- Verified generic to both substrate types (see 9.1's `n_frozen`/
  `p_frozen` generalization): threshold voltage for the same 1e16 cm^-3
  doping comes out at +0.728 V for a p-substrate and the exact mirror
  -0.728 V for an n-substrate, with accumulation/depletion/inversion
  correctly swapping which side of `V_FB` they fall on.
- The band-diagram plot's dedicated oxide-only zoom panel shows a linear
  `psi(x)` across the 1 nm oxide at every gate voltage checked - the
  expected signature of zero oxide charge (D-field continuity, no free
  charge to curve the potential there) - visually confirming the
  eps-per-edge interface treatment from 9.2 is working correctly, not just
  passing the aggregate C-V comparison.

### 9.7 A cross-cutting feature added to both tools: quasi-Fermi-potential plots

Requested to apply to both the diode and the new MOS-cap tool, and to be
configurable rather than hardcoded to a single bias point: `field_save.py`
is a small shared module (`resolve_save_points`, `save_fields`,
`plot_quasi_fermi`) that both `main.py` and `mos_main.py` now use, driven
by an `output.save_bias_points` list in each tool's YAML input (accepting
specific bias values, `"all"`, or `"last"`). For the diode this plots the
actual `phin`/`phip` from the bias sweep directly. For the MOS capacitor,
where the low-frequency curve is equilibrium everywhere (`phin=phip=0` by
construction, nothing to plot) and the precise high-frequency perturbation
is too small (millivolts) to see, `mos_main.py` instead generates a
dedicated, clearly-labeled illustrative plot using a deliberately larger
(0.2 V) frozen-carrier perturbation, so the quasi-Fermi splitting the
high-frequency assumption depends on is actually visible.

## 10. Session 4: renaming the repo, and a common, more flexible input format

### 10.1 Repo rename

The GitHub repo (and its README title) had already outgrown the name
`1D-diode` once the MOS capacitor was added, so it was renamed to
`1D-TCAD` via `gh repo rename` (GitHub keeps the old URL as a redirect;
`git remote -v` confirmed the local `origin` URL updated automatically,
no local reconfiguration needed).

### 10.2 The ask: a common, more flexible input format

Make both input files cover the permutations a student would actually want
to try - not just a fixed doping value per side, but a
device *structure* (thickness, mesh) and a doping *shape* (flat, linear-
graded, or gaussian/implant-like), with the two tools' input files sharing
a common schema wherever they share a concept, and the mesh generator
(`mesh.py`) unified into one shared module rather than a diode-only
`mesh.py` plus a separate `mos_mesh.py`. Explicitly deferred: fully
unifying the diode and MOS-cap into one structure-agnostic "device stack"
description (a list of layers the tool doesn't need to know is a diode or
a MOS-cap) - a good direction, but out of scope for this pass; each tool
still has its own YAML file and its own two-argument mesh-builder entry
point (`build_diode_grid`, `build_mos_grid`), the two of which now share
one mesh *engine* underneath.

Before writing anything, confirmed one scoping question with the user:
whether a graded/gaussian doping profile should be allowed to blend across
what used to be a hard layer boundary (e.g. an implant tailing from the
substrate into the oxide) - the user chose to keep each profile confined
to its own region, layers still meeting at a sharp interface. That
decision simplified the implementation a lot: a profile is defined purely
in a region's own local depth coordinate (0 at its reference edge -
the junction, or the oxide/substrate interface - out to that region's own
thickness), so the existing two-regions-meeting-at-x=0 mesh/geometry
handling didn't need to change at all, only what's sampled *within* each
region.

### 10.3 New shared module: `doping_profiles.py`

A single `DopingProfile` dataclass (`type: flat|linear|gaussian`, plus
type-specific fields) used identically by a diode's p-side/n-side and a
MOS capacitor's substrate. `.sample(depth, thickness)` returns the
unsigned concentration at a given depth into the region; `.reference_
concentration()` returns one representative number (exact for flat, an
approximation - peak, or average - otherwise) for every closed-form
formula in `analytic.py`/`mos_analytic.py`, none of which were touched:
they still just consume a scalar Na/Nd/Cdop_substrate, computed from
whichever profile was configured. This was the key design choice that
kept the blast radius small - the physics/analytic layer is completely
insulated from the new doping-profile machinery.

### 10.4 Mesh unification: a real generalization, not just a file merge

Diode and MOS-cap meshes have always been built the same way - grow the
spacing geometrically outward from a hard interface (finest right at it,
where the electrostatic potential bends sharply from depletion physics
even when doping is perfectly flat) - so unifying them into one `mesh.py`
was mostly mechanical: `mos_mesh.py` was deleted and its logic folded in
as `build_mos_grid`, sharing a new `_region_nodes` helper with the diode's
`build_diode_grid`.

The part that needed genuine new logic: a graded/gaussian profile can
demand a fine mesh somewhere in the *middle* of a region too (e.g. a
gaussian implant peak away from the interface), which the old
distance-from-interface-only geometric growth (`_one_sided_nodes`) can't
see at all - it has no idea the doping is even changing. For a non-flat
profile, `_region_nodes` now dispatches to a new adaptive marching
algorithm (`_graded_nodes`) that at every step takes the tighter of two
local spacing limits: the same geometric interface-distance ramp
(recomputed with the LOCAL doping value at that point, not one number for
the whole region) and a `|d(ln N)/dx|`-based limit that catches wherever
the profile itself is changing quickly, regardless of where that falls.
Flat doping keeps using the exact original `_one_sided_nodes` algorithm
(dispatched on `profile.type == "flat"`), so every existing flat-doping
result was verified bit-for-bit-equivalent after the refactor - both
drivers were re-run end to end and reproduced the documented baselines
exactly (Newton/Gummel 9.02x speedup on the diode sweep; 0.8375 accumulation
C/Cox, V_T=0.7284 V on the MOS-cap sweep). The new adaptive path was
smoke-tested separately with a gaussian n-side implant (5e18 cm^-3 peak,
50 nm deep, 30 nm straggle): the mesh refines around the peak as expected,
and a full equilibrium Poisson solve on it converges and satisfies charge
neutrality to machine precision in the quasi-neutral bulk, with the
expected deviation confined to the (wider than usual, given how lightly
doped the gaussian's background tail is) depletion region. As with the
MOS-cap accumulation-mesh finding in Session 3, this adaptive algorithm is
a heuristic, not a proof of convergence - a new graded-doping case should
still be checked by re-running with a tighter mesh and confirming the
answer doesn't move, the same practice this project has followed
throughout.

### 10.5 Input file changes

`input.yaml` was renamed `input_diode.yaml`. Both YAML files gained a
parallel `doping:` schema (`type: flat|linear|gaussian` per region, with
type-specific keys) and a `mesh:` section exposing every mesh-sizing knob
that was previously hardcoded as a Python default argument
(`growth`, `bulk_spacing_debye_factor`, `junction_spacing_debye_factor` /
`interface_spacing_debye_factor`, `n_ox_points`). `input_mos.yaml` also
gained `oxide.eps_r` (previously only settable in `mos_params.py`) and
renamed its `substrate.type` key to `substrate.polarity`, freeing up
`type` to mean the doping-profile shape consistently in both files (it
was otherwise ambiguous with the same key already meaning "p or n" one
level up). `config.py`/`mos_config.py` both now return an extra
`mesh_opts` dict, unpacked with `**mesh_opts` at the `build_diode_grid`/
`build_mos_grid` call site in `main.py`/`mos_main.py`.

## 11. Session 5: embedding result plots in the README

Purely cosmetic, no code changes: the README rendered on GitHub as plain
text with no visuals, even though it already pointed readers at
`out/03_iv_curve.png` and `out/01_cv_curve.png` by filename in the results
prose. Added a side-by-side preview of both plots near the top of the
README (diode I-V, MOS-cap C-V) and turned the two existing filename
mentions into actual embedded `![...](...)` images inline in their
respective results sections, so the two headline results are visible
without cloning the repo.

## 12. Session 6: a structure+fields file format and a growing plot library, aimed at teaching

The ask: real textbook-style plots (an Ec/Ev/Ei/Ef band diagram, not just
electrostatic potential; a fixed/mobile/net charge-density breakdown; a
literal "here's the device" structure diagram with the mesh visible on
it), organized so the plotting code has somewhere to grow into as more
plot types get added, and so the underlying data can be saved once and
handed to a plotter standalone - without requiring a re-run - by anyone
who has the file. Explicitly scoped to not build 2D/3D yet, but to leave
the hooks for it cheap to add later, since this project is expected to
grow into a 2D/3D TCAD tool eventually.

### 12.1 The design, reviewed before writing code

Per this project's established practice (see Session 1's plain-language
scoping step), the design was laid out and confirmed before implementation:

- `structure_io.py`: schema + `save_structure()`/`load_structure()` for one
  JSON file per driver run (`out/diode_structure.json`, `out/
  mos_structure.json`) - device geometry (`regions`), the mesh (`grid.
  x_um`), doping, and per-bias-point fields (psi/n/p/phin/phip). Additive
  to the existing PNG/CSV outputs, not a replacement for either.
- `plot.py`: the plot library, growing over time. Every function takes an
  already-loaded structure dict, not raw arrays, so it doesn't care
  whether it was called in-memory from `main.py`/`mos_main.py` or via the
  file from the standalone CLI (`python3 plot.py out/diode_structure.json`).
- 2D/3D readiness: the schema carries a `dim` field, and `plot.py`'s
  functions dispatch on it, raising `NotImplementedError` for anything but
  `dim=1` today. The one deliberate design choice for extensibility: a
  region's geometry lives under a dimension-specific key
  (`x_range_um` for 1D), so a 2D region can later add polygon keys instead
  of extending this one - nothing else in the schema needs to change to
  add a dimension.

### 12.2 Three new plots, and a physics bug the MOS-cap case caught

`plot_structure()` draws the device as a colored horizontal strip (by
region: p-Si/n-Si/oxide) with mesh NODE POSITIONS drawn as tick marks
underneath - this is what actually shows adaptive mesh refinement
happening, which none of the existing psi(x)/carrier plots make visible.

`plot_charge()` decomposes rho(x)/q into fixed (ionized dopant) charge
(just `Cdop`, since this project doesn't model partial/compensated
ionization - a given mesh node is unambiguously n-type or p-type doped),
mobile carrier charge (`p-n`), and their sum. For the MOS-cap, each saved
gate voltage is also labeled with its accumulation/depletion/inversion
regime, classified in `mos_main.py` from `VG` vs. `V_FB`/`V_T` (kept out of
the generic `plot.py`/schema, which has no MOS-specific concept of a
threshold voltage) and passed through as an optional `regime` string on
each bias point.

`plot_bands()` is where a real bug showed up. The first implementation
defined an intrinsic level `Ei(x) = -psi(x)` (continuous, shared across
materials) and derived `Ec = Ei + Eg/2`, `Ev = Ei - Eg/2`, `E_vacuum = Ec +
chi` from it - correct for a single uniform material (the diode), where it
was tested first and looked right (matched the textbook equilibrium
band-bending picture exactly, Ef flat, Vbi split correctly at the
junction). Applied to the MOS-cap's oxide/substrate interface (different
chi AND Eg on each side), it produced a ~4 eV *jump in the vacuum level*
at the interface - unphysical; a real material interface (no surface
dipole modeled here) has a continuous vacuum level, with the
conduction/valence-band OFFSET coming from each side's own electron
affinity (Anderson's rule), not from splitting a shared intrinsic level by
+-Eg/2. This was caught by literally looking at the rendered plot
(matplotlib's dotted E_vacuum line had a visible kink at x=0), not from
the diode case, which is why "run it and look at the picture" mattered
here beyond just not-crashing. Fixed by making `E_vacuum(x) = -psi(x)` the
primary, continuous quantity, and deriving `Ec = E_vacuum - chi`, `Ev = Ec
- Eg` (both taking chi/Eg as either a single float or a per-node array -
`mos_main.py` builds a per-node chi/Eg array from `g["is_oxide"]`) - for
the diode's single uniform material this is equivalent up to a constant
additive shift (physically irrelevant, energy references are arbitrary),
so its band diagram's shape didn't change, just its absolute vertical
position (now anchored to a physically meaningful electron-affinity-based
reference instead of an arbitrary zero). After the fix, the MOS-cap band
diagram shows a continuous vacuum level and a conduction-band offset of
exactly `chi_Si - chi_ox = 4.05 - 0.9 = 3.15 eV`, matching the approximate
SiO2 constants added to `mos_params.py` (`CHI_OX_EV`, `EG_OX_EV` -
literature-approximate, used only for this qualitative band picture, never
by the actual Poisson/continuity physics, which only ever sees the oxide
through `eps_ox` and zero carrier density).

### 12.3 A second scale problem, same fix applied twice

The MOS-cap's oxide (1 nm) is invisible next to its substrate (>1 um) on
any single linear x-axis - not just for the structure diagram (mesh dots
bunched invisibly at one edge) but for the band/charge diagrams too (the
oxide's entire width rounds to the same pixel as x=0). Rather than pick
one compromise scale, `plot_structure()`/`plot_bands()`/`plot_charge()`
all gained an optional `xlim_um` zoom argument, and `mos_main.py` calls
each of them twice into a two-panel figure: one panel zoomed on the oxide,
one on the substrate depletion region (mirroring the pattern the original
MOS-cap psi(x) plot from Session 3 already used for the same reason). The
diode's single band/charge diagram doesn't need this split since its
p-side/n-side/junction are all comparable (um) scales.

### 12.4 Two follow-up gaps: output naming belongs in the YAML, and a way to explore one saved file

Feedback on the first pass of this feature raised two things:

1. The structure JSON's filename (`diode_structure.json`/`mos_structure.json`)
   was hardcoded in `main.py`/`mos_main.py`, breaking this project's
   consistent rule that simulation *parameters* (including what gets
   written where) live in `input_diode.yaml`/`input_mos.yaml`, not in the
   driver scripts. Fixed by adding `output.structure_file` to both YAML
   files (default the same names as before; `null`/`~` skips writing the
   JSON entirely) and threading it through `config.py`/`mos_config.py`'s
   `build_from_config()` return tuple. `structure_io.py` was split into a
   pure `build_structure()` (no I/O) and `write_structure()`, so the
   drivers can still build the in-memory doc for the band/charge/structure
   plots even when the JSON write itself is disabled.
2. Someone holding only a `*_structure.json` file (no access to the
   original run) had no way to control what a plot showed - `plot.py`
   could only make one fixed version of each diagram. Added `--band-fields`
   /`--charge-fields` (comma-separated subsets of `BAND_FIELDS`/
   `CHARGE_FIELDS`, e.g. `--band-fields Ec,Ev,Ef` to drop E_vacuum/E_i) to
   the existing static-PNG path, plus a new `--interactive` mode that opens
   a live matplotlib window with a checkbox per curve already drawn on the
   axes (`matplotlib.widgets.CheckButtons`, toggling each line's
   visibility on click rather than re-plotting). `--interactive` requires
   picking exactly one `--which` plot, since the checkboxes are keyed to
   whatever's on one shared axes object.

   Wiring the backend was the one non-obvious part: `plot.py` had
   unconditionally called `matplotlib.use("Agg")` at import time (needed
   for headless use as a library from `main.py`/`mos_main.py`, which
   already select Agg themselves before importing `plot`), but a live
   checkbox window needs a real GUI backend, and matplotlib's backend must
   be chosen before `matplotlib.pyplot` is ever imported - i.e. before
   argparse has even run. Resolved with a raw `"--interactive" in sys.argv`
   check ahead of the `matplotlib.use()` call, before any argument
   parsing; when `plot.py` is imported as a library instead of run as the
   `__main__` script, `sys.argv` belongs to the importing process and
   won't contain that flag, so the Agg path is untouched for driver use.

   Flagged for future attention: this interactive viewer will need
   substantial rework once the project extends to 2D/3D (slice-plane
   selection, blanking individual fields, a mesh/grid-overlay toggle, and
   similar controls that only matter once there's more than one spatial
   dimension to navigate) - `plot_bands`/`plot_charge`/`plot_structure`
   already dispatch on `doc["dim"]` for exactly this reason, but the
   `--interactive` CLI mechanism itself (one shared axes, one checkbox per
   line) is a 1D-only starting point, not a finished design.

   The first cut of `--interactive` still required picking one `--which`
   plot up front (`--which bands --interactive`), on the reasoning that
   the checkboxes were keyed to one shared axes. In practice this was
   exactly backwards from how someone actually wants to use it: handed
   only a `*_structure.json` file, the point of an interactive session is
   to explore it *without* already knowing which plot/fields they want -
   `python3 plot.py out/diode_structure.json --interactive` errored
   immediately asking for a choice that should have been made inside the
   session, not on the command line. Fixed by dropping the one-plot
   restriction: `--interactive` now always builds every `--which` plot
   (all three by default) into one figure with side-by-side subplots, and
   `_interactive_show()` takes the whole list of axes, collecting every
   labeled curve across all of them into a single checkbox panel (toggling
   every line sharing a clicked label, in case a curve is ever drawn on
   more than one of the shown axes). `--which`/`--band-fields`/
   `--charge-fields` still work in interactive mode - they narrow what
   gets loaded in the first place - but are no longer required just to
   get the session open.

## 13. Session 7: a real (depletable) polysilicon gate, two numerics bugs
    it exposed, and a regression test suite

### 13.1 The ask: model poly-gate depletion, not just an ideal metal gate

Every MOS-cap example so far modeled the gate as an ideal metal: a
Dirichlet contact sitting directly on the oxide, with no carriers of its
own (`ni=0` there) and therefore no way for the gate side to develop its
own band-bending. Real CMOS gate stacks (before high-k/metal-gate
processes) use doped polysilicon instead, and a poly gate that isn't
doped heavily enough can itself partially deplete near the oxide interface
under bias - the "polysilicon depletion effect", one of the reasons real
processes eventually moved to metal gates. The ask was to add this as a
genuine third region (metal contact - poly - oxide - substrate) rather
than a metal-gate approximation, and to see it actually show up as a
doping-dependent dent in the C-V curve: near-metal at very high poly
doping (~1e20 cm^-3), visibly depleting at lower doping.

### 13.2 Why this was architecturally cheap, and where it wasn't

`physics.solve_poisson` already accepted arbitrary per-node `ni` and
per-edge `eps` arrays - that's exactly what already let oxide (`ni=0`)
sit next to substrate (`ni=mat.ni`) in every prior MOS-cap example. Adding
a poly-gate region turned out to be "just" a third segment of those same
arrays: `mesh.build_mos_grid` gained an optional `Cdop_gate` parameter
that, when given, meshes a poly region between the outer contact and the
oxide (same interface-refined/geometric-growth scheme as the substrate,
mirrored so the fine spacing sits at the poly/oxide interface), and
`MOSDevice` gained `gate_kind`/`gate_profile`/`t_gate` alongside the
existing metal-only `gate_workfunction_eV`. `mos_config.py` parses a new
`gate.type: poly` YAML block (polarity + doping profile + optional
thickness, same schema shape as `substrate.doping`) independent of the
substrate's own polarity, so an n-substrate/p+-poly combination works
exactly the same way as the p-substrate/n+-poly example that ships by
default.

What was NOT cheap - and is exactly why this session ended up finding two
real numerics bugs rather than zero - is that every previous Dirichlet
boundary condition in this codebase sat on a node with `ni=0` (an ideal
metal, or an ohmic contact whose bias never actually varies). The poly
gate is the first Dirichlet contact in this project sitting on a node
with real carriers AND a bias-dependent target potential, and that
combination broke two assumptions that had never been exercised before.

### 13.3 Bug 1: Dirichlet-row pivoting dilution in `solve_poisson`

First symptom: `psi` at the poly/oxide interface came out bit-for-bit
identical across the entire VG sweep, as if the boundary condition simply
wasn't propagating past the first couple of mesh nodes. Tracing a single
Newton iteration by hand (see the debugging note in `physics.py`) found
the actual cause: `physics.solve_poisson`'s Dirichlet rows used a bare
`diag[0] = 1.0`, relying on that row's own equation (`1*delta_0 = 0`) to
keep the boundary's Newton update at exactly zero. That's fine when
nearby coefficients are order-1, but the poly's short Debye length forces
an extremely fine mesh at the interface (sub-Angstrom spacing at
`interface_spacing_debye_factor=0.001`), and a bad early Newton iterate
(psi far from local equilibrium at that first carrier-bearing node) drove
the electron density there to ~1e34 cm^-3 - astronomically past anything
physical. `scipy.sparse.linalg.spsolve`'s partial pivoting, seeing a
neighboring row's coefficient vastly exceed the Dirichlet row's bare 1.0
in that column, pivoted onto the neighboring row instead, silently
leaking a nonzero value into what had to be an exact zero. Fixed by
scaling the Dirichlet diagonal to the local Laplacian-coefficient
magnitude (`max(1.0, lap_coeff_m[0])`), which guarantees it stays the
largest entry in its column regardless of how badly conditioned the
charge term elsewhere gets. This is a general robustness fix (applies to
every example, not just the poly gate) and was verified not to change any
existing diode/metal-gate numeric result at all.

### 13.4 Bug 2: quasi-Fermi levels can't be flat 0 once the gate has carriers

Fixing bug 1 wasn't enough on its own: `psi` at the contact now correctly
tracked VG, but everything past the first few mesh nodes - including,
mysteriously, the *substrate's* own response - still came out completely
VG-independent. The actual cause was a modeling gap, not a numerics bug:
`solve_mos_equilibrium` set `phin = phip = 0` at every node, applying VG
purely as an electrostatic-potential offset at the boundary. That's
correct ONLY when the gate has no mobile carriers (the ideal-metal case -
`ni=0` there, so `phin` is moot regardless of value). With a real poly
gate, `n = ni*exp((psi-phin)/Vt)` at the contact swings exponentially with
VG while `phin` stays pinned at 0 - an artificial charge spike with no
physical basis, which screens itself out within nanometers (Debye
screening doing exactly what it's supposed to, just in response to a
spurious perturbation) and made the rest of the structure look
untouched. The fix follows directly from what "applying a voltage between
two contacts" actually means physically: since the MOS cap carries zero
current in steady state, each side of the oxide is independently in
local equilibrium with ITS OWN contact, so the two sides' quasi-Fermi
levels should be split by VG, not shared - `phin = phip = np.where(x < 0,
VG, 0.0)`, with the step falling inside the carrier-free oxide where its
exact placement is physically moot. Applying this split unconditionally
(not just when a poly gate is present) is harmless for the metal-gate
case for the same `ni=0` reason, and was confirmed bit-identical there.

A third, smaller instance of the same class of bug turned up while
building a doping-dependent comparison plot (13.5): the high-frequency
C-V calculation freezes the substrate's minority carrier so it can't
respond to a small VG perturbation, but was freezing that carrier
species *everywhere*, including inside the poly - for an n+ poly,
electrons are its own majority carrier, so freezing them there pinned the
whole poly and collapsed every high-frequency curve to ~0 regardless of
doping. Fixed by restricting the freeze mask to the substrate side
(`x >= 0`) only, in `mos_solver.cv_sweep`.

The common thread across all three: this codebase's MOS-cap boundary
conditions had only ever been exercised on carrier-free (metal/oxide)
nodes before. Adding the first real semiconductor-to-semiconductor-via-
insulator boundary condition (poly - oxide - substrate) exposed every
place an assumption ("this node has no carriers, so X doesn't matter")
had quietly been baked in without ever being written down.

### 13.5 Two ways to view the result: one doping, or a doping sweep

Two example scripts ship side by side rather than one replacing the
other, since they answer different questions: `input_mos_poly.yaml` (run
via the existing `mos_main.py input_mos_poly.yaml`) is "what does the C-V
look like for one specific poly doping", while the new `mos_poly_sweep.py`
(driven by the same YAML, overriding only the gate doping concentration)
overlays six doping levels (1e17 through 1e22 cm^-3) on one C-V plot to
show the trend directly. The result matches the requested story cleanly:
at VG=+1V, C/Cox rises monotonically from 0.07 (1e17) through 0.90 (1e21)
to 0.95 (1e22), with diminishing returns each decade (the poly's own
series capacitance improves roughly as sqrt(N), so each additional decade
of doping helps less as it approaches the asymptotic ceiling) - while the
accumulation branch (majority-carrier electrons piling up, which any
reasonable doping handles easily) is nearly doping-independent. Both
example voltage sweeps were widened (`input_mos.yaml` to -1..+2V,
`input_mos_poly.yaml` to -2..+1V) after the first pass showed the C-V
curves hadn't yet saturated toward `C_ox` at the original +-1V ends -
confirmed to be normal asymptotic approach (matches Sze's textbook shape),
not a bug.

One open caveat, tying back to a standing scope note (see the
"tcad1d-known-physics-simplifications" reminder): 1e20-1e22 cm^-3 doping
is above silicon's effective conduction-band density of states
(Nc~2.8e19), i.e. genuinely degenerate - this solver still uses
Maxwell-Boltzmann statistics throughout, so the exact "how close to
ideal-metal" numbers at the highest dopings tested should be trusted
qualitatively (higher doping is closer to metal-like) but not
quantitatively without a Fermi-Dirac correction.

### 13.6 A regression test suite, built to catch exactly the bugs above

With three real, previously-latent bugs found in one session, the next
question was how to avoid needing to re-find the next one by hand. Added
`testsuite/`: `common.py` calls each of the four examples' own
config/mesh/solver functions directly (no plotting, no file I/O) and
returns a small dict of scalar summary metrics - `Cox`, `V_FB`, `V_T`,
`Vbi`, `I0`, and C/I sampled at a few representative sweep points, chosen
deliberately small rather than a full field-by-field dump so the suite
stays robust to harmless changes (mesh tuning, plot styling) while still
catching the order-of-magnitude/collapsed-to-zero signature every bug
this session actually produced. `golden/*.json` holds today's
already-verified numbers; `test_examples.py` reruns each example and
diffs against golden with `rtol=1e-3`; `capture_golden.py` regenerates a
golden file deliberately, meant to be run only after independently
verifying a change is correct, never just to make a failing test go
green.

Verified the suite actually catches something, not just tautologically
passing against its own just-captured snapshot: temporarily reintroduced
the bug 2 fix (`phin = phip = np.zeros(N)`) and reran the suite - both
poly-gate tests failed immediately with exactly the collapsed-to-zero
symptom that had originally been debugged by hand, while the diode and
metal-gate tests correctly stayed green (the bug is poly-specific).
Restored the fix and confirmed all four tests pass again before
committing.

## 14. Session 8: a strongly asymmetric diode, Fermi-Dirac reference curves,
a new C-V capability, and two more real solver bugs

Extended the diode side to a p-side 1e17 / n-side degenerate-doping case
(`input_diode_asymmetric.yaml`), to compare Maxwell-Boltzmann (what the
actual nonlinear PDE solve uses everywhere) against Fermi-Dirac
statistics, and to add a diode C-V curve alongside the existing I-V one.

**Fermi-Dirac, scoped deliberately narrow.** `fermi_dirac.py` implements
the Bednarczyk & Bednarczyk (1978) rational approximation for the F_1/2
Fermi integral, used only for equilibrium/contact reference quantities -
built-in potential, Shockley I0, depletion width/capacitance
(`analytic.py`) - not for the transport PDE itself, which would need a
generalized Einstein relation and is a substantially bigger project. This
was enough to get a real, explainable physics result: at Nd=1e20, Vbi
shifts meaningfully (+31mV, 1.012V to 1.043V) while I0/forward current
barely moves (<0.2%) - injection current is dominated by the
non-degenerate p-side, so n-side degeneracy matters a lot for the
electrostatics (Vbi, depletion width) but barely for current.

**A new diode C-V curve**, using the same quasi-static charge-based
dQ/dV approach already used for the MOS-cap C-V (session 9): integrate
`Q * (n - p - Cdop)` over the p-side at each bias point, then
numerically differentiate against Va. Compared against an analytic
depletion+diffusion capacitance reference curve, with an FD variant when
the doping is degenerate.

**Two more real solver bugs, same bug-class as session 8's `physics.py`
fix, found in the separate `newton_solver.py` implementation:**
- Same Dirichlet-row dilution under `scipy.sparse.linalg.spsolve`'s
  partial pivoting (a bare `diag=1.0` boundary row getting swamped by a
  much larger neighboring coefficient) - fixed by scaling each Dirichlet
  diagonal to at least the largest actual coupling entry in that column.
- A boundary-condition/density-floor mismatch: the solution is clipped to
  a 1.0 cm^-3 floor internally, but the boundary target itself was left
  unclipped - for this doping ratio `p_bc` at the n-side contact came out
  to ~0.1 cm^-3, below the floor, so the residual there could never reach
  zero (a permanent, exactly-0.9 stuck residual). Fixed by clipping the
  boundary targets to the same floor.

**A mesh-sizing bug, and then a further generalization of the fix.** The
first version of this example (n-side at 1e21, not 1e20) produced a
23,192-point mesh, because `build_diode_grid`'s bulk-spacing cap used
`min(L_D_p, L_D_n)` - the shorter side's Debye length - for *both* sides,
even though the p-side's own Debye length is ~100x longer. Fixed to a
per-side cap (23192 -> 8966 points). Prompted by the fix, the more basic
question came up: is Debye length even the right scale for the bulk cap
at all, on *any* example, not just this one? It isn't - Debye length is
an electrostatic screening scale, correct for `h_min` at the junction,
but the bulk region far from the junction needs to resolve the injected
minority carrier's exponential decay under bias, which is set by the
*diffusion length* (`mat.Ln`/`mat.Lp`), typically ~100x longer than the
Debye length. So `h_max` was generalized to
`max(bulk_spacing_debye_factor * L_D, L_diffusion / 10)` on both sides -
doping-gradient regions stay separately protected by the mesh's own
gradient limiter regardless. This dropped the asymmetric example further
to 372 points and the plain default diode example from 317 to 241, with
no change to accuracy on either (confirmed via the regression suite,
`rtol=1e-3`, and via full reruns showing unchanged self-consistency).

**An unresolved convergence limit, elevated to a standing blocker.**
Pushing the n-side doping further, to 1e21 (genuinely 10,000x the p-side,
with mesh spacing collapsing to ~0.01nm at the junction to resolve it),
neither Newton nor Gummel converges robustly across most of a bias
sweep, even with both bug fixes above applied - large charge
non-conservation persists at most bias points. Narrowing to the region
that *does* converge cleanly (forward bias above ~0.4V, self-consistency
well under 1%) isolated the failure to a warm-started sweep's
reverse-bias and near-zero-bias region specifically, where the residual
grows monotonically point to point regardless of solver or mesh density.
Two natural hypotheses were tested and ruled out: a bad cold start
propagating forward (fed the first sweep point 300 Gummel iterations
instead of 15 - identical divergence pattern afterward, and even that
extended warm start itself only reached ~82% self-consistency); and the
depletion width exceeding the auto-sized quasi-neutral domain at deep
reverse bias (directly calculated - depletion width reaches only ~2% of
the p-side domain even at -3V). Root cause not found. Given this doping
ratio and bias regime is *exactly* what a real MOSFET's source/drain-to-
substrate junctions look like (1e20-1e21 cm^-3 against 1e16-1e17 cm^-3,
normally reverse-biased), this was deliberately not routed around again
by dialing doping down further - it's recorded as a standing blocker on
future MOSFET source/drain work, to be root-caused on this simpler 1D
diode testbed before it's needed there. The 1e20 case (clearly
degenerate, but inside the solver's actual convergence range) ships as
the example; `input_diode_asymmetric.yaml` documents the narrowed sweep
range and why in comments.
