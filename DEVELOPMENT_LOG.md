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
