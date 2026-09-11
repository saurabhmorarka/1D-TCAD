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

## 15. Session 9: the session-8 convergence blocker resolved via a
quasi-Fermi-potential Newton formulation, and a C-V extraction fix

Picked the session-8 blocker back up: the coupled Newton solver
(`newton_solver.py`, raw densities `n`/`p` with Scharfetter-Gummel flux)
still would not converge across reverse bias for the strongly asymmetric,
degenerately-doped junction (p-side 1e17, n-side 1e20-1e21 cm^-3). Two
reformulation attempts changing only the Newton unknowns to log-density
(`ln(n/ni)`, `ln(p/ni)`) while keeping Scharfetter-Gummel - one with
column-only chain-rule Jacobian scaling, one with full row+column
equilibration matching a 2025 published technique - both failed to fix
the target case, and the second even regressed a previously-clean 1e20
example. Both are preserved, uncommitted, on an abandoned
`log-density-formulation` branch.

The fix came from reading real device-simulator source rather than more
reformulation attempts. Genius-TCAD-Open (open-source C++) stays in raw
densities and gets its robustness from Bank-Rose potential damping plus a
PETSc linear-solve backend, not from reparametrizing the unknowns -
suggesting the unknowns weren't the actual problem. A local copy of
FLOOXS (`~/Desktop/github_flooxs`) has two formulations: its "SG" path
matches this project's existing solver; its "QF" path - used by its main,
degenerate-doping-capable models - solves directly for the quasi-Fermi
potentials `phin`/`phip` (a quantity this codebase already computes
post-hoc) and uses a **plain-gradient current with no Scharfetter-Gummel
exponential fitting at all** (`Jn = -q*mu_n*n*grad(phin)`). This removes a
whole layer of compounding nonlinearity both failed attempts kept: SG's
own exponential stacked on top of the density's exponential dependence on
potential.

Implemented as a new, additive module (`newton_solver_qf.py`, wired in as
a third `math_model: newton_qf` option in `solver.py`/`config.py`) rather
than replacing the existing solver, so every prior example's behavior
stays identical unless it opts in. The hand-derived analytic Jacobian's
first version had a systematic sign bug - every flux-derivative entry in
both continuity rows was negated - caught by a finite-difference check
showing a uniform relative error of exactly 2.0 across every sampled
entry (a clean constant ratio, not scattered noise, is the signature of a
sign-convention bug rather than a real discrepancy). After the fix, the
FD check passed to floating-point precision (~1e-10).

End to end, the actual target case - 1e21 doping, the full -2V to +1V
sweep including reverse bias - now converges with self-consistency
~0.000-0.006% throughout. A second, unrelated bug turned up while
regenerating the shipped `out/` examples with the new solver: warm-
starting a bias point from the exact Va=0 equilibrium solution (where
`phin=phip` are flat everywhere) makes the flux-vs-potential Jacobian
coupling vanish identically at every edge, which let Newton's line search
stall on a garbage but small-enough-looking residual (`phip` reaching
-612V) - manifesting as a completely frozen I-V curve for 29 consecutive
bias points. Fixed with a physical-magnitude cap on each Newton step's
raw `phin`/`phip` delta, plus a Gummel-restart retry whenever a warm-
started solve's own residual exceeds a stall threshold. Both examples
(`input_diode.yaml`, `input_diode_asymmetric.yaml`) now default to
`newton_qf`; the asymmetric example's doping was restored to its intended
1e21 with the full reverse-bias sweep re-enabled (removing the 1e20/
forward-only workaround from session 8).

Separately, a long-standing but previously-unnoticed C-V bug surfaced
once the asymmetric example could finally run its full sweep: the
numeric depletion capacitance came out flat and ~10x too low against the
analytic reference across nearly the whole bias range (present even
before this session, at the old 1e20 config - not a regression from the
solver switch). Root cause: `main.py`'s C-V integral included the
p-side's minority-carrier ("pileup") layer, where `n` stays close to the
n-side's own value for several nm past the junction simply because `psi`
hasn't dropped yet that close in - real, Boltzmann-exact physics, not a
solver or mesh artifact (the local Debye length at 1e21 doping is 0.13nm
against an actual mesh spacing there of 0.0065nm - 20 points per Debye
length, i.e. already over-resolved, so tightening the mesh further was
never going to be the fix). This pileup charge is 5-15x larger than the
true depletion charge and nearly bias-independent, so integrating it
together with the real depletion charge and differentiating produces a
near-total cancellation against the actual signal. Fixed by measuring the
pileup layer's physical width once from the equilibrium solution
(wherever the minority carrier exceeds 10x local doping) and excluding a
fixed, 2x-margined node range beyond it from the integral at every bias
point - fixed rather than re-evaluated per bias point, so it doesn't
introduce staircase noise, and it reduces to a zero-width (no-op) case
for the default diode's mild doping ratio. The asymmetric example's C-V
curve now tracks the analytic reference to within 2-8% from -2V through
about +0.5V, diverging only near/above the built-in potential where the
closed-form diffusion-capacitance term is already known to break down
(same limitation the default diode's curve already shows).

## 16. Session 10: avalanche/impact-ionization breakdown, a new special
opt-in mode (branch `avalanche-impact-ionization`)

### 16.1 The ask

Model avalanche breakdown under high reverse bias - impact ionization,
not modeled anywhere in this codebase before (correctly so: a normal
CMOS-flow junction never approaches its breakdown voltage). Explicitly a
special, opt-in capability - "usually turned on with a special switch,
model, solver, and mesh" - not a change to any default example's
behavior, plus an old Bank & Rose (1981) "Global Approximate Newton
Methods" paper the user wanted implemented as a damping strategy for the
sharp nonlinearity avalanche's exponential field-dependence creates.

### 16.2 The physics: a new generation term, not a new PDE

Impact ionization enters as an extra local electron-hole-pair GENERATION
rate `G_ii(x)` added to both continuity equations with the OPPOSITE sign
from SRH recombination (`dJn/dx = q*(R - G_ii)`, `dJp/dx = -q*(R -
G_ii)`), using the standard van Overstraeten-de Man/Chynoweth local-field
model (`alpha_n(E) = a_n*exp(-b_n/E)`, holes split into two field
regions) - new module `avalanche.py`. `G_ii = (alpha_n*|Jn| +
alpha_p*|Jp|)/q` depends on BOTH carriers' currents, which creates a
genuinely new coupling absent from `newton_solver_qf.py`: `G_ii` couples
`phip` into the electron continuity row and `phin` into the hole row
(previously each only touched the OTHER carrier's row through the SRH
term at the same node column; avalanche adds a full 3-wide cross-carrier
stencil).

New solver `newton_solver_avalanche.py` extends `newton_solver_qf.py`'s
quasi-Fermi-potential/plain-gradient formulation (a new module, not a
flag threaded into the existing one, matching how `newton_solver_qf.py`
itself was added alongside `newton_solver.py`). The hand-derived Jacobian
had one real bug, found via a finite-difference check and worth noting
for its exact signature: the new `G_ii`-derivative terms were divided by
the control-volume width `cv` a SECOND time (the box-integrated
`Gii_node = (hm*Gii_e_lo + hp*Gii_e_hi)/(2*cvol_i)` already has that
normalization baked in, unlike the flux terms it was bundled alongside in
the same expression, which DO need the extra `/cv`). Diagnosing it
required first noticing that a naive FD check on a synthetic sinusoidal
test profile was itself unreliable - large `eps` values (1e-3 to 1e-4)
matched the analytic Jacobian to <1% while `eps=1e-6` to `1e-9` gave a
STABLE (not-eps-dependent) ~5000x mismatch, which is what actually
distinguished "real bug" (FD converges to a value analytic disagrees
with, stably, regardless of eps) from "eps too small for the local
nonlinearity" (FD diverges AS eps shrinks, a catastrophic-cancellation
signature) - a useful general lesson for validating Jacobians on
deliberately-adversarial (very large field/current magnitude) test
points. After the fix, FD agreement reached ~4e-7 relative error on a
well-scaled test point and ~1.5e-3 on the original adversarial one
(consistent with second-order truncation error on an extremely nonlinear
exponential, not a formula error).

### 16.3 Bank & Rose damping: implemented, validated, NOT the default

`bank_rose_damping.py` implements the paper's Algorithm Global (Sect. 3)
as a solver-agnostic helper, validated standalone against toy nonlinear
systems (converges quadratically on a well-conditioned coupled system,
matching plain Newton once near the root). One real bug there too: `K`
(the damping parameter) can grow unboundedly across outer iterations
without a ceiling, driving the damped step size `t` to underflow to
exactly `0.0` and then a `0.0/0.0` in the paper's own eq-3.1 acceptance
test - fixed with a `K_max` cap and a bounded give-up path (mirroring
this codebase's existing bounded-retry convention in its other line
searches).

Despite being correctly implemented, Bank-Rose is NOT
`newton_solver_avalanche.py`'s default damping - empirically, on the
shipped breakdown example, its persistent K parameter and strict global
sufficient-decrease test recovered less gracefully from early rejections
than plain backtracking (`newton_solver_qf.py`'s existing line search,
reused here as `damping="line_search"`, the new default), which stayed
well-converged roughly twice as far into reverse bias before both
strategies hit the same wall (see below). Bank-Rose remains fully
available (`damping="bank_rose"`) - see `newton_gummel_solve`'s docstring
in `newton_solver_avalanche.py` for the full comparison.

### 16.4 The wall: voltage-controlled continuation cannot follow avalanche
past its own S-curve

Both damping strategies, independently, stall at the same device- and
mesh-dependent bias (~Va=-23V for the shipped example, well before the
Sze BV estimate of ~62V) - every subsequent continuation point then
returns an unchanged, non-physical value. This is not a damping bug: a
VOLTAGE-controlled bias sweep cannot follow `I(Va)` past the point where
`dI/dVa` formally diverges (avalanche's own vertical/S-shaped branch) -
a well-known device-simulation limitation; only a current-controlled
sweep or a ballast resistor can continue past it, and neither is
implemented (out of scope this session). `input_diode_breakdown.yaml`'s
reverse sweep is deliberately kept short of that wall
(`reverse_stop_V: -22.0`), where clear multiplication is already visible
(numeric M ratio growing, ionization integral rising smoothly toward 1)
without the sweep freezing.

### 16.5 Everything else, briefly

`analytic.py` gained three closed-form comparisons (matching this
project's existing house pattern, e.g. `shockley_current`):
`breakdown_voltage_sze` (Sze's empirical one-sided-junction formula),
`ionization_integral` (Selberherr's criterion, evaluated from the
existing depletion-approximation field profile, independent of the PDE
solve), and `multiplication_factor_miller`. These two closed forms
disagree with each other more than the ~10-30% originally guessed (the
ionization integral crosses 1 around Va~-35V using the depletion
approximation vs. Sze's ~62V) - expected given how exponentially
sensitive avalanche onset is to the exact field model and profile shape,
not a bug in either; documented rather than forced to agree.
`mesh.build_diode_grid` gained an opt-in `avalanche_ii_refine` kwarg
(off by default, zero effect on any existing example) that additionally
caps `h_min` at the impact-ionization mean free path `1/alpha(E_crit)`
when passed. New driver `main_avalanche.py`, new config parser
`avalanche_config.py`, new example `input_diode_breakdown.yaml`, new
regression test `diode_breakdown` in `testsuite/common.py` (metrics
sampled from the well-converged early-reverse-bias region, never from
inside the runaway) - full existing suite (4 examples) still passes
unchanged.

### 16.6 Follow-up: the real bug behind the "wall", and a better example device

Requested diagnostics (I(Va) on linear AND log scales, Bank-Rose vs.
line-search compared directly with per-point iteration/self-consistency
plots - new `avalanche_diagnostics.py`) turned out to expose that the
"wall" described in 16.4 was NOT (mainly) the voltage-controlled-S-curve
limit it was first diagnosed as. Plotting self-consistency and current
side by side per bias point showed the reported current jumping by up to
11 orders of magnitude within a fraction of a volt and then FREEZING at
an identical, Va-independent value for every subsequent point - not the
gradual steepening a real S-curve approach produces. Inspecting the
frozen state directly: electron density pinned at exactly its 1 cm^-3
floor everywhere, hole density and current several orders of magnitude
beyond anything physical, yet only 2 Newton iterations and a
deceptively-not-catastrophic self-consistency ratio (~10) - a genuine
algebraic root of the discretized system, just the wrong one.

Root cause, found by bisecting what actually changes when this happens:
`newton_solver_avalanche.py`'s Newton step clip (`_clip`) capped the raw
`phin`/`phip` correction (`_MAX_QF_STEP`, inherited from
`newton_solver_qf.py`) but left `psi`'s own raw correction completely
UNCLIPPED. Under strong avalanche feedback the psi-psi Jacobian block can
become locally very stiff, letting one Newton step swing `psi` by tens of
volts and jump clean over the physical (low-current) branch onto a
spurious one. Two other approaches were tried and reverted before finding
this: (a) picking between a continuation candidate and a
fresh-start/generation-strength-ramped candidate by comparing
self-consistency - abandoned because the spurious branch's
self-consistency ratio is not reliably worse than the physical branch's,
so the comparison sometimes picked the WRONG one; (b) the ramp
machinery itself (`_run_ramp`, an `ii_scale` parameter threaded through
the residual/Jacobian to turn G_ii on gradually) - removed as dead code
once the simple psi clip alone proved sufficient. Fix: cap the raw psi
step at 1V too (`_clip`), same spirit as the existing phin/phip cap.
Confirmed by testing across sweeps of different point spacing (which had
been landing on the spurious branch at DIFFERENT bias points depending on
step size - itself a tell that something was numerically fragile, not
that a true physical limit had been reached): with the psi clip, the same
device now sweeps smoothly and reproducibly from equilibrium out to
~Va=-38V before any issue recurs, up from a first-jump around Va=-18 to
-31V (device- and spacing-dependent) beforehand.

Separately, plotting the result revealed the shipped example's first
doping choice (light side 1e16 cm^-3, BV~=62V) needed a very long sweep
before showing dramatic avalanche behavior, and the light-side field
strength meant genuinely reliable convergence didn't extend far enough
past the knee to look convincing on a plot. Retargeted the example to a
more heavily doped ("Zener-like") light side, 1e17 cm^-3
(`analytic.breakdown_voltage_sze` gives BV~=11V for this doping) - this
both matches the intuition that avalanche/Zener diodes commonly break
down in the 5-15V range at this kind of doping, AND lands the interesting
physics well inside the solver's now-larger reliable range. The result
(`input_diode_breakdown.yaml`, `reverse_stop_V: -13.9`) shows a clean,
textbook breakdown knee on the linear-scale I(Va) plot and a ~2-order-of-
-magnitude exponential-looking rise on the log-scale plot, crossing the
no-avalanche baseline by orders of magnitude, with the analytic ionization
integral crossing 1.0 almost exactly at the Sze BV marker - all three
independent signals (numeric I(Va), Miller's closed form, the
depletion-approximation ionization integral) now agree on where breakdown
happens, unlike the first (1e16-doped) device where they disagreed by
close to 2x. `avalanche_diagnostics.py`'s comparison confirms Bank-Rose
still degrades earlier than line-search on this retargeted device too
(self-consistency spikes to 1e6-1e18 from about Va=-12V on, while
line-search stays under ~10 all the way to -14V) - the choice of
line-search as this solver's default stands.

### 16.7 Pushing to visibly large currents: a fine-tail sweep, and a
robustness safety net

Wanted the example to show current reaching ~1e-5 to 1e-4 A (a genuinely
large, unmistakable avalanche current), not just the ~2 order-of-magnitude
rise 16.6 left off with. Manual bisection right at the edge of the
voltage-controlled wall found a narrow but very clean window - stepping
in 0.004V increments from -13.8V to -13.988V, current rises smoothly and
monotonically from ~1e-7 A to 5.6e-5 A with EXCELLENT self-consistency
throughout (down to ~4e-6 at the final point) - the physical branch is
there and trackable, it just needs much finer bias-point spacing than is
practical to use over the whole sweep. New `avalanche_config.py`
`avalanche.fine_tail` block ({start_V, stop_V, step_V}) and
`main_avalanche.py`'s `build_fine_tail_va_list()` insert that fine
spacing only over the last stretch before breakdown, leaving the coarse
sweep everywhere else unchanged - `input_diode_breakdown.yaml` now
reaches Va=-13.988V with I~5.6e-5 A (M numeric ~118x by -13.93V) while
keeping runtime reasonable (153 total points, not thousands).

Separately, this exposed a real (if rare) crash: at very deep bias the
fresh-start fallback's inner call to newton_solver_qf.py (unmodified,
correctly - it has no reason to expect avalanche's regime) can overflow
`exp((psi-phin)/Vt)` badly enough to hand `spsolve` a Jacobian with NaN
entries, which surfaced as a raw BLAS/LAPACK parameter error instead of a
catchable Python exception in one aggressive test. Added a broad
try/except around both the continuation attempt and the fresh-start
retry in `newton_solver_avalanche.py`'s `newton_gummel_solve`, with a
last-resort fallback to the last known-good state (never silently
propagating a crash into the whole sweep) - `_safe_gummel_retry()`. This
didn't change the production sweep's behavior (it wasn't hitting this
path with the actual fine_tail spacing used), but makes the module
robust against a user pushing a future device/sweep past where even the
fallback solver can cope.

Finally, `main_avalanche.py`'s validation plot now masks (NaN-gates, not
deletes) isolated bias points whose self-consistency exceeds a threshold
(5.0) from the PLOTTED curves only - a voltage-controlled sweep can still
land a single point on a spurious root that the very next point's
continuation recovers from (it isn't carried forward), and one such
point was creating a distracting, misleading spike in the plot. The raw
numbers (self-consistency included) stay in `breakdown_iv.csv`
regardless, so nothing is hidden, just not misleadingly plotted.

### 16.8 Two real numerics bugs found and fixed; one dead end (mesh
loosening) fully characterized; a graded-doping idea explored; a
sub-stepping attempt tried and reverted; the actual fix identified

User pushback on the 5.0-self-consistency mask ("this glitch detector is
very hacky .. why did it even create the wrong solution to begin with")
led to real debugging instead of a sharper heuristic - the right call:
several isolated bias points were landing EXACTLY on the no-avalanche
current (bit-identical), the signature of `_safe_gummel_retry`'s
equilibrium-reset fallback winning over the correct continuation branch
just because it scored a lower raw Newton residual, with no check that
it's the same physical branch.

**Bug 1, real and fixed**: `_clip()` (both damping paths) was clipping
`psi`/`phin`/`phip` corrections component-by-component. `J*delta=-F`
only guarantees `delta` is a descent direction for `||F||^2` as a whole,
undamped vector - a per-component-clipped vector is a DIFFERENT direction
with no such guarantee, and near breakdown the raw correction can span
orders of magnitude between components (a near-zero-density node's QF
potential barely constrained, next to a well-determined one), exactly
where this bites hardest. Fixed to a single global scalar rescale of the
whole vector (preserves direction, only shortens it) - verified via
instrumentation: the corrupted-direction line search burned all 20
backtracking halvings finding zero decrease; after the fix, iteration
counts and self-consistency improved substantially across most of the
sweep.

**Bug 2, real and fixed**: even with direction preserved, some points
still stalled. Traced one directly: the Jacobian's avalanche-coupling
entries reach ~1e31 at this project's sub-nanometer junction mesh
(`alpha(E) * mobility * carrier_density / h`, each factor legitimate, the
product isn't), next to ~1e2-scale residual entries - a ~29-order
spread in one linear solve that erodes double precision's ~16 digits on
the physically meaningful small part of the answer. New
`jacobian_scaling.py` (`equilibrated_spsolve`) does Ruiz row/column
equilibration before every sparse solve in the avalanche path
(`bank_rose_damping.py` and `newton_solver_avalanche.py`'s line-search
fallback) - mathematically exact (recoverable), just better-conditioned
arithmetic. Verified two ways: a synthetic badly-scaled linear system
went from 8.8e-12 to 1.8e-16 relative residual (50,000x), and the tuned
flat-1e19 fine_tail sweep's masked-point count dropped 15->10.

**A methodology bug of my own, caught and corrected**: an early "the fix
works!" result on the hardest point turned out to be because I'd typed
`-13.876` as a literal in a test script instead of using the actual
`np.arange`-produced array value (`-13.875999999999992`, ~8e-15 away).
This device is sensitive enough at that specific point that the
difference changed which branch Newton landed on - a real, if extreme,
illustration of how close to a fold some of these points sit. Re-tested
with the exact array value throughout after catching this.

**Mesh loosening, tried per a direct request ("we can't afford this mesh
in 2D/3D"), fully characterized as a dead end for this physics**: loosening
`junction_spacing_debye_factor` from 0.05 even slightly (to 0.08 - still
"textbook adequate", 5+ cells/Debye length) makes the ENTIRE avalanche
runaway vanish numerically (M(Va) pinned at 1.00 deep into reverse bias
where the fine mesh gives M~118), with no warning - Newton's own
diagnostics look fine throughout. Confirmed the peak field itself is
still well-resolved at moderate bias (matches the fine mesh to 4 sig
figs) - it's specifically that avalanche generation's box-integration
needs the fine mesh in a way ordinary transport doesn't. Real implication
for future 2D/3D work: uniform mesh loosening isn't viable for this
physics; would need local/adaptive refinement tracking wherever the peak
field actually is, not a global density knob. `input_diode_breakdown.yaml`
kept at 0.05 with a comment explaining why.

**Graded p-side doping, explored, informative but not a fix on its
own**: tried grading the heavy side from 1e18 at the junction interface up
to 1e21 over a short transition distance, so `h_min` (now correctly sized
from the INTERFACE doping rather than a region-wide summary - a genuine,
separate mesh.py fix, see below) stays coarse while still reaching 1e21
doping further out. A straight linear-in-concentration ramp badly
front-loads the change (jumps most of the way to 1e21 within the first ~2%
of the transition, since the ramp is dominated by whichever endpoint has
the larger magnitude) and made convergence WORSE (94% masked). A
log-ramp (`doping_profiles.py`'s new `log_ramp` option - linear in
log10(concentration) instead) fixed that specific problem (16% masked,
better than the flat-1e19 baseline) but exposed a DIFFERENT failure: the
avalanche generation feedback loop simply never ignites, even though the
resolved peak field and closed-form ionization integral are nearly
identical to the flat-1e19 device at matching bias (both cross the
Selberherr threshold ~-13V) - confirmed this is a solver limitation, not
real physics, by comparing the two devices' numerically-resolved fields
directly. `mesh.py`'s `h_min` now derives from `p_profile.sample(0.0,...)`
(the interface value) instead of `reference_concentration()` (a region-wide
summary) - correct in general, not just for this experiment, and doesn't
change any existing flat-doping example (verified bit-for-bit via the
golden tests). `doping_profiles.py` gained `transition_um` (bound a
`linear` profile's ramp to less than the full region thickness) and
`log_ramp` (ramp in log-concentration) - both backward compatible,
opt-in, no effect unless set.

**Adaptive bias-step subdivision (sub-stepping), implemented, measured,
and REVERTED**: hypothesis was that the remaining failures were caused by
too-large a jump between consecutive continuation points - added
`_continuation_with_substeps()` to `newton_solver_avalanche.py` to bisect
the Va interval and retry from a closer warm start (up to 4 levels deep)
before falling back to `_safe_gummel_retry`. Measurement (using the
correct array Va values, per the methodology bug above) showed ZERO
benefit on the production sweep (still 10/153 masked, same as
equilibration alone) and ZERO benefit on the broader 1e18-1e21 doping
sweep (1e19: 120->123, 1e20/1e21: no change, still ~97-100% failed) -
while multiplying wall-clock time severalfold on every failing point
(up to 5 nested Newton solves instead of 1), confirmed directly by the
user noticing the doping sweep had gone from ~2-3 minutes to not
finishing a single doping level in over 2 minutes. Verbose tracing showed
why it doesn't help: at EVERY recursion depth, down to 1/16th of the
original step, the very first Newton iteration achieves ZERO residual
improvement at ANY step size - not a step-size-dependent stall, a
persistent local pathology. Reverted cleanly (confirmed no dangling
references, testsuite passes, runtime back to baseline ~11s). The
mechanism itself isn't wrong - it would help a genuinely oversized
user-supplied Va jump - it just isn't what's failing in this device.

**The actual diagnosis, and the planned real fix**: "zero progress at any
step size" is the textbook signature of sitting at or very near a genuine
FOLD POINT in the true I(Va) curve, where `∂F/∂U` at fixed Va is singular
or near-singular - not fixable by better linear-solve precision, a
different damping strategy (verified directly: Bank-Rose does WORSE on
the same point, its K parameter maxing out immediately), or a smaller
step, because the "fix Va, solve for U" formulation is ill-posed exactly
there. This also explains why 1e20/1e21 (smaller, harder-to-resolve
depletion regions) fail almost completely - the fold sits somewhere the
voltage-controlled sweep can't get near. Researched established
alternatives (see plan file / next session): PSEUDO ARC-LENGTH
CONTINUATION (Keller) is the standard technique for tracking a solution
curve through a fold - `Va` becomes an additional unknown solved for
alongside `[psi, phin, phip]`, and the sweep steps along arc length
(always well-defined) instead of along Va (not, at a fold). A genuinely
different paradigm from anything tried this session, not another Newton
solver tuning knob. Scoped as a new, additive `arclength_continuation.py`
module plus a new `main_avalanche_arclength.py` driver, fully opt-in -
see the approved plan (saved separately) for the full design before
implementation begins on a new branch.

## 17. Session 11: drain-to-substrate junction leakage - trap-assisted and
band-to-band tunneling, a new `tat/` package, no damping needed after all

Requested capability: model reverse-bias leakage at a MOSFET's drain-to-
substrate junction (drain ~2e20 cm^-3, substrate ~1e17 cm^-3, opposite
type - the same doping regime `newton_solver_qf.py` was built for), driven
by trap states inside the bandgap assisting tunneling - expected to turn on
at much lower field than avalanche and rise exponentially but more softly.
Full design in `plans/tat_btbt_plan.md` (branch `tat-btbt-leakage`),
scoped to silicon only for this phase (SiGe drain material is an explicit,
deferred follow-on, since its narrower bandgap is expected to make this
mechanism worse).

**Sources actually consulted, not reconstructed from memory**: FLOOXS
(open-source TCAD, local clone) has a working `B2BTunnel/simple.tcl` (Kane
local-field band-to-band tunneling, three field-power variants with real
fitted coefficients) and a `schenk.tcl` (a more microscopic, phonon-assisted
trap-coupled model, not yet implemented - planned as a second, swappable
model once Hurkx is validated). A Sandia National Laboratories conference
paper (Carroll et al., SAND2007-1497C, fetched and read in full) reproduces
Hurkx, Klaassen & Knuvers' 1992 trap-assisted-tunneling formula directly -
confirms the mechanism the task described is a field-enhanced SRH
recombination/generation rate (`tau -> tau/(1+Gamma(F))`), not pure Zener
tunneling. `Gamma(F) = Delta*exp(Delta)*E1(Delta)`, `Delta=(F/F_Gamma)^2`
is the closed form reported consistently across the wider TCAD literature
for that enhancement factor (lower sourcing confidence than the SRH
structure itself - the finite-difference Jacobian check, not the citation,
is what actually gates trust in it). Confirmed via web search that Hurkx is
one of Sentaurus's four standard local BTBT/TAT models (alongside Kane,
Schenk, and the more accurate but structurally different nonlocal
dynamic-path model) - Hurkx was chosen to implement first specifically
because it degrades EXACTLY onto this project's own already-validated
`physics.py:srh_recombination` at zero field, reusing parameters
(`tau_n`, `tau_p`) already in every example, rather than introducing a
large block of new, uncalibratable knobs.

**New package `tat/`** (mirrors `avalanche/`'s structure): `tat.py`
(`KaneBTBTModel`, `HurkxTATModel`, `btbt_generation`, `hurkx_gamma`,
`hurkx_tat_generation`, plus a standalone `sanity_probe()` run via
`python3 -m tat.tat`), `newton_solver_tat.py` (built on
`newton_solver_qf.py`, Hurkx's field-enhanced SRH REPLACES the plain SRH
term already in that solver's continuity rows, Kane's BTBT is purely
additive - subtracted from the effective recombination rate the same way
avalanche's `G_ii` is), `tat_config.py`, `main_tat.py`, plus the new
`configs/input_diode_drain_substrate.yaml` example (exactly the 2e20/1e17
doping the task specified) and `testsuite/golden/diode_tat.json`.
Additive-only changes elsewhere: `core/config.py`'s `math_model` allow-list
and `core/solver.py`'s `voltage_sweep` dispatch both gained `"newton_tat"`;
`testsuite/common.py` gained `run_diode_tat` + one new `EXAMPLES` entry,
sampled at moderate reverse bias (-1V, -5V), well clear of any sharp
transition. Full existing suite (5 examples) still passes byte-for-byte
unchanged; the new 6th test passes too.

**A real numerical bug caught before it reached the solver**: `tat.py`'s
first version computed `Gamma(Delta) = Delta*exp(Delta)*E1(Delta)`
literally, which overflows `exp(Delta)` (well before `scipy.special.exp1`'s
compensating decay brings the product back down to its true, bounded limit
of 1) at fields reachable not just at genuinely extreme bias but
transiently at a rejected Newton line-search trial point. Fixed with the
standard large-`x` asymptotic expansion of `x*exp(x)*E1(x)` (Abramowitz &
Stegun 5.1.51) above `Delta=30`, verified to match the exact formula to
~1e-9 relative error right at the switchover point and to stay accurate
(by construction) at every larger `Delta` where the exact formula would
instead overflow to `inf`/`nan`.

**Unlike avalanche, this solver needed NO new damping strategy** - the
no-feedback argument in `newton_solver_tat.py`'s own module docstring
(both generation terms depend only on local field and local `n, p`, never
on `Jn`/`Jp` the way avalanche's `G_ii` does, so there's no self-
reinforcing loop) held up in practice: a full 0 to -10V reverse sweep on
the actual drain/substrate device converges with plain backtracking
Newton, zero warnings, self-consistency ~0.0005-0.001% throughout, in as
few as 9-13 iterations per point. `bank_rose_damping.py` (already built for
avalanche) was never needed.

**Finite-difference Jacobian check**: passed at ~1e-5 to 1e-6 relative
error across forward bias, moderate reverse bias, and -5V reverse bias,
with two lessons worth keeping for the next such check on this codebase:
(1) the 6 Dirichlet boundary rows must be EXCLUDED from the comparison -
their Jacobian diagonal is deliberately rescaled for linear-solve
conditioning (a pre-existing, unmodified `newton_solver_qf.py` design
choice, "same pivoting-safety scaling as newton_solver.py") and does not
match the literal derivative of the coded residual, which is not a bug;
(2) a single global or per-column significance floor is not enough on a
Jacobian this unevenly scaled (Poisson-block entries dwarf the new
TAT-coupling entries) - use a PER-ROW absolute floor (checked directly:
entries below it were confirmed, via a multi-epsilon convergence check
down to 1e-9, to be correctly computed but genuinely negligible-magnitude
sensitivities in the deep quasi-neutral bulk, not real mismatches) plus a
smaller default perturbation (`1e-7`, not `1e-6`) - both false positives
this session hit were resolved by tightening these two things, not by
finding an actual Jacobian error.

**Numeric result** (original 2e20-drain/-10V version of this example,
since revised - see 17.3/17.4 below for the numbers this project actually
settled on): leakage current enhancement over a plain no-tunneling
baseline grew smoothly and monotonically with reverse bias (no runaway,
no fold, matching the task's own expectation of "still exponential, just
not as sharp" as avalanche). A per-node generation-term breakdown at the
deepest sweep point confirmed the physical ordering directly (not just
via the standalone `tat.py` sanity probe): Hurkx TAT dominates across
nearly the entire depletion width, with Kane BTBT only overtaking right
at the single peak-field point (the metallurgical junction itself).

### 17.1 Band diagrams: showing WHY tunneling turns on, not just that it does

User request: add Ec/Ev/Ei/Efn/Efp/vacuum-level band diagrams to the
validation plots, so the mechanism is visible, not just the resulting
I(Va) curve. This project already had exactly the right tool
(`core/plot.py`'s `plot_bands`/`_band_energies`, built for the MOS-cap
work) - no new plotting code needed, just wiring `main_tat.py` to build a
`structure_io` doc (equilibrium + swept bias points, same pattern
`main.py` already uses) and call it. New `tat_bands.png`: band diagram at
equilibrium, band diagram at the deepest reverse-bias point, and the
G_tat/G_btbt generation-term profile, all sharing one x-axis zoomed to the
depletion region.

**A real physics-caption bug caught before shipping**: the first draft's
caption claimed reverse bias "pulls the drain's E_c down toward the
substrate's E_v," the classic Esaki/tunnel-diode picture. Checking the
actual numbers first (a captioning claim is still a physics claim) showed
this device's substrate (1e17 cm^-3) is not degenerate - its E_f sits well
inside the gap, not inside a band - so there is no literal band-overlap
happening the way there is in a true degenerately-doped tunnel diode; the
drain-side flat region doesn't move at all under bias (it's the grounded
contact), only the substrate side shifts. What actually changes between
the equilibrium and reverse-bias panels is the STEEPNESS of the band
bending across the transition region - the local field, i.e. exactly the
quantity `tat.py`'s Kane `alpha(F)` and Hurkx `Gamma(F)` depend on.
Corrected the caption to describe field/slope steepening rather than a
band-overlap picture that doesn't apply to this doping regime.

### 17.2 Sweep range restricted to -5V, per explicit user direction

User: "we normally don't go beyond that unless dealing with power
devices." Changed `input_diode_drain_substrate.yaml`'s
`reverse_stop_V` from -10.0 to -5.0 (`reverse_points` 60->40 to keep
similar point density) and `save_bias_points` to `[-1,-2,-3,-4,-5]` -
this is a normal-operation logic/memory-junction leakage example, not a
power-device breakdown study, so there was never a reason to sweep or
plot deeper. Golden test re-captured for the new sample points (a range
change, not a physics regression - values track the same smooth curve,
just sampled differently).

### 17.3 Drain-doping sweep (1e18-1e21): one-sided-junction confirmation,
and a real Kane BTBT divergence caught and fixed

User: "skew doping from 1e18-1e21 and see how the model behaves." New
`tat/main_tat_doping_sweep.py` (mirroring `mos_poly_sweep.py`'s
"hold everything fixed, override one doping, overlay results" pattern) -
substrate held at 1e17, drain swept. **A real bug in the sweep script
itself, caught immediately**: overriding `dev.Nd` alone did nothing - all
four doping levels gave BIT-IDENTICAL results. Root cause:
`core.config.build_from_config` already resolves the YAML's doping block
into a concrete `dev.n_profile` `DopingProfile` object, and
`core.mesh.build_diode_grid` reads THAT, not `dev.Nd` - exactly the
pitfall `mos_poly_sweep.py`'s own `run_one()` already works around by
setting `dev.gate_profile` directly. Fixed by also setting
`dev.n_profile = DopingProfile.flat(Nd)`.

Once fixed, the sweep confirmed the expected one-sided-junction physics
DIRECTLY (not just cited from the plan): drain doping 1e18/1e19/1e20 gave
nearly identical peak field (2.2e5 to 5.7e5 V/cm) and leakage - once the
drain is already far heavier than the fixed 1e17 substrate, doping it
further barely narrows the depletion width, which the LIGHT side already
dominates. But 1e21 blew up 6 orders of magnitude to physically
nonsensical milliamp-scale current. Traced before reporting it (not
patched blindly): the plain no-tunneling baseline solver stayed perfectly
well-behaved at the identical doping/mesh, isolating the cause to the
tunneling generation term itself, not a solver/Jacobian bug. Root cause:
1e21's peak field reaches 1.7e6 V/cm (vs 5.7e5 at 1e20), and Kane's
`A*F^P*exp(-B/F)` has NO saturation built in - `exp(-B/F)` alone would
plateau near 1, but the unbounded `F^P` prefactor keeps growing forever,
so a 3x field increase inflated `G_btbt` by 12 orders of magnitude
(4e16 -> 3e28).

Given a direct choice (add a saturation cap / document the limitation and
leave unbounded / drop 1e21 from the study), the user chose to add a cap.
Added `KaneBTBTModel.F_sat_V_cm` (default first tried: 1.5e6, "just above
where 1e20 diverges" - re-derived after checking the actual numbers: at
1.5e6, `G_btbt` is already ~4e27, thirty orders of magnitude beyond
anything this project has ever validated a resulting CURRENT against).
Settled on **9e5 V/cm** instead - just above the peak field the project's
own shipped, validated 2e20-drain example reaches (~7.84e5 V/cm) - so
every doping level's prediction stays anchored to the same order of
magnitude as the one case actually cross-checked against a no-tunneling
baseline, rather than extrapolating an exponential arbitrarily far past
it. Implemented the same hard-floor style `avalanche.py` already uses at
its own (opposite, low-field) limit - `F` capped before evaluating the
formula, so `dG/dF` is exactly 0 beyond the cap, not smoothed. Re-verified:
full testsuite still passes (the shipped 2e20 example's own peak field
sits below the cap, so it is completely unaffected), and the finite-
difference Jacobian check still passes at ~1e-5 to 1e-6 on every
physically realistic bias point tested (a check at Va=+0.3V using a
crude, deliberately non-self-consistent test perturbation DID show a
mismatch at one entry, but was proven - by disabling the cap entirely and
reproducing the identical mismatch - to be a pre-existing artifact of that
crude test construction hitting a ~1e8 V/cm unphysical field spike, not a
regression from the cap). With the cap, 1e21's leakage enhancement is a
bounded, physically believable ~50x at -5V instead of ~1.5 million x.

Also fixed a self-diagnostic bug of my own along the way: the sweep's
`max_selfconsist` column showed an alarming 2.6e7 at 1e18/1e19 - traced to
`J_std/J_mean` being evaluated AT Va=0, where `J_mean` is itself ~0 by
construction (true equilibrium has no net current), making the ratio
blow up for a reason that has nothing to do with solver quality (only 2
Newton iterations needed there) - the same documented artifact
`main_avalanche.py` already works around. Fixed by excluding points
within 0.2V of equilibrium from that diagnostic.

### 17.4 Substrate-doping sweep (1e16-1e18, drain fixed): the informative
half of the doping story, plus a genuine ordinary-diode-physics finding

Drain-doping sweep (17.3) showed the heavy side barely matters below
1e21 - expected for a one-sided junction, but also the LESS interesting
half of the story, since the LIGHT side is what actually sets the
depletion width/field. User: hold drain at 1e20, sweep substrate
1e16/1e17/1e18 instead. New `tat/main_tat_substrate_sweep.py` (same
pattern/bugfix-awareness as 17.3's script). Result: a much stronger,
monotonic effect in the expected direction - peak field rises from 5.4e5
(1e16) to 5.7e5 (1e17) to 8.6e5 V/cm (1e18, no longer negligible next to
the 1e20 drain - only 100:1 now), and leakage enhancement at -5V reaches
~11,000x at 1e18 (vs ~1.1-1.4x at 1e16/1e17). Converged everywhere with
zero warnings.

**User's follow-up question, and a real finding it surfaced**: at shallow
bias, 1e16 (the LIGHTEST substrate) showed MORE absolute leakage current
than 1e17, seemingly contradicting "heavier doping -> more field -> more
leakage." Checked against the pure closed-form `analytic.shockley_I0`
(zero tunneling physics at all) before answering: `I0(1e16)=2.99e-14 A`,
`I0(1e17)=2.99e-15 A`, `I0(1e18)=3.01e-16 A` - exactly 10x apart per
decade, confirming this is ordinary diode physics
(`I0 ~ ni^2/Na`, minority-carrier injection into the lighter side),
present even with the tunneling model switched off entirely. Added a
dedicated, finer 0-to-1V sub-sweep (`tat_substrate_sweep_0to1V.png`,
30 points, not just a re-plot of the coarser full-range sweep) showing
this directly: the no-tunneling dashed baselines alone already show
1e16 > 1e17 across the whole window. The genuinely interesting result is
the CROSSOVER visible in that same plot: 1e18's tunneling enhancement
grows fast enough to overtake both lighter dopings' baselines by about
-0.5V and reach >100x by -1V - i.e. within the practically-relevant
0-to-1V range most non-power devices actually operate in, you can
directly watch ordinary diode leakage (favors lighter doping) lose out to
tunneling enhancement (favors heavier doping/higher field) as bias
deepens.

**Config consolidated per explicit user direction**: the shipped example
now standardizes on drain=1e20 (matching what the informative substrate
sweep was built around, previously 2e20) - `tat/main_tat_doping_sweep.py`
(the drain sweep, confirmed low-information-value below 1e21) removed;
`tat/main_tat_substrate_sweep.py` (the informative one) kept. Golden test
re-captured for the new doping.

**Not done this session, tracked for later** (see `plans/tat_btbt_plan.md`):
Schenk as a second, swappable trap-assisted model (in progress, next
session entry); the SiGe drain-side follow-on; and, longer-term, a
genuine NONLOCAL tunneling-path search once this project extends to 2D/3D
device geometry (an explicit user roadmap item, not this phase's scope) -
`tat/tat.py` is kept structured so that can be added as a new module
alongside it later, not a rewrite.

### 17.5 Schenk model implemented as the planned second trap-assisted
model; a real ill-conditioning bug found and fixed; self-consistency gap
honestly left open

Per the plan's own Hurkx-first rationale (17.0), added
`tat.tat.SchenkTATModel`/`schenk_tat_generation` - the F=E (plain local
field) special case of FLOOXS's `schenk.tcl`, where the density
correction collapses exactly to `n, p` themselves. Derived and
implemented the closed-form derivatives by hand: `SchenkSRH`'s (a
dimensionless sign/magnitude selector, not itself a rate) derivatives
simplify cleanly to `dSchenkSRH/dn = ni/(n+ni)^2` (independent of p, and
symmetric in n/p) after simplifying the general quotient rule - a good
sign the formula was transcribed correctly. `newton_solver_tat.py`
generalized to accept a swappable `trap_model`/`trap_generation_fn` pair
(same call signature for both Hurkx and Schenk) rather than hardcoding
Hurkx, with the external API kept backward compatible (nothing currently
threads a trap model through `voltage_sweep` anyway, matching avalanche's
own precedent noted in 17.0).

**A real FD-Jacobian false alarm, diagnosed rather than silenced**: an
initial FD check (reusing avalanche's own crude, non-self-consistent
ramp-perturbation construction) showed ~1-2% mismatches for Schenk but
not Hurkx. Traced to Schenk's discrete `sign(SchenkSRH)` branch (selects
which of two phonon absorption/emission `Fc` values to use) landing
exactly on its own knife-edge: checking the actual numbers showed
`SchenkSRH` sits at pure floating-point noise (~1e-13 to 1e-26) across
essentially the ENTIRE quasi-neutral bulk on both sides (335 of 372 nodes
for this device) - expected physics (mass-action `n*p~ni^2` holds almost
everywhere except right in the depletion region), but it means a crude
test construction that leaves the bulk still exactly at equilibrium hits
a branch that is, at those specific nodes, genuinely undefined by
floating-point noise. Re-ran the SAME check using a REAL, self-consistent
converged state (the plain `newton_qf` solution) as the base point
instead - passed cleanly at ~1e-6 to 1e-7 across forward and reverse
bias. Lesson for next time: a synthetic ramp perturbation is a fine base
point for smooth models (worked for Hurkx, Kane, and the original QF
Jacobian) but a poor one for any model with a discrete branch tied to a
quantity (like `n*p-ni^2`) that's naturally ~0 almost everywhere in a
non-self-consistent, still-near-equilibrium test state - use a real
converged solution as the base point for those.

**A real, more serious ill-conditioning bug, found and fixed**: even past
the FD-check false alarm, `newton_gummel_solve` with Schenk repeatedly
diverged or stalled (`|F|` stuck at 1e8-1e12, one point hit "Matrix is
exactly singular") starting from a perfectly reasonable Gummel warm
start. Verbose tracing showed the residual barely moving across many
tiny-step backtracking iterations - the same qualitative signature
`DEVELOPMENT_LOG.md` session 16 (avalanche, bug 2) diagnosed as severe
Jacobian ill-conditioning, not a bad search direction. Confirmed directly
by comparison: Schenk's generation rate spans a FAR wider dynamic range
than Hurkx's tau-bounded one (its own standalone comparison in `tat.py`'s
sanity probe already showed this - Schenk stays ~40 orders of magnitude
below Hurkx at low field, then crosses over and grows far more steeply,
matching its closer kinship to Kane's un-tau-bounded exponential than to
Hurkx's saturating `Gamma(F)`). Fix: reuse `core/jacobian_scaling.py`'s
`equilibrated_spsolve` (already built for avalanche's identical class of
problem) in place of the plain `scipy.sparse.linalg.spsolve` call -
applied unconditionally (mathematically exact/recoverable, so no
downside for Hurkx, confirmed bit-for-bit unchanged on Hurkx's own
sweep). Verified directly on the single worst-diverging point (Va=-2V):
un-equilibrated Newton stalled at `|F|~5e8` from a `|F|~1.2e11` cold
start after 18 iterations; the IDENTICAL Newton sequence with
equilibration alone reached `|F|~2.2e-5` in 12 clean, full-step
iterations.

**Left honestly open, not silently shipped as "done"**: even after the
equilibration fix, Schenk's SELF-CONSISTENCY (`J_std/J_mean`, current
uniformity across the device in steady state) on this project's mesh is
visibly worse than Hurkx's (0.1-3 vs. Hurkx's consistent ~1e-5) - this is
NOT a Newton convergence failure (the residual itself converges tightly,
often in single-digit iterations, well below `f_tol`), so raising
`maxiter`/tightening `f_tol` (tried, no effect) doesn't touch it. Working
hypothesis, not yet confirmed: Schenk's much sharper field dependence may
need the same kind of extra mesh refinement near the peak field that
avalanche's own `avalanche_ii_refine` provides for its similarly sharp
generation term - not yet implemented here. New
`tat/main_tat_hurkx_vs_schenk.py` produces the requested comparison
(`out/tat/tat_hurkx_vs_schenk.png`) with this caveat stated directly in
its own docstring and plot title, not hidden - on this device (1e20
drain / 1e17 substrate, 0 to -5V), Schenk's absolute contribution is 4-5
orders of magnitude SMALLER than Hurkx's throughout (expected: this
device's peak field at these biases stays well below the ~9e5 V/cm range
where the standalone sanity probe showed Schenk catching up to Hurkx's
magnitude), so the self-consistency gap doesn't yet visibly corrupt the
headline comparison, but should be resolved (mesh refinement, most
likely) before trusting Schenk at higher fields/heavier doping the way
Hurkx has already been trusted this session.

Full existing testsuite (6/6, including `diode_tat`) still passes
unchanged throughout all of this - none of it touched the Hurkx default
path's own already-validated behavior.

## 18. Session 12: pre-2D/3D architecture prep - a written architecture
boundary, a private-API cleanup, and four locked-in future decisions

After Session 11 shipped, a series of architecture questions came up
(why `core/` isn't `diode/`, whether splitting a `diode/` package out
would help - answered no, since `mos/` already depends on genuinely
shared code in `core/mesh.py` and `avalanche/`/`tat/` import
`newton_solver_qf.py`'s internals in an inheritance-like way that a
rename wouldn't change) and finally: how to think about architecture
before the codebase moves into 2D/3D, given the risk of it becoming
accidentally MOS-centric the way it once became accidentally
diode-centric before the `core/` rename. Rather than let that stay a
verbal answer, the user asked for a concrete plan and to start acting on
it - approved at
`~/.claude/plans/whimsical-stargazing-barto.md`, with four explicit
constraints from the user folded in before implementation started:
avalanche must never be merged into a shared multi-mechanism solver (it's
numerically finicky and not expected to ever run combined with other
generation mechanisms, in 1D or later in 2D/3D); 2D/3D visualization will
need a genuinely different tool (interactive slicing, fields rendered on
the 3D structure, Tecplot-like) rather than an extension of
`core/plot.py`; 2D/3D meshing should use a point-cloud approach for
geometric flexibility rather than extending `core/mesh.py`'s structured
node-array style; and the longer-term ambition is a general PDE framework
(user-specified equations/constants/variables, e.g. thermal simulation),
not a codebase permanently specific to drift-diffusion.

Three things were done this session, all pure prep/cleanup - no physics
changed, no golden outputs changed:

**New root-level `ARCHITECTURE.md`** records, so none of it has to be
re-derived from git archaeology again: which of `core/`'s files are
1D-only and should not be genericized in place (`mesh.py`,
`newton_solver_qf.py`, `physics.py`'s tridiagonal continuity solves); the
actual reusable pure-physics kernel (`srh_recombination`,
`bernoulli`/`bernoulli_deriv`, the TAT/BTBT and avalanche generation-rate
functions, the `Material`/`Device` dataclasses, `doping_profiles.py`) -
noting mobility is currently a constant field, not a function, and any
future mobility model should follow the same pure-function pattern; the
avalanche-stays-standalone rule; and the package-per-mechanism convention
(new physics gets its own top-level sibling package, never nested inside
`mos/`). It also records the four locked-in-but-not-yet-built 2D/3D
decisions above (visualization, meshing, the pre-existing nonlocal-
tunneling-path-search roadmap item, and the general-PDE direction), so
future work doesn't get designed against the wrong assumption.

**Private-API cleanup**: `tat/newton_solver_tat.py` imported four
underscore-prefixed internals directly from `core/newton_solver_qf.py`
(`_poisson_scale`, `_continuity_scale`, `_unpack`, `_MAX_QF_STEP`), while
`avalanche/` never did (it keeps its own private copies instead, by
design, per the standalone rule above) - two different coupling styles
for what's meant to be the same kind of relationship. Standardized on one
public surface: renamed the four to `poisson_row_scale`,
`continuity_row_scale`, `unpack_qf`, `MAX_QF_STEP` in
`core/newton_solver_qf.py` (dropping the leading underscore, adding an
explicit `__all__`), and updated `tat/newton_solver_tat.py`'s import
accordingly. Rename-only, but not entirely mechanical: several call sites
in both files assign a same-named local variable from the function call
(e.g. `poisson_scale = _poisson_scale(...)`) - renaming the function to
match would have made the local variable shadow it and broken the call,
so the new public names were deliberately chosen distinct from those
local variable names to avoid that trap.

**Verification**: full existing test suite (31 tests via
`python3 -m unittest discover -s testsuite`, since `pytest` isn't
installed in this project's `.venv` - `unittest discover` is the suite's
own documented fallback) passes unchanged, including `test_diode`,
`test_diode_breakdown`, and `test_diode_tat` against their existing
golden files - confirming the rename touched no behavior.

Explicitly not done this session, by design: no 2D/3D package, mesh
module, or visualization tool was created - there's no concrete 2D device
target yet, so that would be speculative scaffolding rather than
architecture that's actually needed. The next real engineering step
becomes concrete once a specific 2D device or PDE target is chosen, at
which point it gets its own fresh plan.

## 19. Session 13: a real p-SiGe/n-Si heterojunction, per-node/edge
`MaterialField`, and a Scharfetter-Gummel heterojunction correction the
original plan missed

Goal: a genuine heterojunction diode (p-type relaxed Si0.6Ge0.4, 1e20
cm^-3, against n-type Si, 1e17 cm^-3) run through the existing TAT/BTBT
reverse-leakage solver, per the harmonic-snuggling-puddle plan. Previously
every solver in this project (`core/physics.py`, `core/solver.py`,
`core/newton_solver_qf.py`, `tat/newton_solver_tat.py`) took one scalar
`Material` for the whole device - `core/mesh.py`'s MOS grid builder
(`build_mos_grid`) already had the per-node/edge array precedent
(`eps_edge`, `ni_arr`, `is_oxide`) this generalizes to the diode.

**Architecture, additive not a rewrite** (mirrors the MOS precedent
exactly): new `core.materials.MaterialField` dataclass bundling
`eps_edge`/`mu_n_edge`/`mu_p_edge` (per edge), `ni_arr`/`tau_n_arr`/
`tau_p_arr`/`delta_Ei_arr` (per node), with `.uniform(mat, x)` (constant
arrays, `delta_Ei=0`) and `.from_regions(mat_p, mat_n, x, junction_index)`
(stepped arrays + the Anderson's-rule band-offset term, see below)
constructors. Every touched function normalizes its `mat` argument to a
`MaterialField` as its first step (`_as_field`/inline `isinstance` checks),
so every existing call site passing a plain scalar `Material` gets
bit-for-bit identical arithmetic (elementwise ops on a repeated-constant
array equal the scalar op exactly) - confirmed by the full existing
31-test suite (`python3 -m unittest discover -s testsuite`) passing
unchanged after every step of this work.

Files touched: `core/materials.py` (`MaterialField`), `core/material_db.py`
(`derive_alloy()`, registering an `AlloyMaterial.resolve(x)` under a name -
`bowing_eV=0.36` calibrates `Si0.6Ge0.4` to `Eg~=0.85eV`, the standard
Braunstein/People relaxed-alloy fit), `core/mesh.py` (`build_diode_grid`
gains `mat_n=None`, mirroring `build_mos_grid`'s `Cdop_gate=None` pattern;
returns a new `mat_field`/`interfaces` pair), `core/physics.py`
(`solve_poisson`'s new `delta_Ei=` override; a vectorized
`equilibrium_bulk_potential_arr`; `srh_recombination`/`solve_continuity_n`/
`solve_continuity_p`/`edge_currents` generalized to accept a `MaterialField`
- see the SG correction below), `core/solver.py` (`solve_equilibrium`,
`contact_values`, `gummel_solve` threaded through), `core/newton_solver_qf.py`
and `tat/newton_solver_tat.py` (same diff pattern in both - the plain-
gradient QF flux and Poisson charge term gain `delta_Ei`/per-edge-array
generalizations; `tat/newton_solver_tat.py`'s trap-generation call sites
get a new `_NodeMaterialView`/`_interior_node_view` shim so `tat/tat.py`'s
own Hurkx/Kane/Schenk functions - which read `mat.ni`/`mat.Vt`/`mat.T`/
`mat.tau_n`/`mat.tau_p` as scalars - work UNCHANGED via duck typing against
per-node arrays, no edits to `tat/tat.py` needed), `core/config.py`
(`material.p_side`/`material.n_side` sub-blocks, each shaped like today's
`material:` block plus a new `alloy: {end_member_a, end_member_b, x_a,
bowing_eV}` option; `build_from_config`'s return tuple ARITY is unchanged -
the second material rides inside `mesh_opts["mat_n"]`, defaulting to
`None`, specifically so every existing call site's fixed 7-value unpacking
stays valid without being touched), new example `configs/input_diode_sige_pn.yaml`,
`tat/main_tat.py` updated to thread `build_diode_grid`'s returned
`mat_field` (not the bare scalar `mat`) through to `solve_equilibrium`/
`voltage_sweep` - harmless (bit-identical) for every existing homojunction
config, required for the new heterojunction one to actually get real
per-region physics instead of a same-material silent workaround.

**The physics: a per-node `delta_Ei(x)` Boltzmann-relation offset** (the
plan's own derivation, applied as designed): `Xi(x) = chi(x) +
Vt*ln(Nc(x)/ni(x))`, `delta_Ei(x) = Xi(x) - Xi(n-side reference)`, folded
into `n = ni(x)*exp((psi-phin+delta_Ei)/Vt)`, `p = ni(x)*exp((phip-psi-
delta_Ei)/Vt)` everywhere this project's Boltzmann relation appears
(`solve_poisson`, both QF/TAT Newton solvers' residual, `equilibrium_bulk_
potential`'s vectorized sibling, and each solver's `phin_bc`/`phip_bc`-
from-contact-density derivation, which needs `delta_Ei` ADDED BACK once
inverting `n`/`p`->`phin`/`phip` at a contact - verified by hand and
numerically that `phin_bc=phip_bc=Va` exactly regardless of which
material a contact sits in). `delta_Ei=0` everywhere for a homojunction,
and it never depends on any Newton unknown, so it never touches an
existing Jacobian entry (confirmed by the FD checks below) - exactly as
planned.

**What the plan got wrong, found and fixed this session**: the plan
asserted `solve_continuity_n`/`solve_continuity_p` (the Scharfetter-Gummel
flux `core/solver.py`'s Gummel warm-start relies on) needed "no new term,"
reasoning only about the SRH mass-action term. Wrong - and the bug was
loud: the very first heterojunction sweep showed a ~4-order-of-magnitude
spurious current spike at exactly the mesh edge straddling the SiGe/Si
interface, with `Jtot` everywhere else clean. Root cause, worked out by
hand and confirmed numerically: SG's flux `Jn = coef*(n_{i+1}*B(u_{i+1}-
u_i) - n_i*B(u_i-u_{i+1}))` is exactly zero iff `n_{i+1}/n_i =
exp(u_{i+1}-u_i)` (using the Bernoulli identity `B(-y)=exp(y)*B(y)`).
Plain `u=psi/Vt` satisfies this at equilibrium for a HOMOJUNCTION because
`n0_i = ni*exp(u_i)` has the SAME constant prefactor `ni` at every node,
which cancels in the ratio. At a heterojunction `ni(x)` itself steps
(SiGe's ni is ~150x Si's here) - so even adding `delta_Ei` into `u` (a
narrower, first-guess fix that turned out necessary but NOT sufficient)
still leaves a node-dependent prefactor and a residual spurious current.
The correct, verified-by-hand fix folds `ln(ni(x))` into the Bernoulli
argument too, with an OPPOSITE sign for electrons vs. holes:
`u_n(x) = (psi(x)+delta_Ei(x))/Vt + ln(ni(x))`,
`u_p(x) = (psi(x)+delta_Ei(x))/Vt - ln(ni(x))`
(new `physics._sg_potential_n`/`_sg_potential_p`, used by
`solve_continuity_n`/`_p` and `edge_currents`, each needing their OWN `u`
now instead of one shared array). Both reduce to `psi/Vt` plus a *global*
additive constant for a homojunction (`ln(ni)` is the same everywhere),
which Bernoulli's difference-only argument cancels exactly - bit-identical
to before, confirmed by the unchanged 31-test suite. This is the value of
this project's own "test the discretization against a known analytic
fixed point" discipline (equilibrium zero-current) - it caught a genuine,
non-obvious gap in the hand-derived plan before it corrupted every Gummel
warm start silently.

**Also needed, once traced**: the heterojunction Newton(QF) Jacobian has
a MUCH larger entry-magnitude spread than any homojunction case this
project has hit before (SiGe's ~150x larger `ni` directly amplifies
`dJn/dphin ~ -q*mu*n` terms unevenly across the two materials) - a plain
`spla.spsolve` stalled the line search at a fixed, non-decreasing residual
even after the SG fix above. Switched `core/newton_solver_qf.py`'s solve
step to `core.jacobian_scaling.equilibrated_spsolve` (Ruiz row/column
equilibration, already used by `tat/newton_solver_tat.py` for the same
reason with Schenk's generation term) - mathematically exact, just
better-conditioned arithmetic; applied unconditionally, matching
`newton_solver_tat.py`'s own "one solve path" choice. Confirmed via a
finite-difference Jacobian check done AT a near-stalled point (not just a
cold start) that this was never a Jacobian correctness bug (FD agreement
~1e-7 to 1e-10 relative, both before and after equilibration) - purely a
conditioning/line-search problem, exactly the class of issue
`jacobian_scaling.py`'s own docstring documents.

**A real bug in the new config's own hand-picked mesh thickness**, found
via the same "trace the exact `Jtot` outlier location" discipline: the
first `input_diode_sige_pn.yaml` draft hand-set `Wp_um: 0.3` (mirroring
`input_diode_drain_substrate.yaml`'s light/heavy-side domain-sizing
reasoning) without checking it against SiGe's own minority-carrier
(electron) diffusion length - SiGe's higher electron mobility gives
`Ln~2.5um`, ~8x longer than the hand-picked 0.3um domain, truncating the
injected-carrier profile well before it decayed and producing a
non-flat, ~70x-too-large `Jtot(x)` artifact near the truncated p-contact
that looked exactly like a solver convergence failure. Fixed by removing
the `thickness:` override entirely and trusting `build_diode_grid`'s own
auto-sizing (`Wp=max(5*mat.Ln, 20*L_D_p)`, already using each side's OWN
material correctly) - per this project's own mesh-robustness principle
(pinned memory: solvers/examples should not depend on a manually
guessed domain size).

**Verification results**:
- `MaterialField.uniform()`/`.from_regions(mat_p=mat_n=Silicon)` both give
  `delta_Ei_arr` all zero (checked standalone).
- Full existing 31-test suite passes unchanged after every step (materials,
  mesh, physics, solver, both Newton solvers, config) - zero regression on
  every existing scalar-material example.
- Finite-difference Jacobian check, `newton_solver_qf.py`, at a genuinely
  heterogeneous (SiGe/Si) point: 15 random entries, relative error ~2e-10
  to ~2e-9 for all but one (1.15e-7, still tiny) - Jacobian confirmed
  correct.
- Same check, `tat/newton_solver_tat.py` (with Hurkx+Kane active): 15
  random entries, ~3e-10 to ~9e-10 for 14 of 15, one at 7.5e-3 relative
  error on a very small-magnitude entry (traced to a branch-function
  numerical-precision edge, not a sign/structural bug - matches this
  project's own "check the VALUE, not just smallness" discipline: a
  single outlier at a branch boundary, not a consistent 2x/-1x pattern
  across many entries).
- Equilibrium (Va=0) self-consistency for the shipped SiGe/Si example:
  `J_mean` (median interior) exactly 0.0, `J_std~9.7e-5 A/cm^2`,
  `max|Jtot|~8.4e-4 A/cm^2` - negligible against any real operating
  current density (~1e3 A/cm^2 at 0.6V forward), confirming the SG
  heterojunction fix is doing its job at exactly the case it targets.
- Full reverse (-0.05 to -5V) + forward (0 to 0.6V) sweep: forward bias
  converges cleanly (self-consistency ~1e-4 to ~1e-5 by 0.5-0.6V, matching
  homojunction quality). Reverse-bias leakage self-consistency is GREATLY
  improved by the SG+equilibration fixes above (currents are now smooth
  and monotonic in Va, vs. wildly non-monotonic/wrong-sign before) but is
  NOT fully clean - `J_std/J_mean` ratios of ~1-6 remain at several
  reverse points, with `newton_solver_qf.py`'s own stall warning still
  firing at some of them. Traced this as far as time allowed: it is a
  genuine residual floor (confirmed via FD check at the stalled point
  itself, not a Jacobian bug), not fixed by more Gummel warm-start
  iterations (converges to the identical stalled `|F|` regardless of warm-
  start quality past ~60 Gummel iterations) or by removing `MAX_QF_STEP`
  clipping (the actual proposed step there is tiny, ~4e-3V, not clipped).
  **Left as an open item** - this doping/material combination (a ~150x
  `ni` ratio ON TOP OF the ~1e17/1e20 doping asymmetry this project has
  separately validated before) is harder than anything previously
  resolved, and forcing a fix within this session's remaining budget risked
  papering over a real remaining numerics gap rather than fixing it.
- SiGe vs. homogeneous-Si leakage comparison (against
  `configs/input_diode_drain_substrate.yaml`, same 1e17/1e20 doping RATIO):
  at matched |Va|, the new SiGe/Si example's leakage came out slightly
  LOWER, not higher, than the pure-Si comparison (e.g. -0.81V: 3.8e-10A
  vs. 5.7e-10A; -4.24V: 1.33e-9A vs. 2.00e-9A). This is NOT the naively
  expected "narrower gap -> more leakage" result, but it has a sound
  physical explanation rather than being a bug: the user's requested doping
  (p-SiGe HEAVY at 1e20, n-Si LIGHT at 1e17) puts essentially all the
  depletion width - and therefore the high-field tunneling-generation
  region - on the LIGHT n-Si side, not the narrow-gap SiGe side (which,
  being heavily doped, barely depletes at all). SiGe's narrower gap can
  only show up as MORE leakage if the high-field region actually sits in
  the SiGe material - which would need the SiGe side to be the lightly
  doped one, or a more comparably-doped junction. **Flagged as a real,
  not-yet-resolved finding**, not silently reported as a success: proving
  the narrow-gap-leakage effect needs a follow-up doping configuration
  (SiGe on the light side) run as a second, deliberately contrasting
  example, not the one this session shipped.

Not done this session, explicitly deferred: `input_diode_sige_pn.yaml`'s
own reverse-bias self-consistency is not yet as clean as the homojunction
baseline (see above) - revisit once a specific next need justifies more
time on it; no attempt was made at a SiGe-side-light doping variant to
actually demonstrate the narrow-gap leakage enhancement (see above); no
strained-SiGe model (out of scope, this was deliberately the relaxed
alloy only, per the plan).

## 20. Session 13 (continued): closed-form TAT/BTBT leakage validation,
and a real output-clobbering bug in `tat/main_tat.py`

Two follow-up gaps from session 13's SiGe/Si work, raised directly by the
user: no way to actually see the output plots, and no independent
(non-PDE) closed form to validate the TAT/BTBT leakage `I(Va)` curve
against - this project already had closed forms for the plain diode
(`built_in_potential`/`depletion_widths`/`shockley_current`) and avalanche
(`breakdown_voltage_sze`/`ionization_integral`), but nothing for TAT/BTBT.

**Real bug found and fixed**: `tat/main_tat.py` wrote every run's plots/CSV
to the SAME fixed `out/tat/tat_iv.{csv,png}` regardless of which input
config was passed - confirmed directly (checked `tat_iv.csv`'s content
against the two configs' known numeric fingerprints) that running
`input_diode_sige_pn.yaml` after `input_diode_drain_substrate.yaml` would
silently overwrite the Si-only case's outputs in place. Fixed by deriving a
per-config output subdirectory from the input config's own basename
(`out/tat/<config_basename>/`) - not from `output.structure_file`, since
the config filename is always present and always unique per run, while
`structure_file` is an optional config value. Re-ran both configs; each
now has its own complete, independent output set (paths listed below).

**`core/analytic.py` generalized to `mat_p`/`mat_n` (mat_n=None default -
the SAME backward-compat pattern as `mesh.build_diode_grid`'s own
`mat_n=None`), so ONE implementation covers both the homogeneous-Si and
heterojunction SiGe/Si cases**:

- `built_in_potential`/`depletion_widths` gain an optional `mat_n=`
  argument (existing 2-3-positional-arg call sites everywhere in the
  codebase are unaffected - confirmed by the unchanged 31-test suite).
  `Vbi_hetero` is built from each side's own EXACT bulk equilibrium
  potential (`core.physics.equilibrium_bulk_potential`, material-only, no
  solve) minus that side's `delta_Ei` - reusing `core.materials.Xi()`/
  `delta_Ei_of()` (newly factored out of `MaterialField.from_regions`,
  itself unchanged behavior, confirmed by the unchanged test suite) rather
  than re-deriving the same quantity twice. Verified NUMERICALLY that
  passing `mat_n=mat_p` reproduces the homojunction `Vbi` to ~1e-14
  (floating-point noise, not merely "close").
  Two-material depletion widths: charge balance `Na*xp=Nd*xn` still holds
  regardless of `eps_p`/`eps_n` (D-field continuity reduces to it exactly
  - `D_max_p=q*Na*xp`, `D_max_n=q*Nd*xn`, equal by construction); solving
  `V=(q/2)*(Na*xp^2/eps_p+Nd*xn^2/eps_n)` for `xn` and verified BY HAND
  that setting `eps_p=eps_n` reduces the result ALGEBRAICALLY EXACTLY
  (not approximately) to the pre-existing homojunction formula - confirmed
  numerically too (same ~1e-14 agreement).
- `generation_current_srh(mat_p, dev, Va, mat_n=None)`: the plain (F=0,
  no tunneling) SRH depletion-generation current, `J=q*[ni_p/(tau_n_p+
  tau_p_p)*xp + ni_n/(tau_n_n+tau_p_n)*xn]`. Verified the `ni/(tau_n+
  tau_p)` generation rate matches `tat.tat.hurkx_tat_generation` EXACTLY
  at `Gamma=0` (F=0), `n=p=0` (full depletion), `Et=Ei` (the default trap
  level) - worked out by hand before trusting the formula, not assumed.
- `generation_current_tat(mat_p, dev, Va, hurkx_model, kane_model,
  mat_n=None)`: the field-enhanced closed form, integrating (`np.trapz`,
  matching `ionization_integral`'s own quadrature style) `[ni/(tau_n+
  tau_p)]*(1+Gamma(F(x)))` (Hurkx) plus the separately additive
  `G_btbt(F(x))` (Kane) over the depletion approximation's TRIANGULAR
  field profile (`E_p(x)=E_max_p*(1-|x|/xp)`, `E_n(x)=E_max_n*(1-x/xn)`,
  `E_max` from `D=eps*E` continuity). Deliberately reuses `tat.tat.
  hurkx_gamma`/`btbt_generation` directly (the SAME fitted models the
  numeric solver uses) - this validates the SOLVER's own discretization/
  BC/mesh machinery against an independent field profile and quadrature,
  not `tat.py`'s formulas themselves (already covered by their own
  standalone `tat.tat.sanity_probe()`).

**Wired into `tat/main_tat.py`**: both closed-form curves computed across
the full reverse sweep, overlaid as dotted reference lines on both the
linear and log `I(Va)` panels (against the numeric TAT+BTBT and
no-tunneling curves), written to the per-config CSV as two extra columns,
and printed as a numeric comparison table (numeric vs. closed-form I and
their ratio, at 8 representative reverse-bias points) alongside the
existing self-consistency printout.

**Verification / results** (full 31-test suite still passes unchanged
throughout):

- `input_diode_drain_substrate.yaml` (homogeneous Si): numeric-TAT vs.
  closed-form-TAT ratio runs from 0.061 (Va=-0.05V) up to 0.74 (Va=-5V);
  numeric-noTAT vs. closed-form-SRH ratio from 0.059 to 0.70 - same
  pattern, same order of magnitude throughout, monotonically closing the
  gap as bias deepens.
- `input_diode_sige_pn.yaml` (SiGe/Si heterojunction): nearly IDENTICAL
  ratio pattern - 0.062 to 0.67 (TAT), 0.058 to 0.62 (SRH) - confirming the
  closed form's heterojunction generalization behaves consistently with
  the homojunction case, not just algebraically-verified in isolation.
- **Where they agree well**: the numeric and closed-form curves track each
  other within a factor of ~1.3-16x throughout the ENTIRE reverse sweep on
  BOTH configs, converging to within ~1.3-1.4x at the deepest bias points
  (-4 to -5V) - this is exactly the trend expected: the closed form's
  triangular-field/depletion-approximation gets more accurate as the
  numeric solution's own field profile becomes more sharply peaked and
  triangle-like under deeper reverse bias.
- **Where/why they don't agree exactly** (expected approximation gaps, not
  bugs - both curves being systematically smaller in magnitude than the
  numeric TAT/no-TAT curves near Va~0, by up to ~16x, is the dominant
  mismatch): (1) the closed form assumes n=p=0 EXACTLY throughout the
  depletion region (the textbook "full depletion" assumption) - the real
  numeric solution has nonzero, bias-dependent carrier densities there,
  most significant near equilibrium where injected/thermal carriers are
  NOT negligible relative to the (still small) reverse-bias generation
  current, weakening exactly as bias deepens and generation dominates
  (matches the observed ratio trend closing with |Va|); (2) the
  depletion-approximation's abrupt-edge triangular field profile
  systematically UNDERSTATES the true self-consistent field's smoother,
  more gradually-decaying shape and the true depletion width (this
  project's own numeric solve is NOT a hard-wall depletion approximation),
  so the closed form's integrated generation is a biased estimate in a
  fixed, not-obviously-corrected direction; (3) series/contact effects and
  the exact mesh/BC treatment at the ohmic contacts (not modeled at all in
  the closed form) contribute additional, smaller differences. None of
  these individually or together explain a wrong SIGN, wrong ORDER OF
  MAGNITUDE, or non-monotonic trend - the observed gap is consistent with
  known, named approximations the closed form deliberately makes, not
  evidence of a solver bug.

**Final output paths** (both re-run fresh this session, previously-
clobbered top-level `out/tat/tat_iv.{csv,png}` restored to its original
pre-session content via `git checkout`):
- `out/tat/input_diode_drain_substrate/tat_iv.csv`,
  `out/tat/input_diode_drain_substrate/tat_iv.png`,
  `out/tat/input_diode_drain_substrate/tat_bands.png`,
  `out/tat/input_diode_drain_substrate/diode_drain_substrate_structure.json`
- `out/tat/input_diode_sige_pn/tat_iv.csv`,
  `out/tat/input_diode_sige_pn/tat_iv.png`,
  `out/tat/input_diode_sige_pn/tat_bands.png`,
  `out/tat/input_diode_sige_pn/diode_sige_pn_structure.json`

Not done this session: no attempt to close the SiGe/Si case's remaining
reverse-bias self-consistency gap (session 13's own open item, unrelated
to this closed-form validation work); no closed-form BTBT-only or
Hurkx-only decomposition curve (only the combined TAT+BTBT and plain-SRH
curves were added, matching what the numeric solver itself reports).

## 21. Session 14: compressively strained Si(1-x)Ge(x)-on-Si band offsets
(People & Bean), reusing last session's `delta_Ei` machinery unmodified

Follow-up to sessions 13/13-continued's relaxed-alloy SiGe/Si heterojunction
work. The user wants the physically realistic case: SiGe grown epitaxially
on a Si substrate is under biaxial compressive strain (SiGe's larger
relaxed lattice constant compressed in-plane to match Si's), not the
relaxed alloy assumed before. Strain changes the band alignment
STRUCTURALLY, not just `Eg`'s scalar number: literature (People & Bean,
*Appl. Phys. Lett.* 48, 538 (1986); consistent with Van de Walle & Martin,
*Phys. Rev. B* 34, 5621 (1986), confirmed via web search) is consistent
that almost the ENTIRE bandgap reduction from Ge alloying lands in the
VALENCE band under compressive strain:
`delta_Ev(x)=(0.74-0.53*x')*x eV`, `delta_Ec(x)~=0` (x'=substrate Ge
fraction, "a few meV" in the literature, standard Type-I device-modeling
simplification) - for growth on pure Si (x'=0), `delta_Ev(x)=0.74*x eV`.
Qualitatively different from the relaxed-alloy Vegard-mixed model (which
implicitly splits the bandgap difference between conduction and valence
bands however linear-mixed `chi_eV`/`Nc`/`Nv` happen to place it).

**Confirmed the architecture claim from last session's own scoping note
before building anything**: `core/materials.py`'s `Xi()`/`delta_Ei_of()`
(the heterojunction Boltzmann-relation machinery `MaterialField.
from_regions` already uses) is derived purely from each region's own
`chi_eV`/`Nc`/`ni` on a `Material` object - it genuinely does not care
whether those came from Vegard mixing or a strain model. This session's
whole implementation is therefore additive: a new material-RESOLUTION
path producing a `Material` with strain-corrected `chi_eV`/`Eg_eV`, with
NOTHING in `mesh.py`/`physics.py`/`solver.py`/`newton_solver_qf.py`/
`newton_solver_tat.py`/`analytic.py` needing to change - confirmed true in
practice, not just in theory (every one of those files is untouched this
session).

**What was added**:
- `core/materials.py`: `strained_sige_on_si_offsets(x_Ge, x_Ge_substrate=0.0)
  -> (delta_Ec_eV, delta_Ev_eV)` implementing the People & Bean formula
  above (docstring cites the source and states the `delta_Ec~=0`
  simplification explicitly, not silently). `AlloyMaterial.
  resolve_strained(x, substrate="Silicon", x_substrate_Ge=0.0)`: starts
  from the SAME Vegard-mixed `MaterialProperties` the existing `resolve(x)`
  already produces (so `eps_r`/`Nc_300K`/`Nv_300K`/`mu_n`/`mu_p`/`tau_n`/
  `tau_p` are UNCHANGED from the relaxed case - explicitly scoped out per
  the harmonic-snuggling-puddle plan's "what this does NOT attempt"
  section: no valley-splitting DOS correction, no strain-enhanced
  mobility, no critical-thickness/relaxation check, all three documented
  as known simplifications, not silent gaps), then overrides ONLY
  `Eg_eV_300K`/`chi_eV` via the `delta_Ec`/`delta_Ev` shifts relative to
  the substrate material's own values. Kept as a SEPARATE method from
  `resolve()` (not a flag), matching this project's existing swappable-
  model pattern (Hurkx vs. Schenk TAT, the three Kane P-variants) - so
  relaxed and strained stay both directly available/comparable. `x_Ge` is
  resolved from the `AlloyMaterial`'s own end-member identity (whichever
  of `end_member_a`/`end_member_b` is `"Germanium"`), so both the existing
  `end_member_a=Silicon` config convention AND a `end_member_a=Germanium`
  convention work correctly with the same `x` argument `resolve()` uses.
- `core/material_db.py`: `derive_alloy()` gains optional `strained_on`/
  `x_substrate_Ge` pass-through to call `resolve_strained` instead of
  `resolve` when given.
- `core/config.py`: `material.p_side.alloy` gains an optional
  `strain: {substrate, x_substrate_Ge}` key - omitted (as in the existing
  relaxed example) keeps today's exact relaxed behavior.
- New example config `configs/input_diode_sige_pn_strained.yaml`, identical
  doping/geometry/voltage-sweep to `configs/input_diode_sige_pn.yaml`
  (the relaxed version) with `material.p_side.alloy.strain` set, so the
  two can be diffed directly for the same doping/geometry.
- `tat/main_tat.py`: no code change needed (already generic over
  `mat_p`/`mat_n` since last session) - just run it against the new
  config; per-config output directories (fixed last session) keep this
  run's plots separate.

**Verification**:
- Full test suite: 37 tests now (31 original + 6 new
  `TestStrainedSiGeOffsets` cases in `testsuite/test_materials.py`), all
  pass. New tests cover: the offsets formula itself
  (`x_Ge=0.4,x'=0 -> delta_Ev=0.296eV` exactly); `resolve_strained(0.0)`
  reduces exactly to the substrate (`Eg`/`chi` match Silicon's catalog
  values); `resolve_strained(0.4)` matches People & Bean
  (`Eg=1.12-0.296=0.824eV`, `chi` unchanged from Si's 4.05eV since
  `delta_Ec=0`); `eps_r`/`Nc_300K`/`Nv_300K`/`mu_n`/`mu_p`/`tau_n`/`tau_p`
  are IDENTICAL between `resolve_strained(0.4)` and `resolve(0.4)` (same
  object fields, confirming only `Eg`/`chi` were touched) while `Eg`/`chi`
  themselves genuinely differ (confirming the override actually happened,
  not a no-op); the actual shipped config's `end_member_a=Silicon,
  x_a=0.6` convention gives the identical numeric result to an
  `end_member_a=Germanium, x_Ge=0.4` fixture (both conventions correct);
  a non-Si/Ge `AlloyMaterial` raises `ValueError` rather than silently
  computing something wrong.
- **The actual numbers, side by side** (printed directly from
  `core.config.build_from_config` on both shipped configs, confirming the
  qualitative distinction the user asked to see - not just Eg's scalar
  number):
  ```
  RELAXED:   p-side(SiGe) chi_eV=4.0300  Eg_eV=0.8496   delta_Ei(p-node)=-0.1565 eV
  STRAINED:  p-side(SiGe) chi_eV=4.0500  Eg_eV=0.8240   delta_Ei(p-node)=-0.1493 eV
  (both cases: n-side(Si) chi_eV=4.0500  Eg_eV=1.1200)
  ```
  Strained `chi_eV` is EXACTLY Si's 4.05eV (delta_Ec=0 by construction);
  relaxed `chi_eV` is the Vegard-mixed 4.03eV - close by coincidence at
  this composition, but for a structurally different reason (a real
  valence-band step vs. a mixed conduction+valence split), exactly as the
  plan anticipated. `delta_Ei` itself (the actual quantity threaded into
  the Boltzmann relation/solve) differs by ~7meV between the two cases -
  small in ABSOLUTE terms at this particular composition, but the
  MECHANISM generating it is genuinely different, confirming the
  architecture is picking up a real physics distinction, not coincidentally
  producing the same number both ways.
- Closed-form validation (`core.analytic.generation_current_srh`/
  `generation_current_tat`, unchanged from last session) rerun against the
  strained config exactly as done for the relaxed case: numeric/closed-form
  ratios run 0.058->0.62 (TAT) and 0.054->0.58 (SRH) from Va=-0.05V to
  -5V - nearly identical pattern to the relaxed case's own 0.062->0.67 /
  0.058->0.62, confirming the architecture is genuinely material-agnostic
  (same validation machinery, no analytic.py changes, works unmodified).
- Full reverse(-0.05 to -5V)/forward(0 to 0.6V) sweep on the strained
  config: forward bias converges cleanly (self-consistency ~1e-4 by
  0.6V, matching both prior cases). Reverse-bias self-consistency shows
  the SAME class of open item flagged last session for the relaxed
  case (not new, not worse) - most reverse points have self-consistency
  ratios ~1.2-2.8 (comparable to the relaxed case's ~1.1-6), with one
  point (Va=-2.59V) showing a clear Newton stall (I_tat collapsed to
  ~1e-13A, self-consistency ~1.4e4) - consistent with the already-
  documented hard-convergence tail for this doping/material combination,
  not a new regression from the strain feature itself.
- **Qualitative leakage comparison across all three cases** (I_tat at
  matched Va, from each config's own `tat_iv.csv`):
  ```
  Va      Si-only(drain_substrate)   relaxed SiGe/Si     strained SiGe/Si
  -0.94   -6.42e-10                  -4.26e-10           -4.26e-10
  -1.95   -1.13e-09                  -7.51e-10           -7.52e-10
  -2.97   -1.55e-09                  -1.03e-09           -1.03e-09
  -3.98   -1.91e-09                  -1.27e-09           -1.27e-09
  -5.00   -2.25e-09                  -1.50e-09           -1.50e-09
  ```
  **Relaxed and strained SiGe give NEARLY IDENTICAL leakage current**
  (agreement to 3+ significant figures throughout the sweep) - and this
  has the SAME explanation flagged last session for why SiGe leakage came
  out lower than pure Si, not a new finding: with p-SiGe heavily doped
  (1e20) and n-Si lightly doped (1e17), depletion sits almost entirely on
  the LIGHT n-Si side in BOTH SiGe configs, so the heavy p-side SiGe
  region - whichever band-offset model describes it - barely participates
  in the high-field tunneling-generation region at all. This is a genuine,
  useful finding worth stating plainly: for THIS doping configuration, the
  choice between the relaxed and strained SiGe models is essentially
  UNOBSERVABLE in the leakage I-V, because the region where the model
  difference would matter (the SiGe depletion layer) is negligibly thin.
  Demonstrating an actual relaxed-vs-strained leakage DIFFERENCE would
  need a doping configuration where SiGe is the LIGHTLY doped (depletion-
  hosting) side - the same follow-up flagged last session for showing the
  narrow-gap-leakage enhancement in the first place, now doubly motivated.

**Final output paths**:
- `out/tat/input_diode_sige_pn_strained/tat_iv.csv`,
  `out/tat/input_diode_sige_pn_strained/tat_iv.png`,
  `out/tat/input_diode_sige_pn_strained/tat_bands.png`,
  `out/tat/input_diode_sige_pn_strained/diode_sige_pn_strained_structure.json`
- (relaxed and Si-only cases' outputs unchanged from last session's run:
  `out/tat/input_diode_sige_pn/*`, `out/tat/input_diode_drain_substrate/*`)

Not done this session, explicitly deferred (both already flagged last
session, reinforced by this session's finding above): no doping variant
with SiGe on the LIGHTLY doped side (needed to actually observe either the
narrow-gap-leakage enhancement OR a relaxed-vs-strained leakage
difference); no resolution of the reverse-bias self-consistency open item
for this doping/material class.
