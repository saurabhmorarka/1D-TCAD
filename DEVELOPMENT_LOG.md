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
