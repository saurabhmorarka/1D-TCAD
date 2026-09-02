# 1D Diode

A 1D TCAD-style drift-diffusion simulator for a p-n junction diode, built from
scratch in Python/NumPy/SciPy.

## What it does

- Builds a nonuniform 1D mesh across a step p-n junction (sub-nm spacing at
  the junction, geometrically coarsening into the bulk).
- Solves the **equilibrium** nonlinear Poisson equation (Newton's method) for
  the self-consistent built-in potential and space-charge profile.
- Solves the **biased** drift-diffusion system (Poisson + electron/hole
  continuity, Scharfetter-Gummel discretization), sweeping applied voltage
  forward and reverse, using either of two interchangeable solvers:
  - **Gummel iteration**: decoupled, robust, linear convergence (many outer
    iterations).
  - **Coupled Newton**: all three equations (Poisson + both continuity
    equations) solved together with an analytic sparse Jacobian and a direct
    sparse solve per step, quadratic convergence (few outer iterations,
    ~4x faster wall-clock - see `out/05_solver_benchmark.png`).
- Compares against closed-form theory: built-in potential
  `Vbi = Vt*ln(Na*Nd/ni^2)`, the depletion approximation, and the Shockley
  long-base ideal diode law `I = I0*(exp(V/Vt)-1)`.

All simulation parameters (doping, device thickness, voltage sweep range,
which solver to use) are read from `input.yaml`, not hardcoded - edit that
file to change them. `params.py` just holds the defaults it overrides.

## Files

| File | Purpose |
|---|---|
| `input.yaml` | **Edit this** to change doping, thickness, voltage sweep, or solver (`math_model: gummel` \| `newton`) |
| `config.py` | Loads `input.yaml` and builds the `Material`/`Device`/voltage-sweep/solver-choice overrides from it |
| `params.py` | Physical constants and material/device parameter defaults |
| `mesh.py` | Nonuniform grid generator |
| `physics.py` | Bernoulli function (+ its derivative), nonlinear Poisson (Newton), Scharfetter-Gummel continuity solves |
| `analytic.py` | Closed-form comparisons (Vbi, depletion width, Shockley law) |
| `solver.py` | Equilibrium solve + bias sweep (dispatches to either solver below) |
| `newton_solver.py` | Fully coupled Newton solve: analytic sparse Jacobian, direct sparse solve, backtracking line search |
| `main.py` | Driver: runs the sweep with both solvers (for the benchmark) plus the one from `input.yaml`, generates plots and CSVs in `out/` |

## Running

```bash
python3 main.py
```

Requires `numpy`, `scipy`, `matplotlib`, `pyyaml`. Output plots and CSVs are
written to `out/`.

## Results

The simulation reproduces standard diode physics:

- The numerically self-consistent built-in potential matches
  `Vt*ln(Na*Nd/ni^2)` to 8 significant figures.
- The equilibrium potential profile matches the depletion approximation,
  with the expected smoothing (over a Debye length) at the depletion edges
  that the depletion approximation idealizes as abrupt.
- The extracted ideality factor rises toward n≈1.85 near the recombination
  peak (SRH recombination in the depletion region dominates at low forward
  bias) and relaxes toward n≈1 at higher forward bias (bulk diffusion
  current dominates) — the textbook two-regime diode I-V curve.
- Reverse leakage current is orders of magnitude above the ideal Shockley
  I0, correctly reflecting depletion-region generation current that the
  simple long-base ideal-diode formula does not model.
- The coupled Newton solver matches Gummel's current to ~4 significant
  figures at every bias point while using roughly 5x fewer outer iterations
  (quadratic vs. linear convergence) and running about 4x faster overall.
