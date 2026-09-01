# 1D Diode

A 1D TCAD-style drift-diffusion simulator for a p-n junction diode, built from
scratch in Python/NumPy/SciPy.

## What it does

- Builds a nonuniform 1D mesh across a step p-n junction (sub-nm spacing at
  the junction, geometrically coarsening into the bulk).
- Solves the **equilibrium** nonlinear Poisson equation (Newton's method) for
  the self-consistent built-in potential and space-charge profile.
- Solves the **biased** drift-diffusion system (Poisson + electron/hole
  continuity, Scharfetter-Gummel discretization) via Gummel iteration,
  sweeping applied voltage forward and reverse.
- Compares against closed-form theory: built-in potential
  `Vbi = Vt*ln(Na*Nd/ni^2)`, the depletion approximation, and the Shockley
  long-base ideal diode law `I = I0*(exp(V/Vt)-1)`.

Doping (Na, Nd), mobility, SRH lifetime, and the voltage sweep range are all
parameters in `params.py` / `main.py`.

## Files

| File | Purpose |
|---|---|
| `params.py` | Physical constants and material/device parameters |
| `mesh.py` | Nonuniform grid generator |
| `physics.py` | Bernoulli function, nonlinear Poisson (Newton), Scharfetter-Gummel continuity solves |
| `analytic.py` | Closed-form comparisons (Vbi, depletion width, Shockley law) |
| `solver.py` | Equilibrium solve + Gummel-iteration bias sweep |
| `main.py` | Driver: runs the sweep, generates plots and `iv_sweep.csv` in `out/` |

## Running

```bash
python3 main.py
```

Requires `numpy`, `scipy`, `matplotlib`. Output plots and `iv_sweep.csv` are
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
