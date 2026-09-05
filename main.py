"""Driver script: 1D TCAD diode simulation.

Builds a nonuniform mesh for a step p-n junction, solves the equilibrium
(built-in) potential, then sweeps applied bias (forward and reverse) using a
Scharfetter-Gummel drift-diffusion solver (Gummel iteration). Compares
results against closed-form expressions: built-in potential, depletion
approximation, and the Shockley ideal-diode law.

Doping, mobility, lifetime, mesh, and voltage sweep range are read from
input_diode.yaml (see config.py) - edit that file rather than this one to
change the simulation's parameters. Material()/Device() here are just the
defaults input_diode.yaml overrides.
"""
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh import build_diode_grid
from solver import solve_equilibrium, voltage_sweep
import analytic as an
import config as cfg
import field_save as fsave

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ---- Color palette (qualitative, colorblind-friendly-ish, consistent across plots) ----
C_NUM = "#1f6feb"     # numeric simulation - blue
C_AN = "#e8590c"      # analytic / closed-form - orange
C_GRID = "#c9c9c9"


def main():
    input_cfg = cfg.load_config()
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts = cfg.build_from_config(input_cfg)
    print(f"Loaded input_diode.yaml (solver.math_model={math_model!r})" if input_cfg
          else "No input_diode.yaml found - using params.py defaults")

    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]
    print(f"Grid: {len(x)} points, Wp={g['Wp']*1e4:.2f} um, Wn={g['Wn']*1e4:.2f} um, "
          f"h_min={g['h_min']*1e7:.2f} nm, h_max={g['h_max']*1e7:.2f} nm")
    print(f"Doping: Na={dev.Na:.2e} cm^-3 (p-side), Nd={dev.Nd:.2e} cm^-3 (n-side)")
    print(f"Ln={mat.Ln*1e4:.2f} um, Lp={mat.Lp*1e4:.2f} um (tau_n=tau_p={mat.tau_n*1e9:.2g} ns)")

    # ---- Equilibrium ----
    psi_eq, n_eq, p_eq, eq_iters = solve_equilibrium(x, Cdop, mat)
    Vbi_num = psi_eq[-1] - psi_eq[0]
    Vbi_an = an.built_in_potential(mat, dev)
    xp_an, xn_an, W_an = an.depletion_widths(mat, dev, 0.0)
    print(f"\nEquilibrium: Newton iters={eq_iters}")
    print(f"  Built-in potential: numeric={Vbi_num:.4f} V, analytic Vt*ln(Na*Nd/ni^2)={Vbi_an:.4f} V "
          f"(diff={abs(Vbi_num-Vbi_an)*1e6:.3g} uV)")
    print(f"  Depletion width (analytic, step-junction approx): xp={xp_an*1e4:.4f} um, "
          f"xn={xn_an*1e4:.4f} um, W={W_an*1e4:.4f} um")

    # ---- Voltage sweep (solver chosen by input_diode.yaml: solver.math_model) ----
    print(f"\nRunning bias sweep ({math_model} drift-diffusion solve per point)...")
    _, _, _, results = voltage_sweep(x, Cdop, mat, dev, Va_list, verbose=False, method=math_model)

    Va_arr = np.array([r["Va"] for r in results])
    I_num = np.array([r["I"] for r in results])
    Jres_arr = np.array([r["J_std"] / max(abs(r["J_mean"]), 1e-30) for r in results])
    I_an = an.shockley_current(mat, dev, Va_arr)
    I0 = an.shockley_I0(mat, dev)

    print(f"\nShockley I0 (long-base ideal diode) = {I0:.4e} A")
    print(f"{'Va (V)':>8} {'I_num (A)':>14} {'I_shockley (A)':>16} {'ratio':>10} {'self-consist(%)':>16}")
    for Va, In, Ia, res in zip(Va_arr, I_num, I_an, results):
        ratio = In / Ia if abs(Ia) > 1e-30 else float("nan")
        print(f"{Va:8.3f} {In:14.4e} {Ia:16.4e} {ratio:10.3f} {res['J_std']/max(abs(res['J_mean']),1e-30)*100:16.3f}")

    # ---- Ideality factor n(V) = Vt * d(ln I)/dV, on forward branch ----
    fwd_mask = Va_arr > 0.02
    Vf = Va_arr[fwd_mask]
    If = I_num[fwd_mask]
    order = np.argsort(Vf)
    Vf, If = Vf[order], If[order]
    # ideality factor n defined via I = I0*exp(V/(n*Vt))  =>  n = 1 / (Vt * d(lnI)/dV)
    dlnI_dV = np.gradient(np.log(If), Vf)
    n_ideal = 1.0 / (mat.Vt * dlnI_dV)

    # ================= Plots =================

    # 1. Equilibrium band diagram: numeric vs depletion-approximation (full extent + zoom on the junction)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x_um = x * 1e4
    psi_an = an.depletion_potential_profile(mat, dev, x, Va=0.0)
    zoom_half_width_um = max(3 * W_an * 1e4, 0.5)
    for ax, xlim, title in (
        (axes[0], None, "Full device"),
        (axes[1], (-zoom_half_width_um, zoom_half_width_um), "Zoom on junction/depletion region"),
    ):
        ax.plot(x_um, psi_eq, color=C_NUM, lw=2, label="Numeric (full nonlinear Poisson)")
        ax.plot(x_um, psi_an, color=C_AN, lw=1.8, ls="--", label="Depletion approximation")
        ax.axvline(0, color=C_GRID, lw=1, zorder=0)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("Electrostatic potential psi (V)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, color=C_GRID)
    fig.suptitle(f"Equilibrium band bending  (Vbi: numeric={Vbi_num:.4f} V, analytic={Vbi_an:.4f} V)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_equilibrium_potential.png"), dpi=150)
    plt.close(fig)

    # 2. Carrier concentration profiles: equilibrium and a forward bias case
    fwd_pick = min(results, key=lambda r: abs(r["Va"] - 0.5))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(x_um, n_eq, color=C_NUM, lw=2, label="n, equilibrium")
    ax.semilogy(x_um, p_eq, color=C_AN, lw=2, label="p, equilibrium")
    ax.semilogy(x_um, fwd_pick["n"], color=C_NUM, lw=1.5, ls="--",
                label=f"n, Va={fwd_pick['Va']:.2f} V")
    ax.semilogy(x_um, fwd_pick["p"], color=C_AN, lw=1.5, ls="--",
                label=f"p, Va={fwd_pick['Va']:.2f} V")
    ax.axvline(0, color=C_GRID, lw=1, zorder=0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Carrier concentration (cm^-3)")
    ax.set_title("Carrier profiles: equilibrium vs forward bias")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, color=C_GRID, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_carrier_profiles.png"), dpi=150)
    plt.close(fig)

    # 3. I-V curve: linear (forward only) and semilog |I| (full range)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    fmask = Va_arr >= 0
    ax.plot(Va_arr[fmask], I_num[fmask] * 1e3, color=C_NUM, lw=2, marker="o", ms=3,
            label="Numeric drift-diffusion")
    ax.plot(Va_arr[fmask], I_an[fmask] * 1e3, color=C_AN, lw=1.8, ls="--",
            label="Shockley ideal diode")
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Forward I-V (linear)")
    ax.legend()
    ax.grid(alpha=0.3, color=C_GRID)

    ax = axes[1]
    ax.semilogy(Va_arr, np.abs(I_num), color=C_NUM, lw=2, marker="o", ms=3,
                label="Numeric drift-diffusion")
    ax.semilogy(Va_arr, np.abs(I_an), color=C_AN, lw=1.8, ls="--",
                label="Shockley ideal diode")
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("|Current| (A)")
    ax.set_title("I-V (semilog magnitude, incl. reverse bias)")
    ax.legend()
    ax.grid(alpha=0.3, color=C_GRID, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_iv_curve.png"), dpi=150)
    plt.close(fig)

    # 4. Ideality factor vs voltage
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Vf, n_ideal, color=C_NUM, lw=2)
    ax.axhline(1.0, color=C_AN, lw=1.5, ls="--", label="n=1 (ideal diffusion current)")
    ax.axhline(2.0, color="#8a8a8a", lw=1.5, ls=":", label="n=2 (depletion-region SRH recombination)")
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("Ideality factor  n = Vt d(ln I)/dV")
    ax.set_title("Extracted ideality factor: recombination-dominated -> diffusion-dominated")
    ax.legend()
    ax.grid(alpha=0.3, color=C_GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_ideality_factor.png"), dpi=150)
    plt.close(fig)

    # 5. Quasi-Fermi potentials (non-equilibrium) at a configurable set of bias points
    # (input_diode.yaml: output.save_bias_points - a list of Va values, "all", or "last")
    save_idx = fsave.resolve_save_points(save_bias_points, Va_arr)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_um, psi_eq, color=C_GRID, lw=1.2, ls=":", label="psi, equilibrium (V_a=0 reference)")
    fsave.plot_quasi_fermi(ax, results, save_idx, "Va", x_um)
    ax.axvline(0, color=C_GRID, lw=1, zorder=0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Quasi-Fermi potential (V)")
    ax.set_title("Electron/hole quasi-Fermi potentials at saved bias points")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, color=C_GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_quasi_fermi_potentials.png"), dpi=150)
    plt.close(fig)

    fields_csv = os.path.join(OUT, "fields_by_bias.csv")
    fsave.save_fields(results, save_idx, "Va", fields_csv, x)
    print(f"\nSaved full field profiles for Va = {[round(results[i]['Va'], 4) for i in save_idx]} "
          f"to {fields_csv}")

    # ================= Solver runtime benchmark: Gummel vs Newton =================
    # Runs BOTH solvers across the same voltage sweep (independent of which one
    # produced the results/plots above) so their wall-clock cost is directly
    # comparable point-by-point, not just at one bias.
    print("\nBenchmarking solvers (Gummel iteration vs coupled Newton) across the full sweep...")
    bench = {}
    for method in ("gummel", "newton"):
        t0 = time.perf_counter()
        _, _, _, bres = voltage_sweep(x, Cdop, mat, dev, Va_list, verbose=False, method=method)
        total_t = time.perf_counter() - t0
        bench[method] = {
            "results": bres,
            "total_time": total_t,
            "times": np.array([r["solve_time_s"] for r in bres]),
            "iters": np.array([r["iters"] for r in bres]),
        }
        print(f"  {method:8s}: total={total_t:.3f}s over {len(Va_list)} points "
              f"(avg {total_t/len(Va_list)*1e3:.1f} ms/point, "
              f"avg {bench[method]['iters'].mean():.1f} iters/point)")

    speedup = bench["gummel"]["total_time"] / bench["newton"]["total_time"]
    print(f"  Newton is {speedup:.2f}x faster than Gummel over the full sweep "
          f"({bench['gummel']['total_time']:.3f}s vs {bench['newton']['total_time']:.3f}s)")

    # 5. Solver benchmark plot: per-point time and iteration count, Gummel vs Newton
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot(Va_arr, bench["gummel"]["times"] * 1e3, color=C_AN, lw=2, marker="o", ms=3, label="Gummel")
    ax.plot(Va_arr, bench["newton"]["times"] * 1e3, color=C_NUM, lw=2, marker="o", ms=3, label="Newton (coupled)")
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("Solve time per bias point (ms)")
    ax.set_title(f"Per-point solve time  (total: Gummel {bench['gummel']['total_time']:.2f}s, "
                 f"Newton {bench['newton']['total_time']:.2f}s, {speedup:.1f}x speedup)")
    ax.legend()
    ax.grid(alpha=0.3, color=C_GRID)

    ax = axes[1]
    ax.plot(Va_arr, bench["gummel"]["iters"], color=C_AN, lw=2, marker="o", ms=3, label="Gummel")
    ax.plot(Va_arr, bench["newton"]["iters"], color=C_NUM, lw=2, marker="o", ms=3, label="Newton (coupled)")
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("Outer iterations to converge")
    ax.set_title("Iteration count  (Newton: quadratic convergence; Gummel: linear)")
    ax.legend()
    ax.grid(alpha=0.3, color=C_GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "06_solver_benchmark.png"), dpi=150)
    plt.close(fig)

    bench_csv = os.path.join(OUT, "solver_benchmark.csv")
    with open(bench_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "gummel_time_s", "gummel_iters", "newton_time_s", "newton_iters"])
        for i, Va in enumerate(Va_arr):
            w.writerow([f"{Va:.4f}",
                        f"{bench['gummel']['times'][i]:.5f}", bench["gummel"]["iters"][i],
                        f"{bench['newton']['times'][i]:.5f}", bench["newton"]["iters"][i]])

    # ---- CSV export ----
    csv_path = os.path.join(OUT, "iv_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "I_numeric_A", "I_shockley_A", "self_consistency_pct"])
        for Va, In, Ia, res in zip(Va_arr, I_num, I_an, results):
            w.writerow([f"{Va:.4f}", f"{In:.6e}", f"{Ia:.6e}",
                        f"{res['J_std']/max(abs(res['J_mean']),1e-30)*100:.4f}"])

    print(f"\nPlots and data written to: {OUT}")
    print("  01_equilibrium_potential.png")
    print("  02_carrier_profiles.png")
    print("  03_iv_curve.png")
    print("  04_ideality_factor.png")
    print("  05_quasi_fermi_potentials.png")
    print("  06_solver_benchmark.png")
    print("  fields_by_bias.csv")
    print("  iv_sweep.csv")
    print("  solver_benchmark.csv")


if __name__ == "__main__":
    main()
