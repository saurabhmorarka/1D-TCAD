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
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh import build_diode_grid
from solver import solve_equilibrium, voltage_sweep
from params import Q
import analytic as an
import config as cfg
import field_save as fsave
import structure_io as sio
import plot as tplot

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# ---- Color palette (qualitative, colorblind-friendly-ish, consistent across plots) ----
C_NUM = "#1f6feb"     # numeric simulation - blue
C_AN = "#e8590c"      # analytic / closed-form - orange
C_AN_FD = "#9d4edd"   # analytic, Fermi-Dirac-corrected reference - purple
C_GRID = "#c9c9c9"


def main():
    # Optional CLI arg selects which input YAML to run (default
    # input_diode.yaml) - e.g. `python3 main.py input_diode_asymmetric.yaml`.
    # Output filenames are prefixed by the input file's own name (stripped
    # of the "input_diode_" prefix) for anything other than the default, so
    # multiple examples' plots/CSVs can coexist in out/ without clobbering
    # each other - same convention mos_main.py/mos_poly_sweep.py use.
    input_path = sys.argv[1] if len(sys.argv) > 1 else cfg.DEFAULT_PATH
    base = os.path.splitext(os.path.basename(input_path))[0]
    prefix = "" if base == "input_diode" else base.replace("input_diode_", "").replace("input_diode", "diode") + "_"

    def outp(name):
        return os.path.join(OUT, prefix + name)

    input_cfg = cfg.load_config(input_path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = cfg.build_from_config(input_cfg)
    print(f"Loaded {os.path.basename(input_path)} (solver.math_model={math_model!r})" if input_cfg
          else f"No {os.path.basename(input_path)} found - using params.py defaults")

    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]
    print(f"Grid: {len(x)} points, Wp={g['Wp']*1e4:.2f} um, Wn={g['Wn']*1e4:.2f} um, "
          f"h_min={g['h_min']*1e7:.2f} nm, h_max_p={g['h_max_p']*1e7:.2f} nm, h_max_n={g['h_max_n']*1e7:.2f} nm")
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

    # Fermi-Dirac comparison: only meaningful (and only computed) when a
    # side is degenerate enough for the closed-form reference to actually
    # shift - see fermi_dirac.py for exactly what is/isn't corrected (the
    # numeric PDE solve above stays Boltzmann-only either way).
    is_degenerate = dev.Na > 0.1 * mat.Nv or dev.Nd > 0.1 * mat.Nc
    if is_degenerate:
        Vbi_an_fd = an.built_in_potential_fd(mat, dev)
        print(f"  Fermi-Dirac reference (doping approaches/exceeds Nc={mat.Nc:.1e}/Nv={mat.Nv:.1e} "
              f"cm^-3 - Boltzmann may be inaccurate here):")
        print(f"    Built-in potential (Fermi-Dirac) = {Vbi_an_fd:.4f} V "
              f"(vs Boltzmann {Vbi_an:.4f} V, diff={Vbi_an_fd-Vbi_an:+.4f} V)")

    # ---- Voltage sweep (solver chosen by input_diode.yaml: solver.math_model) ----
    print(f"\nRunning bias sweep ({math_model} drift-diffusion solve per point)...")
    _, _, _, results = voltage_sweep(x, Cdop, mat, dev, Va_list, verbose=False, method=math_model)

    Va_arr = np.array([r["Va"] for r in results])
    I_num = np.array([r["I"] for r in results])
    Jres_arr = np.array([r["J_std"] / max(abs(r["J_mean"]), 1e-30) for r in results])
    I_an = an.shockley_current(mat, dev, Va_arr)
    I0 = an.shockley_I0(mat, dev)

    print(f"\nShockley I0 (long-base ideal diode) = {I0:.4e} A")
    if is_degenerate:
        I0_fd = an.shockley_I0_fd(mat, dev)
        I_an_fd = I0_fd * (np.exp(Va_arr / mat.Vt) - 1.0)
        print(f"  Shockley I0 (Fermi-Dirac-corrected minority reference) = {I0_fd:.4e} A "
              f"(ratio to Boltzmann: {I0_fd/I0:.4f})")
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
    fig.savefig(outp("01_equilibrium_potential.png"), dpi=150)
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
    fig.savefig(outp("02_carrier_profiles.png"), dpi=150)
    plt.close(fig)

    # 3. I-V curve: linear (forward only) and semilog |I| (full range)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    fmask = Va_arr >= 0
    ax.plot(Va_arr[fmask], I_num[fmask] * 1e3, color=C_NUM, lw=2, marker="o", ms=3,
            label="Numeric drift-diffusion")
    ax.plot(Va_arr[fmask], I_an[fmask] * 1e3, color=C_AN, lw=1.8, ls="--",
            label="Shockley ideal diode (Boltzmann)")
    if is_degenerate:
        ax.plot(Va_arr[fmask], I_an_fd[fmask] * 1e3, color=C_AN_FD, lw=1.8, ls=":",
                label="Shockley ideal diode (Fermi-Dirac ref.)")
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Forward I-V (linear)")
    ax.legend()
    ax.grid(alpha=0.3, color=C_GRID)

    ax = axes[1]
    ax.semilogy(Va_arr, np.abs(I_num), color=C_NUM, lw=2, marker="o", ms=3,
                label="Numeric drift-diffusion")
    ax.semilogy(Va_arr, np.abs(I_an), color=C_AN, lw=1.8, ls="--",
                label="Shockley ideal diode (Boltzmann)")
    if is_degenerate:
        ax.semilogy(Va_arr, np.abs(I_an_fd), color=C_AN_FD, lw=1.8, ls=":",
                    label="Shockley ideal diode (Fermi-Dirac ref.)")
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("|Current| (A)")
    ax.set_title("I-V (semilog magnitude, incl. reverse bias)")
    ax.legend()
    ax.grid(alpha=0.3, color=C_GRID, which="both")
    fig.tight_layout()
    fig.savefig(outp("03_iv_curve.png"), dpi=150)
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
    fig.savefig(outp("04_ideality_factor.png"), dpi=150)
    plt.close(fig)

    # 4b. Diode C-V: quasi-static charge-based dQ/dVa, both mechanisms in
    # one curve (depletion capacitance in reverse/near-zero bias, diffusion
    # capacitance in forward bias, once minority-carrier storage current
    # dominates). Unlike the MOS-cap (a true equilibrium capacitor, no
    # current), the diode carries real current at every swept bias, so
    # there's no literal small-signal AC solve here - Q(Va) is the total
    # charge per unit area (net space charge, INCLUDING the majority/
    # minority carrier redistribution the steady-state solve already
    # computed) integrated over the p-side only. By global charge
    # neutrality of the whole device (Poisson's own boundary conditions
    # enforce it), the n-side integral gives the same magnitude with the
    # opposite sign, so either side's dQ/dVa is the same terminal
    # capacitance - the p-side is used here by convention, mirroring how
    # mos_solver.semiconductor_charge always integrates/evaluates on one
    # consistent side. This is a LOW-FREQUENCY (quasi-static) approximation:
    # a true frequency-domain small-signal solve would show the diffusion
    # capacitance rolling off above roughly 1/tau_minority-lifetime, which
    # a quasi-static DC sweep can't capture.
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2.0 renamed trapz
    p_side_mask = x <= 0.0
    Q_pside = np.array([
        _trapz((Q * (r["n"] - r["p"] - Cdop))[p_side_mask], x[p_side_mask]) for r in results
    ])
    order = np.argsort(Va_arr)
    Va_sorted, Q_sorted = Va_arr[order], Q_pside[order]
    # Sign check: capacitance must come out positive - fix the overall sign
    # of the numeric derivative to match that, rather than assume it. Prefer
    # checking in reverse bias if the sweep includes it (unambiguously
    # positive there per the depletion approximation, eps/W); fall back to
    # the mean over whatever range IS swept (e.g. a forward-bias-only sweep)
    # since diffusion capacitance is positive by the same dQ/dV convention.
    dQdV = np.gradient(Q_sorted, Va_sorted)
    reverse_mask = Va_sorted < -0.5
    sign_ref = np.mean(dQdV[reverse_mask]) if np.any(reverse_mask) else np.mean(dQdV)
    C_num_sorted = dQdV if sign_ref > 0 else -dQdV
    C_num = np.empty_like(C_num_sorted)
    C_num[order] = C_num_sorted

    C_an = an.cv_curve_analytic(mat, dev, Va_arr, use_fd=False)
    if is_degenerate:
        C_an_fd = an.cv_curve_analytic(mat, dev, Va_arr, use_fd=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(Va_arr, np.abs(C_num), color=C_NUM, lw=2, marker="o", ms=3,
                label="Numeric (quasi-static dQ/dVa)")
    ax.semilogy(Va_arr, C_an, color=C_AN, lw=1.8, ls="--",
                label="Analytic: C_depletion + C_diffusion (Boltzmann)")
    if is_degenerate:
        ax.semilogy(Va_arr, C_an_fd, color=C_AN_FD, lw=1.8, ls=":",
                    label="Analytic: C_depletion + C_diffusion (Fermi-Dirac ref.)")
    ax.axvline(0, color=C_GRID, lw=1, zorder=0)
    ax.set_xlabel("Applied voltage Va (V)")
    ax.set_ylabel("Capacitance per unit area (F/cm^2)")
    ax.set_title("Diode C-V: depletion capacitance (reverse/near-zero bias)\n"
                 "-> diffusion capacitance (forward bias, minority-carrier storage)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, color=C_GRID, which="both")
    fig.tight_layout()
    fig.savefig(outp("10_cv_curve.png"), dpi=150)
    plt.close(fig)

    cv_csv = outp("cv_sweep.csv")
    with open(cv_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Va_V", "Q_pside_C_cm2", "C_numeric_F_cm2", "C_analytic_boltzmann_F_cm2"]
        if is_degenerate:
            header.append("C_analytic_fd_F_cm2")
        w.writerow(header)
        for i, Va in enumerate(Va_arr):
            row = [f"{Va:.4f}", f"{Q_pside[i]:.6e}", f"{C_num[i]:.6e}", f"{C_an[i]:.6e}"]
            if is_degenerate:
                row.append(f"{C_an_fd[i]:.6e}")
            w.writerow(row)
    print(f"\nSaved diode C-V data to {cv_csv}")

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
    fig.savefig(outp("05_quasi_fermi_potentials.png"), dpi=150)
    plt.close(fig)

    fields_csv = outp("fields_by_bias.csv")
    fsave.save_fields(results, save_idx, "Va", fields_csv, x)
    print(f"\nSaved full field profiles for Va = {[round(results[i]['Va'], 4) for i in save_idx]} "
          f"to {fields_csv}")

    # ================= Structure + textbook band/charge diagrams =================
    struct_bias_points = [{
        "label": "equilibrium (built-in)", "bias": 0.0,
        "fields": {"psi": psi_eq, "n": n_eq, "p": p_eq,
                   "phin": np.zeros_like(x), "phip": np.zeros_like(x)},
    }]
    for i in save_idx:
        r = results[i]
        struct_bias_points.append({
            "label": f"Va={r['Va']:.2f} V", "bias": r["Va"],
            "fields": {"psi": r["psi"], "n": r["n"], "p": r["p"], "phin": r["phin"], "phip": r["phip"]},
        })

    struct_doc = sio.build_structure(
        device="diode",
        material={"eps_r": mat.eps_r, "ni_cm3": mat.ni, "T": mat.T,
                  "chi_eV": mat.chi_eV, "Eg_eV": mat.Eg_eV},
        regions=[
            {"name": f"p-side (Na~{dev.Na:.1e})", "x_range_um": [-g["Wp"] * 1e4, 0.0],
             "kind": "semiconductor", "doping_type": "p"},
            {"name": f"n-side (Nd~{dev.Nd:.1e})", "x_range_um": [0.0, g["Wn"] * 1e4],
             "kind": "semiconductor", "doping_type": "n"},
        ],
        x_um=x_um, doping_cm3=Cdop, bias_points=struct_bias_points,
    )
    if structure_file:
        sio.write_structure(os.path.join(OUT, structure_file), struct_doc)

    band_charge_xlim = (-zoom_half_width_um, zoom_half_width_um)
    fig = tplot.plot_structure(struct_doc).figure
    fig.tight_layout()
    fig.savefig(outp("07_structure.png"), dpi=150)
    plt.close(fig)

    fig = tplot.plot_bands(struct_doc, bias_index=0, xlim_um=band_charge_xlim).figure
    fig.tight_layout()
    fig.savefig(outp("08_band_diagram.png"), dpi=150)
    plt.close(fig)

    fig = tplot.plot_charge(struct_doc, bias_index=0, xlim_um=band_charge_xlim).figure
    fig.tight_layout()
    fig.savefig(outp("09_charge_density.png"), dpi=150)
    plt.close(fig)

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
    fig.savefig(outp("06_solver_benchmark.png"), dpi=150)
    plt.close(fig)

    bench_csv = outp("solver_benchmark.csv")
    with open(bench_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "gummel_time_s", "gummel_iters", "newton_time_s", "newton_iters"])
        for i, Va in enumerate(Va_arr):
            w.writerow([f"{Va:.4f}",
                        f"{bench['gummel']['times'][i]:.5f}", bench["gummel"]["iters"][i],
                        f"{bench['newton']['times'][i]:.5f}", bench["newton"]["iters"][i]])

    # ---- CSV export ----
    csv_path = outp("iv_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "I_numeric_A", "I_shockley_A", "self_consistency_pct"])
        for Va, In, Ia, res in zip(Va_arr, I_num, I_an, results):
            w.writerow([f"{Va:.4f}", f"{In:.6e}", f"{Ia:.6e}",
                        f"{res['J_std']/max(abs(res['J_mean']),1e-30)*100:.4f}"])

    print(f"\nPlots and data written to: {OUT}")
    for name in ["01_equilibrium_potential.png", "02_carrier_profiles.png", "03_iv_curve.png",
                 "04_ideality_factor.png", "05_quasi_fermi_potentials.png", "06_solver_benchmark.png",
                 "07_structure.png", "08_band_diagram.png", "09_charge_density.png", "10_cv_curve.png",
                 "fields_by_bias.csv", "iv_sweep.csv", "cv_sweep.csv", "solver_benchmark.csv"]:
        print(f"  {prefix}{name}")
    if structure_file:
        print(f"  {structure_file}  (standalone: python3 plot.py out/{structure_file})")


if __name__ == "__main__":
    main()
