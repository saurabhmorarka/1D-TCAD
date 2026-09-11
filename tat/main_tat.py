"""Driver script: 1D TCAD drain-to-substrate junction leakage simulation
(trap-assisted + band-to-band tunneling).

Special, opt-in driver (see newton_solver_tat.py's module docstring for why
this is a separate module from main.py rather than a flag in it). Builds the
mesh, runs the TAT+BTBT bias sweep (newton_tat) alongside a plain newton_qf
sweep on the SAME device (no tunneling leakage) for comparison, and plots
I(Va) on both linear and log scale so the leakage knee is visible relative
to the no-tunneling baseline - see input_diode_drain_substrate.yaml and
plans/tat_btbt_plan.md for the full rationale.

Doping/mesh/voltage-sweep parameters are read from
input_diode_drain_substrate.yaml (see config.py/tat_config.py) - edit that
file rather than this one.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.mesh import build_diode_grid
from core.solver import solve_equilibrium, voltage_sweep
from core import analytic as an
from core import config as cfg
from tat import tat_config as tcfg
from tat.tat import btbt_generation, hurkx_tat_generation
from tat.newton_solver_tat import _node_field
from core import structure_io as sio
from core import plot as tplot

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "tat")
os.makedirs(OUT, exist_ok=True)

C_TAT = "#1f6feb"     # numeric, with TAT+BTBT leakage - blue
C_NOTAT = "#6c757d"   # numeric, no tunneling - grey
C_TATTERM = "#e8590c"  # G_tat profile - orange
C_BTBTTERM = "#2f9e44"  # G_btbt profile - green


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "input_diode_drain_substrate.yaml")

    input_cfg = cfg.load_config(input_path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        cfg.build_from_config(input_cfg)
    t = tcfg.parse_tat_config(input_cfg)
    if math_model != "newton_tat":
        print(f"Note: input file's solver.math_model={math_model!r} ignored - "
              "main_tat.py always runs the TAT+BTBT solver.")

    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]
    print(f"Grid: {len(x)} points, Wp={g['Wp']*1e4:.2f} um, Wn={g['Wn']*1e4:.2f} um, "
          f"h_min={g['h_min']*1e7:.2f} nm")
    print(f"Doping: Na={dev.Na:.2e} cm^-3 (substrate), Nd={dev.Nd:.2e} cm^-3 (drain)")

    psi_eq, n_eq, p_eq, eq_iters = solve_equilibrium(x, Cdop, mat)
    Vbi = an.built_in_potential(mat, dev)
    print(f"Equilibrium: Newton iters={eq_iters}, Vbi={Vbi:.4f} V")

    print(f"\nRunning TAT+BTBT bias sweep (newton_tat, {len(Va_list)} points)...")
    _, _, _, res_tat = voltage_sweep(x, Cdop, mat, dev, Va_list, verbose=False,
                                      method="newton_tat")

    print("Running comparison bias sweep (newton_qf, no tunneling leakage)...")
    _, _, _, res_notat = voltage_sweep(x, Cdop, mat, dev, Va_list, verbose=False,
                                        method="newton_qf")

    Va_arr = np.array([r["Va"] for r in res_tat])
    I_tat = np.array([r["I"] for r in res_tat])
    I_notat = np.array([r["I"] for r in res_notat])
    Jres_tat = np.array([r["J_std"] / max(abs(r["J_mean"]), 1e-30) for r in res_tat])

    with np.errstate(divide="ignore", invalid="ignore"):
        enhancement = np.where(np.abs(I_notat) > 0, I_tat / I_notat, np.nan)

    print("\nSelf-consistency (J_std/J_mean, TAT sweep) and leakage enhancement at a few points:")
    for i in np.linspace(0, len(Va_arr) - 1, min(8, len(Va_arr))).astype(int):
        print(f"  Va={Va_arr[i]:+7.2f} V  I_tat={I_tat[i]:.4e} A  "
              f"I_no_tat={I_notat[i]:.4e} A  enhancement={enhancement[i]:.3g}x  "
              f"selfconsist={Jres_tat[i]:.2e}")

    csv_path = os.path.join(OUT, "tat_iv.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "I_tat_A", "I_no_tat_A", "enhancement", "J_std_over_mean"])
        for i in range(len(Va_arr)):
            w.writerow([Va_arr[i], I_tat[i], I_notat[i], enhancement[i], Jres_tat[i]])
    print(f"\nWrote {csv_path}")

    # ---- Validation plot: I(Va) leakage knee, linear + log ----
    rev_mask = Va_arr < 0
    Va_rev = Va_arr[rev_mask]

    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(7, 8))

    ax_lin.plot(Va_rev, I_tat[rev_mask], "-o", ms=3, color=C_TAT, label="numeric I (TAT+BTBT)")
    ax_lin.plot(Va_rev, I_notat[rev_mask], "--", color=C_NOTAT, label="numeric I (no tunneling)")
    ax_lin.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_lin.set_ylabel("I (A), linear")
    ax_lin.set_title("Drain-substrate leakage: I(Va) - linear scale")
    ax_lin.legend(fontsize=8)
    ax_lin.grid(color="#dddddd")

    ax_log.semilogy(Va_rev, np.abs(I_tat[rev_mask]), "-o", ms=3, color=C_TAT,
                     label="numeric I (TAT+BTBT)")
    ax_log.semilogy(Va_rev, np.abs(I_notat[rev_mask]), "--", color=C_NOTAT,
                     label="numeric I (no tunneling)")
    ax_log.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_log.set_ylabel("|I| (A), log")
    ax_log.set_title("Drain-substrate leakage: I(Va) - log scale (softer, earlier-onset knee)")
    ax_log.legend(fontsize=8)
    ax_log.grid(color="#dddddd")

    fig.tight_layout()
    plot_path = os.path.join(OUT, "tat_iv.png")
    fig.savefig(plot_path, dpi=130)
    print(f"Wrote {plot_path}")

    # ---- Structure+fields doc (equilibrium + a few sweep bias points,
    # same convention as main.py) - built once, used both for the saved
    # structure_file AND the band-diagram figure below. ----
    struct_bias_points = [{
        "label": "equilibrium (Va~0)", "bias": 0.0,
        "fields": {"psi": psi_eq, "n": n_eq, "p": p_eq,
                   "phin": np.zeros_like(x), "phip": np.zeros_like(x)},
    }]
    for target in save_bias_points:
        i = int(np.argmin(np.abs(Va_arr - target)))
        r = res_tat[i]
        struct_bias_points.append({
            "label": f"Va={r['Va']:+.2f}V", "bias": r["Va"],
            "fields": {"psi": r["psi"], "n": r["n"], "p": r["p"],
                       "phin": r["phin"], "phip": r["phip"]},
        })

    struct_doc = sio.build_structure(
        device="diode",
        material={"eps_r": mat.eps_r, "ni_cm3": mat.ni, "chi_eV": mat.chi_eV, "Eg_eV": mat.Eg_eV},
        regions=[
            {"name": "substrate (p)", "x_range_um": [x[0] * 1e4, 0.0], "kind": "semiconductor", "doping_type": "p"},
            {"name": "drain (n)", "x_range_um": [0.0, x[-1] * 1e4], "kind": "semiconductor", "doping_type": "n"},
        ],
        x_um=x * 1e4, doping_cm3=Cdop, bias_points=struct_bias_points,
    )
    sio.write_structure(os.path.join(OUT, structure_file), struct_doc)
    print(f"Wrote {os.path.join(OUT, structure_file)}")

    # ---- Band-diagram plot: WHY tunneling initiates. Ec/Ev/Ei/Evac/Ef at
    # equilibrium vs. at the deepest reverse-bias point, zoomed to the
    # depletion region and sharing the same x-axis as the generation-term
    # breakdown below them - so the reader can directly see the band
    # bending that turns the tunneling terms on, not just the resulting
    # I(Va) curve.
    #
    # NOTE on what to actually read off these two panels: this substrate
    # is only 1e17 cm^-3 (non-degenerate - its Ef sits well inside the
    # gap, not inside a band), so this is NOT the classic Esaki/tunnel-
    # diode picture where reverse bias brings the n-side Ec into literal
    # spatial/energy overlap with the p-side Ev. What actually changes
    # between the two panels is the STEEPNESS of the band bending across
    # the transition region - the local electric field, i.e. the SLOPE of
    # Ec/Ev vs. x - which grows sharply under reverse bias as the
    # depletion width narrows relative to the growing voltage it must
    # drop. That slope is exactly the quantity both tat.py's Kane
    # alpha(F)=A*F^P*exp(-B/F) and Hurkx Gamma(F) depend on - so the
    # visibly steeper band bending in the reverse-bias panel IS the
    # band-diagram picture of why the generation terms below turn on,
    # even though the bands themselves don't cross.
    deepest_idx = int(np.argmin(Va_arr))
    deepest = res_tat[deepest_idx]
    xp, xn, _ = an.depletion_widths(mat, dev, deepest["Va"])
    zoom_half_width_um = max(3 * max(xp, xn) * 1e4, 0.05)
    band_xlim = (-zoom_half_width_um, zoom_half_width_um)

    fig, (ax_bands_eq, ax_bands_rev, ax_gen) = plt.subplots(
        3, 1, figsize=(7, 12), sharex=True)

    tplot.plot_bands(struct_doc, bias_index=0, ax=ax_bands_eq, xlim_um=band_xlim)
    ax_bands_eq.set_title("Band diagram at equilibrium (Va~0) - gentle band bending, low field")

    deepest_bp_index = 1 + int(np.argmin(np.abs(
        np.array([bp["bias"] for bp in struct_bias_points[1:]]) - deepest["Va"])))
    tplot.plot_bands(struct_doc, bias_index=deepest_bp_index, ax=ax_bands_rev, xlim_um=band_xlim)
    ax_bands_rev.set_title(
        f"Band diagram at Va={deepest['Va']:+.2f} V - much steeper band bending\n"
        "(higher local field) across the same depletion region")

    # Per-node G_tat vs G_btbt at the SAME deepest reverse-bias point and
    # the SAME x-zoom as the band diagram directly above it - shows
    # directly (not just asserted) both that TAT dominates at this
    # device's actual operating field range (per tat/tat.py's
    # sanity_probe ordering) and exactly where, relative to the band
    # bending above, the generation actually happens.
    F_node, *_ = _node_field(deepest["psi"], x)
    n_i, p_i = deepest["n"][1:-1], deepest["p"][1:-1]
    G_tat, *_ = hurkx_tat_generation(n_i, p_i, F_node, mat, t["hurkx_model"])
    G_btbt, _ = btbt_generation(F_node, t["kane_model"])
    x_um_interior = x[1:-1] * 1e4

    ax_gen.semilogy(x_um_interior, np.clip(G_tat, 1e-10, None), color=C_TATTERM,
                     label="G_tat (Hurkx)")
    ax_gen.semilogy(x_um_interior, np.clip(G_btbt, 1e-10, None), color=C_BTBTTERM,
                     label="G_btbt (Kane)")
    ax_gen.set_xlim(*band_xlim)
    ax_gen.set_xlabel("Position x (um)")
    ax_gen.set_ylabel("Generation rate (cm^-3 s^-1), log")
    ax_gen.set_title("Generation-term breakdown, same bias and x-range as the band diagram above\n"
                      "(TAT dominates across most of the depletion width;\n"
                      "BTBT only overtakes right at the single peak-field point)")
    ax_gen.legend(fontsize=8)
    ax_gen.grid(color="#dddddd")

    fig.tight_layout()
    bands_plot_path = os.path.join(OUT, "tat_bands.png")
    fig.savefig(bands_plot_path, dpi=130)
    print(f"Wrote {bands_plot_path}")


if __name__ == "__main__":
    main()
