"""Driver script: metal-gate MOS-cap C-V across several substrate doping
levels, and separately across several oxide thicknesses.

Companion to mos_main.py's single-point metal-gate run (input_mos.yaml) and
to mos_poly_sweep.py's poly-GATE-doping sweep: this one answers "how does
the C-V curve's accumulation-to-inversion transition change with SUBSTRATE
doping" and "...with OXIDE thickness" by overlaying several values of each
on their own plot, holding every other parameter fixed at whatever
input_mos.yaml specifies (same pattern as mos_poly_sweep.py - the sweep
values below are hardcoded here, not read from the YAML).

Motivated by a real question: does raising substrate doping narrow the
wide, shallow dip between accumulation and inversion? It doesn't - V_T
rises with doping, which widens the gap in V_G even as the dip itself gets
shallower (thinner max depletion width). Oxide thickness is the stronger
lever: a thicker oxide (smaller C_ox) makes the depletion capacitance a
much larger fraction of the total series capacitance, deepening and
widening the dip substantially more than doping does over a comparable
range.

Run with: python3 mos_metal_sweep.py [input_mos.yaml]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mos import mos_config as cfg
from core.mesh import build_mos_grid
from mos.mos_solver import cv_sweep
from mos import mos_analytic as man

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "mos")
os.makedirs(OUT, exist_ok=True)

DOPINGS_CM3 = [1e16, 3.16e16, 1e17, 3.16e17, 1e18, 5e18]
OXIDES_NM = [0.5, 1.0, 2.0, 5.0, 10.0]
OXIDE_SWEEP_DOPING_CM3 = 1e17  # substrate doping held fixed for the oxide sweep


def run_one(mat, dev, Cdop_substrate, VG_list, mesh_opts):
    g = build_mos_grid(mat, dev, Cdop_substrate, Cdop_gate=None, **mesh_opts)
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]

    results = cv_sweep(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate, VG_list, g["oxide_index"],
                        interfaces=g["interfaces"])
    VG_arr = np.array([r["VG"] for r in results])
    Cox = man.C_ox(dev)
    C_lf = np.array([r["C_lf"] for r in results]) / Cox
    C_hf = np.array([r["C_hf"] for r in results]) / Cox
    V_FB = man.flatband_voltage(dev, mat, Cdop_substrate)
    V_T = man.threshold_voltage(mat, dev, Cdop_substrate)
    return VG_arr, C_lf, C_hf, V_FB, V_T


def _plot_and_save(sweep_label, sweep_values, fmt_label, results, fixed_desc, png_name, csv_name, csv_header):
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(sweep_values)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    rows = []
    print(f"\n{sweep_label:>16} {'V_FB (V)':>10} {'V_T (V)':>10} "
          f"{'C_lf/Cox @ VGmin':>17} {'C_lf/Cox @ VGmax':>17}")
    for val, color, (VG_arr, C_lf, C_hf, V_FB, V_T) in zip(sweep_values, colors, results):
        print(f"{fmt_label(val):>16} {V_FB:10.4f} {V_T:10.4f} {C_lf[0]:17.4f} {C_lf[-1]:17.4f}")
        rows.append([val, V_FB, V_T, C_lf[0], C_lf[-1]])
        label = fmt_label(val)
        axes[0].plot(VG_arr, C_lf, color=color, lw=2, label=label)
        axes[1].plot(VG_arr, C_hf, color=color, lw=2, label=label)

    for ax, title in zip(axes, ["Low-frequency C-V", "High-frequency C-V"]):
        ax.set_xlabel("Gate voltage V_G (V)")
        ax.set_ylabel("C / C_ox")
        ax.set_title(f"{title}\n({fixed_desc})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, png_name), dpi=150)
    plt.close(fig)

    csv_path = os.path.join(OUT, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)
        for row in rows:
            w.writerow([f"{row[0]:.4e}"] + [f"{v:.6f}" for v in row[1:]])

    print(f"Saved: {os.path.join(OUT, png_name)}")
    print(f"Saved: {csv_path}")


DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "configs", "input_mos.yaml")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    input_cfg = cfg.load_config(input_path)
    mat, dev, Cdop_substrate_base, VG_list, save_bias_points, mesh_opts, structure_file, Cdop_gate_cfg = \
        cfg.build_from_config(input_cfg)
    if dev.gate_kind != "metal":
        raise ValueError(f"{input_path}: gate.type must be 'metal' for mos_metal_sweep.py "
                          f"(got {dev.gate_kind!r}) - use mos_poly_sweep.py for a poly-gate sweep instead")
    sub_sign = 1.0 if Cdop_substrate_base >= 0 else -1.0
    t_ox_base_nm = dev.t_ox * 1e7

    print(f"Loaded {os.path.basename(input_path)} ({'p' if sub_sign < 0 else 'n'}-sub base, "
          f"t_ox={t_ox_base_nm:.2f} nm) - sweeping substrate doping over {[f'{d:.0e}' for d in DOPINGS_CM3]} "
          f"and oxide thickness over {OXIDES_NM} nm")

    # --- Sweep 1: substrate doping, oxide fixed at the YAML's value ---
    doping_results = []
    for Nsub in DOPINGS_CM3:
        Cdop_substrate = sub_sign * Nsub
        doping_results.append(run_one(mat, dev, Cdop_substrate, VG_list, mesh_opts))
    _plot_and_save(
        "Nsub (cm^-3)", DOPINGS_CM3, lambda v: f"N_sub={v:.1e} cm^-3", doping_results,
        f"{'p' if sub_sign < 0 else 'n'}-sub doping sweep, t_ox={t_ox_base_nm:.1f} nm",
        "psub_doping_sweep_comparison.png", "psub_doping_sweep_summary.csv",
        ["Nsub_cm3", "V_FB_V", "V_T_V", "C_lf_over_Cox_at_VGmin", "C_lf_over_Cox_at_VGmax"])

    # --- Sweep 2: oxide thickness, substrate doping fixed ---
    import copy
    Cdop_substrate_ox = sub_sign * OXIDE_SWEEP_DOPING_CM3
    oxide_results = []
    for t_nm in OXIDES_NM:
        dev_t = copy.copy(dev)
        dev_t.t_ox = t_nm * 1e-7  # nm -> cm
        oxide_results.append(run_one(mat, dev_t, Cdop_substrate_ox, VG_list, mesh_opts))
    _plot_and_save(
        "t_ox (nm)", OXIDES_NM, lambda v: f"t_ox={v:g} nm", oxide_results,
        f"{'p' if sub_sign < 0 else 'n'}-sub {OXIDE_SWEEP_DOPING_CM3:.0e} cm^-3, oxide thickness sweep",
        "gate_tox_sweep_comparison.png", "gate_tox_sweep_summary.csv",
        ["t_ox_nm", "V_FB_V", "V_T_V", "C_lf_over_Cox_at_VGmin", "C_lf_over_Cox_at_VGmax"])


if __name__ == "__main__":
    main()
