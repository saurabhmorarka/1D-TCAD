"""Driver script: poly-gate MOS-cap C-V across several gate doping levels.

Companion to mos_main.py's single-doping poly-gate run (input_mos_poly.yaml):
that example answers "what does the C-V look like for ONE poly doping",
this one answers "how does the poly doping level change the C-V" by
overlaying several dopings (see DOPINGS below) on one plot, holding every
other parameter (substrate, oxide, mesh, voltage sweep) fixed at whatever
input_mos_poly.yaml specifies. Only the gate doping concentration is
overridden per run; its polarity (n/p) and everything else comes from the
YAML as usual.

Run with: python3 mos_poly_sweep.py [input_mos_poly.yaml]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mos_config as cfg
from mesh import build_mos_grid
from mos_solver import cv_sweep
import mos_analytic as man
import physics as ph
from doping_profiles import DopingProfile

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

DOPINGS = [1e17, 1e18, 1e19, 1e20, 1e21, 1e22]


def run_one(mat, dev, Cdop_substrate, VG_list, mesh_opts, gate_polarity_sign, Ngate):
    """One doping point: returns (VG_arr, C_lf/Cox, C_hf/Cox, V_FB, V_T, poly_bending_at_last_VG)."""
    dev.gate_profile = DopingProfile.flat(Ngate)
    Cdop_gate = gate_polarity_sign * Ngate

    g = build_mos_grid(mat, dev, Cdop_substrate, Cdop_gate=Cdop_gate, **mesh_opts)
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]

    results = cv_sweep(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate, VG_list,
                        g["oxide_index"], Cdop_gate=Cdop_gate, gate_oxide_index=g["gate_oxide_index"])

    VG_arr = np.array([r["VG"] for r in results])
    Cox = man.C_ox(dev)
    C_lf = np.array([r["C_lf"] for r in results]) / Cox
    C_hf = np.array([r["C_hf"] for r in results]) / Cox

    psi_bulk_gate = ph.equilibrium_bulk_potential(mat, Cdop_gate)
    poly_bending_last = results[-1]["psi"][g["gate_oxide_index"]] - psi_bulk_gate

    # "Ideal, non-depleting poly" reference work function, for V_FB/V_T only
    # - see mos_main.py's dev_ideal_gate for the same reasoning.
    import copy
    dev_ideal = copy.copy(dev)
    dev_ideal.gate_workfunction_eV = man.semiconductor_workfunction_eV(mat, Cdop_gate)
    V_FB = man.flatband_voltage(dev_ideal, mat, Cdop_substrate)
    V_T = man.threshold_voltage(mat, dev_ideal, Cdop_substrate)

    return VG_arr, C_lf, C_hf, V_FB, V_T, poly_bending_last


DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "input_mos_poly.yaml")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    input_cfg = cfg.load_config(input_path)
    mat, dev, Cdop_substrate, VG_list, save_bias_points, mesh_opts, structure_file, Cdop_gate_cfg = \
        cfg.build_from_config(input_cfg)
    if dev.gate_kind != "poly":
        raise ValueError(f"{input_path}: gate.type must be 'poly' for mos_poly_sweep.py "
                          f"(got {dev.gate_kind!r}) - use mos_main.py for a metal-gate sweep instead")
    gate_polarity_sign = 1.0 if Cdop_gate_cfg >= 0 else -1.0

    print(f"Loaded {os.path.basename(input_path)} (substrate={'p' if Cdop_substrate < 0 else 'n'}-type, "
          f"|Nsub|={abs(Cdop_substrate):.2e} cm^-3, t_ox={dev.t_ox*1e7:.2f} nm, "
          f"gate polarity={'p' if gate_polarity_sign < 0 else 'n'}) - sweeping gate doping over "
          f"{[f'{d:.0e}' for d in DOPINGS]}")

    Cox = man.C_ox(dev)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DOPINGS)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    rows = []
    print(f"\n{'Ngate (cm^-3)':>14} {'V_FB (V)':>10} {'V_T (V)':>10} "
          f"{'C_lf/Cox @ VG_min':>18} {'C_lf/Cox @ VG_max':>18} {'poly bending @ VG_max (V)':>27}")
    for Ngate, color in zip(DOPINGS, colors):
        VG_arr, C_lf, C_hf, V_FB, V_T, poly_bending_last = run_one(
            mat, dev, Cdop_substrate, VG_list, mesh_opts, gate_polarity_sign, Ngate)
        print(f"{Ngate:14.0e} {V_FB:10.4f} {V_T:10.4f} "
              f"{C_lf[0]:18.4f} {C_lf[-1]:18.4f} {poly_bending_last:27.4f}")
        rows.append([Ngate, V_FB, V_T, C_lf[0], C_lf[-1], poly_bending_last])

        label = f"N_gate={Ngate:.0e} cm^-3"
        axes[0].plot(VG_arr, C_lf, color=color, lw=2, label=label)
        axes[1].plot(VG_arr, C_hf, color=color, lw=2, label=label)

    for ax, title in zip(axes, ["Low-frequency C-V", "High-frequency C-V"]):
        ax.set_xlabel("Gate voltage V_G (V)")
        ax.set_ylabel("C / C_ox")
        ax.set_title(f"{title}, poly gate doping sweep\n"
                     f"({'p' if Cdop_substrate < 0 else 'n'}-sub {abs(Cdop_substrate):.0e} cm^-3, "
                     f"t_ox={dev.t_ox*1e7:.1f} nm, {'p' if gate_polarity_sign < 0 else 'n'}+ poly)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "poly_doping_comparison.png"), dpi=150)
    plt.close(fig)

    csv_path = os.path.join(OUT, "poly_doping_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ngate_cm3", "V_FB_V", "V_T_V", "C_lf_over_Cox_at_VGmin", "C_lf_over_Cox_at_VGmax",
                    "poly_bending_at_VGmax_V"])
        for row in rows:
            w.writerow([f"{row[0]:.4e}"] + [f"{v:.6f}" for v in row[1:]])

    print(f"\nSaved: {os.path.join(OUT, 'poly_doping_comparison.png')}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
