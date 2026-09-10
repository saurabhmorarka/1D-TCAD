"""Driver script: makes the impact of a fixed oxide/substrate interface
charge (Qit_cm2, see core/interfaces.py and input_mos_qit.yaml) visually
obvious by overlaying several Qit levels (including Qit=0, i.e. no
interface charge at all) on the same axes - the C-V curve, the potential
profile near the interface, and the charge-density profile near the
interface - plus a numeric flat-band-voltage shift for each level.

A single "with vs without" plot showing the Qit=0 baseline against one
nonzero value (input_mos_qit.yaml's own out/mos_qit/*.png) already exists
from mos_main.py's normal per-example output, but the interface-charge
marker/arrow added there doesn't by itself make clear how much the REST
of the device's electrostatics moved as a result - that requires actually
comparing curves, which this script does directly, across a small sweep
of Qit rather than just one on/off pair (so the trend, not just the
end-points, is visible too).

Run with: python3 -m mos.mos_qit_compare [input_mos.yaml]
"""
import csv
import os
import sys

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mos import mos_config as cfg
from core.mesh import build_mos_grid
from core.interfaces import Interface
from core import physics as ph
from mos.mos_solver import cv_sweep, solve_mos_equilibrium
from mos import mos_analytic as man
from core.params import Q

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "mos_qit_compare")
os.makedirs(OUT, exist_ok=True)

# 0 (no interface charge - the "without" case) plus a few realistic fixed
# interface-charge densities (in units of elementary charges/cm^2, a
# standard way Qit/Dit numbers are quoted in the literature).
QIT_LEVELS_PER_CM2 = [0.0, 5.0e10, 1.0e11, 3.0e11, 1.0e12]
VG_SNAPSHOT = 0.3  # bias point for the potential/charge-profile comparison


def _run(mat, dev, Cdop_substrate, VG_list, mesh_opts, Qit_cm2):
    import copy
    dev_q = copy.copy(dev)
    dev_q.Qit_cm2 = Qit_cm2
    g = build_mos_grid(mat, dev_q, Cdop_substrate, Cdop_gate=None, **mesh_opts)
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]

    results = cv_sweep(x, Cdop, eps_edge, ni_arr, mat, dev_q, Cdop_substrate, VG_list,
                        g["oxide_index"], interfaces=g["interfaces"])
    VG_arr = np.array([r["VG"] for r in results])
    Cox = man.C_ox(dev_q)
    C_lf = np.array([r["C_lf"] for r in results]) / Cox

    psi_bulk = ph.equilibrium_bulk_potential(mat, Cdop_substrate)

    def psi_s_of_VG(VG):
        r = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev_q, Cdop_substrate,
                                   VG, psi_bulk, interfaces=g["interfaces"])
        return r["psi"][g["oxide_index"]] - psi_bulk

    V_FB = brentq(psi_s_of_VG, VG_arr[0], VG_arr[-1], xtol=1e-8)

    snap = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev_q, Cdop_substrate,
                                  VG_SNAPSHOT, psi_bulk, interfaces=g["interfaces"])
    n, p = snap["n"], snap["p"]
    net_charge = Cdop + (p - n)  # elementary charges / cm^3

    return {
        "VG_arr": VG_arr, "C_lf": C_lf, "V_FB": V_FB,
        "x_um": x * 1e4, "psi": snap["psi"], "net_charge": net_charge,
        "oxide_index": g["oxide_index"],
    }


def main():
    default_input = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "configs", "input_mos.yaml")
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    input_cfg = cfg.load_config(input_path)
    mat, dev, Cdop_substrate, VG_list, save_bias_points, mesh_opts, structure_file, Cdop_gate = \
        cfg.build_from_config(input_cfg)
    if dev.gate_kind != "metal":
        raise ValueError(f"{input_path}: gate.type must be 'metal' for mos_qit_compare.py")

    print(f"Loaded {os.path.basename(input_path)} - comparing Qit = "
          f"{[f'{v:.1e} cm^-2' for v in QIT_LEVELS_PER_CM2]} at VG snapshot={VG_SNAPSHOT} V")

    colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(QIT_LEVELS_PER_CM2)))
    results = []
    rows = []
    print(f"\n{'Nit (cm^-2)':>14} {'Qit (C/cm^2)':>14} {'V_FB (V)':>12} {'dV_FB (mV)':>12}")
    for Nit in QIT_LEVELS_PER_CM2:
        Qit_cm2 = Nit * Q
        r = _run(mat, dev, Cdop_substrate, VG_list, mesh_opts, Qit_cm2)
        results.append(r)
        if Nit == 0.0:
            V_FB0 = r["V_FB"]
        rows.append([Nit, Qit_cm2, r["V_FB"]])

    for row in rows:
        Nit, Qit_cm2, V_FB = row
        print(f"{Nit:14.2e} {Qit_cm2:14.4e} {V_FB:12.5f} {(V_FB - V_FB0) * 1e3:12.3f}")

    # ---- Figure 1: C-V curve, with vs. across several Qit levels ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for Nit, color, r in zip(QIT_LEVELS_PER_CM2, colors, results):
        label = "Qit = 0 (no interface charge)" if Nit == 0.0 else f"N_it = {Nit:.1e} cm$^{{-2}}$"
        lw = 2.6 if Nit == 0.0 else 1.8
        ls = "-" if Nit == 0.0 else "--"
        ax.plot(r["VG_arr"], r["C_lf"], color=("#333333" if Nit == 0.0 else color), lw=lw, ls=ls, label=label)
        ax.axvline(r["V_FB"], color=("#333333" if Nit == 0.0 else color), lw=1.0, ls=":", alpha=0.6)
    ax.set_xlabel("Gate voltage V_G (V)")
    ax.set_ylabel("C_lf / C_ox")
    ax.set_title("Low-frequency C-V: impact of fixed interface charge\n"
                 "(dotted vertical lines mark each curve's own flat-band voltage)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qit_cv_comparison.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: potential profile near the interface, same VG for all ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for Nit, color, r in zip(QIT_LEVELS_PER_CM2, colors, results):
        label = "Qit = 0" if Nit == 0.0 else f"N_it = {Nit:.1e} cm$^{{-2}}$"
        lw = 2.6 if Nit == 0.0 else 1.8
        ls = "-" if Nit == 0.0 else "--"
        ax.plot(r["x_um"], r["psi"], color=("#333333" if Nit == 0.0 else color), lw=lw, ls=ls, label=label)
    ax.axvline(0.0, color="#999999", lw=1.0, ls=":")
    ax.set_xlim(-dev.t_ox * 1e4 * 1.5, 0.3)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Potential psi (V)")
    ax.set_title(f"Band bending near the interface at V_G={VG_SNAPSHOT} V\n"
                 f"(same applied V_G for every curve - the shift IS the Qit effect)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qit_potential_comparison.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 3: charge density near the interface, same VG for all ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(0, color="#333333", lw=0.8)
    for Nit, color, r in zip(QIT_LEVELS_PER_CM2, colors, results):
        label = "Qit = 0" if Nit == 0.0 else f"N_it = {Nit:.1e} cm$^{{-2}}$"
        lw = 2.6 if Nit == 0.0 else 1.8
        ls = "-" if Nit == 0.0 else "--"
        ax.plot(r["x_um"], r["net_charge"], color=("#333333" if Nit == 0.0 else color), lw=lw, ls=ls, label=label)
    ax.axvline(0.0, color="#999999", lw=1.0, ls=":")
    ax.set_xlim(-dev.t_ox * 1e4 * 1.5, 0.3)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Net (mobile + doping) charge density / q (cm^-3)")
    ax.set_title(f"Semiconductor charge response near the interface at V_G={VG_SNAPSHOT} V\n"
                 f"(the fixed Qit sheet charge itself sits exactly at x=0, not shown on this cm^-3 axis)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qit_charge_comparison.png"), dpi=150)
    plt.close(fig)

    csv_path = os.path.join(OUT, "qit_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Nit_cm2", "Qit_C_cm2", "V_FB_V", "dV_FB_mV"])
        for Nit, Qit_cm2, V_FB in rows:
            w.writerow([f"{Nit:.4e}", f"{Qit_cm2:.6e}", f"{V_FB:.6f}", f"{(V_FB - V_FB0) * 1e3:.4f}"])

    print(f"\nSaved: {os.path.join(OUT, 'qit_cv_comparison.png')}")
    print(f"Saved: {os.path.join(OUT, 'qit_potential_comparison.png')}")
    print(f"Saved: {os.path.join(OUT, 'qit_charge_comparison.png')}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
