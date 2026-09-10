"""Driver script: makes the impact of a MATERIAL change (Silicon vs
Germanium, same temperature) and a TEMPERATURE change (Silicon at room
temperature vs Silicon at 100C, same material) visually obvious by
overlaying the two cases on the same axes - equilibrium band bending,
carrier profiles, and the I-V curve.

Each comparison changes exactly ONE thing and holds everything else
(doping, geometry, solver) fixed, so the two runs isolate that one
variable's effect cleanly:
  - Material comparison: Silicon vs Germanium, BOTH at T=300K (not the
    T=350K used by the standalone configs/input_diode_ge.yaml demo,
    since mixing material AND temperature there would conflate the two
    effects this script is specifically trying to separate).
  - Temperature comparison: Silicon at T=300K vs T=373.15K (100C), same
    material, using core.materials.resolve_material so ni/Nc/Nv/Eg are
    genuinely recomputed at each T (not frozen defaults).

Both use core.material_db's catalog (see core/materials.py,
core/material_db.py) - this script is also a working demonstration of the
material-database + temperature features together, not just Feature 1/2
in isolation.

Run with: python3 main_diode_compare.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import material_db
from core.materials import resolve_material
from core.params import Device
from core.mesh import build_diode_grid
from core.solver import voltage_sweep

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

VA_LIST = np.concatenate([
    np.linspace(-1.0, -0.05, 10),
    np.linspace(0.0, 0.35, 25),
])


def _run(mat, dev):
    g = build_diode_grid(mat, dev)
    x, Cdop = g["x"], g["Cdop"]
    psi_eq, n_eq, p_eq, results = voltage_sweep(x, Cdop, mat, dev, VA_LIST, method="newton_qf")
    Va_arr = np.array([r["Va"] for r in results])
    I_arr = np.array([r["I"] for r in results])
    return {"x_um": x * 1e4, "psi_eq": psi_eq, "n_eq": n_eq, "p_eq": p_eq,
             "Cdop": Cdop, "junction_index": g["junction_index"],
             "Va": Va_arr, "I": I_arr}


def _compare_plot(out_subdir, title_prefix, label_a, r_a, label_b, r_b,
                   color_a="#1f6feb", color_b="#c2410c", xlim_um=None):
    OUT = os.path.join(OUT_ROOT, out_subdir)
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.plot(r_a["x_um"], r_a["psi_eq"], color=color_a, lw=2.2, label=label_a)
    ax.plot(r_b["x_um"], r_b["psi_eq"], color=color_b, lw=2.2, ls="--", label=label_b)
    ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
    if xlim_um is not None:
        ax.set_xlim(*xlim_um)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Equilibrium potential psi (V)")
    ax.set_title("Equilibrium band bending")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(r_a["x_um"], np.clip(r_a["n_eq"], 1e-3, None), color=color_a, lw=2.0, label=f"{label_a}, n")
    ax.semilogy(r_a["x_um"], np.clip(r_a["p_eq"], 1e-3, None), color=color_a, lw=2.0, ls=":", label=f"{label_a}, p")
    ax.semilogy(r_b["x_um"], np.clip(r_b["n_eq"], 1e-3, None), color=color_b, lw=2.0, ls="--", label=f"{label_b}, n")
    ax.semilogy(r_b["x_um"], np.clip(r_b["p_eq"], 1e-3, None), color=color_b, lw=2.0, ls="-.", label=f"{label_b}, p")
    ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
    if xlim_um is not None:
        ax.set_xlim(*xlim_um)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("Carrier concentration (cm^-3)")
    ax.set_title("Equilibrium carrier profiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.semilogy(r_a["Va"], np.abs(r_a["I"]), color=color_a, lw=2.2, marker="o", ms=3, label=label_a)
    ax.semilogy(r_b["Va"], np.abs(r_b["I"]), color=color_b, lw=2.2, marker="s", ms=3, ls="--", label=label_b)
    ax.set_xlabel("Applied bias V_a (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_title("I-V curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(title_prefix, fontsize=13)
    fig.tight_layout()
    png_path = os.path.join(OUT, "comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


def main():
    dev = Device()  # default doping/geometry (Na=1e17, Nd=1e16), shared by every run below

    # ---- Material comparison: Silicon vs Germanium, both at T=300K ----
    si_300 = resolve_material(material_db.get("Silicon"), T=300.0)
    ge_300 = resolve_material(material_db.get("Germanium"), T=300.0)
    print("Running material comparison (Silicon vs Germanium, both T=300K)...")
    r_si = _run(si_300, dev)
    r_ge = _run(ge_300, dev)
    _compare_plot(
        "diode_material_compare", "Material impact: Silicon vs Germanium (both at T=300K)",
        "Silicon", r_si, "Germanium", r_ge, xlim_um=(-0.5, 0.5),
    )

    # ---- Temperature comparison: Silicon at 300K vs 373.15K (100C) ----
    si_hot = resolve_material(material_db.get("Silicon"), T=373.15)
    print("Running temperature comparison (Silicon at 300K vs 373.15K / 100C)...")
    r_si_hot = _run(si_hot, dev)
    _compare_plot(
        "diode_temperature_compare", "Temperature impact: Silicon at 300K (27C) vs 373.15K (100C)",
        "Silicon, 300K (27C)", r_si, "Silicon, 373.15K (100C)", r_si_hot, xlim_um=(-0.5, 0.5),
    )


if __name__ == "__main__":
    main()
