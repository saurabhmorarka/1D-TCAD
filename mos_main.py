"""Driver script: 1D MOS capacitor C-V simulation.

Builds a mesh across a metal gate - thin oxide - uniform substrate stack,
sweeps gate voltage, and computes both the low-frequency (quasi-static
equilibrium) and high-frequency (frozen-minority-carrier quasi-small-signal)
C-V curves. Compares against closed-form depletion-approximation theory:
flat-band voltage, threshold voltage, and the analytic low-/high-frequency
C-V curves.

All simulation parameters are read from input_mos.yaml (see mos_config.py) -
edit that file, not this one, to change substrate doping/type, oxide
thickness, gate work function, or the voltage sweep.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics as ph
from mesh import build_mos_grid
from mos_solver import cv_sweep, solve_mos_equilibrium
import mos_analytic as man
import mos_config as cfg
import field_save as fsave
import mos_params as mparams
import structure_io as sio
import plot as tplot

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

C_NUM_LF = "#1f6feb"   # numeric low-frequency - blue
C_NUM_HF = "#0a9396"   # numeric high-frequency - teal
C_AN_LF = "#e8590c"    # analytic low-frequency - orange
C_AN_HF = "#c2410c"    # analytic high-frequency - darker orange
C_GRID = "#c9c9c9"


def main():
    # Optional CLI arg selects which input YAML to run (default input_mos.yaml,
    # the ideal-metal-gate example) - e.g. `python3 mos_main.py input_mos_poly.yaml`
    # for the real-poly-gate example, so both can be run/compared without
    # editing or overwriting either file.
    input_path = sys.argv[1] if len(sys.argv) > 1 else cfg.DEFAULT_PATH
    input_cfg = cfg.load_config(input_path)
    mat, dev, Cdop_substrate, VG_list, save_bias_points, mesh_opts, structure_file, Cdop_gate = \
        cfg.build_from_config(input_cfg)
    is_p_sub = Cdop_substrate < 0
    is_poly_gate = dev.gate_kind == "poly"
    # Distinct output filenames for the poly-gate example so running both
    # input_mos.yaml and a poly-gate input in sequence doesn't clobber each
    # other's plots/CSVs in the shared out/ directory.
    prefix = "poly_" if is_poly_gate else ""
    def outp(name):
        return os.path.join(OUT, prefix + name)

    print(f"Loaded {os.path.basename(input_path)} (substrate={'p' if is_p_sub else 'n'}-type, "
          f"|Nsub|={abs(Cdop_substrate):.2e} cm^-3, t_ox={dev.t_ox*1e7:.2f} nm, "
          f"gate={'poly ' + ('p' if Cdop_gate < 0 else 'n') + f'+ |Ngate|={abs(Cdop_gate):.2e} cm^-3' if is_poly_gate else 'ideal metal'})"
          if input_cfg else f"No {os.path.basename(input_path)} found - using mos_params.py defaults")

    g = build_mos_grid(mat, dev, Cdop_substrate, Cdop_gate=Cdop_gate, **mesh_opts)
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]
    gate_oxide_index = g["gate_oxide_index"]
    print(f"Grid: {len(x)} points, t_ox={dev.t_ox*1e7:.2f} nm ({g['is_oxide'].sum()} oxide points), "
          f"t_si={g['t_si']*1e4:.2f} um" + (f", t_gate={g['t_gate']*1e4:.3f} um" if is_poly_gate else "") +
          f", h_min={g['h_min']*1e7:.3f} nm, h_max={g['h_max']*1e7:.2f} nm")

    # ---- Derived quantities ----
    # For a poly gate, there is no single closed-form V_FB/V_T (those assume
    # an ideal, non-depleting metal) - instead we build an "ideal poly"
    # comparison device: same structure, but with dev.gate_kind treated as
    # metal and its work function taken from the poly's OWN doping via the
    # same chi+Eg/2+-phi_F formula already used for the substrate's work
    # function (mos_analytic.semiconductor_workfunction_eV). This reuses the
    # existing depletion-approximation machinery unchanged and gives exactly
    # the right reference curve: "what the C-V would look like if this same
    # poly doping never depleted" - the gap between it and the real numeric
    # solve IS the poly-depletion effect.
    dev_ideal_gate = dev
    if is_poly_gate:
        import copy
        dev_ideal_gate = copy.copy(dev)
        dev_ideal_gate.gate_workfunction_eV = man.semiconductor_workfunction_eV(mat, Cdop_gate)

    Cox = man.C_ox(dev)
    V_FB = man.flatband_voltage(dev_ideal_gate, mat, Cdop_substrate)
    V_T = man.threshold_voltage(mat, dev_ideal_gate, Cdop_substrate)
    W_max = man.max_depletion_width(mat, Cdop_substrate)
    phi_F = mat.Vt * np.log(abs(Cdop_substrate) / mat.ni)
    print(f"\nC_ox = {Cox:.4e} F/cm^2   phi_F = {phi_F:.4f} V   V_FB = {V_FB:.4f} V   "
          f"V_T = {V_T:.4f} V   W_max = {W_max*1e4:.4f} um" +
          ("   (V_FB/V_T use the 'ideal, non-depleting poly' reference work function)" if is_poly_gate else ""))

    # Zoom window on the gate side: wide enough to show the poly's own
    # depletion width (same reasoning as W_max on the substrate side), or
    # just a few oxide-thicknesses for an ideal metal gate (nothing to see
    # there beyond the oxide edge itself).
    gate_zoom_um = 5 * man.max_depletion_width(mat, Cdop_gate) * 1e4 if is_poly_gate else dev.t_ox * 1e4 * 2

    # ---- C-V sweep ----
    print(f"\nRunning C-V sweep ({len(VG_list)} points, low- and high-frequency per point)...")
    results = cv_sweep(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate, VG_list, g["oxide_index"],
                        Cdop_gate=Cdop_gate, gate_oxide_index=gate_oxide_index)

    if is_poly_gate:
        psi_bulk_gate = ph.equilibrium_bulk_potential(mat, Cdop_gate)
        print(f"\nPoly gate depletion (band-bending relative to its own bulk potential "
              f"{psi_bulk_gate:.4f} V, at the poly/oxide interface):")
        print(f"{'VG (V)':>8} {'psi_gate_bulk (V)':>18} {'psi_gate_surf (V)':>18} {'poly bending (V)':>17}")
        for r in results:
            psi_gs = r["psi"][gate_oxide_index]
            print(f"{r['VG']:8.3f} {psi_bulk_gate:18.4f} {psi_gs:18.4f} {psi_gs - psi_bulk_gate:17.4f}")

    VG_arr = np.array([r["VG"] for r in results])
    C_lf = np.array([r["C_lf"] for r in results])
    C_hf = np.array([r["C_hf"] for r in results])
    Qs = np.array([r["Qs"] for r in results])
    C_lf_an = man.cv_curve_lowfreq_depletion_approx(mat, dev_ideal_gate, Cdop_substrate, VG_arr)
    C_hf_an = man.cv_curve_highfreq_depletion_approx(mat, dev_ideal_gate, Cdop_substrate, VG_arr)

    print(f"{'VG (V)':>8} {'C_lf/Cox':>10} {'C_lf_an/Cox':>12} {'C_hf/Cox':>10} {'C_hf_an/Cox':>12} {'iters':>6}")
    for r, clf, clfan, chf, chfan in zip(results, C_lf, C_lf_an, C_hf, C_hf_an):
        print(f"{r['VG']:8.3f} {clf/Cox:10.4f} {clfan/Cox:12.4f} {chf/Cox:10.4f} {chfan/Cox:12.4f} {r['iters']:6d}")

    # ================= Plots =================

    # 1. C-V curve: numeric vs analytic, low- and high-frequency
    an_label_suffix = " (ideal, non-depleting poly)" if is_poly_gate else " (depletion approx.)"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(VG_arr, C_lf / Cox, color=C_NUM_LF, lw=2, marker="o", ms=3,
            label="Low-freq, numeric" + (" (real poly, can deplete)" if is_poly_gate else ""))
    ax.plot(VG_arr, C_lf_an / Cox, color=C_AN_LF, lw=1.8, ls="--", label="Low-freq, analytic" + an_label_suffix)
    ax.plot(VG_arr, C_hf / Cox, color=C_NUM_HF, lw=2, marker="s", ms=3,
            label="High-freq, numeric" + (" (real poly, can deplete)" if is_poly_gate else ""))
    ax.plot(VG_arr, C_hf_an / Cox, color=C_AN_HF, lw=1.8, ls=":", label="High-freq, analytic" + an_label_suffix)
    ax.axvline(V_FB, color=C_GRID, lw=1, ls="-", zorder=0)
    ax.axvline(V_T, color=C_GRID, lw=1, ls="-", zorder=0)
    ax.text(V_FB, 1.02, " V_FB", fontsize=8, color="#666")
    ax.text(V_T, 1.02, " V_T", fontsize=8, color="#666")
    ax.set_xlabel("Gate voltage V_G (V)")
    ax.set_ylabel("C / C_ox")
    gate_title = (f", {'p' if Cdop_gate < 0 else 'n'}+ poly gate {abs(Cdop_gate):.0e} cm^-3" if is_poly_gate else "")
    ax.set_title(f"MOS-cap C-V  (t_ox={dev.t_ox*1e7:.1f} nm, "
                 f"{'p' if is_p_sub else 'n'}-sub {abs(Cdop_substrate):.0e} cm^-3{gate_title})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, color=C_GRID)
    fig.tight_layout()
    fig.savefig(outp("01_cv_curve.png"), dpi=150)
    plt.close(fig)

    # 2. Band diagrams + carrier profiles at a few representative gate voltages
    x_um = x * 1e4
    pick_targets = [VG_arr.min(), V_FB, (V_FB + V_T) / 2.0, V_T, VG_arr.max()]
    pick_idx = sorted({int(np.argmin(np.abs(VG_arr - t))) for t in pick_targets})
    cmap = plt.cm.plasma
    colors = [cmap(t) for t in np.linspace(0.1, 0.85, len(pick_idx))]

    zoom_um = 5 * W_max * 1e4
    x_nm = x * 1e7
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for c, i in zip(colors, pick_idx):
        r = results[i]
        axes[0].plot(x_um, r["psi"], color=c, lw=1.8, label=f"V_G={r['VG']:.2f} V")
        axes[1].plot(x_um, r["psi"], color=c, lw=1.8, label=f"V_G={r['VG']:.2f} V")
        axes[2].plot(x_nm, r["psi"], color=c, lw=1.8, marker=".", ms=4, label=f"V_G={r['VG']:.2f} V")
    axes[0].set_xlabel("x (um)"); axes[0].set_ylabel("Electrostatic potential psi (V)")
    axes[0].set_title("Full device"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3, color=C_GRID)
    axes[1].set_xlim(-dev.t_ox * 1e4 - gate_zoom_um, zoom_um)
    axes[1].set_xlabel("x (um)"); axes[1].set_title("Zoom on depletion region")
    axes[1].axvline(0, color=C_GRID, lw=1); axes[1].grid(alpha=0.3, color=C_GRID)
    axes[1].legend(fontsize=7)
    axes[2].set_xlim(-dev.t_ox * 1e7 * 1.2, dev.t_ox * 1e7 * 0.3)
    axes[2].set_xlabel("x (nm)"); axes[2].set_title("Zoom on the oxide itself\n(linear psi(x) confirms zero oxide charge)")
    axes[2].axvline(0, color=C_GRID, lw=1); axes[2].grid(alpha=0.3, color=C_GRID)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outp("02_band_diagram.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for c, i in zip(colors, pick_idx):
        r = results[i]
        ax.semilogy(x_um, r["n"], color=c, lw=1.8, ls="-", label=f"n, V_G={r['VG']:.2f} V")
        ax.semilogy(x_um, r["p"], color=c, lw=1.8, ls="--", label=f"p, V_G={r['VG']:.2f} V")
    ax.set_xlim(-dev.t_ox * 1e4 - gate_zoom_um, zoom_um)
    ax.set_xlabel("x (um)"); ax.set_ylabel("Carrier concentration (cm^-3)")
    ax.axvline(0, color=C_GRID, lw=1)
    ax.set_title("Carrier profiles near the interface")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, color=C_GRID, which="both")
    fig.tight_layout()
    fig.savefig(outp("03_carrier_profiles.png"), dpi=150)
    plt.close(fig)

    # 3. Quasi-Fermi potentials: illustrating the frozen-minority-carrier
    # assumption behind the high-frequency curve. The precise C_hf uses a
    # tiny (dV_hf) perturbation - too small to see on a plot - so this uses
    # a larger, clearly-labeled illustrative perturbation instead, at the
    # configured save_bias_points (input_mos.yaml: output.save_bias_points).
    save_idx = fsave.resolve_save_points(save_bias_points, VG_arr)
    psi_bulk = ph.equilibrium_bulk_potential(mat, Cdop_substrate)
    dV_illustrate = 0.2
    illustrated = []
    for i in save_idx:
        lf = results[i]
        n_frozen = lf["n"].copy() if is_p_sub else None
        p_frozen = lf["p"].copy() if not is_p_sub else None
        pert = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate,
                                      lf["VG"] + dV_illustrate, psi_bulk, psi_init=lf["psi"],
                                      n_frozen=n_frozen, p_frozen=p_frozen)
        pert["VG"] = lf["VG"]  # label by the base bias point, not the perturbed one
        illustrated.append(pert)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors2 = [cmap(t) for t in np.linspace(0.1, 0.85, len(illustrated))]
    for c, r in zip(colors2, illustrated):
        label = f"V_G={r['VG']:.2f} V"
        ax.plot(x_um, r["phin"], color=c, lw=1.8, ls="-", label=f"phin, {label}")
        ax.plot(x_um, r["phip"], color=c, lw=1.8, ls="--", label=f"phip, {label}")
    ax.set_xlim(-dev.t_ox * 1e4 * 3, zoom_um)
    ax.axvline(0, color=C_GRID, lw=1)
    ax.set_xlabel("x (um)"); ax.set_ylabel("Quasi-Fermi potential (V)")
    ax.set_title(f"Quasi-Fermi split under the frozen-minority-carrier assumption\n"
                 f"(illustrative dV_G={dV_illustrate:.2f} V perturbation, exaggerated vs. the "
                 f"actual dV_hf used for C_hf)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, color=C_GRID)
    fig.tight_layout()
    fig.savefig(outp("04_quasi_fermi_potentials.png"), dpi=150)
    plt.close(fig)

    # ---- CSV export ----
    cv_csv = outp("cv_sweep.csv")
    with open(cv_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["VG_V", "Qs_C_cm2", "C_lf_F_cm2", "C_lf_analytic_F_cm2",
                    "C_hf_F_cm2", "C_hf_analytic_F_cm2"])
        for r, clf, clfan, chf, chfan in zip(results, C_lf, C_lf_an, C_hf, C_hf_an):
            w.writerow([f"{r['VG']:.4f}", f"{r['Qs']:.6e}", f"{clf:.6e}", f"{clfan:.6e}",
                        f"{chf:.6e}", f"{chfan:.6e}"])

    fields_csv = outp("mos_fields_by_bias.csv")
    fsave.save_fields(results, save_idx, "VG", fields_csv, x, is_oxide=g["is_oxide"])

    # ================= Structure + textbook band/charge diagrams =================
    def regime_of(VG):
        if is_p_sub:
            if VG < V_FB:
                return "accumulation"
            return "inversion" if VG > V_T else "depletion"
        else:
            if VG > V_FB:
                return "accumulation"
            return "inversion" if VG < V_T else "depletion"

    struct_bias_points = []
    for i in save_idx:
        r = results[i]
        struct_bias_points.append({
            "label": f"V_G={r['VG']:.2f} V", "bias": r["VG"], "regime": regime_of(r["VG"]),
            "fields": {"psi": r["psi"], "n": r["n"], "p": r["p"], "phin": r["phin"], "phip": r["phip"]},
        })

    is_oxide = g["is_oxide"]
    chi_per_node = np.where(is_oxide, mparams.CHI_OX_EV, mat.chi_eV)
    Eg_per_node = np.where(is_oxide, mparams.EG_OX_EV, mat.Eg_eV)

    regions = []
    if is_poly_gate:
        is_p_gate = Cdop_gate < 0
        regions.append({
            "name": f"{'p' if is_p_gate else 'n'}+ poly gate (~{abs(Cdop_gate):.0e})",
            "x_range_um": [(-dev.t_ox - g["t_gate"]) * 1e4, -dev.t_ox * 1e4],
            "kind": "semiconductor", "doping_type": "p" if is_p_gate else "n",
        })
    regions.append({"name": f"oxide ({dev.t_ox*1e7:.1f} nm)", "x_range_um": [-dev.t_ox * 1e4, 0.0],
                     "kind": "insulator", "doping_type": None})
    regions.append({"name": f"{'p' if is_p_sub else 'n'}-substrate (~{abs(Cdop_substrate):.0e})",
                     "x_range_um": [0.0, g["t_si"] * 1e4],
                     "kind": "semiconductor", "doping_type": "p" if is_p_sub else "n"})

    struct_doc = sio.build_structure(
        device="mos",
        material={"eps_r": mat.eps_r, "ni_cm3": mat.ni, "T": mat.T,
                  "chi_eV": chi_per_node, "Eg_eV": Eg_per_node},
        regions=regions,
        x_um=x_um, doping_cm3=Cdop, bias_points=struct_bias_points,
    )
    if structure_file:
        sio.write_structure(os.path.join(OUT, structure_file), struct_doc)

    interface_xlim = (-dev.t_ox * 1e4 - gate_zoom_um, dev.t_ox * 1e4 * 1.5)

    band_charge_xlim = (-dev.t_ox * 1e4 * 3, zoom_um)
    fig, axes = plt.subplots(1, 2, figsize=(14, 2.8))
    tplot.plot_structure(struct_doc, ax=axes[0])
    tplot.plot_structure(struct_doc, ax=axes[1], xlim_um=interface_xlim)
    axes[0].set_title("Full device")
    axes[1].set_title("Zoom on the gate/oxide/substrate interfaces")
    fig.tight_layout()
    fig.savefig(outp("05_structure.png"), dpi=150)
    plt.close(fig)

    # The oxide (nm-scale) is invisible next to the substrate depletion
    # region (um-scale) on one shared x-axis, so - as with the structure
    # plot above - band/charge diagrams get their own oxide-zoom panel
    # alongside the substrate-zoom one.
    mid_idx = len(struct_bias_points) // 2
    oxide_xlim = interface_xlim

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    tplot.plot_bands(struct_doc, bias_index=mid_idx, ax=axes[0], xlim_um=oxide_xlim)
    tplot.plot_bands(struct_doc, bias_index=mid_idx, ax=axes[1], xlim_um=band_charge_xlim)
    axes[0].set_title("Zoom on the gate/oxide interface" if is_poly_gate else "Zoom on the oxide")
    axes[1].set_title("Zoom on the substrate depletion region")
    fig.tight_layout()
    fig.savefig(outp("06_band_diagram.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tplot.plot_charge(struct_doc, bias_index=mid_idx, ax=axes[0], xlim_um=oxide_xlim)
    tplot.plot_charge(struct_doc, bias_index=mid_idx, ax=axes[1], xlim_um=band_charge_xlim)
    axes[0].set_title("Zoom on the gate/oxide interface" if is_poly_gate else "Zoom on the oxide")
    axes[1].set_title("Zoom on the substrate depletion region")
    fig.tight_layout()
    fig.savefig(outp("07_charge_density.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots and data written to: {OUT}")
    for name in ["01_cv_curve.png", "02_band_diagram.png", "03_carrier_profiles.png",
                 "04_quasi_fermi_potentials.png", "05_structure.png",
                 "06_band_diagram.png (textbook Ec/Ev/Ei/Ef)", "07_charge_density.png",
                 "cv_sweep.csv", "mos_fields_by_bias.csv"]:
        print(f"  {prefix}{name}")
    if structure_file:
        print(f"  {structure_file}  (standalone: python3 plot.py out/{structure_file})")


if __name__ == "__main__":
    main()
