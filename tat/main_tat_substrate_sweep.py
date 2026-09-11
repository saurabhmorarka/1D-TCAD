"""Driver script: drain-to-substrate TAT+BTBT leakage across several
SUBSTRATE (light-side) doping levels, drain held fixed.

Companion to main_tat_doping_sweep.py's DRAIN-doping sweep, which found
that varying the heavy side barely changes anything below ~1e21 (a
one-sided step junction's depletion width/peak field is set almost
entirely by the LIGHT side once the heavy side is already much heavier -
see that module's docstring). This sweep tests the other half of that
claim directly: holding the drain fixed at 1e20 cm^-3 and varying the
substrate (dev.Na) instead should show the OPPOSITE - genuinely different
leakage at each level, since the light side is exactly the side that sets
the depletion width/field in a one-sided junction.

Run with: python3 -m tat.main_tat_substrate_sweep [input_diode_drain_substrate.yaml]
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
from core.doping_profiles import DopingProfile
from tat.newton_solver_tat import _node_field
from tat.tat import btbt_generation, KaneBTBTModel

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "tat")
os.makedirs(OUT, exist_ok=True)

DOPINGS = [1e16, 1e17, 1e18]
FIXED_DRAIN_ND = 1e20


def run_one(mat, dev, Va_list, mesh_opts, Na):
    """One substrate-doping point: drain held fixed at FIXED_DRAIN_ND.
    Returns (Va_arr, I_tat, I_notat, self-consistency-away-from-eq mask,
    n_mesh_points, F_peak, G_btbt_at_F_peak)."""
    dev.Na = Na
    dev.p_profile = DopingProfile.flat(Na)
    dev.Nd = FIXED_DRAIN_ND
    dev.n_profile = DopingProfile.flat(FIXED_DRAIN_ND)
    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]

    _, _, _, res_tat = voltage_sweep(x, Cdop, mat, dev, Va_list, method="newton_tat")
    _, _, _, res_notat = voltage_sweep(x, Cdop, mat, dev, Va_list, method="newton_qf")

    Va_arr = np.array([r["Va"] for r in res_tat])
    I_tat = np.array([r["I"] for r in res_tat])
    I_notat = np.array([r["I"] for r in res_notat])
    away_from_eq = np.abs(Va_arr) > 0.2
    Jres_tat = np.array([r["J_std"] / max(abs(r["J_mean"]), 1e-30) for r in res_tat])

    ref = res_tat[int(np.argmin(np.abs(Va_arr - (-1.0))))]
    F_node, *_ = _node_field(ref["psi"], x)
    F_peak = float(np.max(F_node))
    G_btbt_peak = float(btbt_generation(np.array([F_peak]), KaneBTBTModel.si_kane_quadratic())[0][0])

    return (Va_arr, I_tat, I_notat, Jres_tat, away_from_eq, len(x),
            F_peak, G_btbt_peak)


DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "configs", "input_diode_drain_substrate.yaml")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    input_cfg = cfg.load_config(input_path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        cfg.build_from_config(input_cfg)

    print(f"Loaded {os.path.basename(input_path)} (drain Nd={FIXED_DRAIN_ND:.2e} cm^-3, "
          f"fixed) - sweeping substrate doping Na over {[f'{d:.0e}' for d in DOPINGS]}")

    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(DOPINGS)))
    fig, (ax_iv, ax_enh) = plt.subplots(1, 2, figsize=(15, 5.5))

    rows = []
    print(f"\n{'Na (cm^-3)':>12} {'n_mesh':>8} {'max selfconsist':>16} {'F_peak (V/cm)':>14} "
          f"{'I_tat @-1V':>14} {'I_tat @-5V':>14} {'enhancement @-5V':>17}")
    for Na, color in zip(DOPINGS, colors):
        (Va_arr, I_tat, I_notat, Jres, away_from_eq, n_mesh,
         F_peak, G_btbt_peak) = run_one(mat, dev, Va_list, mesh_opts, Na)
        rev_mask = Va_arr < 0
        Va_rev = Va_arr[rev_mask]

        with np.errstate(divide="ignore", invalid="ignore"):
            enhancement = np.where(np.abs(I_notat) > 0, I_tat / I_notat, np.nan)

        idx_neg1 = int(np.argmin(np.abs(Va_arr - (-1.0))))
        idx_neg5 = int(np.argmin(np.abs(Va_arr - (-5.0))))
        max_sc = float(np.max(np.abs(Jres[away_from_eq]))) if np.any(away_from_eq) else float("nan")
        flag = "  <-- Kane BTBT saturation cap active" if F_peak >= 9.0e5 else ""
        print(f"{Na:12.0e} {n_mesh:8d} {max_sc:16.2e} {F_peak:14.4e} "
              f"{I_tat[idx_neg1]:14.4e} {I_tat[idx_neg5]:14.4e} {enhancement[idx_neg5]:17.3g}{flag}")
        rows.append([Na, n_mesh, max_sc, F_peak, G_btbt_peak, Va_arr[idx_neg1], I_tat[idx_neg1],
                     Va_arr[idx_neg5], I_tat[idx_neg5], enhancement[idx_neg5]])

        label = f"N_a={Na:.0e} cm^-3"
        ax_iv.semilogy(Va_rev, np.abs(I_tat[rev_mask]), "-", color=color, lw=2, label=label)
        ax_iv.semilogy(Va_rev, np.abs(I_notat[rev_mask]), "--", color=color, lw=1, alpha=0.6)
        ax_enh.plot(Va_rev, enhancement[rev_mask], "-", color=color, lw=2, label=label)

    ax_iv.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_iv.set_ylabel("|I| (A), log")
    ax_iv.set_title(f"Leakage current vs. substrate doping (drain fixed at {FIXED_DRAIN_ND:.0e} cm^-3)\n"
                     "(solid = TAT+BTBT, dashed = no tunneling, same color per Na)")
    ax_iv.legend(fontsize=8)
    ax_iv.grid(color="#dddddd")

    ax_enh.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_enh.set_ylabel("I_tat / I_no_tat (enhancement factor)")
    ax_enh.set_title("Leakage enhancement vs. substrate doping\n"
                      "(lighter substrate -> WIDER depletion -> LOWER field at the same Va)")
    ax_enh.legend(fontsize=8)
    ax_enh.grid(color="#dddddd")

    fig.tight_layout()
    plot_path = os.path.join(OUT, "tat_substrate_sweep.png")
    fig.savefig(plot_path, dpi=130)
    print(f"\nWrote {plot_path}")

    # ---- Zoomed 0 to -1V view: the range most logic/memory junctions
    # actually operate in. A dedicated, finer Va sweep (not just a re-plot
    # of the coarser full-range one above) so this shallow window is
    # properly resolved. Also overlays the pure closed-form Shockley I0
    # (zero tunneling physics) per doping, since the ORDERING seen here
    # (lighter substrate -> MORE leakage, the opposite of the deep-
    # reverse-bias story above) is inherited entirely from ordinary diode
    # behavior (I0 ~ ni^2/Na, minority-carrier injection into the lighter
    # side) - not from tunneling at all. Tunneling still adds its own
    # enhancement on top; it just hasn't grown large enough in this
    # shallow range to flip that ordering back the other way for every
    # doping (see the printed I0 table for the pure-diode baseline).
    Va_zoom = np.linspace(-0.02, -1.0, 30)
    print(f"\nZoomed 0V to -1V sweep ({len(Va_zoom)} points) - the range most "
          "non-power devices actually operate in:")
    print(f"{'Na (cm^-3)':>12} {'I0_shockley (A)':>16} {'I_tat @-1V':>14} "
          f"{'I_notat @-1V':>14} {'enhancement @-1V':>17}")
    fig2, (ax_iv2, ax_enh2) = plt.subplots(1, 2, figsize=(15, 5.5))
    for Na, color in zip(DOPINGS, colors):
        (Va_arr, I_tat, I_notat, Jres, away_from_eq, n_mesh,
         F_peak, G_btbt_peak) = run_one(mat, dev, Va_zoom, mesh_opts, Na)
        with np.errstate(divide="ignore", invalid="ignore"):
            enhancement = np.where(np.abs(I_notat) > 0, I_tat / I_notat, np.nan)
        dev.Na = Na
        I0 = float(an.shockley_I0(mat, dev))
        print(f"{Na:12.0e} {I0:16.4e} {I_tat[-1]:14.4e} {I_notat[-1]:14.4e} "
              f"{enhancement[-1]:17.3g}")

        label = f"N_a={Na:.0e} cm^-3"
        ax_iv2.semilogy(Va_arr, np.abs(I_tat), "-", color=color, lw=2, label=label)
        ax_iv2.semilogy(Va_arr, np.abs(I_notat), "--", color=color, lw=1, alpha=0.6)
        ax_enh2.plot(Va_arr, enhancement, "-", color=color, lw=2, label=label)

    ax_iv2.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_iv2.set_ylabel("|I| (A), log")
    ax_iv2.set_title(f"Leakage current, 0 to -1V (drain fixed at {FIXED_DRAIN_ND:.0e} cm^-3)\n"
                      "(solid = TAT+BTBT, dashed = no tunneling - dashed lines alone\n"
                      "already show lighter substrate = more leakage, ordinary diode physics)")
    ax_iv2.legend(fontsize=8)
    ax_iv2.grid(color="#dddddd")

    ax_enh2.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_enh2.set_ylabel("I_tat / I_no_tat (enhancement factor)")
    ax_enh2.set_title("Tunneling enhancement, 0 to -1V\n"
                       "(stays near 1x for 1e16/1e17 across this whole window, but\n"
                       "1e18 - closest to the drain's own 1e20 doping - already\n"
                       "grows past 100x by -1V, crossing the other two near -0.5V)")
    ax_enh2.legend(fontsize=8)
    ax_enh2.grid(color="#dddddd")

    fig2.tight_layout()
    zoom_plot_path = os.path.join(OUT, "tat_substrate_sweep_0to1V.png")
    fig2.savefig(zoom_plot_path, dpi=130)
    print(f"\nWrote {zoom_plot_path}")

    csv_path = os.path.join(OUT, "tat_substrate_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Na_cm3", "n_mesh_points", "max_selfconsist_away_from_eq", "F_peak_V_cm",
                    "G_btbt_at_F_peak", "Va_neg1_V", "I_tat_neg1_A",
                    "Va_neg5_V", "I_tat_neg5_A", "enhancement_neg5"])
        w.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
