"""Driver script: 1D TCAD avalanche-breakdown simulation.

Special, opt-in driver (see newton_solver_avalanche.py's module docstring
for why this is a separate module from main.py rather than a flag in it).
Builds the mesh (with the optional impact-ionization mean-free-path
refinement), runs the AVALANCHE bias sweep (newton_avalanche - impact
ionization + Bank-Rose damping) alongside a plain newton_qf sweep on the
SAME device (no avalanche) for comparison, and cross-checks the numeric
current runaway against two independent closed-form estimates
(analytic.breakdown_voltage_sze, analytic.multiplication_factor_miller,
analytic.ionization_integral) - see input_diode_breakdown.yaml and
plans/avalanche_breakdown_plan.md Section 5 for the full rationale.

Doping/mesh/voltage-sweep parameters are read from input_diode_breakdown.yaml
(see config.py/avalanche_config.py) - edit that file rather than this one.
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
from avalanche import avalanche_config as acfg
from core import structure_io as sio

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "avalanche")
os.makedirs(OUT, exist_ok=True)

C_AVAL = "#1f6feb"    # numeric, avalanche - blue
C_NOAVAL = "#6c757d"  # numeric, no avalanche - grey
C_MILLER = "#e8590c"  # Miller closed form - orange
C_II = "#2f9e44"      # ionization integral - green
C_BV = "#c92a2a"      # Sze BV marker - red


def build_fine_tail_va_list(Va_list, fine_tail):
    """Insert a finely-spaced reverse-bias tail (fine_tail =
    {start_V, stop_V, step_V}, both V negative, stop_V more negative than
    start_V) in place of whatever coarser points Va_list already has past
    start_V, then keep any points beyond stop_V (e.g. the forward leg)
    appended after. Right at avalanche onset the current can rise by many
    orders of magnitude within a fraction of a volt - resolving that
    narrow transition needs point spacing far finer than is practical to
    use over the WHOLE bias range, so this concentrates the extra density
    only where it's needed (see avalanche_config.parse_avalanche_config's
    fine_tail docstring)."""
    Va_list = np.asarray(Va_list, dtype=float)
    start_V, stop_V, step_V = fine_tail["start_V"], fine_tail["stop_V"], fine_tail["step_V"]
    coarse = Va_list[Va_list > start_V]  # less negative than start_V (unaffected)
    fine = np.arange(start_V, stop_V - 1e-12, -abs(step_V))
    forward = Va_list[Va_list >= 0]  # forward leg, if any, kept and appended after
    return np.concatenate([coarse, fine, forward])


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "input_diode_breakdown.yaml")

    input_cfg = cfg.load_config(input_path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        cfg.build_from_config(input_cfg)
    av = acfg.parse_avalanche_config(input_cfg)
    if math_model != "newton_avalanche":
        print(f"Note: input file's solver.math_model={math_model!r} ignored - "
              "main_avalanche.py always runs the avalanche solver.")
    if av["fine_tail"] is not None:
        Va_list = build_fine_tail_va_list(Va_list, av["fine_tail"])
        print(f"Fine tail: {av['fine_tail']['start_V']}V -> {av['fine_tail']['stop_V']}V "
              f"in {av['fine_tail']['step_V']}V steps ({len(Va_list)} total sweep points)")

    avalanche_ii_refine = None
    if av["enabled"]:
        avalanche_ii_refine = {
            "E_crit_V_cm": av["E_crit_V_cm"],
            "ii_model": av["ii_model"],
            "cells_per_mfp": av["cells_per_mfp"],
        }
    g = build_diode_grid(mat, dev, avalanche_ii_refine=avalanche_ii_refine, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]
    print(f"Grid: {len(x)} points, Wp={g['Wp']*1e4:.2f} um, Wn={g['Wn']*1e4:.2f} um, "
          f"h_min={g['h_min']*1e7:.2f} nm")
    print(f"Doping: Na={dev.Na:.2e} cm^-3 (p-side), Nd={dev.Nd:.2e} cm^-3 (n-side)")

    BV_sze = an.breakdown_voltage_sze(mat, dev)
    print(f"\nSze empirical breakdown voltage (closed form): BV ~= {BV_sze:.2f} V")

    psi_eq, n_eq, p_eq, eq_iters = solve_equilibrium(x, Cdop, mat)
    Vbi = an.built_in_potential(mat, dev)
    print(f"Equilibrium: Newton iters={eq_iters}, Vbi={Vbi:.4f} V")

    print(f"\nRunning avalanche bias sweep (newton_avalanche, "
          f"{len(Va_list)} points, Bank-Rose damping)...")
    _, _, _, res_aval = voltage_sweep(x, Cdop, mat, dev, Va_list, verbose=False,
                                       method="newton_avalanche")

    print("Running comparison bias sweep (newton_qf, no avalanche)...")
    _, _, _, res_noaval = voltage_sweep(x, Cdop, mat, dev, Va_list, verbose=False,
                                         method="newton_qf")

    Va_arr = np.array([r["Va"] for r in res_aval])
    I_aval = np.array([r["I"] for r in res_aval])
    I_noaval = np.array([r["I"] for r in res_noaval])
    Jres_aval = np.array([r["J_std"] / max(abs(r["J_mean"]), 1e-30) for r in res_aval])

    # Numeric multiplication factor: ratio of avalanche to non-avalanche
    # current (only meaningful in reverse bias, where the non-avalanche
    # curve is the flat saturated-leakage baseline M is multiplying).
    with np.errstate(divide="ignore", invalid="ignore"):
        M_numeric = np.where(np.abs(I_noaval) > 0, I_aval / I_noaval, np.nan)
    M_miller = an.multiplication_factor_miller(Va_arr, BV_sze)
    ii_integral = np.array([
        an.ionization_integral(mat, dev, Va, av["ii_model"]) if Va < 0 else np.nan
        for Va in Va_arr
    ])

    print("\nSelf-consistency (J_std/J_mean, avalanche sweep) at a few points:")
    for i in np.linspace(0, len(Va_arr) - 1, min(8, len(Va_arr))).astype(int):
        print(f"  Va={Va_arr[i]:+7.2f} V  I_aval={I_aval[i]:.4e} A  "
              f"I_no_aval={I_noaval[i]:.4e} A  M_num={M_numeric[i]:.3g}  "
              f"M_miller={M_miller[i]:.3g}  ii_integral={ii_integral[i]:.3g}  "
              f"selfconsist={Jres_aval[i]:.2e}")

    csv_path = os.path.join(OUT, "breakdown_iv.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "I_avalanche_A", "I_no_avalanche_A", "M_numeric",
                    "M_miller", "ionization_integral", "J_std_over_mean"])
        for i in range(len(Va_arr)):
            w.writerow([Va_arr[i], I_aval[i], I_noaval[i], M_numeric[i],
                        M_miller[i], ii_integral[i], Jres_aval[i]])
    print(f"\nWrote {csv_path}")

    # ---- Validation plot: I(Va) runaway (linear + log) + ionization integral ----
    # x-axis is Va itself (NEGATIVE for reverse bias), not |Va| - reverse
    # bias reads right-to-left as increasingly negative, matching how the
    # bias is actually applied/reported everywhere else in this codebase.
    fig, (ax_lin, ax_log, ax_ii) = plt.subplots(3, 1, figsize=(7, 11), sharex=True)

    rev_mask = Va_arr < 0
    Va_rev = Va_arr[rev_mask]

    # A voltage-controlled continuation sweep occasionally lands a single
    # isolated bias point on a spurious, non-physical root (self-consistency
    # J_std/|J_mean| >> the ~0.01-1 range every well-converged point in this
    # sweep shows - see newton_solver_avalanche.py's newton_gummel_solve
    # docstring) that the VERY NEXT point recovers from (continuation warm-
    # starts from that point, not from the bad one, since it isn't carried
    # forward - only PLOTTING it is misleading). Mask those out of the
    # plotted curves only - the raw numbers, self-consistency included, are
    # still in breakdown_iv.csv for anyone who wants to look at exactly
    # which points were dropped and why.
    #
    # This is deliberately the ONLY masking criterion here - an earlier
    # version of this code also flagged points whose |I| dipped or spiked
    # relative to both neighbors, which "worked" empirically but was purely
    # a symptom-side heuristic with no connection to why a point goes bad.
    # The real mechanism (traced with an instrumented Jacobian/line-search
    # check, not guessed) is that the avalanche generation term's Jacobian
    # entries reach ~1e31 at this device's sub-angstrom junction mesh
    # spacing, next to ~1e2-scale residual entries - a severely ill-scaled
    # linear system that can defeat the direct solver's effective precision
    # at specific bias points, stalling Newton; the equilibrium-reset
    # fallback then accepts whichever candidate has the lower RAW residual
    # with no check that it's the same physical branch, silently swapping
    # in the no-avalanche trivial solution. self-consistency (this solver's
    # own convergence diagnostic) is what actually flags that failure mode,
    # so it stays the only filter; the durable fix is Jacobian
    # scaling/equilibration around the avalanche coupling terms (see
    # DEVELOPMENT_LOG.md), not a sharper output-side heuristic.
    bad_sc_threshold = 5.0
    good = Jres_aval[rev_mask] <= bad_sc_threshold
    n_masked = int(np.sum(~good))
    if n_masked:
        print(f"\nPlot note: masking {n_masked} isolated bias point(s) with "
              f"self-consistency > {bad_sc_threshold:.0f} (see breakdown_iv.csv "
              "for the raw values) - these are single-point Newton hiccups "
              "that the very next bias point's continuation recovers from.")
    I_aval_plot = np.where(good, I_aval[rev_mask], np.nan)

    ax_lin.plot(Va_rev, I_aval_plot, "-o", ms=3, color=C_AVAL, label="numeric I (avalanche)")
    ax_lin.plot(Va_rev, I_noaval[rev_mask], "--", color=C_NOAVAL, label="numeric I (no avalanche)")
    ax_lin.axvline(-BV_sze, color=C_BV, linestyle="-.", label=f"Sze BV~=-{BV_sze:.1f}V")
    ax_lin.set_ylabel("I (A), linear")
    ax_lin.set_title("Avalanche breakdown: I(Va) - linear scale")
    ax_lin.legend(fontsize=8)
    ax_lin.grid(color="#dddddd")

    ax_log.semilogy(Va_rev, np.abs(I_aval_plot), "-o", ms=3, color=C_AVAL,
                     label="numeric I (avalanche)")
    ax_log.semilogy(Va_rev, np.abs(I_noaval[rev_mask]), "--", color=C_NOAVAL,
                     label="numeric I (no avalanche)")
    I0_leak = np.median(np.abs(I_noaval[rev_mask]))
    ax_log.semilogy(Va_rev, I0_leak * M_miller[rev_mask], ":", color=C_MILLER,
                     label="Miller M(Va) x leakage (closed form)")
    ax_log.axvline(-BV_sze, color=C_BV, linestyle="-.", label=f"Sze BV~=-{BV_sze:.1f}V")
    ax_log.set_ylabel("|I| (A), log")
    ax_log.set_title("Avalanche breakdown: I(Va) - log scale")
    ax_log.legend(fontsize=8)
    ax_log.grid(color="#dddddd")

    ax_ii.plot(Va_rev, ii_integral[rev_mask], color=C_II,
               label="ionization integral (analytic, depletion approx.)")
    ax_ii.axhline(1.0, color="k", linestyle=":", linewidth=1)
    ax_ii.axvline(-BV_sze, color=C_BV, linestyle="-.")
    ax_ii.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_ii.set_ylabel("integral alpha_eff dx")
    ax_ii.legend(fontsize=8)
    ax_ii.grid(color="#dddddd")

    fig.tight_layout()
    plot_path = os.path.join(OUT, "breakdown_iv.png")
    fig.savefig(plot_path, dpi=130)
    print(f"Wrote {plot_path}")

    # ---- Structure+fields file (a few bias points, same convention as main.py) ----
    bias_points = []
    for target in save_bias_points:
        i = int(np.argmin(np.abs(Va_arr - target)))
        r = res_aval[i]
        bias_points.append({
            "label": f"Va={r['Va']:+.2f}V", "bias": r["Va"],
            "fields": {"psi": r["psi"], "n": r["n"], "p": r["p"],
                       "phin": r["phin"], "phip": r["phip"]},
        })
    sio.save_structure(
        os.path.join(OUT, structure_file),
        device="diode", dim=1,
        material={"eps_r": mat.eps_r, "ni_cm3": mat.ni, "chi_eV": mat.chi_eV, "Eg_eV": mat.Eg_eV},
        regions=[
            {"name": "p-side", "x_range_um": [x[0] * 1e4, 0.0], "kind": "semiconductor", "doping_type": "p"},
            {"name": "n-side", "x_range_um": [0.0, x[-1] * 1e4], "kind": "semiconductor", "doping_type": "n"},
        ],
        x_um=x * 1e4, doping_cm3=Cdop, bias_points=bias_points,
    )
    print(f"Wrote {os.path.join(OUT, structure_file)}")


if __name__ == "__main__":
    main()
