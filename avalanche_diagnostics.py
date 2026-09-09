"""Diagnostic driver: compares Bank-Rose vs. line-search damping for the
avalanche solver, and plots I(Va) on both linear and log scales alongside
per-point iteration counts and self-consistency - the numbers behind
newton_gummel_solve's own printed warnings, made visual.

Not part of the main example (main_avalanche.py) - this is a deeper look at
HOW the avalanche solve converges, point by point, run on demand:

    python3 avalanche_diagnostics.py [input_diode_breakdown.yaml]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh import build_diode_grid
from solver import solve_equilibrium
import newton_solver_avalanche as nsa
import analytic as an
import config as cfg
import avalanche_config as acfg

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

C_BR = "#1f6feb"       # Bank-Rose - blue
C_LS = "#e8590c"       # line search - orange
C_NOAVAL = "#6c757d"   # no avalanche - grey
C_BV = "#c92a2a"


def run_sweep(x, Cdop, mat, dev, psi_eq, n_eq, p_eq, Va_list, damping, ii_model):
    """Continuation sweep at a fixed damping strategy, recording per-point
    I, iteration count, and self-consistency (J_std/J_mean) - the same
    continuation/warm-start pattern solver.voltage_sweep uses, but kept
    local here so both damping strategies see an IDENTICAL warm-start
    history (voltage_sweep's own per-method dispatch would otherwise use
    two independent sweeps that could drift apart for unrelated reasons)."""
    I_arr, iters_arr, selfc_arr = [], [], []
    psi_prev = phin_prev = phip_prev = None
    for Va in Va_list:
        res = nsa.newton_gummel_solve(
            x, Cdop, mat, Va, psi_eq, n_eq, p_eq,
            psi_init=psi_prev, phin_init=phin_prev, phip_init=phip_prev,
            damping=damping, ii_model=ii_model)
        I_arr.append(res["J_mean"] * dev.area)
        iters_arr.append(res["iters"])
        selfc_arr.append(res["J_std"] / max(abs(res["J_mean"]), 1e-30))
        psi_prev, phin_prev, phip_prev = res["psi"], res["phin"], res["phip"]
    return np.array(I_arr), np.array(iters_arr), np.array(selfc_arr)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "input_diode_breakdown.yaml")

    input_cfg = cfg.load_config(input_path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        cfg.build_from_config(input_cfg)
    av = acfg.parse_avalanche_config(input_cfg)
    ii_model = av["ii_model"]

    avalanche_ii_refine = None
    if av["enabled"]:
        avalanche_ii_refine = {"E_crit_V_cm": av["E_crit_V_cm"], "ii_model": ii_model,
                                "cells_per_mfp": av["cells_per_mfp"]}
    g = build_diode_grid(mat, dev, avalanche_ii_refine=avalanche_ii_refine, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]
    psi_eq, n_eq, p_eq, _ = solve_equilibrium(x, Cdop, mat)

    # Reverse-bias leg only (Va_list's forward leg isn't part of the
    # breakdown story and just clutters the comparison).
    Va_rev = Va_list[Va_list < 0]

    print(f"Sweeping {len(Va_rev)} reverse-bias points with line_search damping...")
    I_ls, iters_ls, selfc_ls = run_sweep(x, Cdop, mat, dev, psi_eq, n_eq, p_eq,
                                          Va_rev, "line_search", ii_model)
    print(f"Sweeping {len(Va_rev)} reverse-bias points with bank_rose damping...")
    I_br, iters_br, selfc_br = run_sweep(x, Cdop, mat, dev, psi_eq, n_eq, p_eq,
                                          Va_rev, "bank_rose", ii_model)

    print(f"Sweeping {len(Va_rev)} reverse-bias points, no avalanche (newton_qf)...")
    from newton_solver_qf import newton_gummel_solve as qf_solve
    I_noaval = []
    psi_prev = phin_prev = phip_prev = None
    for Va in Va_rev:
        res = qf_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq,
                        psi_init=psi_prev, phin_init=phin_prev, phip_init=phip_prev)
        I_noaval.append(res["J_mean"] * dev.area)
        psi_prev, phin_prev, phip_prev = res["psi"], res["phin"], res["phip"]
    I_noaval = np.array(I_noaval)

    print("\n Va(V)   I_line_search    iters  selfc_ls  |  I_bank_rose     iters  selfc_br")
    for i in range(len(Va_rev)):
        print(f"{Va_rev[i]:7.2f}  {I_ls[i]: .4e}  {iters_ls[i]:5d}  {selfc_ls[i]:.2e}  |  "
              f"{I_br[i]: .4e}  {iters_br[i]:5d}  {selfc_br[i]:.2e}")

    total_iters_ls, total_iters_br = int(iters_ls.sum()), int(iters_br.sum())
    stall_ls = int(np.sum(iters_ls >= 45))
    stall_br = int(np.sum(iters_br >= 45))
    print(f"\nTotal Newton iterations: line_search={total_iters_ls}, bank_rose={total_iters_br}")
    print(f"Points that ran to (near) maxiter=50: line_search={stall_ls}, bank_rose={stall_br}")

    # ---- Figure: 5 panels ----
    fig, axes = plt.subplots(5, 1, figsize=(8, 18), sharex=True)
    ax_lin, ax_log, ax_cmp, ax_it, ax_sc = axes

    ax_lin.plot(Va_rev, I_ls, "-o", ms=3, color=C_LS, label="avalanche (line_search)")
    ax_lin.plot(Va_rev, I_noaval, "--", color=C_NOAVAL, label="no avalanche")
    ax_lin.set_ylabel("I (A), linear")
    ax_lin.set_title("Current vs. reverse bias - linear scale")
    ax_lin.legend(fontsize=8)
    ax_lin.grid(color="#dddddd")

    ax_log.semilogy(Va_rev, np.abs(I_ls), "-o", ms=3, color=C_LS, label="avalanche (line_search)")
    ax_log.semilogy(Va_rev, np.abs(I_noaval), "--", color=C_NOAVAL, label="no avalanche")
    ax_log.set_ylabel("|I| (A), log")
    ax_log.set_title("Current vs. reverse bias - log scale")
    ax_log.legend(fontsize=8)
    ax_log.grid(color="#dddddd")

    ax_cmp.semilogy(Va_rev, np.abs(I_ls), "-o", ms=3, color=C_LS, label="line_search damping")
    ax_cmp.semilogy(Va_rev, np.abs(I_br), "-s", ms=3, color=C_BR, label="bank_rose damping")
    ax_cmp.set_ylabel("|I| (A), log")
    ax_cmp.set_title("Damping strategy comparison: same physics, same continuation")
    ax_cmp.legend(fontsize=8)
    ax_cmp.grid(color="#dddddd")

    ax_it.plot(Va_rev, iters_ls, "-o", ms=3, color=C_LS, label="line_search")
    ax_it.plot(Va_rev, iters_br, "-s", ms=3, color=C_BR, label="bank_rose")
    ax_it.axhline(50, color="k", linestyle=":", linewidth=1, label="maxiter")
    ax_it.set_ylabel("Newton iterations")
    ax_it.set_title("Iterations to convergence per bias point")
    ax_it.legend(fontsize=8)
    ax_it.grid(color="#dddddd")

    ax_sc.semilogy(Va_rev, selfc_ls, "-o", ms=3, color=C_LS, label="line_search")
    ax_sc.semilogy(Va_rev, selfc_br, "-s", ms=3, color=C_BR, label="bank_rose")
    ax_sc.axhline(1.0, color="k", linestyle=":", linewidth=1)
    ax_sc.set_ylabel("J_std / |J_mean|")
    ax_sc.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_sc.set_title("Self-consistency (current conservation) per bias point")
    ax_sc.legend(fontsize=8)
    ax_sc.grid(color="#dddddd")

    fig.tight_layout()
    plot_path = os.path.join(OUT, "avalanche_diagnostics.png")
    fig.savefig(plot_path, dpi=130)
    print(f"\nWrote {plot_path}")


if __name__ == "__main__":
    main()
