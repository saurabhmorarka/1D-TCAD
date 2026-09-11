"""Driver script: Hurkx vs. Schenk trap-assisted-tunneling model comparison,
same device (input_diode_drain_substrate.yaml), same no-tunneling baseline.

Per plans/tat_btbt_plan.md's design: Hurkx was implemented and validated
first (lower-risk - it degrades exactly onto this project's own already-
validated SRH code at zero field); Schenk (more microscopic, phonon-
assisted, FLOOXS-sourced) is the planned second, swappable cross-check
model. Both share tat/newton_solver_tat.py's identical
(n, p, F_abs, mat, model) -> (G, dG_dn, dG_dp, dG_dF) call signature (see
tat/tat.py), selected here via `trap_model`/`trap_generation_fn`.

KNOWN LIMITATION, reported honestly rather than hidden: Schenk's
self-consistency (J_std/J_mean) on this mesh is visibly worse than
Hurkx's (0.1-3 vs. Hurkx's ~1e-5) even after Ruiz equilibration fixed an
outright divergence (see DEVELOPMENT_LOG.md) - the Newton RESIDUAL itself
converges tightly (well below f_tol in a handful of iterations), so this
is not a Newton convergence failure, but the resulting solution's current
continuity (Jn+Jp should be spatially uniform in steady state) isn't as
tightly satisfied as Hurkx's. Working hypothesis, not yet confirmed:
Schenk's field dependence is much sharper (closer to Kane BTBT's own
exponential scale than to Hurkx's smoother Gamma(F) - see tat.py's
SchenkTATModel docstring and its standalone sanity-probe comparison),
which may need the same kind of extra mesh refinement near the peak field
that avalanche/avalanche.py's own `avalanche_ii_refine` provides for its
similarly sharp impact-ionization generation - not yet implemented for
Schenk. Treat Schenk's numbers here as a first, honestly-flagged working
comparison, not yet as trustworthy as the Hurkx numbers this project has
validated more thoroughly.

Run with: python3 -m tat.main_tat_hurkx_vs_schenk [input_diode_drain_substrate.yaml]
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
from core import config as cfg
from tat.newton_solver_tat import newton_gummel_solve
from tat.tat import (KaneBTBTModel, HurkxTATModel, SchenkTATModel,
                      hurkx_tat_generation, schenk_tat_generation)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "tat")
os.makedirs(OUT, exist_ok=True)

C_HURKX = "#1f6feb"
C_SCHENK = "#e8590c"
C_NOTAT = "#6c757d"

DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "configs", "input_diode_drain_substrate.yaml")


def run_sweep(x, Cdop, mat, dev, Va_list, kane_model, trap_model, trap_generation_fn):
    psi_eq, n_eq, p_eq, _ = solve_equilibrium(x, Cdop, mat)
    results = []
    pi = phi_ni = phi_pi = None
    for Va in Va_list:
        r = newton_gummel_solve(x, Cdop, mat, Va, psi_eq, n_eq, p_eq, pi, phi_ni, phi_pi,
                                 kane_model=kane_model, trap_model=trap_model,
                                 trap_generation_fn=trap_generation_fn)
        r["Va"] = Va
        r["I"] = r["J_mean"] * dev.area
        results.append(r)
        pi, phi_ni, phi_pi = r["psi"], r["phin"], r["phip"]
    return results


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    input_cfg = cfg.load_config(input_path)
    mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file = \
        cfg.build_from_config(input_cfg)

    g = build_diode_grid(mat, dev, **mesh_opts)
    x, Cdop = g["x"], g["Cdop"]
    print(f"Device: Na={dev.Na:.2e} (substrate), Nd={dev.Nd:.2e} (drain), {len(x)} mesh points")

    kane = KaneBTBTModel.si_kane_quadratic()
    hurkx = HurkxTATModel()
    schenk = SchenkTATModel()

    print("Running Hurkx sweep...")
    res_hurkx = run_sweep(x, Cdop, mat, dev, Va_list, kane, hurkx, hurkx_tat_generation)
    print("Running Schenk sweep...")
    res_schenk = run_sweep(x, Cdop, mat, dev, Va_list, kane, schenk, schenk_tat_generation)
    print("Running no-tunneling baseline...")
    _, _, _, res_notat = voltage_sweep(x, Cdop, mat, dev, Va_list, method="newton_qf")

    Va_arr = np.array([r["Va"] for r in res_hurkx])
    I_hurkx = np.array([r["I"] for r in res_hurkx])
    I_schenk = np.array([r["I"] for r in res_schenk])
    I_notat = np.array([r["I"] for r in res_notat])
    sc_hurkx = np.array([r["J_std"] / max(abs(r["J_mean"]), 1e-30) for r in res_hurkx])
    sc_schenk = np.array([r["J_std"] / max(abs(r["J_mean"]), 1e-30) for r in res_schenk])

    rev_mask = Va_arr < 0
    Va_rev = Va_arr[rev_mask]

    print(f"\n{'Va':>7} {'I_hurkx':>14} {'I_schenk':>14} {'I_no_tat':>14} "
          f"{'sc_hurkx':>10} {'sc_schenk':>10}")
    for i in np.linspace(0, len(Va_arr) - 1, min(10, len(Va_arr))).astype(int):
        print(f"{Va_arr[i]:7.2f} {I_hurkx[i]:14.4e} {I_schenk[i]:14.4e} {I_notat[i]:14.4e} "
              f"{sc_hurkx[i]:10.2e} {sc_schenk[i]:10.2e}")

    csv_path = os.path.join(OUT, "tat_hurkx_vs_schenk.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Va_V", "I_hurkx_A", "I_schenk_A", "I_no_tat_A", "sc_hurkx", "sc_schenk"])
        for i in range(len(Va_arr)):
            w.writerow([Va_arr[i], I_hurkx[i], I_schenk[i], I_notat[i], sc_hurkx[i], sc_schenk[i]])
    print(f"\nWrote {csv_path}")

    fig, (ax_iv, ax_sc) = plt.subplots(1, 2, figsize=(15, 5.5))

    ax_iv.semilogy(Va_rev, np.abs(I_hurkx[rev_mask]), "-o", ms=3, color=C_HURKX, label="Hurkx")
    ax_iv.semilogy(Va_rev, np.abs(I_schenk[rev_mask]), "-s", ms=3, color=C_SCHENK, label="Schenk")
    ax_iv.semilogy(Va_rev, np.abs(I_notat[rev_mask]), "--", color=C_NOTAT, label="no tunneling")
    ax_iv.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_iv.set_ylabel("|I| (A), log")
    ax_iv.set_title("Hurkx vs. Schenk trap-assisted tunneling, same device")
    ax_iv.legend(fontsize=8)
    ax_iv.grid(color="#dddddd")

    ax_sc.semilogy(Va_rev, np.abs(sc_hurkx[rev_mask]), "-o", ms=3, color=C_HURKX, label="Hurkx")
    ax_sc.semilogy(Va_rev, np.abs(sc_schenk[rev_mask]), "-s", ms=3, color=C_SCHENK, label="Schenk")
    ax_sc.axhline(1.0, color="k", linestyle=":", linewidth=1)
    ax_sc.set_xlabel("Applied bias Va (V) - reverse is negative")
    ax_sc.set_ylabel("self-consistency |J_std/J_mean|, log")
    ax_sc.set_title("Self-consistency comparison - Schenk is visibly worse\n"
                     "(known limitation, see this module's docstring)")
    ax_sc.legend(fontsize=8)
    ax_sc.grid(color="#dddddd")

    fig.tight_layout()
    plot_path = os.path.join(OUT, "tat_hurkx_vs_schenk.png")
    fig.savefig(plot_path, dpi=130)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
