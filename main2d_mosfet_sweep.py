"""2D planar NMOS transistor driver: same oxide+gate stack as the MOS
capacitor, now with n+ source/drain regions connected to their own ohmic
contacts, so a real channel current exists (modulated by the gate) -
requires the full coupled Poisson+continuity solver
(solver2d/newton_solver_qf_2d.py, the same one the 2D diode uses), NOT the
MOS capacitor's Poisson-only solver.

Produces the two standard MOSFET characterization curves:
  Ids-Vgs (transfer curve): fixed Vds, sweep Vgs, gate contact continuation.
  Ids-Vds (output curves): a family of fixed Vgs, sweep Vds each.

The gate contact sits on the oxide (an insulator node, ni=0) so its
Dirichlet BC can't use the generic ohmic mass-action relation
newton_solve_2d applies to every other contact by default - this driver
computes it explicitly (mos.mos_analytic.flatband_voltage, same formula
the MOS capacitor's solver uses) and passes it through newton_solve_2d's
psi_bc_override/phin_bc_override/phip_bc_override.

Usage: python3 main2d_mosfet_sweep.py [configs/input_mosfet_2d.yaml]
"""
import argparse
import csv
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from core import structure_io as sio
from core.config import _resolve_material_block
from core.field_save import resolve_save_points
from core.physics import equilibrium_bulk_potential
from mesh2d.config2d import build_domain_from_config
from mesh2d.mesh2d import build_mesh2d
from mos import mos_analytic as man
from mos.mos_params import MOSDevice
from solver2d.current import contact_current_density
from solver2d.efield2d import electric_field_2d
from solver2d.newton_solver_qf_2d import newton_solve_2d
from viz2d.plot2d import plot_field2d

_CM_TO_UM = 1.0e4
_UM_TO_CM = 1.0e-4

DEFAULT_PATH = os.path.join("configs", "input_mosfet_2d.yaml")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_from_config(cfg):
    domain = build_domain_from_config(cfg)
    material = _resolve_material_block(cfg.get("material") or {})

    substrate = cfg["geometry"]["substrate"]
    Na = float(substrate["concentration_cm3"])
    Cdop_substrate = -Na if substrate["doping_type"] == "p" else Na

    oxide_region = next(r for r in domain.regions if r.kind == "insulator")
    t_ox_cm = oxide_region.y_range_cm[1] - oxide_region.y_range_cm[0]
    dev = MOSDevice(Na=Na, t_ox=t_ox_cm, eps_ox_r=oxide_region.eps_r,
                     gate_workfunction_eV=(cfg.get("gate") or {}).get("workfunction_eV"))

    mesh_cfg = cfg.get("mesh") or {}
    mesh_opts = dict(
        h_min_cm=float(mesh_cfg.get("h_min_um", 0.2)) * _UM_TO_CM,
        h_max_cm=float(mesh_cfg.get("h_max_um", 2.0)) * _UM_TO_CM,
        growth=float(mesh_cfg.get("growth", 1.3)),
        min_angle_deg=int(mesh_cfg.get("min_angle_deg", 32)),
    )

    bias_cfg = cfg.get("bias") or {}
    output_cfg = cfg.get("output") or {}
    return domain, material, dev, Cdop_substrate, mesh_opts, bias_cfg, output_cfg


def gate_bc(dev, mat, Cdop_substrate, Vgs):
    """(psi_bc, phin_bc, phip_bc) for the gate contact - an ideal metal
    sitting directly on the oxide (see mos/mos_solver.py's Cdop_gate=None
    convention). phin/phip are moot where ni=0 (the whole oxide, the gate
    node included) but still need a finite value for the Dirichlet row -
    Vgs, matching the "VG on the insulator side" convention used
    throughout the 1D and 2D MOS capacitor solvers."""
    psi_bulk = equilibrium_bulk_potential(mat, Cdop_substrate)
    V_FB = man.flatband_voltage(dev, mat, Cdop_substrate)
    psi_bc = psi_bulk + (Vgs - V_FB)
    return psi_bc, Vgs, Vgs


# Caughey-Thomas velocity-saturation mobility (solver2d/newton_solver_qf_2d.py)
# couples psi into the continuity Jacobian more strongly than the constant-
# mobility case this driver's default maxiter=50 (newton_solve_2d's own
# default) was tuned against - direct testing (Vgs=0 to 0.7V at Vds=0.05V,
# stepping through every 0.1V) showed EVERY point in that range was still
# steadily, monotonically reducing its residual at iteration 50 (not stuck,
# just needing ~55-85 iterations instead of <50) - i.e. this is an iteration-
# budget issue, not a fundamentally harder basin-of-attraction problem.
# Raising maxiter to 200 here (a driver-level choice, not changed globally
# in newton_solve_2d's own default, since the diode/other examples already
# converge fine within 50) resolved the entire sweep except one single
# isolated stall point (Vgs=1.2V) - left to the adaptive bisection
# continuation (`_bisecting_continuation`) below to route around.
MOSFET_MAXITER = 200


def solve_mosfet(mesh, mat, dev, Cdop_substrate, Vgs, Vds, psi_init=None, phin_init=None, phip_init=None,
                  verbose=False, maxiter=MOSFET_MAXITER):
    psi_bc, phin_bc, phip_bc = gate_bc(dev, mat, Cdop_substrate, Vgs)
    bias_by_contact = {"source": 0.0, "drain": Vds, "gate": Vgs, "body": 0.0}
    return newton_solve_2d(
        mesh, mat, bias_by_contact,
        psi_init=psi_init, phin_init=phin_init, phip_init=phip_init,
        psi_bc_override={"gate": psi_bc}, phin_bc_override={"gate": phin_bc},
        phip_bc_override={"gate": phip_bc}, verbose=verbose, maxiter=maxiter,
    )


def sweep_vgs(mesh, mat, dev, Cdop_substrate, Vgs_list, Vds_fixed, verbose=True):
    """Ids-Vgs continuation: solve Vgs=0 first (cold start, the gentlest
    bias point - close to equilibrium, same reasoning as the diode's own
    Va=0 anchor), then walk outward in both directions warm-starting each
    point from its already-solved neighbor.

    A monotonic sweep starting cold at one extreme (e.g. Vgs=-0.5V
    directly) was found to reliably stall a few hundred mV in (a large,
    well-behaved-looking but ultimately stuck Newton trajectory - not
    divergence, just no basin of attraction reachable in one jump from a
    cold start that far from equilibrium). Warm-starting through small
    (~ the sweep's own step size) increments from a good starting point
    resolved every one of the same bias points cleanly - exactly the
    reasoning behind the diode's own bidirectional-from-Va=0 sweep,
    applied here since Vgs (like Va) naturally spans both sides of 0.

    A plain fixed-step walk (this driver's original approach - the fixed
    Vgs_list step size, whatever it is) was found to no longer reliably
    converge across most of the sweep once the Caughey-Thomas velocity-
    saturation mobility was added (see sweep_vds's own matching docstring
    note) - the field-dependent mobility couples psi into the continuity
    Jacobian in a new, sharper way that the fixed step size wasn't designed
    to resolve. Each requested Vgs step that fails to converge directly is
    now retried via the same adaptive bisection helper (`_bisecting_continuation`)
    sweep_vds uses, rather than accepting the stall."""
    Vgs_arr = np.asarray(sorted(Vgs_list), dtype=float)
    zero_idx = int(np.argmin(np.abs(Vgs_arr)))
    results = {}

    def solve_at(Vgs, state):
        t0 = time.perf_counter()
        if state is None:
            r = solve_mosfet(mesh, mat, dev, Cdop_substrate, float(Vgs), Vds_fixed)
        else:
            r = solve_mosfet(mesh, mat, dev, Cdop_substrate, float(Vgs), Vds_fixed,
                              psi_init=state["psi"], phin_init=state["phin"], phip_init=state["phip"])
        r["solve_time_s"] = time.perf_counter() - t0
        return r

    label_fmt = lambda v: f"Vgs={v:+.4f}V"

    def _solve(Vgs, prev, prev_Vgs, label):
        r, _ = _bisecting_continuation(solve_at, prev, prev_Vgs, float(Vgs),
                                        verbose=verbose, label_fmt=label_fmt)
        status = "OK" if r["res_norm"] < 1e-4 else "NOT CONVERGED"
        if verbose:
            print(f"  Vgs={Vgs:+.3f} V (Vds={Vds_fixed:.3f}V): iters={r['iters']:3d}, "
                  f"res_norm={r['res_norm']:.3e}, t={r['solve_time_s']:.4f}s  [{status}]  {label}")
        return r

    r0 = _solve(Vgs_arr[zero_idx], None, 0.0, "(anchor)")
    results[float(Vgs_arr[zero_idx])] = r0

    for idx_range in (range(zero_idx + 1, len(Vgs_arr)), range(zero_idx - 1, -1, -1)):
        prev = r0
        prev_Vgs = float(Vgs_arr[zero_idx])
        for idx in idx_range:
            Vgs = Vgs_arr[idx]
            r = _solve(Vgs, prev, prev_Vgs, "")
            results[float(Vgs)] = r
            prev = r
            prev_Vgs = float(Vgs)
    return results


def _bisecting_continuation(solve_at, prev_state, prev_param, target_param,
                             tol=1e-4, max_depth=8, verbose=True, label_fmt=None):
    """Generic adaptive-step continuation: try to solve directly at
    target_param warm-started from prev_state (converged at prev_param); if
    that fails to converge, bisect the interval (solve the midpoint first,
    itself recursively via the same bisection, then continue from the
    midpoint's converged state to target_param) - this project's standing
    "anchor-at-gentlest-point-and-warm-start-outward" continuation pattern,
    applied adaptively instead of a fixed step size so the (much smaller)
    step actually needed only near a hard nonlinearity (e.g. onset of
    velocity-saturation-driven pinch-off) is found automatically, without
    paying that finer resolution's cost everywhere else in the sweep.
    Gives up (returns the best/last attempt, still marked NOT CONVERGED)
    after max_depth halvings - a real, not infinite, retry budget.
    Returns (result_dict, list_of_(param, result_dict)_pairs_solved_along_the_way,
    sorted by param) so the caller can keep every intermediate point too."""
    r = solve_at(target_param, prev_state)
    if r["res_norm"] < tol or max_depth <= 0:
        if verbose and r["res_norm"] >= tol:
            print(f"    [continuation] giving up at {label_fmt(target_param) if label_fmt else target_param} "
                  f"(res_norm={r['res_norm']:.3e}) after exhausting bisection depth")
        return r, [(target_param, r)]

    mid = 0.5 * (prev_param + target_param)
    if verbose:
        print(f"    [continuation] {label_fmt(target_param) if label_fmt else target_param} did not converge "
              f"(res_norm={r['res_norm']:.3e}) - bisecting via {label_fmt(mid) if label_fmt else mid}")
    r_mid, points_mid = _bisecting_continuation(solve_at, prev_state, prev_param, mid,
                                                 tol, max_depth - 1, verbose, label_fmt)
    if r_mid["res_norm"] >= tol:
        # Even the midpoint failed after exhausting its own budget - no
        # point continuing further out from a state that isn't converged.
        return r_mid, points_mid
    r_final, points_final = _bisecting_continuation(solve_at, r_mid, mid, target_param,
                                                     tol, max_depth - 1, verbose, label_fmt)
    return r_final, points_mid + points_final


def sweep_vds(mesh, mat, dev, Cdop_substrate, Vgs_fixed, Vds_list, init_state=None, verbose=True):
    """Ids-Vds continuation: sequential warm-start walking up the sorted
    Vds list, with adaptive bisection (see `_bisecting_continuation`) at
    each requested Vds step that fails to converge directly.

    A cold start directly at (Vgs_fixed, Vds=0) was found to fail badly for
    some Vgs_fixed values (e.g. Vgs=1.5V stalled at |F|_inf~3.6e4) - the
    same "too big a jump from a flat guess" failure mode sweep_vgs hit,
    just here it's the GATE bias jumping straight to its target instead of
    ramping through it. `init_state`, if given (a converged result dict
    from a NEARBY bias, e.g. the transfer sweep's own point at the closest
    Vgs to Vgs_fixed - see main()), warm-starts the very first (Vds=0)
    point from that instead of a cold start; omitting it falls back to a
    cold start (only safe for a Vgs_fixed close to 0).

    A plain fixed 0.1V-per-step walk (this driver's original approach) was
    found to reliably diverge partway up EVERY Vds_fixed curve once the
    Caughey-Thomas velocity-saturation mobility (see
    solver2d/newton_solver_qf_2d.py::mobility_field) was added - the field-
    dependent mobility is a much sharper nonlinearity (mu drops steeply
    once E approaches vsat/mu0) than the Vds=0.1V step size was ever
    designed to resolve; a direct single-point check confirmed a much
    finer 0.02V step converges cleanly at the very same bias the 0.1V jump
    diverged at, hence adaptive bisection here rather than a blanket finer
    fixed step (which would slow down the whole sweep, not just the
    nonlinear part of it)."""
    Vds_arr = np.asarray(sorted(Vds_list), dtype=float)
    results = {}

    def solve_at(Vds, state):
        t0 = time.perf_counter()
        if state is not None:
            r = solve_mosfet(mesh, mat, dev, Cdop_substrate, Vgs_fixed, float(Vds),
                              psi_init=state["psi"], phin_init=state["phin"], phip_init=state["phip"])
        else:
            r = solve_mosfet(mesh, mat, dev, Cdop_substrate, Vgs_fixed, float(Vds))
        r["solve_time_s"] = time.perf_counter() - t0
        return r

    label_fmt = lambda v: f"Vds={v:+.4f}V"
    prev_state = init_state
    prev_Vds = float(Vds_arr[0]) if init_state is not None else 0.0
    for Vds in Vds_arr:
        r, extra_points = _bisecting_continuation(solve_at, prev_state, prev_Vds, float(Vds),
                                                   verbose=verbose, label_fmt=label_fmt)
        results[float(Vds)] = r
        status = "OK" if r["res_norm"] < 1e-4 else "NOT CONVERGED"
        if verbose:
            print(f"  Vds={Vds:+.3f} V (Vgs={Vgs_fixed:.3f}V): iters={r['iters']:3d}, "
                  f"res_norm={r['res_norm']:.3e}, t={r['solve_time_s']:.4f}s  [{status}]")
        prev_state = r
        prev_Vds = float(Vds)
    return results


def main():
    parser = argparse.ArgumentParser(description="tcad1d 2D planar NMOS driver")
    parser.add_argument("config", nargs="?", default=DEFAULT_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    domain, mat, dev, Cdop_substrate, mesh_opts, bias_cfg, output_cfg = build_from_config(cfg)

    mesa = domain.top_mesas[0]
    mesa_top_segment = ((mesa.x_range_cm[0], -mesa.height_cm), (mesa.x_range_cm[1], -mesa.height_cm))
    interface_segments = domain.junction_segments() + [mesa_top_segment]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = build_mesh2d(domain, mat=mat, interface_segments=interface_segments, **mesh_opts)
    print(f"mesh: {len(mesh.points)} points, {len(mesh.triangles)} triangles "
          f"({int(np.sum(mesh.is_insulator))} insulator points)")

    out_dir = os.path.join("out", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)

    Vgs_list = np.linspace(bias_cfg.get("vgs_start_V", -0.5), bias_cfg.get("vgs_stop_V", 2.0),
                            int(bias_cfg.get("vgs_points", 26)))
    Vds_transfer = float(bias_cfg.get("vds_for_transfer_V", 0.05))
    Vds_list = np.linspace(bias_cfg.get("vds_start_V", 0.0), bias_cfg.get("vds_stop_V", 2.0),
                            int(bias_cfg.get("vds_points", 21)))
    Vgs_for_output = [float(v) for v in bias_cfg.get("vgs_for_output_V", [1.0, 1.5, 2.0])]

    print(f"--- Ids-Vgs transfer sweep (Vds={Vds_transfer}V) ---")
    t0 = time.perf_counter()
    transfer_results = sweep_vgs(mesh, mat, dev, Cdop_substrate, Vgs_list, Vds_transfer)
    print(f"transfer sweep: total {time.perf_counter() - t0:.3f}s")
    Id_transfer = {Vgs: contact_current_density(mesh, mat, r, "drain") for Vgs, r in transfer_results.items()}

    transfer_csv = os.path.join(out_dir, "ids_vgs.csv")
    with open(transfer_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Vgs_V", "Ids_A_cm", "res_norm", "iters"])
        for Vgs in sorted(Id_transfer):
            r = transfer_results[Vgs]
            w.writerow([f"{Vgs:.4f}", f"{Id_transfer[Vgs]:.6e}", f"{r['res_norm']:.3e}", r["iters"]])
    print(f"wrote {transfer_csv}")

    # Ids-Vds output sweeps are SKIPPED for now (2026-09-13, per explicit
    # user direction: "I don't even care about Ids-Vds curve first" while
    # the doping-dependent-mobility Ids-Vgs basics are being sorted out -
    # do not spend time debugging output-curve convergence this round).
    # Set RUN_IDS_VDS=True below to re-enable.
    RUN_IDS_VDS = False
    output_results = {}
    Id_output = {}
    if RUN_IDS_VDS:
        print(f"--- Ids-Vds output sweeps (Vgs={Vgs_for_output}) ---")
        for Vgs in Vgs_for_output:
            print(f" Vgs={Vgs}V:")
            # Warm-start each output sweep's own Vds=0 point from the transfer
            # sweep's already-converged result at the closest Vgs (and a
            # nearby Vds=0.05V) instead of a cold start directly at the full
            # target Vgs - see sweep_vds's own docstring for why that cold
            # start failed.
            nearest_transfer_vgs = min(transfer_results, key=lambda v: abs(v - Vgs))
            init_state = transfer_results[nearest_transfer_vgs]
            output_results[Vgs] = sweep_vds(mesh, mat, dev, Cdop_substrate, Vgs, Vds_list, init_state=init_state)
        Id_output = {Vgs: {Vds: contact_current_density(mesh, mat, r, "drain") for Vds, r in res.items()}
                     for Vgs, res in output_results.items()}

        output_csv = os.path.join(out_dir, "ids_vds.csv")
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Vgs_V", "Vds_V", "Ids_A_cm", "res_norm", "iters"])
            for Vgs in Vgs_for_output:
                for Vds in sorted(Id_output[Vgs]):
                    r = output_results[Vgs][Vds]
                    w.writerow([f"{Vgs:.4f}", f"{Vds:.4f}", f"{Id_output[Vgs][Vds]:.6e}",
                                f"{r['res_norm']:.3e}", r["iters"]])
        print(f"wrote {output_csv}")

    # --- Transfer curve plot (linear + subthreshold semilog) ---
    Vgs_sorted = np.array(sorted(Id_transfer))
    Id_arr = np.array([Id_transfer[v] for v in Vgs_sorted])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(Vgs_sorted, Id_arr, "o-", color="#1f77b4")
    axes[0].set_xlabel("Vgs (V)"); axes[0].set_ylabel("Ids (A/cm)")
    axes[0].set_title(f"Ids-Vgs transfer (Vds={Vds_transfer}V)"); axes[0].grid(alpha=0.3)
    axes[1].semilogy(Vgs_sorted, np.abs(Id_arr) + 1e-30, "o-", color="#1f77b4")
    axes[1].set_xlabel("Vgs (V)"); axes[1].set_ylabel("|Ids| (A/cm)")
    axes[1].set_title("subthreshold semilog"); axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "ids_vgs.png")
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"wrote {out_path}")

    # --- Output curve plot (family of Vgs) - skipped along with the sweep
    # itself when RUN_IDS_VDS is False (see above). ---
    if RUN_IDS_VDS:
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.get_cmap("viridis")
        for i, Vgs in enumerate(Vgs_for_output):
            Vds_sorted = np.array(sorted(Id_output[Vgs]))
            Id_arr_v = np.array([Id_output[Vgs][v] for v in Vds_sorted])
            ax.plot(Vds_sorted, Id_arr_v, "o-", color=cmap(i / max(1, len(Vgs_for_output) - 1)),
                    label=f"Vgs={Vgs:.2f}V")
        ax.set_xlabel("Vds (V)"); ax.set_ylabel("Ids (A/cm)")
        ax.set_title("Ids-Vds output"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, "ids_vds.png")
        fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f"wrote {out_path}")

    # --- Structure+fields JSON (transfer sweep's own bias points) ---
    points_um = mesh.points * _CM_TO_UM
    save_spec = output_cfg.get("save_bias_points", "last")
    save_idx = resolve_save_points(save_spec, Vgs_sorted)
    regions = [
        {
            "name": r.name,
            "x_range_um": [r.x_range_cm[0] * _CM_TO_UM, r.x_range_cm[1] * _CM_TO_UM],
            "y_range_um": [r.y_range_cm[0] * _CM_TO_UM, r.y_range_cm[1] * _CM_TO_UM],
            "kind": r.kind,
            "doping_type": r.doping_type,
        }
        for r in domain.regions
    ]
    boundary = [
        {"point_index": int(i), "bc_type": bc_type}
        for i, bc_type in zip(mesh.boundary_point_index, mesh.boundary_bc_type)
    ]
    material_dict = dict(eps_r=mat.eps_r, ni=mat.ni, mu_n=mat.mu_n, mu_p=mat.mu_p,
                          tau_n=mat.tau_n, tau_p=mat.tau_p, chi_eV=mat.chi_eV, Eg_eV=mat.Eg_eV)
    bias_points = []
    for idx in save_idx:
        Vgs = Vgs_sorted[idx]
        r = transfer_results[Vgs]
        Ex, Ey = electric_field_2d(mesh.points, mesh.triangles, r["psi"])
        fields = {k: r[k] for k in ("psi", "n", "p", "phin", "phip")}
        fields["Ex"] = Ex; fields["Ey"] = Ey
        bias_points.append({"label": f"Vgs={Vgs:+.3f}V (Vds={Vds_transfer}V)", "bias": float(Vgs), "fields": fields})
    doc = sio.build_structure(
        device="mosfet2d", material=material_dict, regions=regions,
        x_um=points_um[:, 0], y_um=points_um[:, 1], doping_cm3=mesh.Cdop,
        bias_points=bias_points, dim=2,
        mesh2d={"triangles": mesh.triangles.tolist(), "boundary": boundary},
    )
    structure_file = output_cfg.get("structure_file")
    if structure_file:
        struct_path = os.path.join(out_dir, structure_file)
        sio.write_structure(struct_path, doc)
        print(f"wrote {struct_path} ({len(bias_points)} bias point(s) saved)")


if __name__ == "__main__":
    main()
