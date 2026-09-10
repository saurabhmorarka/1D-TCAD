"""Shared helper for saving/plotting full spatial field profiles (potential,
carrier densities, quasi-Fermi potentials) at a configurable subset of the
swept bias points, used by both main.py (diode) and mos_main.py (MOS
capacitor).

A bias sweep's `results` list already holds the full psi/n/p/phin/phip
arrays for every point in memory (nothing is discarded during the sweep) -
this module just controls which of those get written to disk/plotted, since
saving every point's full spatial profile for a long/fine sweep would be
wasteful, but the user should be able to ask for exactly that when useful.
"""
import csv

import numpy as np


def resolve_save_points(spec, bias_arr: np.ndarray):
    """spec: a list of bias values (nearest-match), or the string "all", or
    the string "last". Returns the indices into bias_arr to save."""
    if spec is None:
        spec = "last"
    if isinstance(spec, str):
        if spec == "all":
            return list(range(len(bias_arr)))
        if spec == "last":
            return [len(bias_arr) - 1]
        raise ValueError(f"output.save_bias_points string must be 'all' or 'last', got {spec!r}")
    return sorted({int(np.argmin(np.abs(bias_arr - v))) for v in spec})


def save_fields(results, indices, bias_key, out_csv_path, x, is_oxide=None):
    """Write a tidy CSV of (bias, x, psi, n, p, phin, phip) for the selected
    result indices - one row per (bias point, mesh node)."""
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [bias_key, "x_cm", "psi_V", "n_cm-3", "p_cm-3", "phin_V", "phip_V"]
        if is_oxide is not None:
            header.append("is_oxide")
        w.writerow(header)
        for i in indices:
            r = results[i]
            for j in range(len(x)):
                row = [f"{r[bias_key]:.6g}", f"{x[j]:.6e}", f"{r['psi'][j]:.6f}",
                       f"{r['n'][j]:.6e}", f"{r['p'][j]:.6e}",
                       f"{r['phin'][j]:.6f}", f"{r['phip'][j]:.6f}"]
                if is_oxide is not None:
                    row.append(int(is_oxide[j]))
                w.writerow(row)


def plot_quasi_fermi(ax, results, indices, bias_key, x_um, bias_unit="V", bias_label="Va",
                      colors=None):
    """Plot phin(x) (solid) and phip(x) (dashed) for the selected bias
    points on one axes, one color per bias point."""
    if colors is None:
        cmap = __import__("matplotlib.pyplot", fromlist=["cm"]).cm.viridis
        colors = [cmap(t) for t in np.linspace(0.15, 0.9, max(len(indices), 1))]
    for c, i in zip(colors, indices):
        r = results[i]
        label = f"{bias_label}={r[bias_key]:.3f} {bias_unit}"
        ax.plot(x_um, r["phin"], color=c, lw=1.8, ls="-", label=f"phin, {label}")
        ax.plot(x_um, r["phip"], color=c, lw=1.8, ls="--", label=f"phip, {label}")
