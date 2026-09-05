"""Structure+fields file format: schema, save_structure(), load_structure().

Written as one small JSON file per device run (out/<device>_structure.json),
alongside the existing plot PNGs and per-bias CSVs - it's additive, not a
replacement for either. It exists so plot.py can make textbook-style
structure/band/charge plots from a form a human can open and read directly,
and so a future 2D/3D version of this project can extend it without
changing the shape of any existing key.

Schema (schema_version=1, dim=1):
  schema_version : int
  dim            : number of spatial dimensions the grid/regions describe.
                   Only dim=1 is implemented by plot.py today; a 2D/3D
                   structure would add "y_um"/"z_um" grid keys and give
                   regions polygon/volume keys instead of x_range_um -
                   nothing else in this schema has to change for that.
  device         : "diode" | "mos"
  material       : dict of material constants. chi_eV/Eg_eV (electron
                   affinity, bandgap) may each be a single float (uniform
                   material) or a per-node list (e.g. MOS oxide+substrate)
                   - plot.py broadcasts either.
  regions        : list of {"name", "x_range_um": [x0, x1], "kind":
                   "semiconductor"|"insulator", "doping_type": "p"|"n"|None}
                   describing the device's geometry for the structure plot.
  grid           : {"x_um": [...]} node positions.
  doping_cm3     : net doping (Nd-Na) at each node, cm^-3.
  bias_points    : list of {"label", "bias", "regime" (optional, e.g.
                   "accumulation"/"depletion"/"inversion"), "fields": {
                   "psi": [...], "n": [...], "p": [...], "phin": [...],
                   "phip": [...]}} - fields are per-node, same length as
                   grid.x_um. NaN (e.g. phin/phip inside an oxide, where
                   they're undefined) is written as JSON null.
"""
import json

import numpy as np

SCHEMA_VERSION = 1


def _sig(v, ndigits=6):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(f"{v:.{ndigits}g}")


def _sig_list(arr, ndigits=6):
    return [_sig(v, ndigits) for v in np.asarray(arr, dtype=float).tolist()]


def save_structure(path, *, device, material, regions, x_um, doping_cm3, bias_points, dim=1):
    """Write the structure+fields JSON. See the module docstring for the
    schema; `bias_points` items are {"label", "bias", "fields": {...}} with
    an optional "regime" key, and `regions` items are {"name",
    "x_range_um", "kind", "doping_type"}."""
    material_out = {}
    for k, v in material.items():
        material_out[k] = _sig_list(v, 8) if np.ndim(v) > 0 else _sig(v, 8)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "dim": dim,
        "device": device,
        "material": material_out,
        "regions": regions,
        "grid": {"x_um": _sig_list(x_um)},
        "doping_cm3": _sig_list(doping_cm3),
        "bias_points": [
            {
                "label": bp["label"],
                "bias": _sig(bp["bias"], 8),
                **({"regime": bp["regime"]} if bp.get("regime") else {}),
                "fields": {k: _sig_list(v, 8) for k, v in bp["fields"].items()},
            }
            for bp in bias_points
        ],
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def load_structure(path):
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {doc.get('schema_version')!r} "
                          f"(this tool understands {SCHEMA_VERSION})")
    return doc
