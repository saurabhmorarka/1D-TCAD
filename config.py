"""Loads input.yaml and builds Material/Device/voltage-sweep/solver-choice
overrides from it. Every value in the input file overrides the
corresponding default from params.py; the input file itself is optional
(defaults are used for anything missing or if the file doesn't exist)."""
import os

import numpy as np
import yaml

from params import Material, Device

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "input.yaml")


def load_config(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_from_config(cfg: dict):
    """Returns (Material, Device, Va_array, math_model)."""
    mat = Material()
    dev = Device()

    doping = cfg.get("doping", {})
    if "Na_cm3" in doping:
        dev.Na = float(doping["Na_cm3"])
    if "Nd_cm3" in doping:
        dev.Nd = float(doping["Nd_cm3"])

    thickness = cfg.get("thickness", {})
    if thickness.get("Wp_um") is not None:
        dev.Wp = float(thickness["Wp_um"]) * 1e-4
    if thickness.get("Wn_um") is not None:
        dev.Wn = float(thickness["Wn_um"]) * 1e-4

    vs = cfg.get("voltage_sweep", {})
    rev = np.linspace(
        vs.get("reverse_start_V", -2.0),
        vs.get("reverse_stop_V", -0.05),
        int(vs.get("reverse_points", 10)),
    )
    fwd = np.linspace(
        vs.get("forward_start_V", 0.0),
        vs.get("forward_stop_V", 0.65),
        int(vs.get("forward_points", 27)),
    )
    Va_list = np.concatenate([rev, fwd])

    math_model = cfg.get("solver", {}).get("math_model", "gummel")
    if math_model not in ("gummel", "newton"):
        raise ValueError(f"solver.math_model must be 'gummel' or 'newton', got {math_model!r}")

    save_bias_points = cfg.get("output", {}).get("save_bias_points", "last")

    return mat, dev, Va_list, math_model, save_bias_points
