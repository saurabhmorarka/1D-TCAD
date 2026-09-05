"""Loads input_mos.yaml and builds Material/MOSDevice/voltage-sweep
overrides from it - the MOS-capacitor analog of config.py."""
import os

import numpy as np
import yaml

from params import Material
from mos_params import MOSDevice

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "input_mos.yaml")


def load_config(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_from_config(cfg: dict):
    """Returns (Material, MOSDevice, Cdop_substrate, VG_array, save_bias_points)."""
    mat = Material()
    dev = MOSDevice()

    sub = cfg.get("substrate", {})
    sub_type = sub.get("type", "p")
    doping = float(sub.get("doping_cm3", dev.Na))
    dev.Na = doping
    if sub_type not in ("p", "n"):
        raise ValueError(f"substrate.type must be 'p' or 'n', got {sub_type!r}")
    Cdop_substrate = -doping if sub_type == "p" else doping

    oxide = cfg.get("oxide", {})
    if "thickness_nm" in oxide:
        dev.t_ox = float(oxide["thickness_nm"]) * 1e-7

    gate = cfg.get("gate", {})
    dev.gate_workfunction_eV = gate.get("workfunction_eV", None)

    vs = cfg.get("voltage_sweep", {})
    VG_list = np.linspace(vs.get("start_V", -0.5), vs.get("stop_V", 1.0), int(vs.get("points", 61)))

    save_bias_points = cfg.get("output", {}).get("save_bias_points", "last")

    return mat, dev, Cdop_substrate, VG_list, save_bias_points
