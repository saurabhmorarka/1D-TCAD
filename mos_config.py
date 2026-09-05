"""Loads input_mos.yaml and builds Material/MOSDevice/voltage-sweep/mesh
overrides from it - the MOS-capacitor analog of config.py."""
import os

import numpy as np
import yaml

from params import Material
from mos_params import MOSDevice
import doping_profiles as dp

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "input_mos.yaml")


def load_config(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_from_config(cfg: dict):
    """Returns (Material, MOSDevice, Cdop_substrate, VG_array, save_bias_points, mesh_opts, structure_file)."""
    mat = Material()
    dev = MOSDevice()

    sub = cfg.get("substrate") or {}
    polarity = sub.get("polarity", "p")
    if polarity not in ("p", "n"):
        raise ValueError(f"substrate.polarity must be 'p' or 'n', got {polarity!r}")
    dev.substrate_profile = dp.parse_doping_profile(sub.get("doping") or {}, dev.Na)
    dev.Na = dev.substrate_profile.reference_concentration()
    Cdop_substrate = -dev.Na if polarity == "p" else dev.Na

    oxide = cfg.get("oxide") or {}
    if "thickness_nm" in oxide:
        dev.t_ox = float(oxide["thickness_nm"]) * 1e-7
    if oxide.get("eps_r") is not None:
        dev.eps_ox_r = float(oxide["eps_r"])

    gate = cfg.get("gate") or {}
    dev.gate_workfunction_eV = gate.get("workfunction_eV", None)

    mesh_cfg = cfg.get("mesh") or {}
    mesh_opts = dict(
        n_ox_points=int(mesh_cfg.get("n_ox_points", 15)),
        growth=float(mesh_cfg.get("growth", 1.08)),
        bulk_spacing_debye_factor=float(mesh_cfg.get("bulk_spacing_debye_factor", 5.0)),
        interface_spacing_debye_factor=float(mesh_cfg.get("interface_spacing_debye_factor", 0.001)),
    )

    vs = cfg.get("voltage_sweep", {})
    VG_list = np.linspace(vs.get("start_V", -0.5), vs.get("stop_V", 1.0), int(vs.get("points", 61)))

    output_cfg = cfg.get("output", {})
    save_bias_points = output_cfg.get("save_bias_points", "last")
    structure_file = output_cfg.get("structure_file", "mos_structure.json")

    return mat, dev, Cdop_substrate, VG_list, save_bias_points, mesh_opts, structure_file
