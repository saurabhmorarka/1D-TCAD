"""Loads input_diode.yaml and builds Material/Device/voltage-sweep/solver-
choice/mesh overrides from it. Every value in the input file overrides the
corresponding default from params.py; the input file itself is optional
(defaults are used for anything missing or if the file doesn't exist)."""
import os

import numpy as np
import yaml

from params import Material, Device
import doping_profiles as dp

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "input_diode.yaml")


def load_config(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_from_config(cfg: dict):
    """Returns (Material, Device, Va_array, math_model, save_bias_points, mesh_opts, structure_file)."""
    mat = Material()
    dev = Device()

    doping = cfg.get("doping") or {}
    dev.p_profile = dp.parse_doping_profile(doping.get("p_side") or {}, dev.Na)
    dev.n_profile = dp.parse_doping_profile(doping.get("n_side") or {}, dev.Nd)
    dev.Na = dev.p_profile.reference_concentration()
    dev.Nd = dev.n_profile.reference_concentration()

    thickness = cfg.get("thickness") or {}
    if thickness.get("Wp_um") is not None:
        dev.Wp = float(thickness["Wp_um"]) * 1e-4
    if thickness.get("Wn_um") is not None:
        dev.Wn = float(thickness["Wn_um"]) * 1e-4

    material = cfg.get("material") or {}
    if material.get("eps_r") is not None:
        mat.eps_r = float(material["eps_r"])
    if material.get("ni_cm3") is not None:
        mat.ni = float(material["ni_cm3"])

    mesh_cfg = cfg.get("mesh") or {}
    mesh_opts = dict(
        growth=float(mesh_cfg.get("growth", 1.06)),
        bulk_spacing_debye_factor=float(mesh_cfg.get("bulk_spacing_debye_factor", 5.0)),
        junction_spacing_debye_factor=float(mesh_cfg.get("junction_spacing_debye_factor", 0.05)),
    )

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

    output_cfg = cfg.get("output", {})
    save_bias_points = output_cfg.get("save_bias_points", "last")
    structure_file = output_cfg.get("structure_file", "diode_structure.json")

    return mat, dev, Va_list, math_model, save_bias_points, mesh_opts, structure_file
