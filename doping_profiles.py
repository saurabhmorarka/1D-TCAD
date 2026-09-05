"""Doping-profile shapes for a device region: flat (uniform), linear
(graded), or gaussian (implant-like). Shared between the diode (p-side/
n-side) and the MOS capacitor (substrate) so both input files use the same
schema and both mesh builders use the same sampling/mesh-sizing logic.

Each profile lives entirely inside one region, in that region's own local
depth coordinate: depth=0 at the region's reference edge (the metallurgical
junction for a diode's p/n side, the oxide/substrate interface for a MOS
substrate) and depth=thickness at its far edge. A profile does not straddle
a layer boundary - regions still meet at a sharp physical interface, only
the doping *within* each region can be graded.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class DopingProfile:
    type: str = "flat"                # "flat" | "linear" | "gaussian"
    concentration_cm3: float = None   # flat
    start_cm3: float = None           # linear: concentration at depth=0
    end_cm3: float = None             # linear: concentration at depth=thickness
    peak_cm3: float = None            # gaussian: peak concentration
    peak_depth_um: float = 0.0        # gaussian: depth of the peak, from depth=0
    straggle_um: float = None         # gaussian: standard deviation ("straggle")
    background_cm3: float = 0.0       # gaussian: floor/background concentration

    def sample(self, depth_cm, thickness_cm):
        """Unsigned concentration (cm^-3) at depth_cm (scalar or array),
        depth=0 at the region's reference edge."""
        depth_cm = np.asarray(depth_cm, dtype=float)
        if self.type == "flat":
            return np.full_like(depth_cm, self.concentration_cm3)
        if self.type == "linear":
            frac = np.clip(depth_cm / thickness_cm, 0.0, 1.0)
            return self.start_cm3 + (self.end_cm3 - self.start_cm3) * frac
        if self.type == "gaussian":
            peak_depth_cm = self.peak_depth_um * 1e-4
            straggle_cm = self.straggle_um * 1e-4
            return self.background_cm3 + self.peak_cm3 * np.exp(
                -0.5 * ((depth_cm - peak_depth_cm) / straggle_cm) ** 2
            )
        raise ValueError(f"doping type must be 'flat', 'linear', or 'gaussian', got {self.type!r}")

    def reference_concentration(self) -> float:
        """A single representative concentration (cm^-3), used everywhere a
        closed-form/analytic formula wants one number (Vbi, depletion width,
        Shockley I0, V_FB, V_T, ...): exact for a flat profile, an
        approximation (peak, for gaussian; average, for linear) otherwise.
        Every analytic comparison in this project assumes uniform doping, so
        treat it as approximate whenever a non-flat profile is in use."""
        if self.type == "flat":
            return self.concentration_cm3
        if self.type == "linear":
            return 0.5 * (self.start_cm3 + self.end_cm3)
        if self.type == "gaussian":
            return self.peak_cm3 + self.background_cm3
        raise ValueError(f"doping type must be 'flat', 'linear', or 'gaussian', got {self.type!r}")

    @staticmethod
    def flat(concentration_cm3: float) -> "DopingProfile":
        return DopingProfile(type="flat", concentration_cm3=concentration_cm3)


def parse_doping_profile(cfg: dict, default_flat_cm3: float) -> "DopingProfile":
    """Parse a `doping: {...}` YAML block into a DopingProfile. An empty/
    missing block falls back to flat at default_flat_cm3 (the params.py
    default), keeping every existing input file valid with no changes."""
    if not cfg:
        return DopingProfile.flat(default_flat_cm3)
    ptype = cfg.get("type", "flat")
    if ptype == "flat":
        return DopingProfile(type="flat",
                              concentration_cm3=float(cfg.get("concentration_cm3", default_flat_cm3)))
    if ptype == "linear":
        return DopingProfile(type="linear",
                              start_cm3=float(cfg["start_cm3"]), end_cm3=float(cfg["end_cm3"]))
    if ptype == "gaussian":
        return DopingProfile(type="gaussian",
                              peak_cm3=float(cfg["peak_cm3"]),
                              peak_depth_um=float(cfg.get("peak_depth_um", 0.0)),
                              straggle_um=float(cfg["straggle_um"]),
                              background_cm3=float(cfg.get("background_cm3", 0.0)))
    raise ValueError(f"doping.type must be 'flat', 'linear', or 'gaussian', got {ptype!r}")
