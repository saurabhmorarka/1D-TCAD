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
    transition_um: float = None       # linear only: if given, the ramp from
                                       # start_cm3 to end_cm3 completes within
                                       # this depth (not the full region
                                       # thickness), then holds flat at
                                       # end_cm3 beyond - e.g. a lightly-doped
                                       # junction interface grading into a
                                       # heavily-doped bulk/contact region a
                                       # short distance away. None (default)
                                       # keeps the original behavior: ramp
                                       # spans the whole region thickness.
    log_ramp: bool = False             # linear only: ramp linearly in
                                       # log10(concentration) instead of in
                                       # concentration itself - the physically
                                       # standard shape for grading between two
                                       # doping levels that differ by orders of
                                       # magnitude (e.g. a retrograde implant).
                                       # A straight linear-in-N ramp between
                                       # very different start/end values is
                                       # badly front-loaded (jumps most of the
                                       # way to the larger value within the
                                       # first few percent of the transition,
                                       # since the ramp is dominated by
                                       # whichever endpoint has the larger
                                       # magnitude); log_ramp spreads the
                                       # change evenly in decades instead.
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
            ramp_depth_cm = (self.transition_um * 1e-4
                              if self.transition_um is not None else thickness_cm)
            frac = np.clip(depth_cm / ramp_depth_cm, 0.0, 1.0)
            if self.log_ramp:
                log_start, log_end = np.log10(self.start_cm3), np.log10(self.end_cm3)
                return 10.0 ** (log_start + (log_end - log_start) * frac)
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
        treat it as approximate whenever a non-flat profile is in use.

        For a linear profile with transition_um set, the ramp only occupies
        a small fraction of the region (that's the whole point - see its
        field docstring), so the arithmetic mean of start/end would badly
        misrepresent the region: end_cm3 is what almost all of the region's
        volume - and its "quasi-neutral bulk" physics - actually sits at,
        so that is the representative value instead. Without transition_um,
        the ramp spans the full region and the original midpoint average is
        kept (unchanged behavior)."""
        if self.type == "flat":
            return self.concentration_cm3
        if self.type == "linear":
            if self.transition_um is not None:
                return self.end_cm3
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
                              start_cm3=float(cfg["start_cm3"]), end_cm3=float(cfg["end_cm3"]),
                              transition_um=(float(cfg["transition_um"])
                                             if cfg.get("transition_um") is not None else None),
                              log_ramp=bool(cfg.get("log_ramp", False)))
    if ptype == "gaussian":
        return DopingProfile(type="gaussian",
                              peak_cm3=float(cfg["peak_cm3"]),
                              peak_depth_um=float(cfg.get("peak_depth_um", 0.0)),
                              straggle_um=float(cfg["straggle_um"]),
                              background_cm3=float(cfg.get("background_cm3", 0.0)))
    raise ValueError(f"doping.type must be 'flat', 'linear', or 'gaussian', got {ptype!r}")
