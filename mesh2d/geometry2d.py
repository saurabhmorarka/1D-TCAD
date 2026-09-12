"""Blocky 2D device geometry: axis-aligned rectangular regions with a
painter's-algorithm override order (a later region overrides an earlier one
wherever they overlap - this is how a p+ square gets "dug into" an n
substrate rectangle), plus contact placement on the top/bottom domain
boundary.

All lengths are stored in cm (matching core/params.py's/core/mesh.py's
convention) - only the YAML config layer (mesh2d/config2d.py) works in um.

Domain convention: x in [0, width_cm], y in [0, height_cm] with y=0 the top
(free) surface and y=height_cm the bottom of the substrate. Left (x=0) and
right (x=width_cm) are always the `symmetry` boundary - see the module
docstring in mesh2d/boundary.py for why a side is never a contact surface.
Shape vocabulary is rectangles-only for now; tapered/other shapes are
explicitly future work (see the tcad1d 2D/3D plan).
"""
from dataclasses import dataclass, field

import numpy as np

_EPS_REL = 1e-9  # relative tolerance for "on the rectangle boundary" tests


@dataclass
class Region:
    name: str
    x_range_cm: tuple      # (x0, x1)
    y_range_cm: tuple      # (y0, y1)
    doping_type: str       # "p" | "n"
    concentration_cm3: float
    kind: str = "semiconductor"   # "semiconductor" | "insulator" (insulator unused until MOS-in-2D)


@dataclass
class Contact:
    name: str
    surface: str            # "top" | "bottom"
    x_range_cm: tuple
    bias_role: str           # "anode" | "cathode"


@dataclass
class Domain2D:
    width_cm: float
    height_cm: float
    regions: list = field(default_factory=list)   # index 0 = base region (e.g. substrate)
    contacts: list = field(default_factory=list)

    def doping_at(self, x_cm, y_cm):
        """Signed net doping (Nd - Na, cm^-3) at point(s) (x_cm, y_cm) via
        last-region-wins rectangle membership. Points not covered by any
        region (shouldn't happen if region[0] spans the whole domain) get 0."""
        x = np.asarray(x_cm, dtype=float)
        y = np.asarray(y_cm, dtype=float)
        net = np.zeros_like(x)
        tol_x = _EPS_REL * max(self.width_cm, 1e-30)
        tol_y = _EPS_REL * max(self.height_cm, 1e-30)
        for region in self.regions:
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            mask = ((x >= x0 - tol_x) & (x <= x1 + tol_x)
                    & (y >= y0 - tol_y) & (y <= y1 + tol_y))
            sign = 1.0 if region.doping_type == "n" else -1.0
            net = np.where(mask, sign * region.concentration_cm3, net)
        return net

    def junction_segments(self):
        """Line segments ((x0,y0),(x1,y1)) forming the metallurgical-junction
        boundary of every non-base region, excluding the parts that coincide
        with the domain's own outer boundary (those are a free surface / a
        contact, not a doping-type junction). Used to drive point-cloud
        refinement in mesh2d/pointcloud.py."""
        segs = []
        for region in self.regions[1:]:
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            candidates = [
                ((x0, y0), (x1, y0)),  # top edge of the region
                ((x0, y1), (x1, y1)),  # bottom edge
                ((x0, y0), (x0, y1)),  # left edge
                ((x1, y0), (x1, y1)),  # right edge
            ]
            for (p0, p1) in candidates:
                if self._on_domain_boundary(p0) and self._on_domain_boundary(p1):
                    continue
                segs.append((p0, p1))
        return segs

    def _on_domain_boundary(self, p):
        x, y = p
        tol_x = _EPS_REL * max(self.width_cm, 1e-30)
        tol_y = _EPS_REL * max(self.height_cm, 1e-30)
        return (abs(x) <= tol_x or abs(x - self.width_cm) <= tol_x
                or abs(y) <= tol_y or abs(y - self.height_cm) <= tol_y)

    def region_corners(self):
        """All rectangle corner points (domain + every region), used to make
        sure the point cloud has vertices exactly on every geometric feature
        rather than relying on quadtree refinement alone to land near one."""
        pts = [(0.0, 0.0), (self.width_cm, 0.0),
               (0.0, self.height_cm), (self.width_cm, self.height_cm)]
        for region in self.regions:
            x0, x1 = region.x_range_cm
            y0, y1 = region.y_range_cm
            pts += [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        for contact in self.contacts:
            x0, x1 = contact.x_range_cm
            y = 0.0 if contact.surface == "top" else self.height_cm
            pts += [(x0, y), (x1, y)]
        return pts
