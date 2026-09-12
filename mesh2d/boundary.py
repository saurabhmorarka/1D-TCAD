"""Boundary-condition tagging for a 2D point cloud, and the normal/tangential
geometric utility used to integrate contact current and (later) decompose
flux at a material interface.

Every mesh point that lies on the domain's outer rectangle gets exactly one
BC tag:
  - "contact:<name>"  - an ideal ohmic Dirichlet contact (ideal-ohmic
    mass-action/charge-neutrality relation, same as core/solver.py's
    contact_values - the local relation is dimension-agnostic).
  - "symmetry"        - left/right sides. The domain is a truncated slice of
    a structure that repeats sideways, so this is where we chose to cut, not
    a real device edge: zero normal field and zero normal current by
    construction, forever. No future physics (surface recombination,
    interface charge, a contact) should ever attach to a `symmetry` point.
  - "free_surface"    - top/bottom points not covered by a contact: a real
    physical boundary (semiconductor meeting air/vacuum). With no surface
    charge modeled yet, normal-D continuity across the near-zero-
    permittivity air gap gives the same zero-normal-field/zero-normal-
    current equations as `symmetry` today, so the two tags are numerically
    identical for now - but this is the seam future physics (surface
    recombination velocity, interface Qit, a partial-top gate contact for
    MOS-in-2D) attaches to, and only to. Keeping the tag distinct from
    `symmetry` means that future work can't accidentally change symmetry
    behavior.

Tag priority at a corner point (contact > symmetry > free_surface): a
contact is checked first because a contact spanning the full top/bottom
width (as in the first diode example) should claim its corners; failing
that, being on the left/right domain edge always means `symmetry`.
"""
from dataclasses import dataclass

import numpy as np

_EPS_REL = 1e-9


@dataclass
class BoundaryTags:
    point_index: np.ndarray   # int array, indices into the mesh's point array
    bc_type: list              # str per boundary point: "contact:<name>" | "symmetry" | "free_surface"


def tag_boundary_points(points_cm, domain):
    """points_cm: (N,2) array. Returns BoundaryTags for every point lying on
    the domain's outer rectangle (interior points are not included)."""
    x, y = points_cm[:, 0], points_cm[:, 1]
    tol_x = _EPS_REL * max(domain.width_cm, 1e-30)
    tol_y = _EPS_REL * max(domain.height_cm, 1e-30)

    on_left = np.abs(x) <= tol_x
    on_right = np.abs(x - domain.width_cm) <= tol_x
    on_top = np.abs(y) <= tol_y
    on_bottom = np.abs(y - domain.height_cm) <= tol_y
    on_boundary = on_left | on_right | on_top | on_bottom

    idx = np.nonzero(on_boundary)[0]
    tags = []
    for i in idx:
        tag = None
        if on_top[i] or on_bottom[i]:
            surface = "top" if on_top[i] else "bottom"
            for contact in domain.contacts:
                if contact.surface != surface:
                    continue
                x0, x1 = contact.x_range_cm
                if x0 - tol_x <= x[i] <= x1 + tol_x:
                    tag = f"contact:{contact.name}"
                    break
        if tag is None and (on_left[i] or on_right[i]):
            tag = "symmetry"
        if tag is None:
            tag = "free_surface"
        tags.append(tag)
    return BoundaryTags(point_index=idx, bc_type=tags)


def outward_normal(p0, p1, domain):
    """Outward unit normal of a boundary segment (p0, p1) whose two endpoints
    lie on the domain's outer rectangle. Picks whichever of the two
    perpendiculars to the segment points away from the domain's centroid -
    correct for the convex rectangular domains this module builds; a future
    non-convex/irregular-shape domain would need a per-segment membership
    test instead of a single global centroid."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    edge = p1 - p0
    n1 = np.array([edge[1], -edge[0]])
    norm = np.linalg.norm(n1)
    if norm == 0.0:
        raise ValueError("outward_normal: degenerate segment (p0 == p1)")
    n1 = n1 / norm
    mid = 0.5 * (p0 + p1)
    center = np.array([domain.width_cm / 2.0, domain.height_cm / 2.0])
    return n1 if np.dot(n1, mid - center) > 0 else -n1


def decompose_normal_tangential(vec, normal):
    """Split a 2D vector `vec` into (tangential_vector, normal_component)
    with respect to a (not necessarily unit) `normal`. General-purpose: used
    today for contact current integration (normal component only) and kept
    general for future interface tangential-flux decomposition."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    vec = np.asarray(vec, dtype=float)
    vn = float(np.dot(vec, normal))
    tangential = vec - vn * normal
    return tangential, vn
