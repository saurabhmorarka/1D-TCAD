"""Top-level 2D mesh builder: combines mesh2d/geometry2d.py (blocky regions/
contacts), mesh2d/pointcloud.py (triangle-library constrained quality
triangulation), mesh2d/fvgeometry.py (Voronoi-box FV geometry),
mesh2d/mesh_quality.py (mandatory quality gate) and mesh2d/boundary.py (BC
tagging) into one mesh object - the 2D analog of
core/mesh.py::build_diode_grid, but device-generic (build_mesh2d takes any
Domain2D, not a diode specifically) since the quality gate and FV assembly
have nothing diode-specific about them and every future 2D device (MOS,
etc.) should get the same mandatory checks for free by going through this
one function."""
import warnings
from dataclasses import dataclass

import numpy as np

from mesh2d.boundary import tag_boundary_points
from mesh2d.fvgeometry import build_fv_geometry
from mesh2d.mesh_quality import check_mesh_quality
from mesh2d.pointcloud import build_point_cloud


@dataclass
class Mesh2D:
    domain: object                # geometry2d.Domain2D
    points: np.ndarray             # (N,2) cm
    triangles: np.ndarray          # (M,3) int
    edges: np.ndarray              # (E,2) int, internal edges only
    edge_weight: np.ndarray        # (E,) float
    facet_length: np.ndarray       # (E,) float, cm
    cv_area: np.ndarray            # (N,) cm^2
    boundary_edges: set
    n_negative_subareas: int
    n_floored: int
    quality_report: dict           # see mesh2d/mesh_quality.py::mesh_quality_report
    Cdop: np.ndarray               # (N,) cm^-3, signed net doping per point
    boundary_point_index: np.ndarray   # int, indices into points/Cdop
    boundary_bc_type: list             # str per boundary point
    h_min_cm: float                # the mesh generator's own intended nominal minimum
                                     # spacing - NOT the same as the smallest actual edge
                                     # length; solvers should use this, not a raw
                                     # np.min(edge_lengths), as their normalization
                                     # length scale (see newton_solver_qf_2d.py)


def _drop_isolated_points(points, triangles, cv_area_floor, max_passes=3):
    """A rectangle corner's triangle can occasionally end up with its only
    non-boundary edge not shared by any neighboring triangle - a point with
    ZERO internal edges gets no Poisson/continuity coupling to the rest of
    the system at all, making its own 3x3 local Jacobian block exactly
    singular (confirmed directly: scipy.sparse.linalg.spsolve raised
    MatrixRankWarning: Matrix is exactly singular with such a point
    present). These points carry no physically meaningful area anyway, so
    the fix is to drop them and rebuild the FV geometry from the remaining
    triangles (iterating in case that ever isolates another point)."""
    for _ in range(max_passes):
        fv = build_fv_geometry(points, triangles=triangles, cv_area_floor=cv_area_floor)
        deg = np.zeros(len(points), dtype=int)
        np.add.at(deg, fv.edges[:, 0], 1)
        np.add.at(deg, fv.edges[:, 1], 1)
        isolated = deg == 0
        if not np.any(isolated):
            return fv
        keep = ~isolated
        new_index = np.cumsum(keep) - 1
        points = points[keep]
        tri_keep = ~np.any(isolated[triangles], axis=1)
        triangles = new_index[triangles[tri_keep]]
    raise RuntimeError(
        f"mesh2d: {np.sum(isolated)} isolated point(s) remained after {max_passes} "
        "drop-and-rebuild passes - a persistent degeneracy, not a one-off corner quirk; "
        "needs investigation rather than dropping further points.")


def build_mesh2d(domain, h_min_cm, h_max_cm, growth=1.3, cv_area_floor_factor=0.1,
                  min_angle_deg=32, interface_segments=None):
    """Build a Mesh2D for any Domain2D (blocky regions/contacts) - the
    single entry point every 2D device driver should go through, so the
    mandatory quality gate (mesh2d/mesh_quality.py) and FV-robustness
    regularizations (cv_area_floor, isolated-point removal) apply to every
    kind of run, not just the diode example that first exercised them.

    `interface_segments`, if given, overrides the default grading target
    (domain.junction_segments(), i.e. every doping-type boundary) - e.g. a
    future oxide/semiconductor interface that should drive its own tight-
    near/relaxed-away mesh grading independent of doping-junction geometry.

    cv_area_floor_factor sets the box-method control-volume area floor
    (mesh2d/fvgeometry.py::build_fv_geometry) as a fraction of h_min_cm^2."""
    points, triangles = build_point_cloud(
        domain, h_min_cm, h_max_cm, growth=growth, min_angle_deg=min_angle_deg,
        interface_segments=interface_segments)

    quality_report = check_mesh_quality(points, triangles)

    fv = _drop_isolated_points(points, triangles, cv_area_floor=cv_area_floor_factor * h_min_cm ** 2)
    Cdop = domain.doping_at(fv.points[:, 0], fv.points[:, 1])
    tags = tag_boundary_points(fv.points, domain)
    return Mesh2D(
        domain=domain,
        points=fv.points,
        triangles=fv.triangles,
        edges=fv.edges,
        edge_weight=fv.edge_weight,
        facet_length=fv.facet_length,
        cv_area=fv.cv_area,
        boundary_edges=fv.boundary_edges,
        n_negative_subareas=fv.n_negative_subareas,
        n_floored=fv.n_floored,
        quality_report=quality_report,
        Cdop=Cdop,
        boundary_point_index=tags.point_index,
        boundary_bc_type=tags.bc_type,
        h_min_cm=h_min_cm,
    )


# Backwards-compatible alias - build_mesh2d is device-generic, but the name
# used throughout main2d.py/configs/input_diode_2d.yaml predates that
# generalization.
build_diode2d_mesh = build_mesh2d
