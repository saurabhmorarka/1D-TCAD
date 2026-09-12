"""Constrained, quality-guaranteed, graded point-cloud/triangulation
generator for a Domain2D, built on Shewchuk's `triangle` library (Ruppert/
Chew-style Delaunay refinement - the standard, well-established tool for
exactly this problem; see mesh2d/mesh_quality.py's module docstring for why
a hand-rolled quadtree+scipy.Delaunay approach was abandoned in favor of
this).

Two things `triangle` gives that the old quadtree approach could not:
  1. A PROVABLE minimum-angle guarantee (Ruppert's theorem) - not just an
     approximation of the device geometry, but EXACT conformity to every
     region/domain boundary segment (a constrained Delaunay triangulation
     never crosses a segment), so the mesh is not just non-obtuse-ish but
     also geometrically exact at every doping-step boundary.
  2. Graded sizing via triangle's own refine mode ('r' + `triangle_max_area`
     per existing triangle) rather than a hand-rolled quadtree sizing
     function - `interface_segments` (defaulting to the device's doping
     junctions, but overridable with any user-specified interface, e.g. a
     future oxide/semiconductor interface) drives a distance-based target
     area exactly the way the old quadtree's `_target_spacing` did, just
     applied as a refinement constraint `triangle` itself enforces.
"""
import numpy as np
import triangle as tr


def _dist_point_to_segment(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _dist_to_segments(p, segments):
    if not segments:
        return np.inf
    return min(_dist_point_to_segment(p, a, b) for (a, b) in segments)


def _target_spacing(dist, h_min, h_max, growth):
    if not np.isfinite(dist):
        return h_max
    return float(np.clip(h_min * growth ** (dist / h_min), h_min, h_max))


def _domain_pslg(domain):
    """Vertices + segments for the domain's outer rectangle and every
    region's rectangle boundary - a Planar Straight Line Graph `triangle`
    triangulates conformingly (segments are never crossed)."""
    verts = [(0.0, 0.0), (domain.width_cm, 0.0),
             (domain.width_cm, domain.height_cm), (0.0, domain.height_cm)]
    segs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def add_rect(x0, y0, x1, y1):
        base = len(verts)
        verts.extend([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        segs.extend([(base, base + 1), (base + 1, base + 2),
                     (base + 2, base + 3), (base + 3, base)])

    for region in domain.regions[1:]:
        x0, x1 = region.x_range_cm
        y0, y1 = region.y_range_cm
        add_rect(x0, y0, x1, y1)

    return np.array(verts, dtype=float), np.array(segs, dtype=int)


def build_point_cloud(domain, h_min_cm, h_max_cm, growth=1.3, min_angle_deg=32,
                       interface_segments=None, max_refine_passes=8):
    """Returns (points, triangles): a constrained conforming-Delaunay
    triangulation (min_angle_deg respected everywhere, by Ruppert's
    theorem, up to floating-point/PSLG-input-angle limits) graded from
    h_min_cm at `interface_segments` (default: domain.junction_segments(),
    i.e. every doping-type boundary) out to h_max_cm in the bulk, via
    repeated refine passes until the triangle count stabilizes."""
    segments = (domain.junction_segments() if interface_segments is None
                else interface_segments)
    verts, segs = _domain_pslg(domain)

    mesh = tr.triangulate(dict(vertices=verts, segments=segs),
                          f'pq{min_angle_deg}Da{h_max_cm ** 2}')

    prev_n = -1
    for _ in range(max_refine_passes):
        tris = mesh['triangles']
        pts = mesh['vertices']
        if len(tris) == prev_n:
            break
        prev_n = len(tris)
        centroids = pts[tris].mean(axis=1)
        target = np.array([_target_spacing(_dist_to_segments(c, segments), h_min_cm, h_max_cm, growth)
                            for c in centroids])
        areas = target ** 2
        mesh = tr.triangulate(
            dict(vertices=pts, segments=segs, triangles=tris, triangle_max_area=areas),
            f'rpq{min_angle_deg}Da')

    return mesh['vertices'], mesh['triangles']
