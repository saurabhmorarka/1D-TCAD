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
    # Cap the exponent before evaluating growth**exponent, not after: for a
    # very small h_min relative to the domain size (e.g. a MOS capacitor's
    # nanometer-thin oxide graded against a micron-scale domain), dist/h_min
    # can run into the thousands, and growth**that raises a plain Python
    # OverflowError (unlike numpy, which would just return inf) well before
    # np.clip ever gets a chance to bring it back down to h_max.
    exponent = dist / h_min
    max_exponent = np.log(h_max / h_min) / np.log(growth)
    if exponent >= max_exponent:
        return h_max
    return float(np.clip(h_min * growth ** exponent, h_min, h_max))


def _domain_pslg(domain):
    """Vertices + segments for the domain's outer boundary (a plain
    rectangle, or a rectangle with a mesa protrusion spliced into its top
    edge - see geometry2d.py::Domain2D.outer_boundary) plus every interior
    region's own boundary - a Planar Straight Line Graph `triangle`
    triangulates conformingly (segments are never crossed).

    Any of a region's 4 sides that lies entirely on the domain's own outer
    boundary (domain.boundary_point_role() is non-None at the side's
    MIDPOINT - e.g. a source/drain region placed flush against the
    domain's left/right edge, or a mesa oxide's top/side walls) is SKIPPED
    here - that side is already part of the outer polygon added above, and
    re-adding it would give `triangle` two exactly-coincident overlapping
    segments, which is undefined input and was observed to crash the C
    extension outright (segfault) rather than raise a Python exception.
    The midpoint (not the two endpoints) is what's tested: a mesa oxide's
    own y=0 interface side has BOTH its endpoints sitting exactly at the
    mesa's side-wall corners (which individually do test as "on boundary"
    - a false positive if only the endpoints were checked), yet the
    segment connecting them cuts straight across underneath the mesa and
    is genuinely interior; since every region here is an axis-aligned
    rectangle, a straight side can only be fully on vs. fully off the
    boundary (it can't dip in and out along its own length), so the
    midpoint alone is a sufficient test either way. Only a region's
    genuinely interior sides (e.g. the diode's p_well's 4 sides, all
    strictly inside the domain; or a mesa oxide's y=0 interface) get a
    fresh segment, reusing an existing vertex at either endpoint (from the
    outer polygon or an earlier region) instead of creating a duplicate
    coincident one."""
    outer_verts, outer_edges, _ = domain.outer_boundary()
    verts = [tuple(v) for v in outer_verts]
    segs = [tuple(e) for e in outer_edges]

    def find_or_add_vertex(pt, tol=1e-9):
        for k, v in enumerate(verts):
            if abs(v[0] - pt[0]) <= tol and abs(v[1] - pt[1]) <= tol:
                return k
        verts.append(pt)
        return len(verts) - 1

    def add_rect(x0, y0, x1, y1):
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for k in range(4):
            p0, p1 = corners[k], corners[(k + 1) % 4]
            mid = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
            if domain.boundary_point_role(*mid) is not None:
                continue  # already represented by the outer polygon
            i0 = find_or_add_vertex(p0)
            i1 = find_or_add_vertex(p1)
            segs.append((i0, i1))

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
