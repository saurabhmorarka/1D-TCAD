"""Electric field E = -grad(psi) at every mesh point, recovered from the
converged potential the same way any P1 (piecewise-linear-per-triangle)
finite-element field's gradient is recovered: each triangle's own vertex
values determine ONE constant gradient over that triangle (a linear
function has a constant gradient), and each point's field is the area-
weighted average of every incident triangle's gradient - the standard,
general "vertex gradient recovery" technique, independent of the box-FV
edge/circumcenter machinery solver2d/newton_solver_qf_2d.py and
solver2d/poisson2d_mos.py use to solve for psi in the first place. Works
identically for any 2D device (diode, MOS capacitor, ...) since it only
needs the triangulation and a per-point scalar field.
"""
import numpy as np


def _triangle_gradient(p0, p1, p2, f0, f1, f2):
    e1 = p1 - p0
    e2 = p2 - p0
    det = e1[0] * e2[1] - e1[1] * e2[0]
    if det == 0.0:
        return np.array([0.0, 0.0])
    df1 = f1 - f0
    df2 = f2 - f0
    gx = (df1 * e2[1] - df2 * e1[1]) / det
    gy = (e1[0] * df2 - e2[0] * df1) / det
    return np.array([gx, gy])


def electric_field_2d(points, triangles, psi):
    """Returns (Ex, Ey), each (N,) arrays in V/cm (matching this project's
    cm-based length convention), E = -grad(psi)."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    psi = np.asarray(psi, dtype=float)
    N = len(points)

    tri_pts = points[triangles]            # (M,3,2)
    tri_psi = psi[triangles]                # (M,3)
    e1 = tri_pts[:, 1] - tri_pts[:, 0]
    e2 = tri_pts[:, 2] - tri_pts[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    df1 = tri_psi[:, 1] - tri_psi[:, 0]
    df2 = tri_psi[:, 2] - tri_psi[:, 0]

    safe_det = np.where(det == 0.0, 1.0, det)
    gx = (df1 * e2[:, 1] - df2 * e1[:, 1]) / safe_det
    gy = (e1[:, 0] * df2 - e2[:, 0] * df1) / safe_det
    gx = np.where(det == 0.0, 0.0, gx)
    gy = np.where(det == 0.0, 0.0, gy)
    tri_area = 0.5 * np.abs(det)

    grad_sum = np.zeros((N, 2))
    area_sum = np.zeros(N)
    for k in range(3):
        idx = triangles[:, k]
        np.add.at(grad_sum, idx, np.stack([gx, gy], axis=1) * tri_area[:, None])
        np.add.at(area_sum, idx, tri_area)

    area_safe = np.where(area_sum == 0.0, 1.0, area_sum)
    grad = grad_sum / area_safe[:, None]
    return -grad[:, 0], -grad[:, 1]
