"""Ruiz row/column equilibration for a sparse linear solve - a numerically
exact preconditioning step (it does not change what equation is being
solved, only how accurately floating-point arithmetic can solve it),
applied right before every direct sparse solve in the avalanche breakdown
Newton solvers (newton_solver_avalanche.py, bank_rose_damping.py).

WHY THIS EXISTS: the avalanche generation term's Jacobian entries
(newton_solver_avalanche.py's dGii_node_d* block) scale as
alpha(E) * mobility * carrier_density / mesh_spacing - each factor
individually legitimate, but together they can reach ~1e31 at this
project's finest junction meshes (sub-nanometer h, heavily doped
degenerate carrier densities, exponentially large alpha near breakdown),
sitting in the SAME sparse matrix as residual/flux-divergence entries
around ~1e2. A direct sparse solve (scipy's spsolve, effectively a sparse
LU factorization) computes en exact answer for the matrix AS REPRESENTED
IN FLOATING POINT - but double precision only carries ~16 significant
digits, so when some entries are 29 orders of magnitude larger than
others, the small-magnitude entries (where the physically meaningful part
of the answer often lives) can be swamped to the point of losing all
useful precision, even though the solve completes formally without error.
Traced directly on a specific failing bias point: the solver's own
predicted improvement from a Newton step (computed from F and J) disagreed
by many orders of magnitude with the ACTUAL residual change measured after
taking that step - the signature of a Jacobian whose solve has lost
precision, not a genuinely hard nonlinear landscape.

Ruiz equilibration (Ruiz 2001, "A scaling algorithm to equilibrate both
rows and columns norms in matrices") iteratively rescales each row and
column by 1/sqrt(max|entry|) in that row/column, converging in a few
passes to a matrix where every row and column has entries within roughly
[0.1, 10] of each other in magnitude - dramatically better conditioned for
a direct solve, at negligible cost relative to the factorization itself.
This project's own history already used exactly this technique (row+column
equilibration, matching Perez-Escudero et al. 2025's approach) to fix an
unrelated ill-conditioning problem in an earlier, abandoned log-density
Newton formulation - see the tcad1d-extreme-doping-convergence-limit
memory/DEVELOPMENT_LOG.md session 9 for that precedent.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def equilibrate(J: sp.spmatrix, n_passes: int = 2):
    """Returns (J_scaled, row_scale, col_scale) such that
    J_scaled = diag(row_scale) @ J @ diag(col_scale) has row/column max
    magnitudes close to 1. J_scaled is a new csr_matrix; J itself is not
    modified. A zero row/column (no nonzeros - shouldn't happen for this
    project's Jacobians, which always have a Dirichlet-row diagonal, but
    guarded anyway) gets a scale factor of 1 (left untouched)."""
    Jw = J.tocsr(copy=True).astype(float)
    n = Jw.shape[0]
    row_scale = np.ones(n)
    col_scale = np.ones(n)

    for _ in range(n_passes):
        row_max = np.maximum(np.abs(Jw).max(axis=1).toarray().ravel(), 0.0)
        r = np.where(row_max > 0, 1.0 / np.sqrt(row_max), 1.0)
        Jw = sp.diags(r) @ Jw
        row_scale *= r

        col_max = np.maximum(np.abs(Jw).max(axis=0).toarray().ravel(), 0.0)
        c = np.where(col_max > 0, 1.0 / np.sqrt(col_max), 1.0)
        Jw = Jw @ sp.diags(c)
        col_scale *= c

    return Jw.tocsc(), row_scale, col_scale


def equilibrated_spsolve(J: sp.spmatrix, rhs: np.ndarray, n_passes: int = 2) -> np.ndarray:
    """Drop-in replacement for scipy.sparse.linalg.spsolve(J, rhs): solves
    the EXACT SAME linear system (mathematically), just via an equilibrated
    matrix for better floating-point accuracy. Returns the same delta
    spsolve(J, rhs) would return in exact arithmetic."""
    J_scaled, row_scale, col_scale = equilibrate(J, n_passes=n_passes)
    rhs_scaled = row_scale * rhs
    y = spla.spsolve(J_scaled, rhs_scaled)
    return col_scale * y
