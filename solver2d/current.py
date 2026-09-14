"""Terminal current extraction: integrate the converged electron+hole
current across a named contact's boundary - the normal-current-integration
use case mesh2d/boundary.py's outward_normal/decompose_normal_tangential
utilities were written for. Lives in solver2d/ (not mesh2d/) since this is
solver post-processing (needs a converged psi/n/p/phin/phip result), not
mesh geometry.

The result is a current PER UNIT DEVICE DEPTH IN Z (A/cm, since this is a
genuinely 2D cross-section, translationally invariant in z, unlike 1D's
"per unit cross-sectional area" convention which folds an assumed area in
from the start) - see contact_current_density() for the natural quantity
to compare against a 1D solve's own current density.
"""
from core.params import Q


def contact_current(mesh, mat, result, contact_name):
    """Total current (A/cm) flowing OUT of the named contact into the
    device, summed over every mesh edge connecting a contact point to a
    non-contact neighbor. Uses the SAME plain-gradient edge-current formula
    the solver's own residual assembly uses (solver2d/newton_solver_qf_2d.py),
    evaluated at the converged (psi, n, p, phin, phip) - not a separate,
    possibly-inconsistent post-processing formula."""
    import numpy as np

    boundary_bc_type = np.array(mesh.boundary_bc_type)
    is_contact_boundary = boundary_bc_type == f"contact:{contact_name}"
    contact_point_idx = mesh.boundary_point_index[is_contact_boundary]
    if len(contact_point_idx) == 0:
        raise ValueError(f"solver2d.current: no contact named {contact_name!r} in this mesh")
    N = len(mesh.points)
    is_contact = np.zeros(N, dtype=bool)
    is_contact[contact_point_idx] = True

    n, p, phin, phip = result["n"], result["p"], result["phin"], result["phip"]
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    edge_len = np.linalg.norm(mesh.points[jj] - mesh.points[ii], axis=1)
    n_avg = 0.5 * (n[ii] + n[jj])
    p_avg = 0.5 * (p[ii] + p[jj])
    # Must use the SAME per-edge mobility the solver's own residual
    # assembly used (solver2d/newton_solver_qf_2d.py::_mesh_mobility_nodal)
    # - a plain scalar mat.mu_n/mat.mu_p here would silently mismatch the
    # doping-dependent mobility actually driving the converged (n,p,phin,
    # phip) state, giving an inconsistent post-processed current.
    from solver2d.newton_solver_qf_2d import _mesh_mobility_nodal
    mu_n_node, mu_p_node = _mesh_mobility_nodal(mesh, mat)
    mu_n_e = 0.5 * (mu_n_node[ii] + mu_n_node[jj])
    mu_p_e = 0.5 * (mu_p_node[ii] + mu_p_node[jj])
    Jn_e = -Q * mu_n_e * n_avg * (phin[jj] - phin[ii]) / edge_len
    Jp_e = -Q * mu_p_e * p_avg * (phip[jj] - phip[ii]) / edge_len
    I_e = (Jn_e + Jp_e) * mesh.facet_length  # A/cm, direction i -> j

    # 2026-09-13: exclude any edge whose non-contact endpoint is an
    # insulator node from this sum. Those edges' phin/phip on the oxide
    # side are an arbitrary pinned placeholder (0), not a real potential
    # (see newton_solver_qf_2d.py's is_oxide_free comments) - a contact
    # sitting right at an oxide/mesa corner (as this project's first
    # MOSFET geometry does) picks up one such edge whose "current" is
    # driven entirely by that placeholder and the contact's own fixed
    # bias, completely independent of Vgs. That single edge was found to
    # account for ~99.995% of the entire off-state terminal current,
    # flooring what should be an exponentially falling subthreshold
    # diffusion current. This exclusion only changes THIS post-processing
    # sum, not the Newton solver's own residual/Jacobian (those rows are
    # already overwritten by the Dirichlet/oxide-pin logic regardless of
    # this edge, so masking it here has no effect on convergence) - unlike
    # the broader _semiconductor_edge_mask() masking tried and reverted in
    # the solver itself, which changed real channel-surface nodes' own
    # continuity equations and made Newton convergence far worse.
    is_insulator = mesh.is_insulator if mesh.is_insulator is not None \
        else np.zeros(N, dtype=bool)
    i_is_contact = is_contact[ii] & ~is_contact[jj] & ~is_insulator[jj]
    j_is_contact = is_contact[jj] & ~is_contact[ii] & ~is_insulator[ii]
    return float(np.sum(I_e[i_is_contact]) - np.sum(I_e[j_is_contact]))


def contact_current_density(mesh, mat, result, contact_name):
    """contact_current(...) divided by the contact's own width (cm) - an
    average current density (A/cm^2) directly comparable to a 1D solve's
    own J(Va), since near a wide-enough contact the 2D cross-section is
    locally translation-invariant and should behave like the equivalent 1D
    structure."""
    contact = next(c for c in mesh.domain.contacts if c.name == contact_name)
    width_cm = contact.x_range_cm[1] - contact.x_range_cm[0]
    return contact_current(mesh, mat, result, contact_name) / width_cm
