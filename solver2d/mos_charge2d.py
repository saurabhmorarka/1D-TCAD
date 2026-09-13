"""Gate charge / low-frequency (quasi-static) C-V extraction for the 2D MOS
capacitor - the point-cloud/box-FV analog of mos/mos_solver.py's
semiconductor_charge + the low-frequency half of cv_sweep. High-frequency
(frozen-minority-carrier) C-V is not implemented here yet - see
DEVELOPMENT_LOG.md for the scoping decision.

The gate is an ideal conductor (a Dirichlet node), so its own induced free
charge is exactly whatever local flux imbalance the Poisson residual would
otherwise show at that node if it weren't a Dirichlet row (n=p=Cdop=0
identically at every oxide/gate node, since ni_arr=0 there, so this reduces
to a pure sum of edge conductances times potential differences - the same
Dirichlet-node "reaction flux" trick solver2d/current.py::contact_current
uses for terminal current, applied to charge instead)."""
import numpy as np


def gate_charge(mesh, result, gate_contact_name="gate"):
    """Total charge induced on the named gate contact, per unit device
    depth in z (C/cm - same 2D cross-section convention as
    solver2d/current.py). Divide by the gate's own width for a C/cm^2
    directly comparable to the 1D MOS capacitor's Qs(VG)."""
    boundary_bc_type = np.array(mesh.boundary_bc_type)
    is_gate_boundary = boundary_bc_type == f"contact:{gate_contact_name}"
    gate_point_idx = mesh.boundary_point_index[is_gate_boundary]
    if len(gate_point_idx) == 0:
        raise ValueError(f"solver2d.mos_charge2d: no contact named {gate_contact_name!r} in this mesh")
    N = len(mesh.points)
    is_gate = np.zeros(N, dtype=bool)
    is_gate[gate_point_idx] = True

    psi = result["psi"]
    ii, jj = mesh.edges[:, 0], mesh.edges[:, 1]
    g_e = mesh.edge_g
    flux_e = g_e * (psi[jj] - psi[ii])  # C/cm, direction i -> j

    i_is_gate = is_gate[ii] & ~is_gate[jj]
    j_is_gate = is_gate[jj] & ~is_gate[ii]
    # Charge induced at a gate node i = sum over its edges of g_e*(psi_j - psi_i)
    # = -flux_e for an edge oriented i->j (gate on the i side), or +flux_e
    # for an edge oriented i->j with the gate on the j side.
    return float(np.sum(-flux_e[i_is_gate]) + np.sum(flux_e[j_is_gate]))


def gate_charge_density(mesh, result, gate_contact_name="gate"):
    """gate_charge(...) divided by the gate contact's own width (cm) -
    C/cm^2, directly comparable to the 1D MOS capacitor's Qs(VG)."""
    contact = next(c for c in mesh.domain.contacts if c.name == gate_contact_name)
    width_cm = contact.x_range_cm[1] - contact.x_range_cm[0]
    return gate_charge(mesh, result, gate_contact_name) / width_cm


def cv_sweep_2d(mesh, mat, dev, Cdop_substrate, VG_list, gate_contact_name="gate", verbose=False):
    """Low-frequency (quasi-static) C-V sweep: a sequence of equilibrium
    solves (warm-started from the previous point, mirroring
    main2d_sweep.py's diode bias sweep) differentiated numerically,
    C_lf(VG) = -dQ_gate/dVG (sign per mos_solver.py::cv_sweep's own note:
    gate charge increases with VG, giving the conventional positive
    capacitance - Qs returned here is the SEMICONDUCTOR-side charge
    (gate_charge on the gate contact, which is exactly -Q_semiconductor by
    two-terminal charge neutrality, so -dQ_gate/dVG is likewise
    -d(-Qs)/dVG = dQs/dVG... to match mos_solver.py's own convention we
    report Qs = -Q_gate and C = -np.gradient(Qs, VG) = np.gradient(Q_gate, VG))."""
    from solver2d.poisson2d_mos import solve_mos_equilibrium_2d

    results = []
    psi_prev = None
    for VG in VG_list:
        r = solve_mos_equilibrium_2d(mesh, mat, dev, Cdop_substrate, VG,
                                      gate_contact_name=gate_contact_name,
                                      psi_init=psi_prev, verbose=verbose)
        Q_gate = gate_charge_density(mesh, r, gate_contact_name)
        results.append({"VG": VG, "psi": r["psi"], "n": r["n"], "p": r["p"],
                         "iters": r["iters"], "res_norm": r["res_norm"],
                         "Qs": -Q_gate})
        psi_prev = r["psi"]

    Q_arr = np.array([r["Qs"] for r in results])
    VG_arr = np.array([r["VG"] for r in results])
    C_lf = -np.gradient(Q_arr, VG_arr)
    for r, c in zip(results, C_lf):
        r["C_lf"] = c
    return results
