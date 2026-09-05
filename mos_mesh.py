"""Mesh, doping, permittivity, and ni profile builder for the 1D MOS
capacitor: metal gate - thin oxide - uniform substrate - ohmic back contact.

Substrate type (p or n) is generic here: pass Cdop_substrate = -Na (p-type,
electrons are the minority/inversion carrier) or +Nd (n-type, holes are the
minority/inversion carrier). mos_solver.py picks which carrier to freeze for
the high-frequency C-V trick based on the sign of this doping.
"""
import numpy as np

from params import Q, Material
from mos_params import MOSDevice


def build_mos_grid(mat: Material, dev: MOSDevice, Cdop_substrate: float,
                    n_ox_points: int = 15, growth: float = 1.08,
                    bulk_spacing_debye_factor: float = 5.0,
                    interface_spacing_debye_factor: float = 0.001):
    """interface_spacing_debye_factor default is much tighter than the
    diode's equivalent (0.05): a mesh-convergence check showed the MOS-cap
    C-V curve is genuinely under-resolved at the coarser factor in strong
    accumulation. The accumulation-layer screening length shrinks well
    below the bulk-doping Debye length (which is what sizes the mesh) as
    surface hole/electron concentration rises, and low-frequency
    capacitance there converged from ~0.875*Cox (factor=0.02) to a stable
    ~0.843*Cox only once h_min was pushed down to ~0.01-0.04 nm - a real
    physical effect (finite accumulation-layer capacitance in series with
    Cox, not the idealized C=Cox the depletion approximation assumes), but
    one that needs this finer default mesh to actually converge to."""
    """Build the oxide+substrate mesh and per-node doping/eps/ni arrays.

    Returns dict with:
      x          : node positions, cm, x=0 at the gate/oxide interface,
                   oxide occupying x in [-t_ox, 0), substrate x in [0, t_si]
      Cdop       : net doping, cm^-3 (0 in oxide, Cdop_substrate in substrate)
      eps_edge   : permittivity per mesh edge, length len(x)-1
      ni_arr     : intrinsic concentration per node (0 in oxide, mat.ni in substrate)
      is_oxide   : boolean mask, True for oxide nodes
      oxide_index: index of the oxide/substrate interface node (first substrate node)
      t_si       : substrate thickness actually used
    """
    from mesh import _one_sided_nodes, _debye_length

    L_D = _debye_length(mat, abs(Cdop_substrate))
    h_min = interface_spacing_debye_factor * L_D
    h_max = bulk_spacing_debye_factor * L_D

    t_si = dev.t_si if dev.t_si is not None else max(20 * L_D, 2.0e-4)  # >=2 um floor

    x_ox = -np.linspace(dev.t_ox, 0.0, n_ox_points)[::-1]  # -t_ox .. 0, n_ox_points points, uniform
    x_si = _one_sided_nodes(t_si, h_min, h_max, growth)     # 0 .. t_si, fine at 0, coarsening

    x = np.concatenate([x_ox[:-1], x_si])  # drop duplicate 0 from x_ox
    x = np.unique(x)

    oxide_index = int(np.searchsorted(x, 0.0))  # first node with x>=0 -> substrate side
    is_oxide = x < 0.0

    Cdop = np.where(is_oxide, 0.0, Cdop_substrate)
    ni_arr = np.where(is_oxide, 0.0, mat.ni)

    x_mid = (x[:-1] + x[1:]) / 2.0
    eps_edge = np.where(x_mid < 0.0, dev.eps_ox, mat.eps)

    return {
        "x": x,
        "Cdop": Cdop,
        "eps_edge": eps_edge,
        "ni_arr": ni_arr,
        "is_oxide": is_oxide,
        "oxide_index": oxide_index,
        "t_si": t_si,
        "h_min": h_min,
        "h_max": h_max,
    }
