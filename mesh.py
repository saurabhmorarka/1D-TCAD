"""Nonuniform 1D mesh generator for the diode.

Strategy: place the finest spacing at the metallurgical junction (x=0), where
the space charge and field vary fastest, and grow the spacing geometrically
away from the junction until it saturates at a maximum bulk spacing, then stay
uniform out to each contact. This keeps the depletion region well resolved
(spacing << Debye length) without wasting points in the long, nearly flat
quasi-neutral bulk.
"""
import numpy as np

from params import Material, Device, EPS0


def _debye_length(mat: Material, N: float) -> float:
    from params import Q
    return (mat.eps * mat.Vt / (Q * N)) ** 0.5


def _one_sided_nodes(length: float, h_min: float, h_max: float, growth: float) -> np.ndarray:
    """Spacings starting at h_min, growing geometrically to h_max, then constant,
    covering total distance `length`. Returns cumulative node positions (0..length),
    NOT including the starting 0 (caller prepends it)."""
    xs = [0.0]
    h = h_min
    x = 0.0
    while x < length:
        x_next = x + h
        remaining_after = length - x_next
        if remaining_after <= 0.5 * h:
            # The final step would either overshoot or leave a tiny sliver
            # smaller than half a step (a huge, ill-conditioned
            # Scharfetter-Gummel flux coefficient q*D/h waiting to happen).
            # Land exactly on `length` instead of adding that sliver as its
            # own segment.
            xs.append(length)
            break
        xs.append(x_next)
        x = x_next
        h = min(h * growth, h_max)
    return np.array(xs)


def build_grid(mat: Material, dev: Device, growth: float = 1.06,
                bulk_spacing_debye_factor: float = 5.0,
                junction_spacing_debye_factor: float = 0.05):
    """Build the nonuniform grid, doping profile, and useful lengths.

    Returns dict with:
      x        : node positions, cm, x=0 at the junction, p-side negative
      Cdop     : net doping Nd-Na at each node, cm^-3 (step at junction)
      junction_index : index of node closest to x=0
      Wp, Wn   : region lengths actually used
    """
    L_D_p = _debye_length(mat, dev.Na)
    L_D_n = _debye_length(mat, dev.Nd)
    L_D_min = min(L_D_p, L_D_n)

    h_min = junction_spacing_debye_factor * L_D_min
    h_max = bulk_spacing_debye_factor * L_D_min

    Wp = dev.Wp if dev.Wp is not None else max(dev.n_diffusion_lengths * mat.Ln, 20 * L_D_p)
    Wn = dev.Wn if dev.Wn is not None else max(dev.n_diffusion_lengths * mat.Lp, 20 * L_D_n)

    x_n_side = _one_sided_nodes(Wn, h_min, h_max, growth)          # 0 .. Wn
    x_p_side = -_one_sided_nodes(Wp, h_min, h_max, growth)[::-1]   # -Wp .. 0 (excludes duplicate 0 at end)

    x = np.concatenate([x_p_side[:-1], x_n_side])
    x = np.unique(x)  # sorted, dedup the shared 0

    Cdop = np.where(x >= 0.0, dev.Nd, -dev.Na)
    junction_index = int(np.argmin(np.abs(x)))

    return {
        "x": x,
        "Cdop": Cdop,
        "junction_index": junction_index,
        "Wp": Wp,
        "Wn": Wn,
        "h_min": h_min,
        "h_max": h_max,
        "L_D_p": L_D_p,
        "L_D_n": L_D_n,
    }
