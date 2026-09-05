"""Nonuniform 1D mesh generator, shared by the diode and the MOS capacitor.

Both devices are built the same way: one or two semiconductor "regions"
(diode: p-side + n-side; MOS: just the substrate) meeting a hard interface
(the junction, or the oxide/substrate boundary) at x=0, each region meshed
outward from that interface with `_region_nodes` below. That function picks
the mesh spacing to resolve two effects that can each demand a fine mesh
independently of the other:

  1. Proximity to the interface itself, where the electrostatic potential
     bends sharply (depletion physics) even when the local doping is
     perfectly flat - the classic reason this project's meshes have always
     been finest at x=0 and geometrically coarsen away from it.
  2. The local doping PROFILE changing quickly (relevant once doping.type is
     "linear" or "gaussian" rather than "flat") - resolved via a
     |d(ln N)/dx| based limit, wherever it happens to fall in the region.

For a flat profile, effect (2) is inactive everywhere, and `_region_nodes`
reduces exactly to the original geometric-growth algorithm (bit-for-bit) -
so nothing about the existing flat-doping diode/MOS results changes.
"""
import numpy as np

from params import Material, Device
from doping_profiles import DopingProfile


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


def _graded_nodes(length: float, profile: DopingProfile, mat: Material,
                   h_min: float, h_max: float, growth: float,
                   gradient_safety_factor: float = 0.25, n_probe: int = 4000) -> np.ndarray:
    """Adaptive marching mesh for a region with a non-flat doping profile.
    Local target spacing is the tighter of:
      - the same geometric ramp _one_sided_nodes uses (h_min at the interface,
        growing to h_max), but with the LOCAL Debye length (from the doping
        value at that point) rather than a single region-wide value, and
      - gradient_safety_factor / |d(ln N)/dx|, so a cell never spans a region
        where the doping itself changes by more than ~25% (default) in log
        space - this is what actually resolves a gaussian implant peak or a
        linear grade, wherever in the region it falls.
    This is a heuristic, not a proof of mesh convergence - as with the MOS
    accumulation-layer mesh study (DEVELOPMENT_LOG.md), always sanity-check
    a new graded-doping case by re-running with a tighter mesh and confirming
    the result doesn't move."""
    xs_probe = np.linspace(0.0, length, n_probe)
    N_probe = np.clip(profile.sample(xs_probe, length), 1e10, None)
    dlogN_probe = np.gradient(np.log(N_probe), xs_probe)

    def target_h(x: float) -> float:
        N_here = np.interp(x, xs_probe, N_probe)
        dlnN_here = np.interp(x, xs_probe, dlogN_probe)
        L_D = _debye_length(mat, N_here)
        ramp = h_min + (h_max - h_min) * min(x / max(10.0 * L_D, 1e-30), 1.0)
        grad_limit = gradient_safety_factor / max(abs(dlnN_here), 1e-30)
        return float(np.clip(min(ramp, grad_limit), h_min, h_max))

    xs = [0.0]
    x = 0.0
    h_prev = None
    while x < length:
        h = target_h(x)
        if h_prev is not None:
            h = min(h, h_prev * growth)  # cap how fast spacing can grow step-to-step
        x_next = x + h
        remaining_after = length - x_next
        if remaining_after <= 0.5 * h:
            xs.append(length)
            break
        xs.append(x_next)
        x = x_next
        h_prev = h
    return np.array(xs)


def _region_nodes(length: float, profile: DopingProfile, mat: Material,
                   h_min: float, h_max: float, growth: float) -> np.ndarray:
    """Mesh nodes (0..length) for one region, dispatching to the exact
    original algorithm for flat doping, or the adaptive one otherwise."""
    if profile.type == "flat":
        return _one_sided_nodes(length, h_min, h_max, growth)
    return _graded_nodes(length, profile, mat, h_min, h_max, growth)


def build_diode_grid(mat: Material, dev: Device, growth: float = 1.06,
                      bulk_spacing_debye_factor: float = 5.0,
                      junction_spacing_debye_factor: float = 0.05):
    """Build the nonuniform grid and doping profile for a step (or graded)
    p-n junction diode.

    Returns dict with:
      x        : node positions, cm, x=0 at the junction, p-side negative
      Cdop     : net doping Nd-Na at each node, cm^-3
      junction_index : index of node closest to x=0
      Wp, Wn   : region lengths actually used
    """
    p_profile = dev.p_profile if dev.p_profile is not None else DopingProfile.flat(dev.Na)
    n_profile = dev.n_profile if dev.n_profile is not None else DopingProfile.flat(dev.Nd)

    L_D_p = _debye_length(mat, p_profile.reference_concentration())
    L_D_n = _debye_length(mat, n_profile.reference_concentration())
    L_D_min = min(L_D_p, L_D_n)

    h_min = junction_spacing_debye_factor * L_D_min
    h_max = bulk_spacing_debye_factor * L_D_min

    Wp = dev.Wp if dev.Wp is not None else max(dev.n_diffusion_lengths * mat.Ln, 20 * L_D_p)
    Wn = dev.Wn if dev.Wn is not None else max(dev.n_diffusion_lengths * mat.Lp, 20 * L_D_n)

    x_n_side = _region_nodes(Wn, n_profile, mat, h_min, h_max, growth)          # 0 .. Wn
    x_p_side = -_region_nodes(Wp, p_profile, mat, h_min, h_max, growth)[::-1]   # -Wp .. 0

    x = np.concatenate([x_p_side[:-1], x_n_side])
    x = np.unique(x)  # sorted, dedup the shared 0

    Cdop = np.where(x >= 0.0, n_profile.sample(x, Wn), -p_profile.sample(-x, Wp))
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
        "p_profile": p_profile,
        "n_profile": n_profile,
    }


def build_mos_grid(mat: Material, dev, Cdop_substrate: float,
                    n_ox_points: int = 15, growth: float = 1.08,
                    bulk_spacing_debye_factor: float = 5.0,
                    interface_spacing_debye_factor: float = 0.001):
    """Build the oxide+substrate mesh and per-node doping/eps/ni arrays for
    the 1D MOS capacitor: metal gate - thin oxide - substrate.

    Substrate type (p or n) is generic here: Cdop_substrate's SIGN selects
    it (negative=p, positive=n; magnitude is the reference substrate doping
    used by the closed-form comparisons) - mos_solver.py picks which
    carrier to freeze for the high-frequency C-V trick from that sign.
    `interface_spacing_debye_factor` defaults much tighter than the diode's
    equivalent (0.05): a mesh-convergence check showed the MOS-cap C-V curve
    is genuinely under-resolved at the coarser factor in strong
    accumulation - see DEVELOPMENT_LOG.md.

    Returns dict with:
      x          : node positions, cm, x=0 at the gate/oxide interface,
                   oxide occupying x in [-t_ox, 0), substrate x in [0, t_si]
      Cdop       : net doping, cm^-3 (0 in oxide, signed substrate profile
                   sample in the substrate)
      eps_edge   : permittivity per mesh edge, length len(x)-1
      ni_arr     : intrinsic concentration per node (0 in oxide, mat.ni in substrate)
      is_oxide   : boolean mask, True for oxide nodes
      oxide_index: index of the oxide/substrate interface node (first substrate node)
      t_si       : substrate thickness actually used
    """
    substrate_profile = (dev.substrate_profile if dev.substrate_profile is not None
                          else DopingProfile.flat(abs(Cdop_substrate)))
    sign = 1.0 if Cdop_substrate >= 0 else -1.0

    L_D = _debye_length(mat, substrate_profile.reference_concentration())
    h_min = interface_spacing_debye_factor * L_D
    h_max = bulk_spacing_debye_factor * L_D

    t_si = dev.t_si if dev.t_si is not None else max(20 * L_D, 2.0e-4)  # >=2 um floor

    x_ox = -np.linspace(dev.t_ox, 0.0, n_ox_points)[::-1]  # -t_ox .. 0, n_ox_points points, uniform
    x_si = _region_nodes(t_si, substrate_profile, mat, h_min, h_max, growth)  # 0 .. t_si

    x = np.concatenate([x_ox[:-1], x_si])  # drop duplicate 0 from x_ox
    x = np.unique(x)

    oxide_index = int(np.searchsorted(x, 0.0))  # first node with x>=0 -> substrate side
    is_oxide = x < 0.0

    Cdop = np.where(is_oxide, 0.0, sign * substrate_profile.sample(np.clip(x, 0.0, None), t_si))
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
        "substrate_profile": substrate_profile,
    }
