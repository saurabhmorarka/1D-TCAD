"""Closed-form (textbook) results for a step p-n junction, for comparison
against the numerical drift-diffusion simulation.

References: Sze, "Physics of Semiconductor Devices"; Pierret, "Semiconductor
Device Fundamentals".
"""
import numpy as np

from params import Q, Material, Device


def built_in_potential(mat: Material, dev: Device) -> float:
    """Vbi = Vt * ln(Na*Nd/ni^2)."""
    return mat.Vt * np.log(dev.Na * dev.Nd / mat.ni ** 2)


def depletion_widths(mat: Material, dev: Device, Va: float = 0.0):
    """Depletion-approximation widths on each side, step junction, under
    applied bias Va (forward positive on the p-side). Returns (xp, xn, W)."""
    Vbi = built_in_potential(mat, dev)
    V = Vbi - Va  # total potential dropped across the junction under bias
    V = max(V, 1e-6)  # guard against forward bias collapsing the depletion width in this simple formula
    W = np.sqrt(2 * mat.eps * V / Q * (1.0 / dev.Na + 1.0 / dev.Nd))
    xp = W * dev.Nd / (dev.Na + dev.Nd)   # depletion extent into p-side
    xn = W * dev.Na / (dev.Na + dev.Nd)   # depletion extent into n-side
    return xp, xn, W


def depletion_potential_profile(mat: Material, dev: Device, x: np.ndarray, Va: float = 0.0):
    """Analytic (depletion-approximation) electrostatic potential profile,
    referenced the same way as the simulation (psi=0 where n=p=ni), for a
    step junction at x=0 under bias Va."""
    from physics import equilibrium_bulk_potential
    xp, xn, W = depletion_widths(mat, dev, Va)
    psi_p_bulk = equilibrium_bulk_potential(mat, -dev.Na)
    psi_n_bulk = equilibrium_bulk_potential(mat, dev.Nd)

    psi = np.empty_like(x)
    left_dep = (x >= -xp) & (x < 0)
    right_dep = (x >= 0) & (x <= xn)
    bulk_p = x < -xp
    bulk_n = x > xn

    psi[bulk_p] = psi_p_bulk
    psi[bulk_n] = psi_n_bulk + Va
    # quadratic potential in depletion region (integrate the triangular field twice)
    psi[left_dep] = psi_p_bulk + (Q * dev.Na / (2 * mat.eps)) * (x[left_dep] + xp) ** 2
    psi[right_dep] = psi_n_bulk + Va - (Q * dev.Nd / (2 * mat.eps)) * (x[right_dep] - xn) ** 2
    return psi


def shockley_I0(mat: Material, dev: Device) -> float:
    """Long-base ideal diode saturation current: I0 = q*A*ni^2*(Dp/(Lp*Nd) + Dn/(Ln*Na))."""
    J0 = Q * mat.ni ** 2 * (mat.Dp / (mat.Lp * dev.Nd) + mat.Dn / (mat.Ln * dev.Na))
    return J0 * dev.area


def shockley_current(mat: Material, dev: Device, Va: np.ndarray) -> np.ndarray:
    """Ideal Shockley diode law I(Va) = I0*(exp(Va/Vt)-1)."""
    I0 = shockley_I0(mat, dev)
    return I0 * (np.exp(Va / mat.Vt) - 1.0)
