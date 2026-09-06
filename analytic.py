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


def built_in_potential_fd(mat: Material, dev: Device) -> float:
    """Fermi-Dirac analog of built_in_potential(): Vbi from the two sides'
    Fermi-Dirac (not Boltzmann) equilibrium bulk potentials - see
    fermi_dirac.py. Reduces to built_in_potential() wherever neither side
    is degenerate enough for the difference to matter."""
    import fermi_dirac as fd
    psi_n = fd.equilibrium_bulk_potential_fd(mat, dev.Nd)
    psi_p = fd.equilibrium_bulk_potential_fd(mat, -dev.Na)
    return psi_n - psi_p


def shockley_I0_fd(mat: Material, dev: Device) -> float:
    """Fermi-Dirac-corrected Shockley I0: the two minority-carrier
    equilibrium reference densities (p0 in the n-side bulk, n0 in the
    p-side bulk) are computed directly from Fermi-Dirac statistics at each
    side's own doping (fermi_dirac.p_fd/n_fd) instead of the Boltzmann
    mass-action shortcut ni^2/N - everything else (the "law of the
    junction" boundary condition, the long-base diffusion solution this
    formula's Dp/Lp, Dn/Ln prefactors come from) is unchanged, so this
    stays a Boltzmann-consistent MINORITY-carrier-injection picture; only
    the equilibrium reference each side's exponential injection is
    measured FROM is corrected for the majority side's own degeneracy."""
    import fermi_dirac as fd
    psi_n = fd.equilibrium_bulk_potential_fd(mat, dev.Nd)
    psi_p = fd.equilibrium_bulk_potential_fd(mat, -dev.Na)
    p0_n_side = fd.p_fd(mat, psi_n)   # minority holes in the n-side bulk
    n0_p_side = fd.n_fd(mat, psi_p)   # minority electrons in the p-side bulk
    J0 = Q * (mat.Dp / mat.Lp * p0_n_side + mat.Dn / mat.Ln * n0_p_side)
    return J0 * dev.area


def depletion_capacitance(mat: Material, dev: Device, Va: np.ndarray) -> np.ndarray:
    """Depletion (junction) capacitance per unit area, F/cm^2: C_dep =
    eps/W(Va), from the same depletion-approximation width already used
    for the equilibrium band-diagram comparison. Dominates in reverse
    bias and near zero bias; becomes an increasingly poor approximation
    approaching/exceeding Vbi (W formally -> 0), same caveat depletion_widths()
    already carries."""
    Va = np.atleast_1d(np.asarray(Va, dtype=float))
    W = np.array([depletion_widths(mat, dev, v)[2] for v in Va])
    return mat.eps / W


def diffusion_capacitance(mat: Material, dev: Device, Va: np.ndarray,
                           p0_n_side: float = None, n0_p_side: float = None) -> np.ndarray:
    """Diffusion capacitance per unit area, F/cm^2, from the standard
    long-base minority-charge-storage result: each side's excess stored
    minority charge Q_stored = q*L*p0*(exp(Va/Vt)-1) (the same profile
    integrated to give the Shockley current's diffusion term), differentiated
    wrt Va. Dominates in forward bias once diffusion current exceeds the
    (here-ignored) generation/recombination-only depletion-region current.

    p0_n_side/n0_p_side: equilibrium minority densities to use (defaults to
    the Boltzmann ni^2/N values, matching shockley_I0(); pass the
    Fermi-Dirac equivalents from shockley_I0_fd()'s own calculation for the
    FD-consistent comparison curve)."""
    if p0_n_side is None:
        p0_n_side = mat.ni ** 2 / dev.Nd
    if n0_p_side is None:
        n0_p_side = mat.ni ** 2 / dev.Na
    Va = np.asarray(Va, dtype=float)
    return (Q / mat.Vt) * (mat.Lp * p0_n_side + mat.Ln * n0_p_side) * np.exp(Va / mat.Vt)


def cv_curve_analytic(mat: Material, dev: Device, Va: np.ndarray, use_fd: bool = False):
    """Combined depletion + diffusion capacitance per unit area, F/cm^2 -
    an approximate closed-form reference (the two mechanisms are simply
    added, which is standard pedagogically but only accurate away from the
    transition region where both are comparable). use_fd=True substitutes
    the Fermi-Dirac equilibrium minority references (shockley_I0_fd's
    p0_n_side/n0_p_side) into the diffusion term, and the Fermi-Dirac Vbi
    into the depletion term (via a per-call Device-like override), leaving
    everything else identical - the gap between the two curves is exactly
    the closed-form counterpart of the Boltzmann-vs-Fermi-Dirac comparison
    the numeric solve can't show directly (see fermi_dirac.py's module
    docstring for why the numeric PDE solve itself stays Boltzmann-only)."""
    import fermi_dirac as fd
    if not use_fd:
        return depletion_capacitance(mat, dev, Va) + diffusion_capacitance(mat, dev, Va)

    psi_n = fd.equilibrium_bulk_potential_fd(mat, dev.Nd)
    psi_p = fd.equilibrium_bulk_potential_fd(mat, -dev.Na)
    p0_n_side = fd.p_fd(mat, psi_n)
    n0_p_side = fd.n_fd(mat, psi_p)
    Vbi_fd = psi_n - psi_p

    Va = np.atleast_1d(np.asarray(Va, dtype=float))
    V = np.maximum(Vbi_fd - Va, 1e-6)
    W = np.sqrt(2 * mat.eps * V / Q * (1.0 / dev.Na + 1.0 / dev.Nd))
    C_dep_fd = mat.eps / W
    C_diff_fd = diffusion_capacitance(mat, dev, Va, p0_n_side=p0_n_side, n0_p_side=n0_p_side)
    return C_dep_fd + C_diff_fd
