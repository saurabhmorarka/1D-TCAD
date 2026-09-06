"""Fermi-Dirac (degenerate) equilibrium carrier statistics - a SCOPED
alternative to the Maxwell-Boltzmann relations physics.py uses everywhere
else in this project.

Scope (deliberately limited - see DEVELOPMENT_LOG.md and the
"tcad1d-known-physics-simplifications" note): this module only answers
"what would the EQUILIBRIUM contact/built-in potential and minority-carrier
reference densities look like under proper Fermi-Dirac statistics, instead
of Boltzmann". It does NOT touch the active nonlinear Poisson/continuity
PDE solve (physics.py, newton_solver.py) - that still assumes Boltzmann
throughout, including its Scharfetter-Gummel flux discretization, whose
exact-exponential-fitting derivation implicitly assumes the Boltzmann
psi<->n relation. Doing this properly for a real current-carrying
transport solve would mean replacing that discretization's Einstein
relation with a degenerate/generalized one everywhere - a substantially
bigger numerics project, not attempted here. What IS captured here is
exactly what matters for a doping level like Nd=1e21 cm^-3 (well above
silicon's Nc~2.8e19, i.e. genuinely degenerate): how the equilibrium
built-in potential, depletion width, and closed-form Shockley I0 shift
once the heavily-doped side's contact is treated correctly.

Reference: F_{1/2} is the complete Fermi-Dirac integral of order 1/2,
n = Nc * F_{1/2}(eta), eta = (Ef-Ec)/kT. Approximated here via the
Bednarczyk & Bednarczyk (1978) rational fit, accurate to within ~0.4% over
the full range and a common choice in TCAD tools for exactly this reason
(closed-form, cheap, and differentiable without a numerical integral).
"""
import numpy as np
from scipy.optimize import brentq

from params import Material


def fermi_half(eta):
    """F_{1/2}(eta), the complete Fermi-Dirac integral of order 1/2
    (Bednarczyk & Bednarczyk 1978 approximation). Reduces to exp(eta) for
    eta << 0 (the non-degenerate/Boltzmann limit) by construction."""
    eta = np.asarray(eta, dtype=float)
    a = eta ** 4 + 50.0 + 33.6 * eta * (1.0 - 0.68 * np.exp(-0.17 * (eta + 1.0) ** 2))
    return 1.0 / (np.exp(-eta) + (3.0 * np.sqrt(np.pi) / 4.0) * a ** (-3.0 / 8.0))


def n_fd(mat: Material, psi, phin=0.0):
    """Electron density under Fermi-Dirac statistics, referenced so it
    matches physics.py's Boltzmann n=ni*exp((psi-phin)/Vt) exactly in the
    non-degenerate limit (see module docstring for the Ec-Ei offset
    derivation: Ec_i = Vt*ln(Nc/ni))."""
    eta = (psi - phin) / mat.Vt - np.log(mat.Nc / mat.ni)
    return mat.Nc * fermi_half(eta)


def p_fd(mat: Material, psi, phip=0.0):
    """Hole density under Fermi-Dirac statistics, mirrored from n_fd."""
    eta = (phip - psi) / mat.Vt - np.log(mat.Nv / mat.ni)
    return mat.Nv * fermi_half(eta)


def equilibrium_bulk_potential_fd(mat: Material, Cdop: float) -> float:
    """Fermi-Dirac analog of physics.equilibrium_bulk_potential: the
    charge-neutral bulk potential psi0 solving n_fd(psi0) - p_fd(psi0) =
    Cdop exactly (both carriers evaluated with Fermi-Dirac statistics, so
    this is the fully general solve - it reduces to the Boltzmann closed
    form automatically wherever |Cdop| isn't large enough for degeneracy
    to matter, since fermi_half(eta)->exp(eta) there).

    Root-found (no closed form for F_{1/2}'s inverse) via brentq, bracketed
    generously since the Boltzmann solution is always a good starting
    estimate of where the root lives.
    """
    from physics import equilibrium_bulk_potential
    psi0_boltzmann = equilibrium_bulk_potential(mat, Cdop)

    def residual(psi0):
        return float(n_fd(mat, psi0) - p_fd(mat, psi0) - Cdop)

    # Bracket around the Boltzmann estimate - Fermi-Dirac's psi0 is always
    # on the same side of 0 and never wildly different in magnitude (F_1/2
    # saturates SLOWER than Boltzmann diverges, so |psi0_FD| >= |psi0_Boltzmann|
    # for majority-carrier concentrations approaching/exceeding Nc or Nv).
    lo, hi = psi0_boltzmann - 2.0, psi0_boltzmann + 2.0
    return brentq(residual, lo, hi, xtol=1e-12)
