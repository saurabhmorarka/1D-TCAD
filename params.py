"""Physical constants and default device/material parameters for the 1D diode TCAD solver.

All quantities are in CGS-practical semiconductor units:
  length   -> cm
  time     -> s
  charge   -> C
  potential-> V
  concentration -> cm^-3
  current density -> A/cm^2
"""
from dataclasses import dataclass, field

from doping_profiles import DopingProfile


# ---- Universal constants ----
Q = 1.602176634e-19        # elementary charge, C
KB = 1.380649e-23          # Boltzmann constant, J/K
EPS0 = 8.8541878128e-14    # vacuum permittivity, F/cm


@dataclass
class Material:
    T: float = 300.0                # temperature, K
    eps_r: float = 11.7             # Si relative permittivity
    ni: float = 1.0e10              # intrinsic carrier concentration, cm^-3
    mu_n: float = 1350.0            # electron mobility, cm^2/V/s (constant)
    mu_p: float = 480.0             # hole mobility, cm^2/V/s (constant)
    tau_n: float = 1.0e-9           # SRH electron lifetime, s
    tau_p: float = 1.0e-9           # SRH hole lifetime, s
    chi_eV: float = 4.05            # Si electron affinity, eV (band diagrams only)
    Eg_eV: float = 1.12             # Si bandgap, eV (300K; band diagrams only)
    Nc: float = 2.8e19              # conduction-band effective density of states, cm^-3
                                     # (300K Si) - only used by fermi_dirac.py's degenerate
                                     # (Fermi-Dirac statistics) equilibrium relations, not by
                                     # the Boltzmann relations solve_poisson/solve_continuity_*
                                     # use throughout the actual PDE solve.
    Nv: float = 1.04e19             # valence-band effective density of states, cm^-3 (300K Si)

    @property
    def Vt(self) -> float:
        return KB * self.T / Q

    @property
    def eps(self) -> float:
        return self.eps_r * EPS0

    @property
    def Dn(self) -> float:
        return self.mu_n * self.Vt

    @property
    def Dp(self) -> float:
        return self.mu_p * self.Vt

    @property
    def Ln(self) -> float:
        return (self.Dn * self.tau_n) ** 0.5

    @property
    def Lp(self) -> float:
        return (self.Dp * self.tau_p) ** 0.5


@dataclass
class Device:
    """Step p-n junction. Region x<0 is p-type (Na), x>=0 is n-type (Nd).

    Lengths of the quasi-neutral regions are set automatically (in Grid) to a
    multiple of the relevant minority-carrier diffusion length unless overridden.
    """
    Na: float = 1.0e17   # p-side acceptor doping, cm^-3
    Nd: float = 1.0e16   # n-side donor doping, cm^-3
    area: float = 1.0e-4  # device cross-sectional area, cm^2 (1e-4 cm^2 = 100 um x 100 um)

    Wp: float = None      # p-side length, cm (None -> auto)
    Wn: float = None      # n-side length, cm (None -> auto)
    n_diffusion_lengths: float = 5.0  # how many L's of quasi-neutral region to keep

    p_profile: DopingProfile = None  # None -> flat at Na (see mesh.build_diode_grid)
    n_profile: DopingProfile = None  # None -> flat at Nd
