"""Material/device parameters for the 1D MOS capacitor C-V simulator.

Reuses Material (params.py) for the silicon substrate properties (ni, Vt,
eps_si, mobility/lifetime - the latter two are unused here, since the C-V
solves below never need continuity equations, only the equilibrium Poisson
solve; see mos_solver.py's module docstring for why).
"""
from dataclasses import dataclass

from params import EPS0
from doping_profiles import DopingProfile


EPS_OX_R = 3.9  # SiO2 relative permittivity
CHI_SI_EV = 4.05  # silicon electron affinity, eV
EG_SI_EV = 1.12   # silicon bandgap, eV (300K)
# Approximate literature SiO2 values, used only for the qualitative
# oxide-side band picture in plot.py's band diagram - not used anywhere in
# this project's actual physics (Poisson/continuity solves treat the oxide
# purely via eps_ox and zero carrier density, never via chi/Eg).
CHI_OX_EV = 0.9
EG_OX_EV = 9.0

# A few common real gate work functions, eV - pass one as gate_workfunction_eV
# to compute a realistic (non-zero) flat-band voltage instead of the ideal
# phi_ms=0 default.
GATE_WORKFUNCTION_N_POLY_EV = 4.05    # n+ polysilicon (~conduction band edge)
GATE_WORKFUNCTION_P_POLY_EV = 5.17    # p+ polysilicon (~valence band edge)
GATE_WORKFUNCTION_MIDGAP_EV = 4.61    # midgap metal (chi + Eg/2)


@dataclass
class MOSDevice:
    Na: float = 1.0e16       # uniform p-type substrate doping, cm^-3 (use negative
                              # Cdop_substrate = -Na for p-type, or pass Nd/+Cdop for n-type)
    t_ox: float = 1.0e-7     # oxide thickness, cm (default: 1 nm)
    t_si: float = None       # substrate thickness, cm (None -> auto-size, see mos_mesh.py)
    area: float = 1.0e-4     # gate area, cm^2
    gate_workfunction_eV: float = None  # metal/poly gate work function, eV; None = "ideal"
                              # MOS assumption (flat-band voltage = 0, i.e. the gate's
                              # Fermi level is defined to align with the substrate's OWN
                              # equilibrium Fermi level at V_G=0). Pass a value (e.g. one
                              # of the GATE_WORKFUNCTION_* constants above) for a realistic,
                              # generally nonzero V_FB - see mos_analytic.flatband_voltage.
    eps_ox_r: float = EPS_OX_R
    substrate_profile: DopingProfile = None  # None -> flat at Na (see mesh.build_mos_grid)

    @property
    def eps_ox(self) -> float:
        return self.eps_ox_r * EPS0
