"""Closed-form (depletion-approximation) MOS capacitor C-V theory, for
comparison against the numerical solve. Formulas follow the standard
treatment (e.g. Sze, "Physics of Semiconductor Devices"; Pierret,
"Semiconductor Device Fundamentals"), generalized to either substrate type
via Cdop_substrate (negative = p-type, positive = n-type).
"""
import numpy as np
from scipy.optimize import brentq

from params import Q, Material
from mos_params import MOSDevice, CHI_SI_EV, EG_SI_EV
from physics import equilibrium_bulk_potential


def C_ox(dev: MOSDevice) -> float:
    """Oxide capacitance per unit area, F/cm^2."""
    return dev.eps_ox / dev.t_ox


def bulk_potential(mat: Material, Cdop_substrate: float) -> float:
    """psi_bulk: substrate's own equilibrium potential (intrinsic-level
    reference) - phi_F=|psi_bulk| is the standard "Fermi potential"."""
    return equilibrium_bulk_potential(mat, Cdop_substrate)


def depletion_charge(mat: Material, Cdop_substrate: float, psi_s: float) -> float:
    """Depletion-approximation semiconductor charge per unit area (C/cm^2)
    for surface band-bending psi_s (relative to the bulk), depletion
    approximation: |Q_dep| = sqrt(2*eps_si*q*|Nsub|*|psi_s|).
    Sign: positive charge for p-type in depletion/inversion (psi_s>0,
    ionized acceptors are negative... - we return the SEMICONDUCTOR charge
    with the physically correct sign: opposite sign to Cdop_substrate for
    band-bending that depletes majority carriers."""
    Nsub = abs(Cdop_substrate)
    sign = -np.sign(Cdop_substrate)  # p-type (Cdop<0) depletes to expose negative acceptor charge... see docstring
    return sign * np.sqrt(2 * mat.eps * Q * Nsub * abs(psi_s))


def semiconductor_workfunction_eV(mat: Material, Cdop_substrate: float) -> float:
    """phi_S = chi_Si + Eg/2 + phi_F (p-type, Fermi level phi_F below
    midgap) or chi_Si + Eg/2 - phi_F (n-type, Fermi level phi_F above
    midgap), the standard silicon work-function formula referenced to the
    vacuum level."""
    Nsub = abs(Cdop_substrate)
    phi_F = mat.Vt * np.log(Nsub / mat.ni)  # >0 magnitude of the Fermi level's offset from midgap
    sign = 1.0 if Cdop_substrate < 0 else -1.0  # p-type: Ef below midgap -> larger work function
    return CHI_SI_EV + EG_SI_EV / 2.0 + sign * phi_F


def flatband_voltage(dev: MOSDevice, mat: Material = None, Cdop_substrate: float = None) -> float:
    """Flat-band voltage V_FB = phi_M - phi_S (metal work function minus
    semiconductor work function), the gate voltage at which there is no
    band bending at all.

    If dev.gate_workfunction_eV is None, this is the "ideal MOS" assumption
    used throughout this module: V_FB=0, i.e. the gate's Fermi level is
    simply defined to align with the substrate's own equilibrium Fermi
    level at V_G=0 (flat bands there by construction). Passing a real gate
    work function (mat and Cdop_substrate also required then) computes the
    actual, generally nonzero V_FB from the work-function difference -
    this is what determines exactly where the accumulation/depletion/
    inversion regions fall along the V_G axis for a real gate material.
    """
    if dev.gate_workfunction_eV is None:
        return 0.0
    phi_S = semiconductor_workfunction_eV(mat, Cdop_substrate)
    return dev.gate_workfunction_eV - phi_S


def surface_potential_depletion_approx(mat: Material, dev: MOSDevice, Cdop_substrate: float, VG: float) -> float:
    """Solve VG - V_FB = psi_s - Q_dep(psi_s)/Cox for psi_s (depletion
    approximation; ignores the accumulation-layer and inversion-layer
    charge, so only valid in depletion, up to the onset of strong
    inversion)."""
    Cox_ = C_ox(dev)
    V_FB = flatband_voltage(dev, mat, Cdop_substrate)

    def residual(psi_s):
        return (VG - V_FB) - (psi_s - depletion_charge(mat, Cdop_substrate, psi_s) / Cox_)

    # bracket a reasonable range for psi_s (band-bending magnitude bounded by a few volts)
    lo, hi = -2.0, 2.0
    return brentq(residual, lo, hi, xtol=1e-12)


def threshold_voltage(mat: Material, dev: MOSDevice, Cdop_substrate: float) -> float:
    """Classic V_T formula: onset of strong inversion at psi_s = 2*phi_F."""
    Nsub = abs(Cdop_substrate)
    phi_F = mat.Vt * np.log(Nsub / mat.ni)
    psi_s_th = 2 * phi_F
    Q_dep_max = np.sqrt(2 * mat.eps * Q * Nsub * psi_s_th)
    V_FB = flatband_voltage(dev, mat, Cdop_substrate)
    sign = 1.0 if Cdop_substrate < 0 else -1.0  # p-sub: V_T > V_FB; n-sub: V_T < V_FB
    return V_FB + sign * (psi_s_th + Q_dep_max / C_ox(dev))


def max_depletion_width(mat: Material, Cdop_substrate: float) -> float:
    Nsub = abs(Cdop_substrate)
    phi_F = mat.Vt * np.log(Nsub / mat.ni)
    return np.sqrt(4 * mat.eps * phi_F / (Q * Nsub))


def cv_curve_lowfreq_depletion_approx(mat: Material, dev: MOSDevice, Cdop_substrate: float, VG_arr):
    """Analytic low-frequency C-V using the depletion approximation:
    accumulation/depletion from the implicit surface-potential relation,
    strong inversion clamped at psi_s=2*phi_F with C->Cox (inversion layer
    charge screens the semiconductor, low-frequency = it can respond)."""
    Nsub = abs(Cdop_substrate)
    phi_F = mat.Vt * np.log(Nsub / mat.ni)
    V_T = threshold_voltage(mat, dev, Cdop_substrate)
    Cox_ = C_ox(dev)
    is_p = Cdop_substrate < 0

    C = np.empty_like(VG_arr, dtype=float)
    for i, VG in enumerate(VG_arr):
        past_threshold = (VG >= V_T) if is_p else (VG <= V_T)
        if past_threshold:
            C[i] = Cox_  # low-frequency: inversion charge fully responds, screens like accumulation
        else:
            psi_s = surface_potential_depletion_approx(mat, dev, Cdop_substrate, VG)
            if (is_p and psi_s <= 0) or (not is_p and psi_s >= 0):
                C[i] = Cox_  # accumulation: majority carriers screen right at the interface
            else:
                Q_dep = abs(depletion_charge(mat, Cdop_substrate, psi_s))
                W = Q_dep / (Q * Nsub)
                C_dep = mat.eps / max(W, 1e-30)
                C[i] = 1.0 / (1.0 / Cox_ + 1.0 / C_dep)
    return C


def cv_curve_highfreq_depletion_approx(mat: Material, dev: MOSDevice, Cdop_substrate: float, VG_arr):
    """Analytic high-frequency C-V: depletion width (and hence capacitance)
    saturates at its value at threshold (W_max) once in strong inversion,
    since the inversion charge can't respond to a fast small-signal probe."""
    Nsub = abs(Cdop_substrate)
    V_T = threshold_voltage(mat, dev, Cdop_substrate)
    Cox_ = C_ox(dev)
    W_max = max_depletion_width(mat, Cdop_substrate)
    C_min = 1.0 / (1.0 / Cox_ + W_max / mat.eps)
    is_p = Cdop_substrate < 0

    C = np.empty_like(VG_arr, dtype=float)
    for i, VG in enumerate(VG_arr):
        past_threshold = (VG >= V_T) if is_p else (VG <= V_T)
        if past_threshold:
            C[i] = C_min
        else:
            psi_s = surface_potential_depletion_approx(mat, dev, Cdop_substrate, VG)
            if (is_p and psi_s <= 0) or (not is_p and psi_s >= 0):
                C[i] = Cox_
            else:
                Q_dep = abs(depletion_charge(mat, Cdop_substrate, psi_s))
                W = Q_dep / (Q * Nsub)
                C_dep = mat.eps / max(W, 1e-30)
                C[i] = 1.0 / (1.0 / Cox_ + 1.0 / C_dep)
    return C
