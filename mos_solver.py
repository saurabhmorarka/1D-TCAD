"""MOS capacitor equilibrium solve and C-V sweep.

Unlike the diode, a MOS capacitor has no current path at all in steady
state (the gate is an ideal insulator) - so at every DC gate voltage the
whole structure sits at a single, uniform Fermi level (phin=phip=0
throughout, same convention as the diode's equilibrium solve), *provided*
enough time has passed for generation-recombination to fully populate any
inversion layer. That assumption is exactly the "quasi-static" or
"low-frequency" C-V measurement: no continuity equations, no G-R kinetics,
and no AC/small-signal analysis are needed for it - just a sequence of
equilibrium nonlinear-Poisson solves (physics.solve_poisson, reused as-is
with an oxide/semiconductor eps(x) and ni(x) profile), one per gate
voltage, differentiated numerically: C(V_G) = dQ_s/dV_G.

The high-frequency C-V curve, where minority (inversion) carriers can't
follow a fast small-signal probe, is a genuine "quasi-small-signal"
calculation: take the low-frequency solution's minority-carrier density at
a given DC gate bias, freeze it (via physics.solve_poisson's n_frozen/
p_frozen), perturb the gate voltage by a small delta, and let only the
majority carrier and potential respond. See the module docstring exchange
in DEVELOPMENT_LOG.md for the reasoning behind not needing a literal
frequency-domain AC solve for either curve.
"""
import numpy as np

import physics as ph
from params import Material
from mos_params import MOSDevice
import mos_analytic as man


def solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat: Material, dev: MOSDevice,
                           Cdop_substrate, VG, psi_bulk,
                           psi_init=None, n_frozen=None, p_frozen=None,
                           damping_cap=0.5, max_iter=200):
    """Single-gate-voltage equilibrium (or quasi-small-signal, if
    n_frozen/p_frozen given) solve. Returns dict with psi, n, p, iters."""
    N = len(x)
    V_FB = man.flatband_voltage(dev, mat, Cdop_substrate)
    psi_gate_bc = psi_bulk + (VG - V_FB)

    if psi_init is None:
        frac = np.linspace(1.0, 0.0, N)
        psi_guess = psi_bulk + (psi_gate_bc - psi_bulk) * frac
    else:
        psi_guess = psi_init.copy()
    psi_guess[0] = psi_gate_bc
    psi_guess[-1] = psi_bulk

    phin = np.zeros(N)
    phip = np.zeros(N)
    psi, n, p, iters = ph.solve_poisson(
        x, Cdop, mat, phin, phip, psi_guess,
        eps=eps_edge, ni=ni_arr, damping_cap=damping_cap, max_iter=max_iter,
        n_frozen=n_frozen, p_frozen=p_frozen,
    )
    # Quasi-Fermi potentials are only meaningful where there are carriers to
    # define them for (ni_arr>0, i.e. the semiconductor) - in the oxide,
    # n=p=0 identically (it's an insulator) and phin/phip are undefined, not
    # some huge/meaningless number, so they're masked to NaN there.
    is_semi = ni_arr > 0
    ni_safe = np.where(is_semi, ni_arr, 1.0)
    phin = np.where(is_semi, psi - mat.Vt * np.log(np.clip(n, 1e-300, None) / ni_safe), np.nan)
    phip = np.where(is_semi, psi + mat.Vt * np.log(np.clip(p, 1e-300, None) / ni_safe), np.nan)
    return {"psi": psi, "n": n, "p": p, "iters": iters, "VG": VG, "phin": phin, "phip": phip}


def semiconductor_charge(x, psi, dev: MOSDevice, oxide_index):
    """Total semiconductor-side charge per unit area (C/cm^2), from Gauss's
    law applied to the (uniform, since charge-free) field across the oxide:
    Q_s = eps_ox * (psi_interface - psi_gate) / t_ox. (Sign checked against
    accumulation, where excess majority-carrier charge at the surface must
    come out positive for a p-substrate.)"""
    psi_gate = psi[0]
    psi_interface = psi[oxide_index]
    return dev.eps_ox * (psi_interface - psi_gate) / dev.t_ox


def cv_sweep(x, Cdop, eps_edge, ni_arr, mat: Material, dev: MOSDevice,
             Cdop_substrate, VG_list, oxide_index, dV_hf=2e-3):
    """Low- and high-frequency C-V sweep across VG_list (with continuation
    for robustness/speed). Returns a list of per-point result dicts."""
    psi_bulk = ph.equilibrium_bulk_potential(mat, Cdop_substrate)
    is_p_sub = Cdop_substrate < 0

    results = []
    psi_prev = None
    for VG in VG_list:
        lf = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate,
                                    VG, psi_bulk, psi_init=psi_prev)
        Qs_lf = semiconductor_charge(x, lf["psi"], dev, oxide_index)

        # High-frequency point: freeze the MINORITY carrier (electrons for a
        # p-substrate, holes for an n-substrate) at its low-frequency value,
        # perturb VG by a small dV, and only let the majority carrier and
        # potential respond.
        n_frozen = p_frozen = None
        if is_p_sub:
            n_frozen = lf["n"].copy()
        else:
            p_frozen = lf["p"].copy()

        hf_pert = solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate,
                                         VG + dV_hf, psi_bulk, psi_init=lf["psi"],
                                         n_frozen=n_frozen, p_frozen=p_frozen)
        Qs_hf_pert = semiconductor_charge(x, hf_pert["psi"], dev, oxide_index)
        # C = dQ_gate/dVG = -dQ_semiconductor/dVG (charge neutrality of the
        # two-terminal capacitor: gate charge is the negative of the
        # semiconductor charge, and it's the gate charge that increases
        # with VG, giving the conventional positive capacitance).
        C_hf = -(Qs_hf_pert - Qs_lf) / dV_hf

        results.append({
            "VG": VG, "psi": lf["psi"], "n": lf["n"], "p": lf["p"],
            "phin": lf["phin"], "phip": lf["phip"],
            "iters": lf["iters"], "Qs": Qs_lf, "C_hf": C_hf,
        })
        psi_prev = lf["psi"]

    Q_arr = np.array([r["Qs"] for r in results])
    VG_arr = np.array([r["VG"] for r in results])
    # Low-frequency capacitance: numerical dQ_gate/dV = -dQ_semiconductor/dV
    # of the equilibrium sweep itself (see the C_hf sign note above).
    C_lf = -np.gradient(Q_arr, VG_arr)
    for r, c in zip(results, C_lf):
        r["C_lf"] = c

    return results
