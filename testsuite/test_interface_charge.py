"""Verification of the interface-charge source term added to
core.physics.solve_poisson (core/interfaces.py's Interface, wired through
core/mesh.py's build_mos_grid and mos/mos_solver.py).

The check is a direct application of Gauss's law's boundary-charge jump
condition at the oxide/substrate interface node: the D-field just inside
the substrate minus the D-field just inside the oxide must equal the local
mobile/doping charge (as it always did) MINUS the added Qit_cm2 - this is
exactly what solve_poisson's new residual term computes, so verifying it
against the raw converged psi values is an exact check (not an
approximate/analytic-theory comparison), independent of the coarser
oxide-spanning field estimate mos_solver.semiconductor_charge uses (which
carries some residual Newton-tolerance noise of its own, unrelated to this
feature - confirmed by how tightly the direct adjacent-node check below
matches, versus that coarser measure).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.params import Material, Q
from mos.mos_params import MOSDevice
from core.mesh import build_mos_grid
from core import physics as ph
from mos import mos_solver as ms


def _gauss_jump_check(mat, dev, Cdop_substrate, VG):
    """Returns (Dsi - Dox, predicted) at the oxide/substrate interface node
    for a converged equilibrium solve - should match to near machine
    precision."""
    g = build_mos_grid(mat, dev, Cdop_substrate, Cdop_gate=None)
    x, Cdop, eps_edge, ni_arr = g["x"], g["Cdop"], g["eps_edge"], g["ni_arr"]
    psi_bulk = ph.equilibrium_bulk_potential(mat, Cdop_substrate)
    r = ms.solve_mos_equilibrium(x, Cdop, eps_edge, ni_arr, mat, dev, Cdop_substrate,
                                  VG, psi_bulk, interfaces=g["interfaces"])
    oi = g["oxide_index"]
    psi = r["psi"]

    hm = x[oi] - x[oi - 1]
    hp = x[oi + 1] - x[oi]
    cvol = (hm + hp) / 2.0
    D_ox = dev.eps_ox * (psi[oi] - psi[oi - 1]) / hm
    D_si = mat.eps * (psi[oi + 1] - psi[oi]) / hp

    n_oi = mat.ni * np.exp(psi[oi] / mat.Vt)
    p_oi = mat.ni * np.exp(-psi[oi] / mat.Vt)
    charge_term = Q * (n_oi - p_oi - Cdop[oi])
    predicted = cvol * charge_term - dev.Qit_cm2

    return D_si - D_ox, predicted


class TestInterfaceChargeGaussLaw(unittest.TestCase):
    def test_zero_Qit_matches_standard_gauss_law(self):
        mat, dev = Material(), MOSDevice()
        dev.Qit_cm2 = 0.0
        actual, predicted = _gauss_jump_check(mat, dev, -dev.Na, VG=0.3)
        self.assertAlmostEqual(actual / predicted, 1.0, places=4)

    def test_nonzero_Qit_matches_gauss_law_with_interface_charge(self):
        mat, dev = Material(), MOSDevice()
        dev.Qit_cm2 = 1.0e11 * Q  # a realistic fixed interface charge density
        actual, predicted = _gauss_jump_check(mat, dev, -dev.Na, VG=0.3)
        self.assertAlmostEqual(actual / predicted, 1.0, places=4)

    def test_Qit_shifts_the_jump_by_exactly_Qit(self):
        mat, dev = Material(), MOSDevice()
        Qit_test = 1.0e11 * Q
        dev.Qit_cm2 = 0.0
        jump_base, _ = _gauss_jump_check(mat, dev, -dev.Na, VG=0.3)
        dev.Qit_cm2 = Qit_test
        jump_shift, _ = _gauss_jump_check(mat, dev, -dev.Na, VG=0.3)
        # Local charge term at this node is unaffected by Qit (same psi[oi]
        # to a good approximation at fixed VG for this small a Qit), so the
        # jump should shift by essentially -Qit_test.
        self.assertAlmostEqual((jump_shift - jump_base) / (-Qit_test), 1.0, places=2)

    def test_default_Qit_is_zero_and_matches_pre_feature_behavior(self):
        # dev.Qit_cm2 defaults to 0.0 - build_mos_grid's interfaces list is
        # populated either way, but with a zero charge the added residual
        # term is additively zero, so this is just confirming the default
        # construction path is inert.
        dev = MOSDevice()
        self.assertEqual(dev.Qit_cm2, 0.0)
        g = build_mos_grid(Material(), dev, -dev.Na, Cdop_gate=None)
        self.assertEqual(len(g["interfaces"]), 1)
        self.assertEqual(g["interfaces"][0].Qit_cm2, 0.0)


if __name__ == "__main__":
    unittest.main()
