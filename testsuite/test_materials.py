"""Unit tests for the material-properties catalog (core.materials,
core.material_db). Not golden-file based - these are plain assertions
about the DB's own data and the AlloyMaterial mixing formula.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import material_db
from core.materials import AlloyMaterial, strained_sige_on_si_offsets
from core.params import Material


class TestMaterialDB(unittest.TestCase):
    def test_silicon_matches_existing_literal_defaults(self):
        si = material_db.get("Silicon")
        default = Material()
        self.assertEqual(si.eps_r, default.eps_r)
        self.assertEqual(si.chi_eV, default.chi_eV)
        self.assertEqual(si.Eg_eV_300K, default.Eg_eV)
        self.assertEqual(si.Nc_300K, default.Nc)
        self.assertEqual(si.Nv_300K, default.Nv)
        self.assertEqual(si.mu_n, default.mu_n)
        self.assertEqual(si.mu_p, default.mu_p)
        self.assertEqual(si.tau_n, default.tau_n)
        self.assertEqual(si.tau_p, default.tau_p)

    def test_unknown_material_raises_keyerror_with_choices(self):
        with self.assertRaises(KeyError) as ctx:
            material_db.get("Unobtainium")
        self.assertIn("Silicon", str(ctx.exception))

    def test_metal_entries_have_workfunction_only(self):
        for name in ("Ti", "TiN", "Al", "W"):
            mat = material_db.get(name)
            self.assertEqual(mat.category, "metal")
            self.assertIsNotNone(mat.workfunction_eV)
            self.assertIsNone(mat.eps_r)


class TestAlloyMaterial(unittest.TestCase):
    def setUp(self):
        self.sige = AlloyMaterial(name="SixGe1-x", end_member_a="Silicon", end_member_b="Germanium")

    def test_x_equals_1_reduces_to_silicon(self):
        resolved = self.sige.resolve(1.0)
        si = material_db.get("Silicon")
        self.assertAlmostEqual(resolved.eps_r, si.eps_r)
        self.assertAlmostEqual(resolved.Eg_eV_300K, si.Eg_eV_300K)
        self.assertAlmostEqual(resolved.Nc_300K, si.Nc_300K)

    def test_x_equals_0_reduces_to_germanium(self):
        resolved = self.sige.resolve(0.0)
        ge = material_db.get("Germanium")
        self.assertAlmostEqual(resolved.eps_r, ge.eps_r)
        self.assertAlmostEqual(resolved.Eg_eV_300K, ge.Eg_eV_300K)
        self.assertAlmostEqual(resolved.Nc_300K, ge.Nc_300K)

    def test_intermediate_composition_is_between_endpoints(self):
        resolved = self.sige.resolve(0.5)
        si, ge = material_db.get("Silicon"), material_db.get("Germanium")
        self.assertTrue(min(si.eps_r, ge.eps_r) <= resolved.eps_r <= max(si.eps_r, ge.eps_r))
        self.assertTrue(min(si.Eg_eV_300K, ge.Eg_eV_300K) <= resolved.Eg_eV_300K <= max(si.Eg_eV_300K, ge.Eg_eV_300K))


class TestStrainedSiGeOffsets(unittest.TestCase):
    """People & Bean compressively-strained Si(1-x)Ge(x)-on-Si band
    offsets (core.materials.strained_sige_on_si_offsets/
    AlloyMaterial.resolve_strained) - see the harmonic-snuggling-puddle
    plan and DEVELOPMENT_LOG.md for the physics and derivation."""

    def setUp(self):
        # end_member_a=Germanium so this fixture's `x` argument directly
        # equals x_Ge (matching resolve_strained's own docstring: x_Ge is
        # resolved from whichever end-member is "Germanium") - the actual
        # shipped config (input_diode_sige_pn_strained.yaml) uses the
        # OPPOSITE convention (end_member_a=Silicon, x_a=0.6 -> x_Ge=0.4),
        # covered separately below to confirm both conventions work.
        self.ge_si = AlloyMaterial(name="GexSi1-x", end_member_a="Germanium", end_member_b="Silicon")

    def test_offsets_formula_pure_si_substrate(self):
        dEc, dEv = strained_sige_on_si_offsets(0.4, x_Ge_substrate=0.0)
        self.assertEqual(dEc, 0.0)
        self.assertAlmostEqual(dEv, 0.296, places=10)

    def test_resolve_strained_zero_ge_reduces_to_substrate(self):
        resolved = self.ge_si.resolve_strained(0.0, substrate="Silicon")
        si = material_db.get("Silicon")
        self.assertAlmostEqual(resolved.Eg_eV_300K, si.Eg_eV_300K)
        self.assertAlmostEqual(resolved.chi_eV, si.chi_eV)

    def test_resolve_strained_x0p4_matches_people_and_bean(self):
        resolved = self.ge_si.resolve_strained(0.4, substrate="Silicon")
        self.assertAlmostEqual(resolved.Eg_eV_300K, 1.12 - 0.296, places=6)
        si = material_db.get("Silicon")
        self.assertAlmostEqual(resolved.chi_eV, si.chi_eV)  # delta_Ec=0 -> chi unchanged from substrate

    def test_resolve_strained_leaves_eps_Nc_unchanged_from_relaxed(self):
        strained = self.ge_si.resolve_strained(0.4, substrate="Silicon")
        relaxed = self.ge_si.resolve(0.4)
        self.assertEqual(strained.eps_r, relaxed.eps_r)
        self.assertEqual(strained.Nc_300K, relaxed.Nc_300K)
        self.assertEqual(strained.Nv_300K, relaxed.Nv_300K)
        self.assertEqual(strained.mu_n, relaxed.mu_n)
        self.assertEqual(strained.mu_p, relaxed.mu_p)
        self.assertEqual(strained.tau_n, relaxed.tau_n)
        self.assertEqual(strained.tau_p, relaxed.tau_p)
        # Eg/chi are exactly where strained and relaxed DIFFER
        self.assertNotAlmostEqual(strained.Eg_eV_300K, relaxed.Eg_eV_300K)
        self.assertNotAlmostEqual(strained.chi_eV, relaxed.chi_eV)

    def test_resolve_strained_config_convention_end_member_a_silicon(self):
        """The actual shipped config's convention (end_member_a=Silicon,
        x_a=0.6 -> x_Ge=1-0.6=0.4) gives the SAME result as the
        end_member_a=Germanium fixture above at x_Ge=0.4."""
        si_ge = AlloyMaterial(name="SixGe1-x", end_member_a="Silicon", end_member_b="Germanium", bowing_eV=0.36)
        resolved = si_ge.resolve_strained(0.6, substrate="Silicon")
        expected = self.ge_si.resolve_strained(0.4, substrate="Silicon")
        self.assertAlmostEqual(resolved.Eg_eV_300K, expected.Eg_eV_300K)
        self.assertAlmostEqual(resolved.chi_eV, expected.chi_eV)

    def test_resolve_strained_raises_for_non_sige_alloy(self):
        not_sige = AlloyMaterial(name="bogus", end_member_a="Silicon", end_member_b="SiO2")
        with self.assertRaises(ValueError):
            not_sige.resolve_strained(0.5)


if __name__ == "__main__":
    unittest.main()
