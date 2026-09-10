"""Unit tests for the material-properties catalog (core.materials,
core.material_db). Not golden-file based - these are plain assertions
about the DB's own data and the AlloyMaterial mixing formula.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import material_db
from core.materials import AlloyMaterial
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


if __name__ == "__main__":
    unittest.main()
