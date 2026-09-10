"""Unit tests for material inheritance/derivation (core.material_db.derive
and the material.derive_from/overrides YAML wiring in core.config).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import material_db
from core import config as diode_cfg


class TestDerive(unittest.TestCase):
    def test_derive_overrides_only_given_fields(self):
        derived = material_db.derive("TestSi1", "Silicon", tau_n=5.0e-9, tau_p=5.0e-9)
        si = material_db.get("Silicon")
        self.assertEqual(derived.tau_n, 5.0e-9)
        self.assertEqual(derived.tau_p, 5.0e-9)
        # Inherited fields untouched.
        self.assertEqual(derived.eps_r, si.eps_r)
        self.assertEqual(derived.chi_eV, si.chi_eV)
        self.assertEqual(derived.Eg_eV_300K, si.Eg_eV_300K)

    def test_derive_registers_in_the_db(self):
        material_db.derive("TestSi2", "Silicon", mu_n=1000.0)
        self.assertIn("TestSi2", material_db.MATERIALS)
        self.assertIs(material_db.get("TestSi2"), material_db.MATERIALS["TestSi2"])

    def test_derive_from_material_properties_object_directly(self):
        si = material_db.get("Silicon")
        derived = material_db.derive("TestSi3", si, eps_r=12.0)
        self.assertEqual(derived.eps_r, 12.0)
        self.assertEqual(derived.mu_n, si.mu_n)


class TestYamlDerivation(unittest.TestCase):
    def test_inline_derive_from_yaml(self):
        cfg = {
            "material": {
                "name": "TestSi4",
                "derive_from": "Silicon",
                "T_K": 300.0,
                "overrides": {"tau_n_ns": 5.0, "tau_p_ns": 5.0},
            }
        }
        mat, *_ = diode_cfg.build_from_config(cfg)
        self.assertEqual(mat.tau_n, 5.0e-9)
        self.assertEqual(mat.tau_p, 5.0e-9)
        self.assertEqual(mat.eps_r, material_db.get("Silicon").eps_r)

    def test_unknown_name_without_derive_from_raises(self):
        cfg = {"material": {"name": "TotallyMadeUp"}}
        with self.assertRaises(ValueError):
            diode_cfg.build_from_config(cfg)


if __name__ == "__main__":
    unittest.main()
