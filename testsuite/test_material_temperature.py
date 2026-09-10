"""Unit tests for temperature-dependent material properties
(core.materials.eg_at_T/nc_at_T/nv_at_T/ni_at_T/resolve_material) and the
material.name/material.T_K YAML wiring in core.config/mos.mos_config.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import material_db
from core import config as diode_cfg
from core.materials import eg_at_T, ni_at_T, nc_at_T, nv_at_T, resolve_material
from core.params import Material


class TestTemperatureFormulas(unittest.TestCase):
    def test_eg_at_reference_temperature_matches_300K_value(self):
        si = material_db.get("Silicon")
        self.assertAlmostEqual(eg_at_T(si, 300.0), si.Eg_eV_300K, places=9)

    def test_eg_decreases_with_increasing_temperature(self):
        si = material_db.get("Silicon")
        self.assertLess(eg_at_T(si, 400.0), eg_at_T(si, 300.0))

    def test_nc_nv_scale_as_T_1p5(self):
        si = material_db.get("Silicon")
        ratio = (400.0 / 300.0) ** 1.5
        self.assertAlmostEqual(nc_at_T(si, 400.0) / si.Nc_300K, ratio, places=6)
        self.assertAlmostEqual(nv_at_T(si, 400.0) / si.Nv_300K, ratio, places=6)

    def test_ni_increases_with_temperature(self):
        si = material_db.get("Silicon")
        self.assertGreater(ni_at_T(si, 400.0), ni_at_T(si, 300.0))

    def test_silicon_ni_at_300K_close_to_existing_literal_default(self):
        # Formula-derived, not bit-identical to the hardcoded 1.0e10 literal
        # (computed differently) - loose tolerance documents consistency.
        si = material_db.get("Silicon")
        default_ni = Material().ni
        self.assertAlmostEqual(ni_at_T(si, 300.0) / default_ni, 1.0, delta=0.5)

    def test_resolve_material_returns_live_material_with_requested_T(self):
        ge = material_db.get("Germanium")
        mat = resolve_material(ge, T=350.0)
        self.assertIsInstance(mat, Material)
        self.assertEqual(mat.T, 350.0)
        self.assertEqual(mat.ni, ni_at_T(ge, 350.0))
        self.assertEqual(mat.eps_r, ge.eps_r)
        self.assertEqual(mat.mu_n, ge.mu_n)


class TestYamlMaterialWiring(unittest.TestCase):
    def test_no_material_name_keeps_unchanged_default_path(self):
        mat, *_ = diode_cfg.build_from_config({})
        default = Material()
        self.assertEqual(mat.T, default.T)
        self.assertEqual(mat.ni, default.ni)
        self.assertEqual(mat.eps_r, default.eps_r)

    def test_material_name_and_T_K_resolves_from_db(self):
        cfg = {"material": {"name": "Germanium", "T_K": 350.0}}
        mat, *_ = diode_cfg.build_from_config(cfg)
        ge = material_db.get("Germanium")
        self.assertEqual(mat.T, 350.0)
        self.assertEqual(mat.ni, ni_at_T(ge, 350.0))
        self.assertEqual(mat.eps_r, ge.eps_r)

    def test_T_K_without_name_raises(self):
        cfg = {"material": {"T_K": 350.0}}
        with self.assertRaises(ValueError):
            diode_cfg.build_from_config(cfg)

    def test_explicit_override_applies_after_db_resolution(self):
        cfg = {"material": {"name": "Silicon", "T_K": 300.0, "ni_cm3": 5.0e9}}
        mat, *_ = diode_cfg.build_from_config(cfg)
        self.assertEqual(mat.ni, 5.0e9)


if __name__ == "__main__":
    unittest.main()
