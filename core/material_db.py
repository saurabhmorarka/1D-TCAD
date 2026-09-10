"""The material-properties registry: a name -> MaterialProperties lookup,
meant as a common catalog future sims beyond diode/MOS/avalanche can also
draw from.

Values below are seeded from this project's existing hardcoded defaults
(core/params.py's Material, mos/mos_params.py's oxide/band constants) plus
standard literature values for entries not yet used anywhere in the solver
(Germanium, Si3N4, the named metals). mos_params.py's own GATE_WORKFUNCTION_*
floats (poly-Si and idealized-midgap work functions, not real metal
identities) are intentionally left untouched by this module - they aren't
catalog entries, and nothing about them changes here.
"""
from core.materials import MaterialProperties

MATERIALS: dict = {
    "Silicon": MaterialProperties(
        name="Silicon", category="semiconductor",
        eps_r=11.7, chi_eV=4.05, Eg_eV_300K=1.12,
        Nc_300K=2.8e19, Nv_300K=1.04e19,
        mu_n=1350.0, mu_p=480.0, tau_n=1.0e-9, tau_p=1.0e-9,
        varshni_alpha_eV_per_K=4.73e-4, varshni_beta_K=636.0,
    ),
    "Germanium": MaterialProperties(
        name="Germanium", category="semiconductor",
        eps_r=16.2, chi_eV=4.0, Eg_eV_300K=0.66,
        Nc_300K=1.04e19, Nv_300K=6.0e18,
        mu_n=3900.0, mu_p=1900.0, tau_n=1.0e-9, tau_p=1.0e-9,
        varshni_alpha_eV_per_K=4.77e-4, varshni_beta_K=235.0,
    ),
    "SiO2": MaterialProperties(
        name="SiO2", category="insulator",
        eps_r=3.9, chi_eV=0.9, Eg_eV_300K=9.0,
    ),
    "Si3N4": MaterialProperties(
        name="Si3N4", category="insulator",
        eps_r=7.5, Eg_eV_300K=5.0,
    ),
    "Ti": MaterialProperties(name="Ti", category="metal", workfunction_eV=4.33),
    "TiN": MaterialProperties(name="TiN", category="metal", workfunction_eV=4.6),
    "Al": MaterialProperties(name="Al", category="metal", workfunction_eV=4.28),
    "W": MaterialProperties(name="W", category="metal", workfunction_eV=4.55),
}


def get(name: str) -> MaterialProperties:
    """Look up a material by exact name; raises KeyError listing every
    known name on a typo (mirrors this project's existing
    ValueError-with-choices convention for math_model/gate_type)."""
    if name not in MATERIALS:
        raise KeyError(f"Unknown material {name!r}; known materials: {sorted(MATERIALS)}")
    return MATERIALS[name]
