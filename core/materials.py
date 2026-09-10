"""Material-properties catalog schema.

Distinct from core.params.Material on purpose: Material is the *live,
T-resolved, solver-facing* object each simulation actually uses (has a
concrete T, has derived @property quantities like Vt/Dn/Dp). MaterialProperties
is the *catalog* entry a named material is looked up as (no T, no derived
properties) - the same separation this project already uses between
DopingProfile (pure data) and Material (data + derived @property values).

All quantities use this project's existing CGS-practical unit convention
(see params.py's module docstring): length -> cm, energy -> eV where noted,
concentration -> cm^-3.
"""
from dataclasses import dataclass


@dataclass
class MaterialProperties:
    """One catalog entry: a named material's fixed reference-temperature
    properties. Fields that don't apply to a given category (e.g. eps_r for
    a bare metal work-function entry) are left None rather than given a
    fabricated numeric default.
    """
    name: str
    category: str                          # "semiconductor" | "insulator" | "metal"

    eps_r: float = None                    # relative permittivity
    chi_eV: float = None                   # electron affinity, eV (semiconductors)
    Eg_eV_300K: float = None               # bandgap at 300K, eV (semiconductors)
    Nc_300K: float = None                  # conduction-band DOS at 300K, cm^-3
    Nv_300K: float = None                  # valence-band DOS at 300K, cm^-3
    mu_n: float = None                     # electron mobility, cm^2/V/s (constant-mobility model)
    mu_p: float = None                     # hole mobility, cm^2/V/s
    tau_n: float = None                    # SRH electron lifetime, s
    tau_p: float = None                    # SRH hole lifetime, s
    workfunction_eV: float = None          # metals: bare work function (no chi/Eg split)

    # Varshni Eg(T) coefficients - None (no scaling data) leaves Eg(T) flat
    # at Eg_eV_300K wherever it's used (see materials.eg_at_T).
    varshni_alpha_eV_per_K: float = None
    varshni_beta_K: float = None

    # Room for future mechanical-property work (not consumed by any solver
    # yet) - left optional so existing entries need no changes when used.
    youngs_modulus_GPa: float = None
    poisson_ratio: float = None


@dataclass
class AlloyMaterial:
    """A composition-dependent alloy, resolved on demand into a
    MaterialProperties via Vegard's-law linear mixing of its two
    end-members (plus an optional quadratic bowing correction on Eg).

    Deliberately NOT a MATERIALS dict entry itself, and NOT the same
    mechanism as material_db.derive() (discrete copy-and-override of one
    fixed entry): an alloy is a *parametric* family (a continuum of
    possible compositions), while derive() produces one new *discrete*
    named entry. Kept as a separate concept so neither mechanism has to
    understand the other's math.
    """
    name: str
    end_member_a: str    # material_db key, composition fraction x=1
    end_member_b: str    # material_db key, composition fraction x=0
    bowing_eV: float = 0.0

    def resolve(self, x: float) -> MaterialProperties:
        """x = fraction of end_member_a, 0 <= x <= 1. Returns a new
        MaterialProperties with every numeric field linearly interpolated
        between the two end-members (None if either end-member's field is
        None), plus the standard quadratic bowing correction on Eg_eV_300K:
        Eg(x) = x*Eg_a + (1-x)*Eg_b - bowing*x*(1-x).
        """
        from core import material_db

        a = material_db.get(self.end_member_a)
        b = material_db.get(self.end_member_b)

        def mix(field_name):
            va, vb = getattr(a, field_name), getattr(b, field_name)
            if va is None or vb is None:
                return None
            return x * va + (1 - x) * vb

        eg = mix("Eg_eV_300K")
        if eg is not None:
            eg = eg - self.bowing_eV * x * (1 - x)

        return MaterialProperties(
            name=f"Si{x:.2f}Ge{1 - x:.2f}" if self.name == "SixGe1-x" else f"{self.name}(x={x:.2f})",
            category="semiconductor",
            eps_r=mix("eps_r"),
            chi_eV=mix("chi_eV"),
            Eg_eV_300K=eg,
            Nc_300K=mix("Nc_300K"),
            Nv_300K=mix("Nv_300K"),
            mu_n=mix("mu_n"),
            mu_p=mix("mu_p"),
            tau_n=mix("tau_n"),
            tau_p=mix("tau_p"),
            varshni_alpha_eV_per_K=mix("varshni_alpha_eV_per_K"),
            varshni_beta_K=mix("varshni_beta_K"),
        )
