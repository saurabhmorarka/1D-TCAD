"""A physical boundary between two materials in the mesh, distinct from a
bulk node.

The existing mesh.py per-edge/per-node arrays (eps_edge, ni_arr, is_oxide)
already answer "what's the field value at this location" - exactly right
for permittivity/carrier density, which are genuinely continuous fields
that happen to step at a material boundary, and must stay undisturbed
(that step is what makes the finite-volume flux enforce D-field continuity
today). Interface charge is a different kind of object: an areal density
(C/cm^2) that exists only at one point, not a bulk field value - Interface
gives it its own identity instead of being smuggled into a fake near-zero
bulk array.
"""
from dataclasses import dataclass


@dataclass
class Interface:
    node_index: int          # mesh node this interface sits at (mesh.py's
                              # previously-anonymous oxide_index/
                              # gate_oxide_index, now given identity)
    material_a: str          # core.material_db keys, identity only - eps/etc
    material_b: str          # still come from eps_edge, unchanged by this
    Qit_cm2: float = 0.0     # fixed interface charge density, C/cm^2 (sign
                              # as usual: +Q for positive/donor-like trapped
                              # charge). 0.0 = today's exact (electrically
                              # inert) behavior.
    Dit_cm2_eV: float = None  # interface trap DENSITY (states/cm^2/eV) -
                               # placeholder for a future bias-dependent
                               # trap-charge model; unused by any solve today.
