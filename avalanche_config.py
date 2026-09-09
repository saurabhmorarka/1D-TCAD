"""Parses the optional `avalanche:` block of an input YAML into the kwargs
newton_solver_avalanche.py / mesh.build_diode_grid's avalanche_ii_refine
need. Kept separate from config.py (which every example, avalanche or not,
already goes through) so that a normal CMOS-flow YAML with no `avalanche:`
block at all is completely unaffected - this module is only ever imported by
main_avalanche.py."""
from avalanche import AvalancheModel


def parse_avalanche_config(cfg: dict) -> dict:
    """Returns {"enabled": bool, "ii_model": AvalancheModel, "E_crit_V_cm":
    float, "cells_per_mfp": float, "fine_tail": dict|None}. Defaults to the
    standard Si van Overstraeten-de Man model and enabled=False if the
    block (or the whole file) is absent - main_avalanche.py itself always
    requests method="newton_avalanche" regardless, so "enabled" here only
    gates whether mesh.build_diode_grid's avalanche_ii_refine extra mesh
    tightening (and the voltage-sweep fine_tail refinement, if given) is
    applied, not whether the physics is modeled.

    fine_tail (optional `avalanche.fine_tail` block: {start_V, stop_V,
    step_V}) requests extra-fine bias-point spacing over [start_V, stop_V]
    (both negative, stop_V more negative than start_V), APPENDED after the
    YAML's own reverse sweep reaches start_V. Right at the onset of
    avalanche multiplication the current can rise by many orders of
    magnitude within a fraction of a volt (see
    newton_solver_avalanche.py's newton_gummel_solve docstring on the
    voltage-controlled-continuation wall) - a uniform sweep coarse enough
    to be practical over the FULL bias range is usually too coarse to
    track that narrow, sharp transition, so the fine tail exists to
    resolve just that last stretch without paying its point density
    everywhere. See main_avalanche.py's build_fine_tail_va_list()."""
    av = cfg.get("avalanche") or {}
    tail = av.get("fine_tail")
    return {
        "enabled": bool(av.get("enabled", False)),
        "ii_model": AvalancheModel.si_von_overstraeten_de_man(),
        "E_crit_V_cm": float(av.get("E_crit_V_cm", 3.0e5)),
        "cells_per_mfp": float(av.get("cells_per_mfp", 8.0)),
        "fine_tail": ({"start_V": float(tail["start_V"]),
                       "stop_V": float(tail["stop_V"]),
                       "step_V": float(tail["step_V"])}
                      if tail else None),
    }
