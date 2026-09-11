"""Parses the optional `tat:` block of an input YAML into the kwargs
main_tat.py needs. Kept separate from core/config.py (which every example,
TAT or not, already goes through) so a normal CMOS-flow YAML with no `tat:`
block is completely unaffected - this module is only ever imported by
main_tat.py, mirroring avalanche/avalanche_config.py's own pattern."""
from tat.tat import KaneBTBTModel, HurkxTATModel


def parse_tat_config(cfg: dict) -> dict:
    """Returns {"kane_model": KaneBTBTModel, "hurkx_model": HurkxTATModel}.
    Defaults to the standard Si Kane-quadratic and Hurkx (midgap trap,
    m_t=0.25*m0) models if the `tat:` block (or the whole file) is absent -
    main_tat.py itself always requests method="newton_tat" regardless, so
    this only customizes the models' own parameters, not whether the
    physics is modeled."""
    t = cfg.get("tat") or {}
    kane_cfg = t.get("kane") or {}
    hurkx_cfg = t.get("hurkx") or {}

    kane_variant = kane_cfg.get("variant", "quadratic")
    kane_ctor = {
        "linear": KaneBTBTModel.si_kane_linear,
        "three_half": KaneBTBTModel.si_kane_three_half,
        "quadratic": KaneBTBTModel.si_kane_quadratic,
    }[kane_variant]
    kane_model = kane_ctor()

    hurkx_model = HurkxTATModel(
        m_t_over_m0=float(hurkx_cfg.get("m_t_over_m0", 0.25)),
        Et_minus_Ei_eV=float(hurkx_cfg.get("Et_minus_Ei_eV", 0.0)),
    )

    return {"kane_model": kane_model, "hurkx_model": hurkx_model}
