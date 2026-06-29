"""
Laboratory metrology layer for The Liquid Codex balance engine.

Point estimates from Stage 0.6 are deterministic but not instrument-grade.
This module attaches:
  - constant provenance (where ABV/TA/Brix came from)
  - uncertainty bands (what we can defend statistically)
  - measurement slots (for future bench backfill without UI changes)

Schema version 1.0 — backend-only; frontends may ignore `_metrology` until ready.
"""

from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = "1.0"
TIER_FORMULATION = "formulation_estimate"  # not lab_certified

# Nominal dilution by serve style + empirical bar range (ice mass, time, temp)
DILUTION_MODEL: dict[str, dict[str, float]] = {
    "shaken":  {"nominal": 0.22, "low": 0.15, "high": 0.30},
    "stirred": {"nominal": 0.17, "low": 0.12, "high": 0.22},
    "built":   {"nominal": 0.05, "low": 0.02, "high": 0.10},
    "blended": {"nominal": 0.25, "low": 0.18, "high": 0.35},
}

# Field-level uncertainty when props confidence is low / default
_PROPS_FIELD_UNCERTAINTY: dict[str, dict[str, float]] = {
    "high":   {"abv_pct": 0.02, "ta_pct": 0.5, "brix": 3.0, "density": 0.01},
    "medium": {"abv_pct": 0.04, "ta_pct": 1.0, "brix": 6.0, "density": 0.02},
    "low":    {"abv_pct": 0.08, "ta_pct": 2.0, "brix": 12.0, "density": 0.04},
}

_DISCLAIMER_EN = (
    "Computed values are formulation estimates from Codex constants and heuristic dilution. "
    "They are repeatable but not substitute for refractometer, pH meter, or distillation ABV. "
    "Use measurement_slots to record bench data and reconcile."
)
_DISCLAIMER_ZH = (
    "计算值为 Codex 常数与启发式稀释下的配方估算，可重复但不替代折光仪、pH 计或蒸馏测 ABV。"
    "请通过 measurement_slots 回填吧台实测并做对账。"
)


def classify_props_provenance(
    name: str,
    props: dict | None,
    *,
    from_perplexity: bool = False,
    from_codex_exact: bool = False,
) -> dict[str, str]:
    """Tag where physical constants originated."""
    if props is None:
        return {"source": "engine_default", "confidence": "low"}
    if from_perplexity:
        return {"source": "perplexity_cold", "confidence": "low"}
    if from_codex_exact:
        return {"source": "codex_db", "confidence": "high"}
    return {"source": "runtime_cache", "confidence": "medium"}


def _parse_amount_ml(amount: str) -> float | None:
    m = re.match(r"^([\d.]+)\s*ml\b", (amount or "").strip(), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _band(value: float, pct: float, *, floor: float = 0.0) -> dict[str, float]:
    delta = max(abs(value) * pct, 0.01)
    return {
        "value": round(value, 4),
        "low": round(max(value - delta, floor), 4),
        "high": round(value + delta, 4),
    }


def _volume_uncertainty_ml(ml: float) -> dict[str, float]:
    """Jigger rounding + hand-pour: ±2 ml or ±8%, whichever is larger."""
    delta = max(2.0, ml * 0.08)
    return {
        "value": round(ml, 1),
        "low": round(max(ml - delta, 0.0), 1),
        "high": round(ml + delta, 1),
        "unit": "ml",
        "method": "jigger_rounding",
    }


def attach_metrology(balance: dict, selection: dict | None) -> dict:
    """
    Mutates and returns balance with `_metrology` block.
    Does not change point estimates shown in locked tables.
    """
    serve_style = (balance.get("serve_style") or (selection or {}).get("serve_style") or "shaken")
    dil = DILUTION_MODEL.get(serve_style, {"nominal": 0.20, "low": 0.12, "high": 0.28})

    # ── Constant provenance from Stage 0.5 selection ─────────────────────
    constants: list[dict] = []
    low_conf_count = 0
    if selection:
        for ing in selection.get("ingredients", []):
            meta = ing.get("_props_meta") or {"source": "unknown", "confidence": "low"}
            if meta.get("confidence") == "low":
                low_conf_count += 1
            props = ing.get("props") or {}
            conf = meta.get("confidence", "low")
            field_unc = _PROPS_FIELD_UNCERTAINTY.get(conf, _PROPS_FIELD_UNCERTAINTY["low"])
            fields: dict[str, dict] = {}
            for key in ("abv_pct", "ta_pct", "brix", "density"):
                if key in props and props[key] is not None:
                    v = float(props[key])
                    u = field_unc.get(key, 0.0)
                    fields[key] = {"value": v, "uncertainty_abs": u}
            constants.append({
                "ingredient": ing.get("name", ""),
                "category": ing.get("category", ""),
                "source": meta.get("source"),
                "confidence": conf,
                "fields": fields,
            })

    # ── Output uncertainty bands ───────────────────────────────────────────
    vol_nom = float(balance.get("total_volume_ml") or 0)
    vol_low = vol_nom / (1 + dil["low"]) * (1 + dil["high"]) if vol_nom else 0
    vol_high = vol_nom / (1 + dil["high"]) * (1 + dil["low"]) if vol_nom else 0

    abv_nom = float(balance.get("final_abv_pct") or 0)
    abv_spread = 2.0 + low_conf_count * 0.8 + (dil["high"] - dil["nominal"]) * 100
    if not (selection or {}).get("ingredients"):
        abv_spread += 1.5

    acid_nom = float(balance.get("total_acid_g") or 0)
    sugar_nom = float(balance.get("total_sugar_g") or 0)
    acid_pct = 0.30 if low_conf_count else 0.15
    sugar_pct = 0.30 if low_conf_count else 0.15

    ph_nom = balance.get("ph_estimate")
    has_acid = acid_nom > 0
    ph_block: dict[str, Any]
    if has_acid and ph_nom is not None:
        ph_block = {
            "value": ph_nom,
            "low": round(max(float(ph_nom) - 0.25, 2.3), 1),
            "high": round(min(float(ph_nom) + 0.25, 5.5), 1),
            "unit": "pH",
            "method": "empirical_citric_equiv",
            "valid": True,
        }
    else:
        ph_block = {
            "value": ph_nom,
            "low": None,
            "high": None,
            "unit": "pH",
            "method": "placeholder_no_acid",
            "valid": False,
        }

    outputs = {
        "final_abv_pct": {
            "value": abv_nom,
            "low": round(max(abv_nom - abv_spread, 0.0), 1),
            "high": round(abv_nom + abv_spread, 1),
            "unit": "% ABV",
            "method": "algebraic_with_dilution_band",
        },
        "total_volume_ml": {
            "value": vol_nom,
            "low": round(vol_low, 0),
            "high": round(vol_high, 0),
            "unit": "ml",
            "method": "dilution_sensitivity",
        },
        "total_acid_g": {**_band(acid_nom, acid_pct), "unit": "g", "method": "ta_brix_conversion"},
        "total_sugar_g": {**_band(sugar_nom, sugar_pct), "unit": "g", "method": "ta_brix_conversion"},
        "ph_estimate": ph_block,
    }

    # ── Measurement slots (bench backfill) ───────────────────────────────
    slots: list[dict] = []
    for idx, ing in enumerate(balance.get("ingredients", [])):
        ml = _parse_amount_ml(ing.get("amount", ""))
        slot: dict[str, Any] = {
            "id": f"ing_{idx}",
            "ingredient": ing.get("name", ""),
            "role": ing.get("role", ""),
            "computed_amount": ing.get("amount"),
            "computed_ml": ml,
            "measured_ml": None,
            "measured_abv_pct": None,
            "delta_ml": None,
            "delta_abv_pct": None,
        }
        slots.append(slot)

    slots.extend([
        {
            "id": "final_abv",
            "field": "final_abv_pct",
            "computed": abv_nom,
            "measured": None,
            "delta": None,
            "unit": "% ABV",
        },
        {
            "id": "final_volume",
            "field": "total_volume_ml",
            "computed": vol_nom,
            "measured": None,
            "delta": None,
            "unit": "ml",
        },
        {
            "id": "final_ph",
            "field": "ph_estimate",
            "computed": ph_nom,
            "measured": None,
            "delta": None,
            "unit": "pH",
            "valid_only_if_acidic": True,
        },
    ])

    balance["_metrology"] = {
        "schema_version": SCHEMA_VERSION,
        "tier": TIER_FORMULATION,
        "disclaimer_en": _DISCLAIMER_EN,
        "disclaimer_zh": _DISCLAIMER_ZH,
        "dilution_model": {
            "serve_style": serve_style,
            "nominal_pct": round(dil["nominal"] * 100, 1),
            "range_pct": [round(dil["low"] * 100, 1), round(dil["high"] * 100, 1)],
            "parameterized": False,
            "note": "Ice mass (g), stir/shake time (s), ambient temp not yet inputs.",
        },
        "constants": constants,
        "outputs": outputs,
        "measurement_slots": slots,
        "low_confidence_ingredient_count": low_conf_count,
    }
    return balance


def reconcile_measurements(balance: dict, measurements: dict[str, Any]) -> dict:
    """
    Apply bench measurements to measurement_slots and compute deltas.
    measurements: {slot_id: value} e.g. {"ing_0": 50.5, "final_abv": 16.8, "final_ph": 3.35}
    Returns updated _metrology with measured values and reconciliation summary.
    """
    meta = balance.get("_metrology") or attach_metrology(balance, None).get("_metrology", {})
    slots = meta.get("measurement_slots", [])
    applied = 0
    deltas: list[dict] = []

    for slot in slots:
        sid = slot.get("id")
        if sid not in measurements:
            continue
        measured = measurements[sid]
        slot["measured"] = measured if "field" in slot else None

        if "computed_ml" in slot and slot["computed_ml"] is not None:
            slot["measured_ml"] = float(measured)
            slot["delta_ml"] = round(float(measured) - slot["computed_ml"], 2)
            deltas.append({
                "id": sid,
                "ingredient": slot.get("ingredient"),
                "delta_ml": slot["delta_ml"],
            })
            applied += 1
        elif slot.get("field") == "final_abv_pct":
            slot["measured"] = float(measured)
            comp = slot.get("computed") or 0
            slot["delta"] = round(float(measured) - comp, 2)
            deltas.append({"id": sid, "delta_abv_pct": slot["delta"]})
            applied += 1
        elif slot.get("field") == "total_volume_ml":
            slot["measured"] = float(measured)
            comp = slot.get("computed") or 0
            slot["delta"] = round(float(measured) - comp, 1)
            deltas.append({"id": sid, "delta_volume_ml": slot["delta"]})
            applied += 1
        elif slot.get("field") == "ph_estimate":
            slot["measured"] = float(measured)
            if slot.get("computed") is not None:
                slot["delta"] = round(float(measured) - float(slot["computed"]), 2)
                deltas.append({"id": sid, "delta_ph": slot["delta"]})
            applied += 1

    meta["reconciliation"] = {
        "applied_count": applied,
        "deltas": deltas,
        "within_uncertainty": _deltas_within_bands(deltas, meta.get("outputs", {})),
    }
    balance["_metrology"] = meta
    return balance


def _deltas_within_bands(deltas: list[dict], outputs: dict) -> bool | None:
    """True if all recorded deltas fall inside computed uncertainty bands."""
    if not deltas:
        return None
    for d in deltas:
        if "delta_abv_pct" in d:
            out = outputs.get("final_abv_pct", {})
            if d["delta_abv_pct"] < (out.get("low", 0) - out.get("value", 0)):
                return False
            if d["delta_abv_pct"] > (out.get("high", 0) - out.get("value", 0)):
                return False
    return True
