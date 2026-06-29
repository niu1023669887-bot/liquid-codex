"""
Stage 0.55 — Classic cocktail anchor matching and ratio scaling.

Uses canonical IBA/Savoy proportions as calibration anchors.
When selection pattern matches a classic (score >= threshold), volumes are
scaled from the anchor rather than pure algebraic solve.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lab_metrology import attach_metrology
from reference_canon import dilution_for

ANCHOR_THRESHOLD = 0.70

_ANCHORS_PATH = Path(__file__).parent / "data" / "classic_anchors.json"
_MOCKTAIL_PATH = Path(__file__).parent / "data" / "mocktail_anchors.json"
_ANCHORS: list[dict] | None = None
_MOCKTAIL_ANCHORS: list[dict] | None = None


def load_anchors() -> list[dict]:
    global _ANCHORS
    if _ANCHORS is None:
        with open(_ANCHORS_PATH, encoding="utf-8") as f:
            _ANCHORS = json.load(f)
    return _ANCHORS


def load_mocktail_anchors() -> list[dict]:
    global _MOCKTAIL_ANCHORS
    if _MOCKTAIL_ANCHORS is None:
        if _MOCKTAIL_PATH.exists():
            with open(_MOCKTAIL_PATH, encoding="utf-8") as f:
                _MOCKTAIL_ANCHORS = json.load(f)
        else:
            _MOCKTAIL_ANCHORS = []
    return _MOCKTAIL_ANCHORS


def _category_counts(ingredients: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ing in ingredients:
        cat = ing.get("category", "")
        if cat:
            counts[cat] += 1
    return counts


def selection_has_base_spirit(selection: dict) -> bool:
    """True when selection includes spirit or a high-ABV (>20%) modifier base."""
    ingredients = selection.get("ingredients") or []
    if any(i.get("category") == "spirit" for i in ingredients):
        return True
    return any(
        (i.get("props") or {}).get("abv_pct", 0) > 0.2
        for i in ingredients
        if i.get("category") == "modifier"
    )


def _pattern_score(sel_counts: dict[str, int], pattern: dict[str, int]) -> float:
    """
    Score how well selection categories fit anchor pattern.
    Required categories must be present; unexpected extras penalise harder.
    Missing any required category (need > 0, have == 0) disqualifies the anchor.
    """
    if not pattern:
        return 0.0
    hits = 0.0
    checks = 0.0
    all_cats = set(sel_counts) | set(pattern)
    for cat in all_cats:
        need = pattern.get(cat, 0)
        have = sel_counts.get(cat, 0)
        if need > 0:
            if have <= 0:
                return 0.0  # disqualify
            checks += 1.0
            if have >= need:
                hits += 1.0
            else:
                hits += 0.5
        elif need == 0 and have > 0:
            # Unexpected category — heavier penalty than before (was 0.5/0.0)
            # e.g. dilutant in a sour anchor, or sweetener in an Old Fashioned
            checks += 1.0
            hits += 0.0
    return hits / checks if checks else 0.0


def match_classic_anchor(selection: dict, anchors: list[dict] | None = None) -> tuple[dict | None, float]:
    """
    Return (best_anchor, score) for the ingredient selection.
    score in [0, 1]; >= ANCHOR_THRESHOLD triggers anchor scaling.
    """
    ingredients = selection.get("ingredients") or []
    if not ingredients:
        return None, 0.0

    sel_style = selection.get("serve_style", "shaken")
    sel_counts = _category_counts(ingredients)
    pool = anchors if anchors is not None else load_anchors()

    best: dict | None = None
    best_score = 0.0

    for anchor in pool:
        score = 0.0
        if anchor.get("serve_style") == sel_style:
            score += 0.25
        pat_score = _pattern_score(sel_counts, anchor.get("pattern", {}))
        score += 0.65 * pat_score

        # tie-break: prefer anchor whose reference ABV is near selection target
        target_abv = float(selection.get("target_abv_pct", 18))
        ref_abv = float(anchor.get("target_abv_pct", 18))
        abv_delta = abs(target_abv - ref_abv)
        score += 0.10 * max(0.0, 1.0 - abv_delta / 15.0)

        if score > best_score:
            best_score = score
            best = anchor
        elif best and abs(score - best_score) < 0.02:
            # Tie-break: prefer higher spirit ratio when target ABV is high (Martini over Rusty Nail)
            target_abv = float(selection.get("target_abv_pct", 18))
            def _spirit_frac(a: dict) -> float:
                r = a.get("ratio") or {}
                s = sum(float(v) for v in r.values()) or 1.0
                return float(r.get("spirit", 0)) / s
            if target_abv >= 28 and _spirit_frac(anchor) > _spirit_frac(best):
                best = anchor
                best_score = score

    return best, round(best_score, 3)


def match_mocktail_anchor(selection: dict) -> tuple[dict | None, float]:
    """Spirit-free anchor pool — patterns never require spirit."""
    return match_classic_anchor(selection, load_mocktail_anchors())


def _split_volume(total: float, items: list, min_ml: int = 5) -> list[int]:
    if not items:
        return []
    n = len(items)
    each = max(round(total / n), min_ml)
    amounts = [each] * n
    diff = round(total) - sum(amounts)
    amounts[0] = max(amounts[0] + diff, min_ml)
    return amounts


def calculate_balance_from_anchor(selection: dict, anchor: dict) -> dict | None:
    """
    Scale canonical anchor ratios to user's ingredients and target volume.
    Maps selection ingredients to anchor roles by category.
    """
    from routers.ai import _estimate_ph  # lazy — avoids circular import at load time
    try:
        from ingredient_normalize import is_powder_ingredient
    except ImportError:
        def is_powder_ingredient(_: str) -> bool:
            return False

    ingredients = selection.get("ingredients") or []
    if not ingredients:
        return None

    serve_style = anchor.get("serve_style") or selection.get("serve_style") or "shaken"
    dilution = dilution_for(serve_style)
    target_vol = float(selection.get("target_volume_ml", anchor.get("reference_volume_ml", 90)))

    # Built highballs with explicit dilutant: ratios often include top-up in target_vol
    has_dilutant = any(i.get("category") == "dilutant" for i in ingredients)
    if serve_style == "built" and has_dilutant:
        v_liquid = target_vol
    else:
        v_liquid = target_vol / (1 + dilution)

    ratio = anchor.get("ratio") or {}
    ratio_sum = sum(float(v) for v in ratio.values())
    if ratio_sum <= 0:
        return None

    scale = v_liquid / ratio_sum

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for ing in ingredients:
        by_cat[ing.get("category", "modifier")].append(ing)

    result_ings: list[dict] = []
    role_map = {
        "spirit": "base",
        "acid": "acid",
        "sweetener": "sweetener",
        "modifier": "modifier",
        "dilutant": "dilutant",
        "bitters": "bitters",
    }

    total_abv_numerator = 0.0
    total_liquid = 0.0
    total_acid_g = 0.0
    total_sugar_g = 0.0

    for role, ref_ml in ratio.items():
        items = by_cat.get(role, [])
        if not items:
            continue
        total_ml = ref_ml * scale
        use_g_role = anchor.get("modifier_unit") == "g" and role == "modifier"
        min_amt = 4 if use_g_role else (5 if role != "dilutant" else 30)
        amounts = _split_volume(total_ml, items, min_ml=min_amt)
        for item, amt in zip(items, amounts):
            props = item.get("props") or {}
            use_g = (
                item.get("_unit") == "g"
                or is_powder_ingredient(item.get("name", ""))
                or use_g_role
            )
            if use_g:
                amt = max(3, min(6, round(amt))) if use_g_role else max(3, min(6, round(amt * 0.35)))
            amount_str = f"{amt} g" if use_g else f"{amt} ml"
            result_ings.append({
                "amount": amount_str,
                "name": item["name"],
                "role": role_map.get(role, role),
            })
            if not use_g:
                total_liquid += amt
            abv = props.get("abv_pct", 0.0)
            total_abv_numerator += amt * abv
            if role == "acid":
                ta = props.get("ta_pct", 5.0)
                dens = props.get("density", 1.03)
                total_acid_g += amt * ta * dens / 100
                brix = props.get("brix", 0.0)
                if brix > 0:
                    total_sugar_g += amt * brix * dens / 100
            if role == "sweetener":
                brix = props.get("brix", 50.0)
                dens = props.get("density", 1.18)
                total_sugar_g += amt * brix * dens / 100

    bitters = by_cat.get("bitters", [])
    if bitters:
        dashes = anchor.get("bitters_dashes", 2)
        for b in bitters:
            dash_ml = 1.0
            result_ings.append({
                "amount": f"{dashes} dashes",
                "name": b["name"],
                "role": "bitters",
            })
            props = b.get("props") or {}
            total_abv_numerator += dashes * dash_ml * props.get("abv_pct", 0.44)
            total_liquid += dashes * dash_ml

    if not result_ings:
        return None

    final_vol = total_liquid
    if serve_style != "built" or not has_dilutant:
        final_vol = total_liquid * (1 + dilution)

    actual_abv = (
        (total_abv_numerator / total_liquid) / (1 + dilution)
        if total_liquid > 0 and serve_style != "built"
        else (total_abv_numerator / total_liquid if total_liquid > 0 else 0)
    )
    if serve_style == "built" and has_dilutant:
        actual_abv = total_abv_numerator / total_liquid if total_liquid > 0 else 0

    engine = "mocktail-anchor" if anchor.get("target_abv_pct", 1) == 0 else "classic-anchor"
    tag = "MOCKTAIL ANCHOR" if engine == "mocktail-anchor" else "CLASSIC ANCHOR"
    style_tag = f"{serve_style}·anchor={anchor['id']}"
    return {
        "ingredients": result_ings,
        "final_abv_pct": round(actual_abv * 100, 1),
        "total_acid_g": round(total_acid_g, 2),
        "total_sugar_g": round(total_sugar_g, 1),
        "ph_estimate": _estimate_ph(total_acid_g, final_vol),
        "total_volume_ml": round(final_vol),
        "balance_notes": (
            f"[{tag} · {style_tag}] "
            f"ref={anchor.get('reference', '')} · scaled from {anchor.get('name', anchor['id'])}"
        ),
        "serve_style": serve_style,
        "_engine": engine,
        "_anchor": {
            "id": anchor["id"],
            "name": anchor.get("name", anchor["id"]),
            "family": anchor.get("family"),
            "reference": anchor.get("reference"),
        },
    }


def resolve_balance(selection: dict, deterministic_fn) -> dict | None:
    """
    Stage 0.55 + 0.6 entry: try classic anchor first, else algebraic engine.
    deterministic_fn: calculate_balance_deterministic from ai.py
    """
    # Spirit cocktails → classic anchors; mocktails → dedicated anchor pool first.
    if not selection_has_base_spirit(selection):
        mock, mscore = match_mocktail_anchor(selection)
        if mock and mscore >= ANCHOR_THRESHOLD:
            balanced = calculate_balance_from_anchor(selection, mock)
            if balanced:
                balanced["_anchor"]["match_score"] = mscore
                return attach_metrology(balanced, selection)
        return deterministic_fn(selection)

    anchor, score = match_classic_anchor(selection)
    if anchor and score >= ANCHOR_THRESHOLD:
        balanced = calculate_balance_from_anchor(selection, anchor)
        if balanced:
            balanced["_anchor"]["match_score"] = score
            return attach_metrology(balanced, selection)
    return deterministic_fn(selection)
