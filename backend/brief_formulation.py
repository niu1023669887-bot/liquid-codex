"""
Deterministic brief → formulation enrichment.

Runs after LLM ingredient selection, before balance/anchor.
Handles under-specified mocktails and disambiguates common ingredient confusions.
"""

from __future__ import annotations

import re
from typing import Any

from classic_anchor import selection_has_base_spirit
from ingredient_normalize import apply_synonym_hints, canonical_name


def _brief_blob(body: Any) -> str:
    return " ".join([
        getattr(body, "ingredients", "") or "",
        getattr(body, "notes", "") or "",
        " ".join(getattr(body, "flavors", []) or []),
        getattr(body, "alcohol", "") or "",
    ]).lower()


# ── Coconut intent discrimination ──────────────────────────────────────────────
_COCONUT_BREEZE_RE = re.compile(
    r"椰子微风|椰林飘香|椰影|椰香|清冽.*椰|椰.*清冽|"
    r"coconut\s*(?:breeze|cooler|highball|spritz)|coco(?:nut)?\s*breeze",
    re.IGNORECASE,
)
_COCONUT_WATER_RE = re.compile(
    r"椰子水|椰青水|coconut\s+water",
    re.IGNORECASE,
)
_COCONUT_CREAM_RE = re.compile(
    r"椰奶|椰浆|椰子奶|coconut\s+(?:cream|milk)|cream\s+of\s+coconut|pina\s+colada",
    re.IGNORECASE,
)
_COCONUT_MEAT_RE = re.compile(
    r"椰子肉|椰肉|椰丝|coconut\s+meat",
    re.IGNORECASE,
)
_COCONUT_TERM_RE = re.compile(r"椰子|coconut", re.IGNORECASE)


def _attach_props(ing: dict) -> dict:
    from routers.ai import INGREDIENT_PROPS, _lookup_ingredient_props

    out = dict(ing)
    name = out.get("name", "")
    props = _lookup_ingredient_props(name)
    if props is None:
        key = name.lower().strip()
        props = INGREDIENT_PROPS.get(key)
    out["props"] = props
    return out


def _has_category(ings: list[dict], cat: str) -> bool:
    return any(i.get("category") == cat for i in ings)


def _ingredient_names(ings: list[dict]) -> set[str]:
    return {(i.get("name") or "").lower().strip() for i in ings}


def enrich_selection_from_brief(selection: dict, body: Any) -> dict:
    """
    Expand under-specified spirit-free selections to match brief intent.
    Does not add ingredients the user explicitly forbade in strict mode.
    """
    if not selection or selection_has_base_spirit(selection):
        return selection

    blob = _brief_blob(body)
    strict = bool(getattr(body, "strict_ingredients", False))
    out = dict(selection)
    ings = [dict(i) for i in (selection.get("ingredients") or [])]

    # ── Coconut intent: water vs cream vs meat ────────────────────────────────
    # Only apply coconut water highball preset when user clearly means a light cooler,
    # NOT when they mean piña colada (coconut cream) or cooking (coconut meat/milk).
    coconut_is_cream = bool(_COCONUT_CREAM_RE.search(blob))
    coconut_is_water = bool(_COCONUT_WATER_RE.search(blob))
    coconut_is_meat = bool(_COCONUT_MEAT_RE.search(blob))
    coconut_mentioned = bool(_COCONUT_TERM_RE.search(blob))

    needs_coconut_highball = (
        not coconut_is_cream
        and not coconut_is_meat
        and (
            bool(_COCONUT_BREEZE_RE.search(blob))
            or (
                coconut_mentioned
                and not coconut_is_water   # bare "椰子" without qualifier
                and not (_has_category(ings, "acid") and _has_category(ings, "dilutant"))
            )
        )
    )

    if needs_coconut_highball:
        names = _ingredient_names(ings)
        rebuilt: list[dict] = []
        for ing in ings:
            n = (ing.get("name") or "").strip()
            low = n.lower()
            # Upgrade bare "椰子" / "coconut" to coconut water
            if low in ("椰子", "coconut") and "coconut water" not in names and not _COCONUT_MEAT_RE.search(n):
                ing = dict(ing)
                ing["name"] = "coconut water"
                ing["category"] = "dilutant"
                ing["note"] = "base"
            rebuilt.append(ing)
        ings = rebuilt
        names = _ingredient_names(ings)

        if not _has_category(ings, "dilutant") and "coconut water" not in names:
            ings.append({"name": "coconut water", "category": "dilutant", "note": "base"})
        if not _has_category(ings, "acid") and not strict:
            ings.append({"name": "lime juice", "category": "acid", "note": "acid"})
        if not _has_category(ings, "sweetener") and not strict:
            ings.append({"name": "simple syrup", "category": "sweetener", "note": "sweet"})
        out["serve_style"] = "built"
        out["target_abv_pct"] = 0.0
        out["target_volume_ml"] = max(float(out.get("target_volume_ml") or 0), 180)
        out["sweet_acid_ratio"] = float(out.get("sweet_acid_ratio") or 2.0)

    ings = apply_synonym_hints(ings)
    out["ingredients"] = [_attach_props(i) for i in ings]
    return out


def mocktail_structure_ok(selection: dict) -> tuple[bool, str]:
    """
    Validate that a spirit-free selection can form a drinkable serve.
    Shaken mocktails must have BOTH acid AND sweetener (not just one).
    """
    if selection_has_base_spirit(selection):
        return True, "ok"
    ings = selection.get("ingredients") or []
    if len(ings) < 2:
        return False, "Mocktail needs at least two ingredients (e.g. acid + dilutant)."
    cats = {i.get("category") for i in ings}
    style = selection.get("serve_style", "shaken")

    if style == "built" and "dilutant" not in cats:
        return False, "Built mocktail/highball requires a dilutant (coconut water, soda, etc.)."

    if style in ("shaken", "blended"):
        has_acid = "acid" in cats
        has_sweet = "sweetener" in cats
        # High-Brix modifier (e.g. elderflower cordial, juice concentrate) can sub for sweetener
        has_sweet_modifier = any(
            (i.get("props") or {}).get("brix", 0) >= 40
            for i in ings
            if i.get("category") == "modifier"
        )
        if not has_acid:
            return False, "Shaken mocktail requires at least one acid ingredient (citrus juice, etc.)."
        if not has_sweet and not has_sweet_modifier:
            return False, "Shaken mocktail requires sweetener or a high-Brix modifier to balance acidity."

    return True, "ok"


def concept_mentions_forbidden_food(concept: dict, allowed: list[str]) -> list[str]:
    """Detect hallucinated foods in title/tagline outside allowed vocabulary."""
    blob = " ".join([
        concept.get("title_primary", ""),
        concept.get("title_secondary", ""),
        concept.get("tagline", ""),
    ]).lower()
    allowed_blob = " ".join(allowed).lower()
    violations: list[str] = []
    _FORBIDDEN_CONCEPT = re.compile(
        r"牛肉干|beef\s*jerky|培根|bacon(?!\s*fat)|烟熏肉|smoked\s+meat|"
        r"迷迭香|rosemary|薄荷枝|mint\s+sprig|"
        r"烟草|tobacco|烟熏|(?<!\w)smoke(?!\w)",
        re.IGNORECASE,
    )
    for m in _FORBIDDEN_CONCEPT.finditer(blob):
        word = m.group(0)
        if word.lower() not in allowed_blob and not any(word.lower() in a.lower() for a in allowed):
            violations.append(word)
    return violations


def clamp_concept_to_brief(concept: dict, body: Any, allowed: list[str]) -> dict:
    """Rewrite concept fields that hallucinate off-brief ingredients."""
    out = dict(concept)
    violations = concept_mentions_forbidden_food(out, allowed)
    if violations:
        flavors = ", ".join((getattr(body, "flavors", None) or [])[:3])
        ing = ", ".join(allowed[:4])
        if body.language == "zh":
            out["tagline"] = (
                f"以{ing or flavors}为核心的无酒精出品。"
                if (ing or flavors)
                else "锁定配方，专注表内原料。"
            )
        else:
            out["tagline"] = (
                f"A spirit-free serve built on {ing or flavors}."
                if (ing or flavors)
                else "Locked formulation — on-table ingredients only."
            )
        for key in ("title_primary", "title_secondary"):
            t = out.get(key, "")
            t = re.sub(r"牛肉干|beef\s*jerky|jerky|烟草|tobacco", "", t, flags=re.IGNORECASE).strip()
            t = re.sub(r"\s{2,}", " ", t).strip(" ·—-")
            if t:
                out[key] = t
    return out
