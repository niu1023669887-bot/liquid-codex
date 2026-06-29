"""
Ingredient synonym resolution and powder-vs-liquid unit hints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SYN_PATH = Path(__file__).parent / "data" / "ingredient_synonyms.json"
_SYNONYMS: dict | None = None

# Powders measured by mass, not volume (Arnold / Codex)
_POWDER_RE = re.compile(
    r"\b(?:matcha|抹茶|cocoa\s*powder|可可粉|charcoal|竹炭|bentonite|琼脂粉|"
    r"citric\s*acid\s*powder|柠檬酸|抹茶粉|茶粉)\b",
    re.IGNORECASE,
)


def load_synonyms() -> dict:
    global _SYNONYMS
    if _SYNONYMS is None:
        if _SYN_PATH.exists():
            with open(_SYN_PATH, encoding="utf-8") as f:
                _SYNONYMS = json.load(f)
        else:
            _SYNONYMS = {"aliases": {}, "powder_names": []}
    return _SYNONYMS


def canonical_name(name: str) -> str:
    data = load_synonyms()
    key = name.lower().strip()
    aliases = data.get("aliases", {})
    return aliases.get(key, name)


def is_powder_ingredient(name: str) -> bool:
    data = load_synonyms()
    lower = name.lower().strip()
    if lower in {p.lower() for p in data.get("powder_names", [])}:
        return True
    return bool(_POWDER_RE.search(name))


def apply_synonym_hints(ingredients: list[dict]) -> list[dict]:
    """Normalize names and tag powder ingredients for balance engine."""
    data = load_synonyms()
    category_hints = data.get("category_hints", {})
    out = []
    for ing in ingredients:
        ing = dict(ing)
        raw = ing.get("name", "")
        canon = canonical_name(raw)
        if canon != raw:
            ing["name"] = canon
        hint = category_hints.get(canon.lower()) or category_hints.get(raw.lower())
        current_cat = ing.get("category")
        if hint and (not current_cat or current_cat == "unknown"):
            ing["category"] = hint
        if is_powder_ingredient(ing.get("name", "")):
            ing["_unit"] = "g"
        out.append(ing)
    return out
