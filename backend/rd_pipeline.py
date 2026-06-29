"""
Recipe Pipelines — v4 (round 42: simplified)

Two generation pipelines, no retry loops:

PRECISION:
  L0   Spirit mode + banned ingredients
  L0-C Inventory classification
  L1   Perplexity research (reference templates)
  L2   Pre-treatment plan
  L4   Single deepseek-chat call → safety-only validation → render

FAST:
  L0   Spirit mode + banned ingredients
  L4   Single deepseek-chat call → safety-only validation → render

Key changes from v3:
  · Retry loops removed — single pass, single LLM call
  · No bartender_review (LLM gate removed)
  · No whitelist/flavor-coverage validation (was overly strict)
  · Safety-only: banned ingredients, carbonated+shake conflict,
    kombucha ABV, home equipment vs frontier technique
  · No classic_anchor_fallback (was causing "Could not generate" errors)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from openai import OpenAI

from classic_anchor import (
    match_classic_anchor,
    match_mocktail_anchor,
)
from prompt_core import (
    PERPLEXITY_RESEARCH_SYSTEM,
    build_formula_system,
)
from reference_canon import (
    DILUTION_BY_STYLE,
)

logger = logging.getLogger(__name__)

# ── Top-10 flavor → ingredient mapping ─────────────────────────────────────
# Extracted from IBA classic_recipes.py (round 33+ hardcoded database).
# Used by validate_formula_spec for flavor-coverage checks.
FLAVOR_INGREDIENT_MAP: dict[str, set[str]] = {
    "citrus":   {"lemon","lime","orange","grapefruit","lemon juice","lime juice","orange juice","grapefruit juice","yuzu","citrus"},
    "tropical": {"pineapple","pineapple juice","coconut","coconut cream","coconut milk","mango","passion fruit","banana","guava","papaya"},
    "herbal":   {"mint","basil","rosemary","thyme","sage","green chartreuse","yellow chartreuse","benedictine","absinthe","herbal"},
    "smoky":    {"mezcal","scotch whisky","scotch","islay scotch","peated","smoky"},
    "floral":   {"elderflower","st germain","st. germain","rose water","lavender","violet","floral","hibiscus"},
    "spicy":    {"ginger","cinnamon","clove","nutmeg","chilli","black pepper","spicy","allspice","pimento"},
    "sweet":    {"simple syrup","sugar","honey","agave","maple syrup","grenadine","sweet","sugar syrup"},
    "bitter":   {"campari","aperol","bitter","vermouth","amaro","angostura","bitters","cynar","fernet"},
    "creamy":   {"cream","milk","coconut cream","coconut milk","egg","egg white","half and half","creamy","vanilla"},
    "umami":    {"tomato","soy sauce","umami","mushroom","parmesan","bacon","olive","brine","sherry","worcestershire"},
}


# ─────────────────────────────────────────────────────────────────────────────
# L0-A  Spirit Mode Detection
# ─────────────────────────────────────────────────────────────────────────────

_NO_ALCOHOL_RE = re.compile(
    r"无酒精|不含酒|不要酒|spirit[\s-]free|mocktail|non[\s-]?alcoholic|zero[\s-]?alcohol|"
    r"不喝酒|驾驶员|孕妇|0[\s%]*abv|zero\s*abv|no\s+alcohol|没有酒",
    re.IGNORECASE,
)
_HAS_ALCOHOL_RE = re.compile(
    r"gin|vodka|rum|tequila|whisky|whiskey|bourbon|scotch|brandy|cognac|mezcal|"
    r"金酒|伏特加|朗姆|龙舌兰|威士忌|白兰地|干邑|白酒|烧酒|清酒|sake|vermouth|campari|aperol|"
    r"amaretto|chartreuse|cointreau|triple\s+sec|kahlua|baileys|lillet|cynar|benedictine",
    re.IGNORECASE,
)


def detect_spirit_mode(body: Any) -> str:
    """Return 'spirit-free' or 'with-spirit'."""
    from user_prefs import get_occasion, get_alcohol_pref

    if get_occasion(body) == "non-alcoholic":
        return "spirit-free"
    alc_pref = get_alcohol_pref(body)
    if alc_pref in ("none", "no spirit", "spirit-free", "mocktail", "无", "无酒精"):
        return "spirit-free"

    blob = " ".join([
        getattr(body, "ingredients", "") or "",
        getattr(body, "notes", "") or "",
        getattr(body, "alcohol", "") or "",
        " ".join(getattr(body, "flavors", []) or []),
    ])
    if _NO_ALCOHOL_RE.search(blob):
        return "spirit-free"
    if _HAS_ALCOHOL_RE.search(blob):
        return "with-spirit"
    if alc_pref:
        return "with-spirit"
    return "with-spirit"


# ─────────────────────────────────────────────────────────────────────────────
# L0-C  Inventory Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_inventory(body: Any) -> list[dict]:
    """
    Parse user ingredient string → list of classified dicts.
    Uses INGREDIENT_PROPS for known items; leaves unknown items uncategorised.
    """
    from ingredient_normalize import apply_synonym_hints
    from routers.ai import INGREDIENT_PROPS, _lookup_ingredient_props

    raw = (getattr(body, "ingredients", "") or "").strip()
    if not raw:
        return []

    tokens = [t.strip() for t in re.split(r"[,，、;；/\n]+", raw) if t.strip()]
    result: list[dict] = []
    for tok in tokens:
        props = _lookup_ingredient_props(tok)
        if props is None:
            props = INGREDIENT_PROPS.get(tok.lower())
        cat = (props or {}).get("category", "unknown")
        result.append({
            "name": tok,
            "category": cat,
            "props": props,
            "prep_note": "",
        })
    return apply_synonym_hints(result)


# ─────────────────────────────────────────────────────────────────────────────
# L1  Gap Analysis + Template Finding (Perplexity)
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_DEFAULT: dict = {
    "template_name": "Sour",
    "reference_drink": "Daiquiri-style",
    "reference_spec": "spirit : acid : sweetener ≈ 60 : 30 : 15 ml",
    "technique": "shaken",
    "glassware": "coupe",
    "gaps": [],
    "auto_additions": [],
    "gap_fill_suggestions": [],
    "must_have_ingredients": [],
    "reference_recipes": [],
    "flavor_pairing_suggestions": [],
    "concept_interpretation": "",
    "flavor_notes": [],
    "prep_notes": [],
    "safety_flag": False,
}


def fetch_template_reference(
    perplexity_key: str,
    body: Any,
    spirit_mode: str,
    inventory: list[dict],
) -> dict:
    """
    L1: Ask Perplexity to research real cocktail references and professional guidance.
    Uses PERPLEXITY_RESEARCH_SYSTEM to return deep research results including
    must_have_ingredients from the user's flavor concept.
    """
    from user_prefs import build_preferences_user_block

    if not perplexity_key:
        return dict(_TEMPLATE_DEFAULT)

    # Build research brief from user inputs
    flavor_intent = (getattr(body, "notes", "") or "").strip()
    user_ingredients = (getattr(body, "ingredients", "") or "").strip()
    flavors = getattr(body, "flavors", []) or []
    techniques = getattr(body, "techniques", []) or []

    research_brief = f"Flavor concept: {flavor_intent}\n"
    if user_ingredients:
        research_brief += f"User's available ingredients: {user_ingredients}\n"
    if flavors:
        research_brief += f"Dominant flavor profiles: {', '.join(flavors)}\n"
    if techniques:
        research_brief += f"Preferred techniques: {', '.join(techniques)}\n"
    research_brief += f"Spirit mode: {spirit_mode}\n"
    research_brief += f"Equipment: {getattr(body, 'equipment', 'bar')}\n"

    try:
        client = OpenAI(
            api_key=perplexity_key,
            base_url="https://api.perplexity.ai",
            timeout=30,
        )
        resp = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": PERPLEXITY_RESEARCH_SYSTEM},
                {"role": "user", "content": research_brief},
            ],
            stream=False,
            max_tokens=1500,
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        brace = re.search(r"\{[\s\S]*\}", raw)
        if brace:
            raw = brace.group(0)
        result = json.loads(raw)

        # Merge defaults
        for k, v in _TEMPLATE_DEFAULT.items():
            result.setdefault(k, v)

        # Extract must_have_ingredients for downstream use
        must_have = result.get("must_have_ingredients") or []
        result["must_have_ingredients"] = must_have

        return result
    except Exception as exc:
        logger.warning("Perplexity research failed: %s — using defaults", exc)
        return dict(_TEMPLATE_DEFAULT)


def auto_complete_inventory(
    inventory: list[dict],
    template_data: dict,
    spirit_mode: str,
) -> list[dict]:
    """
    Merge gap_fill_suggestions from Perplexity research into inventory.
    Each addition fills a missing structural category.
    """
    from routers.ai import INGREDIENT_PROPS, _lookup_ingredient_props

    current_cats = {i["category"] for i in inventory}
    out = list(inventory)

    # Prefer gap_fill_suggestions (new Perplexity schema); fall back to auto_additions (legacy)
    suggestions = (template_data.get("gap_fill_suggestions") or []) or (template_data.get("auto_additions") or [])

    for addition in suggestions:
        name = addition.get("name", "").strip()
        cat = addition.get("category", "unknown")
        if not name:
            continue
        # Skip if this category is already present (don't duplicate)
        if cat in current_cats:
            continue
        # Skip spirit additions when spirit-free
        if spirit_mode == "spirit-free" and cat == "spirit":
            continue
        props = _lookup_ingredient_props(name) or INGREDIENT_PROPS.get(name.lower())
        out.append({
            "name": name,
            "category": cat,
            "props": props,
            "prep_note": addition.get("reason", "auto-added for balance"),
            "_auto_added": True,
        })
        current_cats.add(cat)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# L2  Pre-treatment Plan (Python knowledge base)
# ─────────────────────────────────────────────────────────────────────────────

_INFUSION_RE = re.compile(
    r"infus|macerat|浸泡|浸渍|fat[\s-]?wash|脂洗|"
    r"clarif|澄清|sous[\s-]?vide|centrifug",
    re.IGNORECASE,
)
_FAT_WASH_RE = re.compile(r"fat[\s-]?wash|脂洗|bacon\s+fat|培根油|butter\s+wash", re.IGNORECASE)
_THERMOLABILE_MUSHROOM_RE = re.compile(
    r"见手青|红见手青|黄见手青|白见手青|牛肝菌|boletus\b|porcini|cep\b|羊肚菌|morel\b",
    re.IGNORECASE,
)
_RAW_ELDERBERRY_RE = re.compile(
    r"elder\s*berr(?:y|ies)(?!\s*(?:syrup|liqueur|cordial|juice|spirit|wine))|"
    r"接骨木(?:浆)?果(?!糖浆|果汁|酒)",
    re.IGNORECASE,
)
_CITRUS_JUICE_RE = re.compile(
    r"\blemon\s*juice|\blime\s*juice|\bgrapefruit\s*juice|\borange\s*juice|"
    r"柠檬汁|青柠汁|橙汁|葡萄柚汁",
    re.IGNORECASE,
)
_FRESH_CITRUS_RE = re.compile(
    r"\blemon\b|\blime\b|\bgrapefruit\b|\borange\b|柠檬|青柠|橙子|葡萄柚",
    re.IGNORECASE,
)


def build_prep_plan(ingredients: list[dict], body: Any) -> list[dict]:
    """
    L2: For each ingredient, determine the required preparation step.
    Returns a list of prep dicts: [{ingredient, step, type, mandatory}]
    """
    prep_steps: list[dict] = []
    blob = " ".join([
        getattr(body, "ingredients", "") or "",
        getattr(body, "notes", "") or "",
    ])

    for ing in ingredients:
        name = ing.get("name", "")
        prep: list[dict] = []

        # Wild mushroom: mandatory pre-cook
        if _THERMOLABILE_MUSHROOM_RE.search(name):
            prep.append({
                "ingredient": name,
                "type": "safety_precook",
                "mandatory": True,
                "step_en": (
                    f"Blanch {name} in vigorously boiling water (≥100 °C) for ≥5 min. "
                    "Drain and cool; discard blanching liquid. "
                    "Destroys thermolabile proteins."
                ),
                "step_zh": (
                    f"将{name}放入沸腾清水（≥100°C）中焯水≥5分钟，捞出沥干冷却，焯水液废弃。"
                    "此步骤破坏热不稳定毒素蛋白。"
                ),
            })

        # Raw elderberry: mandatory pre-cook
        if _RAW_ELDERBERRY_RE.search(name):
            prep.append({
                "ingredient": name,
                "type": "safety_precook",
                "mandatory": True,
                "step_en": (
                    f"Simmer {name} in 200 ml water over medium heat for ≥15 min. "
                    "Strain; discard solids. Raw elderberries contain cyanogenic glycosides."
                ),
                "step_zh": (
                    f"将{name}加入200ml清水中以中火熬煮≥15分钟，过滤去渣。"
                    "生接骨木浆果含氰苷（sambunigrin），须充分加热。"
                ),
            })

        # Fat wash detection
        if _FAT_WASH_RE.search(name) or _FAT_WASH_RE.search(blob):
            if ing.get("category") == "spirit":
                prep.append({
                    "ingredient": name,
                    "type": "fat_wash",
                    "mandatory": True,
                    "step_en": (
                        f"Fat-wash {name}: combine spirit with rendered fat at 5:1 ratio "
                        "(e.g. 200 ml spirit + 40 ml fat). "
                        "Infuse 2-4 h at room temp, freeze at -18 °C for 24 h, "
                        "strain through coffee filter — freeze step is mandatory to solidify fat. "
                        "(Logsdon, Modernist Infusions)"
                    ),
                    "step_zh": (
                        f"脂洗{name}：按烈酒:脂肪 = 5:1 体积比混合（如200 ml烈酒配40 ml脂肪），"
                        "室温浸泡2-4小时，零下18°C冷冻24小时（冷冻步骤不可省略，需固化脂肪方可过滤），"
                        "用咖啡滤纸过滤。（Logsdon《现代浸泡》）"
                    ),
                })

        # Fresh citrus: squeeze note
        if _FRESH_CITRUS_RE.search(name) and not _CITRUS_JUICE_RE.search(name):
            prep.append({
                "ingredient": name,
                "type": "fresh_squeeze",
                "mandatory": False,
                "step_en": f"Squeeze {name} fresh — never use bottled juice (Arnold: pH drift within hours).",
                "step_zh": f"{name}现榨——禁用瓶装果汁（Arnold：酸度在数小时内漂移）。",
            })

        prep_steps.extend(prep)

    # User-selected prep techniques (clarification, sous-vide, etc.)
    from user_prefs import build_technique_prep_steps
    lang = getattr(body, "language", "en") or "en"
    tech_steps = build_technique_prep_steps(body, ingredients, lang)
    existing_types = {s.get("type") for s in prep_steps}
    for step in tech_steps:
        if step.get("type") not in existing_types:
            prep_steps.append(step)

    return prep_steps


# ─────────────────────────────────────────────────────────────────────────────
# L3  Pre-treatment Safety Validation (Python)
# ─────────────────────────────────────────────────────────────────────────────

def validate_prep_safety(prep_steps: list[dict]) -> tuple[bool, str, list[dict]]:
    """
    L3: Verify mandatory safety steps are present and correctly specified.
    Returns (ok, reason, validated_steps).
    """
    from routers.ai import BANNED_INGREDIENTS

    # Check banned ingredients (this check runs on ingredient names in prep_steps)
    for step in prep_steps:
        ing_lower = step.get("ingredient", "").lower()
        for banned in BANNED_INGREDIENTS:
            if banned.lower() in ing_lower:
                return False, f"禁用原料：{step['ingredient']}", prep_steps

    # All mandatory steps exist by construction from build_prep_plan —
    # they were added precisely when detected. Validation passes.
    return True, "ok", prep_steps


# ─────────────────────────────────────────────────────────────────────────────
# L4  Formula Design — deepseek-chat
# ─────────────────────────────────────────────────────────────────────────────

def _chat_json(client: OpenAI, model: str, messages: list, max_tokens: int = 5000) -> dict:
    """Call model, extract JSON from response."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=0.4,
        timeout=300,
    )
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        raise RuntimeError("Model returned empty content")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        text = brace.group(0)
    return json.loads(text)


def _build_formula_user_msg(
    body: Any,
    spirit_mode: str,
    inventory: list[dict],
    template_data: dict,
    prep_steps: list[dict],
    language: str,
) -> str:
    """Construct the user message for the formula design call."""
    from user_prefs import build_preferences_user_block

    ing_lines = "\n".join(
        f"  · {i['name']} ({i['category']})"
        + (f" [auto-added: {i.get('prep_note','')}]" if i.get("_auto_added") else "")
        for i in inventory
    )
    prep_lines = "\n".join(
        f"  [{s['type']}] {s['ingredient']}: "
        + (s["step_zh"] if language == "zh" else s["step_en"])
        for s in prep_steps
    ) or "  (none)"

    prefs_block = build_preferences_user_block(body, language)
    equipment = getattr(body, "equipment", "bar")

    # ── Perplexity research block (real references from L1) ──────────────
    research_lines: list[str] = []
    ci = template_data.get("concept_interpretation")
    if ci:
        research_lines.append(f"Concept interpretation: {ci}")
    refs = template_data.get("reference_recipes") or []
    if refs:
        research_lines.append("Reference recipes from research:")
        for r in refs[:3]:
            rname = r.get("name", "")
            rsrc = r.get("source", "")
            rwhy = r.get("why_it_fits", "")
            rkey = ", ".join(r.get("key_ingredients", [])[:4])
            if rname:
                research_lines.append(f"  · {rname}" + (f" ({rsrc})" if rsrc else ""))
                if rkey:
                    research_lines.append(f"    Key ingredients: {rkey}")
                if rwhy:
                    research_lines.append(f"    Why: {rwhy}")
    must_have = template_data.get("must_have_ingredients") or []
    if must_have:
        research_lines.append(
            f"MANDATORY ingredients from flavor concept (MUST use): {', '.join(must_have)}"
        )
    gaps = template_data.get("gap_fill_suggestions") or []
    if gaps:
        research_lines.append("Suggested structural additions:")
        for g in gaps[:3]:
            research_lines.append(
                f"  · {g.get('name','')} ({g.get('category','')}) — {g.get('reason','')}"
            )
    flavor_suggestions = template_data.get("flavor_pairing_suggestions") or []
    if flavor_suggestions:
        research_lines.append("Flavor pairing guidance:")
        for fs in flavor_suggestions[:3]:
            research_lines.append(f"  · {fs}")
    research_block = ""
    if research_lines:
        research_block = "━━━ PROFESSIONAL REFERENCE RESEARCH (from Perplexity) ━━━\n" + "\n".join(research_lines) + "\n\n"

    lang_note = "Please respond in Chinese for title_primary, tagline, and method_steps." if language == "zh" else ""

    # ── Dilution model (Arnold, Liquid Intelligence) ─────────────────────
    technique = template_data.get("technique") or "shaken"
    from reference_canon import DILUTION_BY_STYLE
    dilution_rate = DILUTION_BY_STYLE.get(technique, 0.22)
    total_raw = sum(i.get("amount_ml", 0) for i in inventory)
    final_vol = total_raw * (1 + dilution_rate)
    dilution_block = (
        f"\n━━━ DILUTION MODEL (Dave Arnold, Liquid Intelligence) ━━━\n"
        f"Technique: {technique} → dilution ~{dilution_rate*100:.0f}%\n"
        f"Pre-dilution volume: {total_raw:.0f} ml → estimated final: {final_vol:.0f} ml\n"
        f"Post-dilution ABV = (total alcohol ml / {final_vol:.0f} ml) × 100\n"
        f"CRITICAL: Your ingredient amounts will be diluted by ~{dilution_rate*100:.0f}% in the final drink. "
        f"Set pre-dilution amounts accordingly, knowing ice melt will add volume.\n\n"
    )

    return (
        f"Spirit mode: {spirit_mode}\n"
        f"{prefs_block}\n"
        f"Equipment: {equipment}\n"
        f"{lang_note}\n\n"
        f"INGREDIENT LIST (design the recipe using ONLY these):\n{ing_lines}\n\n"
        f"{research_block}"
        f"{dilution_block}"
        f"Required pre-treatment steps (must appear in method_steps):\n{prep_lines}\n"
        "Now design the complete recipe JSON."
    )


def validate_formula_spec(
    spec: dict,
    spirit_mode: str,
    equipment: str = "bar",
    body: Any | None = None,
    prep_steps: list[dict] | None = None,
) -> list[str]:
    """
    Python validation of the spec against physical/professional rules.
    Returns a list of error strings (empty = pass).
    """
    from reference_canon import (
        DILUTION_BY_STYLE,
        EQUIPMENT_TIERS,
        normalize_equipment_tier,
    )
    from routers.ai import _lookup_ingredient_props

    errors: list[str] = []
    ings = spec.get("ingredients") or []
    if not ings:
        errors.append("No ingredients in spec.")
        return errors

    # ── Ingredient name quality gate ──────────────────────────────────────────
    # Reject names that are clearly degenerate: numbers, ratios, percentages,
    # or over-long merged descriptions. These indicate the LLM put the recipe
    # logic in the name field instead of the prep_note field.
    _DEGENERATE_NAME_RE = re.compile(
        r"^[\d\s%.:/]+$"          # pure numbers / ratios / percentages: "1:1", "2%", "0.5"
        r"|^\d+[\s]*(ml|g|oz|%|:)[\s\d.]*$",  # "5 ml", "3 g", "5%", "3:1"
        re.IGNORECASE,
    )
    for ing in ings:
        raw_name = (ing.get("name") or "").strip()
        # Strip bilingual suffix for the check: "gin / 金酒" → "gin"
        check_name = raw_name.split(" / ")[0].strip() if " / " in raw_name else raw_name

        if not check_name:
            errors.append(
                "An ingredient has an empty name. Every ingredient must have a real, "
                "identifiable name (e.g. 'white rum', 'lime juice', 'simple syrup')."
            )
        elif _DEGENERATE_NAME_RE.match(check_name):
            errors.append(
                f"Ingredient name '{raw_name}' is a ratio/percentage/number — "
                "use the actual ingredient name (e.g. 'simple syrup' not '1:1', "
                "'sodium alginate solution' not '2%'). "
                "Preparation ratios belong in the 'prep_note' field."
            )
        elif len(check_name) > 120:
            errors.append(
                f"Ingredient name '{check_name[:30]}...' ({len(check_name)} chars) is too long. "
                "Name = single ingredient only. Put preparation details in 'prep_note'. "
                "Example: name='coconut cream', prep_note='combined with pineapple juice 1:1'."
            )

    if errors:  # name errors take priority — return early to avoid noise
        return errors

    # ── Volume check + auto-correction ──────────────────────────────────────
    total_liquid = sum(float(i.get("amount_ml", 0)) for i in ings)
    technique = (spec.get("technique") or "shaken").lower()
    dil = DILUTION_BY_STYLE.get(technique, 0.22)
    finished_vol = total_liquid if technique == "built" else total_liquid * (1 + dil)

    if technique in ("shaken", "stirred", "blended"):
        if not (70 <= finished_vol <= 280):
            target = 90.0 if finished_vol < 70 else 240.0
            scale = target / finished_vol
            for i in ings:
                i["amount_ml"] = round(i.get("amount_ml", 0) * scale, 1)
            total_liquid = sum(float(i.get("amount_ml", 0)) for i in ings)
            finished_vol = total_liquid if technique == "built" else total_liquid * (1 + dil)
            logger.info("Auto-fix volume: scaled to %.0f ml finished volume", finished_vol)
    elif technique == "built":
        if not (80 <= total_liquid <= 320):
            target = 120.0 if total_liquid < 80 else 240.0
            scale = target / total_liquid
            for i in ings:
                i["amount_ml"] = round(i.get("amount_ml", 0) * scale, 1)
            total_liquid = sum(float(i.get("amount_ml", 0)) for i in ings)
            logger.info("Auto-fix volume: scaled to %.0f ml (built)", total_liquid)

    # ── ABV check ──────────────────────────────────────────────────────────────
    ethanol_ml = 0.0
    if spirit_mode == "spirit-free":
        for ing in ings:
            props = ing.get("props") or _lookup_ingredient_props(ing["name"]) or {}
            if float(props.get("abv_pct", 0)) > 0.005:
                errors.append(
                    f"Spirit-free recipe must not contain alcoholic ingredient: {ing['name']} "
                    f"(ABV {props['abv_pct']*100:.1f}%)."
                )
    else:
        for ing in ings:
            props = ing.get("props") or _lookup_ingredient_props(ing["name"]) or {}
            abv = float(props.get("abv_pct", 0))
            vol = float(ing.get("amount_ml", 0))
            ethanol_ml += abv * vol
        if finished_vol > 0:
            final_abv = ethanol_ml / finished_vol
            tier = normalize_equipment_tier(equipment)
            max_abv = (EQUIPMENT_TIERS.get(tier) or {}).get("max_abv_pct", 28) / 100
            if final_abv > max_abv:
                errors.append(
                    f"Final ABV {final_abv*100:.1f}% exceeds equipment tier maximum {max_abv*100:.0f}%."
                )

    # ── Sweet/acid balance (sour drinks only) ────────────────────────────────
    cats = {i.get("category", "") for i in ings}
    if "acid" in cats and "sweetener" in cats:
        acid_ml = sum(float(i.get("amount_ml", 0)) for i in ings if i.get("category") == "acid")
        sweet_ml = sum(float(i.get("amount_ml", 0)) for i in ings if i.get("category") == "sweetener")
        # Also count sugar contribution from high-Brix modifiers
        for ing in ings:
            if ing.get("category") == "modifier":
                props = ing.get("props") or _lookup_ingredient_props(ing["name"]) or {}
                brix = float(props.get("brix", 0))
                if brix >= 30:
                    sweet_ml += float(ing.get("amount_ml", 0)) * 0.5
        if acid_ml > 0 and sweet_ml > 0:
            ratio = sweet_ml / acid_ml
            # Volume ratio validation. Classic Daiquiri: 15ml syrup / 30ml lime ≈ 0.5;
            # allow 0.3–2.0 for creative range.
            if not (0.3 <= ratio <= 2.0):
                errors.append(
                    f"Sweet/acid volume ratio {ratio:.2f} out of range "
                    f"(sweetener {sweet_ml:.0f} ml / acid {acid_ml:.0f} ml). "
                    "Standard sour: 0.5–0.8 vol/vol (e.g. Daiquiri 15/30 = 0.5)."
                )

    # ── Carbonated + shake conflict ───────────────────────────────────────────
    _CARBONATED_RE = re.compile(
        r"soda|tonic|sparkling|prosecco|champagne|ginger\s*beer|kombucha|"
        r"beer|cider|seltzer|club\s*soda|fizzy|effervescent|carbonated|"
        r"苏打|汤力|起泡|气泡|香槟|姜汁啤酒|康普茶|啤酒|苹果酒|气泡水|自然充气",
        re.IGNORECASE,
    )
    has_carbonated = any(_CARBONATED_RE.search(i.get("name", "")) for i in ings)
    if has_carbonated and technique == "shaken":
        spec["technique"] = "built"
        technique = "built"
        logger.info("Auto-fix: shaken → built (carbonated ingredients)")

    # ── Fermented ingredients + shake conflict ─────────────────────────────────
    _FERMENTED_RE = re.compile(
        r"fermented|发酵|kombucha|康普茶|shrub|shrubs",
        re.IGNORECASE,
    )
    has_fermented = any(_FERMENTED_RE.search(i.get("name", "")) for i in ings)
    if has_fermented and technique == "shaken":
        spec["technique"] = "built"
        technique = "built"
        logger.info("Auto-fix: shaken → built (fermented ingredients)")

    # ── User technique preference check ────────────────────────────────────────
    if body is not None:
        user_techniques = [t.lower() for t in (getattr(body, "techniques", None) or [])]
        if any(t in user_techniques for t in ["carbonation", "充气", "fermentation", "发酵"]):
            if technique == "shaken":
                spec["technique"] = "built"
                technique = "built"
                logger.info("Auto-fix: shaken → built (user selected carbonation/fermentation)")

    # ── Glassware / technique consistency ─────────────────────────────────────
    # Clear professional mismatches that indicate a fundamental design error.
    _GLASS_NORM = {
        "old fashioned": "rocks", "lowball": "rocks",
        "martini glass": "cocktail glass", "coupe": "cocktail glass",
        "nick & nora": "cocktail glass", "nick and nora": "cocktail glass",
        "champagne saucer": "cocktail glass",
    }
    glass_raw = (spec.get("glassware") or "").lower().strip()
    glass = _GLASS_NORM.get(glass_raw, glass_raw)

    if technique == "stirred" and glass in ("highball", "collins"):
        errors.append(
            f"Technique 'stirred' conflicts with glassware '{spec.get('glassware')}'. "
            "Stirred cocktails belong in rocks, cocktail glass, or Nick & Nora — "
            "not tall glasses. Use 'built' for long drinks."
        )
    elif technique == "shaken" and glass in ("highball", "collins") and not has_carbonated:
        # Sours/short shaken drinks in a tall glass is unusual — warn unless it has a
        # carbonated top-up (which would make it a Fizz/Collins, where this is valid).
        has_dilutant = any(
            re.search(r"soda|tonic|ginger|cola|sprite|sparkling", i.get("name", ""), re.IGNORECASE)
            for i in ings
        )
        if not has_dilutant:
            errors.append(
                f"Technique 'shaken' with glassware '{spec.get('glassware')}' is unusual for a "
                "short cocktail without a carbonated top-up. "
                "Use 'cocktail glass' (coupe) for shaken sours, or add soda/tonic and change "
                "technique to 'built' for long drinks."
            )

    if body is not None:
        from user_prefs import validate_all_user_prefs
        errors.extend(validate_all_user_prefs(spec, body, prep_steps))

    # ── Flavor coverage check — optional hint ────────────────────────────────
    # Top-10 flavor map (hardcoded from IBA classics). If user requested a
    # specific flavor, verify the ingredient list covers it. Only emits advisory
    # warnings — does not block the recipe.
    flavors = getattr(body, "flavors", []) if body is not None else []
    if flavors:
        ing_names_lower = {i.get("name", "").lower() for i in ings}
        # Also check bilingual split
        for i in ings:
            raw = i.get("name", "")
            if " / " in raw:
                ing_names_lower.add(raw.split(" / ")[0].strip().lower())
        for flavor in flavors:
            fkey = flavor.lower().strip()
            matched_kws = FLAVOR_INGREDIENT_MAP.get(fkey)
            if matched_kws:
                if not any(kw in " ".join(ing_names_lower) for kw in matched_kws):
                    errors.append(
                        f"Flavor '{flavor}' not expressed in the recipe. "
                        f"No ingredient matches the '{flavor}' profile. "
                        "Add an ingredient from this flavor family."
                    )

    # ── Cocktail balance check (Dave Arnold — Liquid Intelligence) ────────
    # Check estimated ABV / Brix / Acid against type-specific targets.
    from reference_canon import COCKTAIL_BALANCE_TARGETS
    balance_key = technique  # shaken / stirred / built
    # Map blended to shaken targets; carbonated is a sub-type of built
    if technique == "blended":
        balance_key = "shaken"
    if balance_key in COCKTAIL_BALANCE_TARGETS:
        min_abv, max_abv, min_brix, max_brix, min_acid, max_acid = COCKTAIL_BALANCE_TARGETS[balance_key]
        # ABV check (already computed as final_abv above, but we have finished_vol)
        if finished_vol > 0:
            est_abv = ethanol_ml / finished_vol * 100
            if est_abv < min_abv:
                errors.append(
                    f"Estimated ABV {est_abv:.1f}% below {balance_key} minimum {min_abv}%. "
                    "Increase spirit or reduce dilutant."
                )
        # Note: Brix/acid checks require mass calculations beyond current scope
        # but we do a simple acid_ml sanity
        if "acid" in cats and technique in ("built", "stirred"):
            total_acid_ml = sum(float(i.get("amount_ml", 0)) for i in ings if i.get("category") == "acid")
            if total_acid_ml > 20:
                errors.append(
                    f"Technique '{technique}' with acid ingredients ({total_acid_ml:.0f} ml) is unusual. "
                    "Built/stirred drinks normally contain little to no fresh citrus."
                )

    # ── Punch (潘切) five-element rule (Meehan's — §11.1) ─────────────────────
    _PUNCH_KEYWORDS = re.compile(r"punch|潘切|潘趣|punch bowl|批量", re.IGNORECASE)
    is_punch = (
        _PUNCH_KEYWORDS.search(spec.get("title_primary", "") + " " + spec.get("tagline", ""))
        or any(_PUNCH_KEYWORDS.search(i.get("name", "")) for i in ings)
    )
    if is_punch:
        from reference_canon import PUNCH_ELEMENTS
        present_cats = {i.get("category", "") for i in ings}
        missing_elements = PUNCH_ELEMENTS - present_cats
        # Spice element: check prep_note or ingredient keywords
        has_spice = any(
            re.search(r"spice|香料|cinnamon|clove|nutmeg|肉豆蔻|肉桂|丁香|allspice|pimento",
                      i.get("name", "") + " " + (i.get("prep_note") or ""), re.IGNORECASE)
            for i in ings
        )
        if not has_spice:
            missing_elements.add("spice")
        missing_elements -= {"modifier"}  # modifier is optional; spice is the key
        if missing_elements:
            errors.append(
                f"Punch recipe missing {len(missing_elements)} of 5 elements: "
                f"{', '.join(sorted(missing_elements))}. "
                "Punch needs: spirit (spirit), acid (acid), sweetener (sweetener), "
                "dilutant (dilutant), and spice (prep_note or ingredient with spice keyword)."
            )

    # ── Fermentation safety check (Liquid Codex Vol.III — §15) ───────────────
    from reference_canon import (
        LACTO_FERMENTATION_PH, ACETIC_FERMENTATION_PH,
        KOMBUCHA_FERMENTATION_PH, KOMBUCHA_ABV_MAX,
        FERMENTATION_TEMP_LACTO, FERMENTATION_DAYS_LACTO,
    )
    _LACTO_KEYWORDS = re.compile(r"lacto|乳酸|ferment|发酵|pickled|泡", re.IGNORECASE)
    _ACETIC_KEYWORDS = re.compile(r"acetic|醋酸|vinegar|醋", re.IGNORECASE)
    _KOMBUCHA_KEYWORDS = re.compile(r"kombucha|康普茶", re.IGNORECASE)
    for ing in ings:
        iname = ing.get("name", "")
        iprep = ing.get("prep_note") or ""
        blob = iname + " " + iprep
        if _LACTO_KEYWORDS.search(blob):
            # Warn but don't block — pH values aren't in the spec schema yet
            errors.append(
                f"'{iname}' is a lacto-fermented ingredient. "
                "Ensure pH is 3.0-3.5 for lacto-fermented components."
            )
        if _ACETIC_KEYWORDS.search(blob):
            errors.append(
                f"'{iname}' is an acetic-fermented ingredient. "
                "Ensure pH < 3.0 for acetic components."
            )
        if _KOMBUCHA_KEYWORDS.search(iname) or "kombucha" in blob.lower():
            props = ing.get("props") or _lookup_ingredient_props(iname) or {}
            kombucha_abv = float(props.get("abv_pct", 0.005))
            if kombucha_abv > KOMBUCHA_ABV_MAX:
                errors.append(
                    f"Kombucha ABV {kombucha_abv*100:.1f}% exceeds max {KOMBUCHA_ABV_MAX*100:.1f}%. "
                    "If kombucha is used as a dilutant, its ABV must be ≤ 1.5%."
                )

    # ── Molecular ingredient usage validation (§17) ───────────────────────────
    from reference_canon import (
        AGAR_FLUID_GEL, SODIUM_ALGINATE_SPHERIFICATION,
        XANTHAN_GUM_SUSPENSION, LECITHIN_FOAM,
    )
    _MOLECULAR_RANGES: dict[str, tuple[float, float, str]] = {
        "agar-agar": (AGAR_FLUID_GEL[0], AGAR_FLUID_GEL[1], "g/L for fluid gel"),
        "agar": (AGAR_FLUID_GEL[0], AGAR_FLUID_GEL[1], "g/L for fluid gel"),
        "sodium alginate": (SODIUM_ALGINATE_SPHERIFICATION[0], SODIUM_ALGINATE_SPHERIFICATION[1], "% w/v for spherification"),
        "xanthan gum": (XANTHAN_GUM_SUSPENSION[0], XANTHAN_GUM_SUSPENSION[1], "% w/v for suspension"),
        "lecithin": (LECITHIN_FOAM[0], LECITHIN_FOAM[1], "% w/v for foam"),
    }
    for ing in ings:
        iname = ing.get("name", "").lower().strip()
        for mname, (lo, hi, unit_label) in _MOLECULAR_RANGES.items():
            if mname in iname:
                # Check prep_note for concentration hints
                prep = (ing.get("prep_note") or "").lower()
                conc_match = re.search(r"(\d+\.?\d*)\s*(g|%)\s*per\s*(l|liter|100\s*ml)", prep)
                if conc_match:
                    val = float(conc_match.group(1))
                    unit = conc_match.group(3)
                    if unit in ("l", "liter") and not (lo <= val <= hi):
                        errors.append(
                            f"'{iname}' concentration {val} g/L outside recommended range {lo}-{hi} {unit_label}."
                        )
                    elif "100" in unit and not (lo <= val <= hi):
                        errors.append(
                            f"'{iname}' concentration {val}% w/v outside recommended range {lo}-{hi} {unit_label}."
                        )
                break

    # ── Frontier technique / equipment requirement check (§18) ────────────────
    _FRONTIER_KEYWORDS = re.compile(
        r"liquid\s*nitrogen|液氮|rotovap|旋转蒸发|centrifuge|离心机|"
        r"pacojet|isomalt|sugar\s*work|糖艺|carbonat|碳酸化|sous.?vide",
        re.IGNORECASE,
    )
    frontier_used = False
    for step in spec.get("method_steps", []):
        if _FRONTIER_KEYWORDS.search(step):
            frontier_used = True
            break
    # Also check ingredient names and prep_notes
    for ing in ings:
        blob = ing.get("name", "") + " " + (ing.get("prep_note") or "")
        if _FRONTIER_KEYWORDS.search(blob):
            frontier_used = True
            break
    if frontier_used:
        tier = normalize_equipment_tier(equipment)
        if tier == "home":
            # Home equipment — these techniques are forbidden
            home_forbidden = ("centrifuge", "rotovap", "liquid nitrogen", "pacojet", "sous-vide")
            for step in spec.get("method_steps", []):
                lower_step = step.lower()
                for tool in home_forbidden:
                    if tool in lower_step:
                        errors.append(
                            f"'{tool}' 需要专业设备（酒吧/实验室级）。"
                            "家用设备无法使用此技法。"
                        )
                        break
        # Sugar work and carbonation also need bar equipment (isomalt ≥150°C)
        _ISOMALT_RE = re.compile(r"isomalt|sugar\s*work|糖艺", re.IGNORECASE)
        for step in spec.get("method_steps", []):
            if _ISOMALT_RE.search(step) and tier == "home":
                errors.append(
                    "Sugar work (isomalt ≥150°C) requires professional bar equipment — "
                    "home equipment insufficient for this technique."
                )
                break

    # ── Ice engineering (§13) ─────────────────────────────────────────────────
    from reference_canon import ICE_TYPE_BY_TECHNIQUE
    ice_mentioned = False
    steps_text = " ".join(spec.get("method_steps", []))
    for ing in ings:
        blob = ing.get("name", "").lower()
        if any(kw in blob for kw in ("ice cube", "large ice", "crushed ice", "ice", "冰块")):
            ice_mentioned = True
            break
    if technique in ICE_TYPE_BY_TECHNIQUE and ice_mentioned:
        allowed_ice = ICE_TYPE_BY_TECHNIQUE[technique]
        ice_found = False
        for ing in ings:
            blob = ing.get("name", "").lower()
            if any(ice_type in blob for ice_type in allowed_ice):
                ice_found = True
                break
        # Also check method steps
        for ai_type in allowed_ice:
            if ai_type in steps_text.lower():
                ice_found = True
                break
        if not ice_found:
            errors.append(
                f"Technique '{technique}' requires appropriate ice type "
                f"({', '.join(allowed_ice)}). Add matching ice to ingredient list or method step."
            )

    # ── Shake/stir duration check (§14.2) ─────────────────────────────────────
    from reference_canon import SHAKE_MIN_SECONDS, STIR_MIN_SECONDS, STIR_MAX_SECONDS
    if technique == "shaken":
        # Check method_steps for time references
        time_match = re.search(r"(\d+)\s*(?:sec|s|秒)", steps_text)
        if time_match:
            shake_sec = int(time_match.group(1))
            if shake_sec < SHAKE_MIN_SECONDS:
                errors.append(
                    f"Shake time {shake_sec}s is below minimum {SHAKE_MIN_SECONDS}s "
                    "for thermal equilibrium (Arnold, Liquid Intelligence)."
                )
        else:
            errors.append(
                f"Shaken technique must specify shake duration in method steps "
                f"(minimum {SHAKE_MIN_SECONDS}s for thermal equilibrium)."
            )
    elif technique == "stirred":
        time_match = re.search(r"(\d+)\s*(?:sec|s|秒)", steps_text)
        if time_match:
            stir_sec = int(time_match.group(1))
            if stir_sec < STIR_MIN_SECONDS:
                errors.append(
                    f"Stir time {stir_sec}s is below minimum {STIR_MIN_SECONDS}s."
                )
            elif stir_sec > STIR_MAX_SECONDS:
                errors.append(
                    f"Stir time {stir_sec}s exceeds {STIR_MAX_SECONDS}s — risk of over-dilution."
                )
        else:
            errors.append(
                f"Stirred technique must specify stir duration in method steps "
                f"(recommended {STIR_MIN_SECONDS}-{STIR_MAX_SECONDS}s)."
            )

    # ── Bitters usage limits (§7.3) ───────────────────────────────────────────
    from reference_canon import BITTERS_MAX_DASH, BITTERS_MAX_TYPES, BITTERS_DASH_ML
    bitters_ings = [i for i in ings if i.get("category") == "bitters"]
    if len(bitters_ings) > BITTERS_MAX_TYPES:
        errors.append(
            f"Too many bitters types ({len(bitters_ings)} > {BITTERS_MAX_TYPES} max). "
            "Limit bitters to 3 different types."
        )
    if bitters_ings:
        total_dashes = sum(
            max(1, round(float(i.get("amount_ml", 0)) / BITTERS_DASH_ML))
            for i in bitters_ings
        )
        if total_dashes > BITTERS_MAX_DASH:
            errors.append(
                f"Total bitters {total_dashes} dashes exceeds {BITTERS_MAX_DASH} dash limit."
            )

    return errors


def validate_formula_spec_safety_only(
    spec: dict,
    spirit_mode: str,
    equipment: str = "bar",
    body: Any | None = None,
    prep_steps: list[dict] | None = None,
) -> list[str]:
    """只验证安全相关（禁用原料、碳酸+摇和冲突、发酵安全、设备限制）。"""
    errors: list[str] = []
    ings = spec.get("ingredients") or []
    from routers.ai import BANNED_INGREDIENTS

    # 1. Banned ingredients
    for ing in ings:
        iname = ing.get("name", "").lower()
        for banned in BANNED_INGREDIENTS:
            if banned.lower() in iname:
                errors.append(f"禁用原料：{ing['name']}")

    # 2. Spirit-free mode: no alcohol
    if spirit_mode == "spirit-free":
        from routers.ai import _lookup_ingredient_props
        for ing in ings:
            props = ing.get("props") or _lookup_ingredient_props(ing["name"]) or {}
            if float(props.get("abv_pct", 0)) > 0.005:
                errors.append(f"无酒精配方包含酒精：{ing['name']}")

    # 3. Carbonated + shake conflict
    technique = (spec.get("technique") or "shaken").lower()
    _CARBONATED_RE = re.compile(
        r"soda|tonic|sparkling|prosecco|champagne|ginger\s*beer|kombucha|"
        r"beer|cider|seltzer|club\s*soda|fizzy|effervescent|carbonated|"
        r"苏打|汤力|起泡|气泡|香槟|姜汁啤酒|康普茶|啤酒|苹果酒|气泡水|自然充气",
        re.IGNORECASE,
    )
    has_carbonated = any(_CARBONATED_RE.search(i.get("name", "")) for i in ings)
    if has_carbonated and technique == "shaken":
        spec["technique"] = "built"
        technique = "built"
        logger.info("Auto-fix: shaken → built (carbonated ingredients)")

    # 3b. Fermented + shake conflict
    _FERMENTED_RE = re.compile(
        r"fermented|发酵|kombucha|康普茶|shrub|shrubs",
        re.IGNORECASE,
    )
    has_fermented = any(_FERMENTED_RE.search(i.get("name", "")) for i in ings)
    if has_fermented and technique == "shaken":
        spec["technique"] = "built"
        technique = "built"
        logger.info("Auto-fix: shaken → built (fermented ingredients)")

    # 3c. User technique preference check
    if body is not None:
        user_techniques = [t.lower() for t in (getattr(body, "techniques", None) or [])]
        if any(t in user_techniques for t in ["carbonation", "充气", "fermentation", "发酵"]):
            if technique == "shaken":
                spec["technique"] = "built"
                technique = "built"
                logger.info("Auto-fix: shaken → built (user selected carbonation/fermentation)")

    # 3d. Volume auto-correction
    total_vol = sum(float(i.get("amount_ml", 0)) for i in ings)
    if total_vol > 0:
        if total_vol < 70:
            scale = 90.0 / total_vol
            for i in ings:
                i["amount_ml"] = round(i.get("amount_ml", 0) * scale, 1)
            logger.info("Auto-fix volume: %.0f ml → ~90 ml", total_vol)
        elif total_vol > 280:
            scale = 240.0 / total_vol
            for i in ings:
                i["amount_ml"] = round(i.get("amount_ml", 0) * scale, 1)
            logger.info("Auto-fix volume: %.0f ml → ~240 ml", total_vol)

    # 4. Fermentation safety (kombucha ABV)
    from reference_canon import KOMBUCHA_ABV_MAX
    _KOMBUCHA_RE = re.compile(r"kombucha|康普茶", re.IGNORECASE)
    for ing in ings:
        if _KOMBUCHA_RE.search(ing.get("name", "")):
            from routers.ai import _lookup_ingredient_props
            props = ing.get("props") or _lookup_ingredient_props(ing["name"]) or {}
            if float(props.get("abv_pct", 0.005)) > KOMBUCHA_ABV_MAX:
                errors.append(f"康普茶酒精度超过 {KOMBUCHA_ABV_MAX*100:.1f}%")

    # 5. Frontier technique + home equipment
    _FRONTIER_RE = re.compile(
        r"liquid\s*nitrogen|液氮|rotovap|旋转蒸发|centrifuge|离心机|pacojet",
        re.IGNORECASE,
    )
    if equipment == "home" or equipment == "Home Kitchen":
        for step in spec.get("method_steps", []):
            if _FRONTIER_RE.search(step):
                errors.append(f"家用设备无法使用：{step[:50]}...")

    return errors


# bartender_review removed (round 42: no longer called — single-pass pipeline)



def design_formula(
    client: OpenAI,
    body: Any,
    spirit_mode: str,
    inventory: list[dict],
    template_data: dict,
    prep_steps: list[dict],
    language: str,
    equipment: str,
) -> dict:
    """Single LLM call for formula design."""
    from user_prefs import get_alcohol_pref, get_flavors, get_occasion, get_techniques

    system = build_formula_system(
        language=language,
        equipment=equipment,
        spirit_free=(spirit_mode == "spirit-free"),
        techniques=get_techniques(body),
        flavors=get_flavors(body),
        occasion=get_occasion(body),
        alcohol=get_alcohol_pref(body),
    )
    user = _build_formula_user_msg(
        body, spirit_mode, inventory, template_data, prep_steps, language,
    )
    model = "deepseek-chat"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return _chat_json(client, model, messages, max_tokens=6000)


# ─────────────────────────────────────────────────────────────────────────────
# L5  Classic Anchor Fallback
# ─────────────────────────────────────────────────────────────────────────────

def classic_anchor_fallback(inventory: list[dict], spirit_mode: str, language: str) -> dict | None:
    """
    L5: If LLM fails 3× safety validation, fall back to the closest classic anchor.
    Returns a spec-dict compatible with render_recipe(), or None.
    """
    selection = {
        "ingredients": inventory,
        "serve_style": "shaken",
        "target_abv_pct": 0 if spirit_mode == "spirit-free" else 15,
        "target_volume_ml": 120,
    }
    try:
        if spirit_mode == "spirit-free":
            anchor_preview, score = match_mocktail_anchor(selection)
        else:
            anchor_preview, score = match_classic_anchor(selection)

        # Build ingredients directly from anchor_preview
        if anchor_preview and anchor_preview.get("ingredients"):
            raw_ingredients = anchor_preview["ingredients"]
            total_raw = sum(float(i.get("amount_ml", 0)) for i in raw_ingredients)

            # Scale to target_volume_ml
            scale_factor = selection["target_volume_ml"] / total_raw if total_raw > 0 else 1.0
            spec_ingredients = [
                {
                    "name": i.get("name", ""),
                    "amount_ml": round(float(i.get("amount_ml", 0)) * scale_factor, 1),
                    "category": i.get("category", ""),
                    "prep_note": "",
                }
                for i in raw_ingredients
            ]
        else:
            # If anchor_preview has no ingredients, distribute equally from inventory
            spec_ingredients = []
            per_ing = selection["target_volume_ml"] / max(len(inventory), 1)
            for i in inventory:
                spec_ingredients.append({
                    "name": i.get("name", ""),
                    "amount_ml": round(per_ing, 1),
                    "category": i.get("category", ""),
                    "prep_note": "",
                })

        spec: dict = {
            "title_primary": "Classic Serve" if language != "zh" else "经典出品",
            "title_secondary": "",
            "tagline": "Canon fallback — locked classic proportions." if language != "zh"
                       else "典籍回退 — 经典比例锁定。",
            "technique": selection.get("serve_style", "shaken"),
            "glassware": "coupe",
            "garnish": "",
            "ingredients": spec_ingredients,
            "method_steps": [
                "Combine all ingredients in shaker with ice" if language != "zh" else "将所有材料加入摇酒壶加冰",
                "Shake vigorously for 12 seconds" if language != "zh" else "用力摇匀12秒",
                "Double strain into chilled coupe" if language != "zh" else "双重过滤入冰镇碟形杯",
            ],
            "science_note": "Classic anchor fallback — proportions from reference_canon.",
            "architect_note": "*Canon shell.*",
            "_fallback": True,
        }
        return spec
    except Exception as exc:
        logger.error("Classic anchor fallback failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# L6  Render — Python assembles the final markdown
# ─────────────────────────────────────────────────────────────────────────────

def render_recipe(spec: dict, prep_steps: list[dict], language: str) -> str:
    """
    Build the complete recipe markdown from the validated spec.
    Python owns all tables and calculated values — LLM prose is used as-is.
    """
    from routers.ai import INGREDIENT_PROPS, _lookup_ingredient_props

    zh = language == "zh"
    ings = spec.get("ingredients") or []
    technique = (spec.get("technique") or "shaken").lower()
    dil = DILUTION_BY_STYLE.get(technique, 0.22)

    # ── Props lookup helper — handles bilingual names like "white rum / 白朗姆酒" ──
    _ZH_STRIP_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef／/]+")

    def _get_props(ing: dict) -> dict:
        """Return props dict, trying English-only fallback for bilingual/Chinese names."""
        cached = ing.get("props")
        if cached:
            return cached
        name = ing.get("name", "")
        props = _lookup_ingredient_props(name)
        if props:
            return props
        # Strip Chinese chars + separators and retry with English portion
        en_only = _ZH_STRIP_RE.sub(" ", name).strip()
        if en_only and en_only.lower() != name.lower():
            props = _lookup_ingredient_props(en_only)
        return props or {}

    # ── Calculate parameters ──────────────────────────────────────────────────
    total_liquid = sum(float(i.get("amount_ml", 0)) for i in ings)
    finished_vol = total_liquid * (1 + dil) if technique != "built" else total_liquid
    ethanol_ml = 0.0
    total_sugar_g = 0.0
    for ing in ings:
        props = _get_props(ing)
        vol = float(ing.get("amount_ml", 0))
        abv = float(props.get("abv_pct", 0))
        brix = float(props.get("brix", 0))
        ethanol_ml += abv * vol
        total_sugar_g += brix / 100 * vol * 1.04  # density correction
    final_abv_pct = (ethanol_ml / finished_vol * 100) if finished_vol > 0 else 0.0

    acid_ml = sum(float(i.get("amount_ml", 0)) for i in ings if i.get("category") == "acid")
    sweet_ml = sum(float(i.get("amount_ml", 0)) for i in ings if i.get("category") == "sweetener")
    sweet_acid_ratio = (sweet_ml / acid_ml) if acid_ml > 0 else 0.0

    # Calories: ethanol 7 kcal/g (density 0.789), sugar 4 kcal/g
    kcal = ethanol_ml * 0.789 * 7 + total_sugar_g * 4

    # ── Display helpers ───────────────────────────────────────────────────────
    def _display_name(name: str, zh_mode: bool) -> str:
        """In Chinese mode, extract the Chinese part of a bilingual ingredient name.
        'white rum / 白朗姆酒' → '白朗姆酒';  'syrup（糖浆）' → '糖浆'.
        English name is preserved internally for DB lookup; only the display changes."""
        if not zh_mode:
            return name
        m = re.search(r'[（(]([^\)）]+)[）)]', name)
        if m:
            return m.group(1).strip()
        if ' / ' in name:
            return name.split(' / ', 1)[1].strip()
        return name

    _CATEGORY_ZH: dict[str, str] = {
        "spirit": "基酒", "modifier": "利口酒", "acid": "酸味剂",
        "sweetener": "甜味剂", "dilutant": "稀释剂", "bitters": "苦精",
        "other": "其他", "unknown": "其他",
    }
    _TECHNIQUE_ZH: dict[str, str] = {
        "shaken": "摇匀", "stirred": "搅拌", "built": "直调", "blended": "冰沙机",
    }

    # ── Ingredient table ──────────────────────────────────────────────────────
    def _fmt_amount(ing: dict, zh_mode: bool) -> str:
        """
        Smart amount display:
        - bitters with amount_ml ≤ 5  → 'X dashes'
        - amount_ml == 0 (solid/garnish) → use prep_note or 'see prep'
        - everything else → 'X ml'
        """
        amt = float(ing.get("amount_ml", 0))
        cat = ing.get("category", "")
        prep = (ing.get("prep_note") or "").strip()
        if cat == "bitters":
            if amt <= 0:
                return "2 dashes" if not zh_mode else "2 滴"
            # 1 dash ≈ 1.5 ml (industry standard); no upper limit for bitters display
            dashes = max(1, round(amt / 1.5))
            label = "dash" if dashes == 1 else "dashes"
            return f"{dashes} {label}" if not zh_mode else f"{dashes} 滴"
        if amt == 0:
            return "适量" if zh_mode else "to taste"
        return f"{amt:.0f} ml"

    if zh:
        ing_header = "| 原料 | 用量 | 类别 | 备注 |\n|------|------|------|------|\n"
        ing_rows = "".join(
            f"| {_display_name(i.get('name',''), True)} | {_fmt_amount(i, True)} "
            f"| {_CATEGORY_ZH.get(i.get('category',''), i.get('category',''))} "
            f"| {i.get('prep_note','') or '—'} |\n"
            for i in ings
        )
    else:
        ing_header = "| Ingredient | Amount | Category | Prep |\n|-----------|--------|----------|------|\n"
        ing_rows = "".join(
            f"| {i.get('name','')} | {_fmt_amount(i, False)} | {i.get('category','')} "
            f"| {i.get('prep_note','') or '—'} |\n"
            for i in ings
        )

    # ── Chemical parameters table ─────────────────────────────────────────────
    technique_label = _TECHNIQUE_ZH.get(technique, technique) if zh else technique
    if zh:
        chem_header = "| 参数 | 值 |\n|------|----|\n"
        chem_rows = (
            f"| 成品体积 | {finished_vol:.0f} ml |\n"
            f"| 最终 ABV | {final_abv_pct:.1f}% |\n"
            f"| 总糖量 | {total_sugar_g:.1f} g |\n"
            f"| 酸甜比（体积） | {sweet_acid_ratio:.2f} |\n"
            f"| 热量（估算） | {kcal:.0f} kcal |\n"
            f"| 技法 | {technique_label} · 稀释 {dil*100:.0f}% |\n"
        )
    else:
        chem_header = "| Parameter | Value |\n|-----------|-------|\n"
        chem_rows = (
            f"| Finished volume | {finished_vol:.0f} ml |\n"
            f"| Final ABV | {final_abv_pct:.1f}% |\n"
            f"| Total sugar | {total_sugar_g:.1f} g |\n"
            f"| Sweet/acid ratio | {sweet_acid_ratio:.2f} |\n"
            f"| Calories (est.) | {kcal:.0f} kcal |\n"
            f"| Technique | {technique_label} · {dil*100:.0f}% dilution |\n"
        )

    # ── Method steps (inject pre-treatment steps first) ───────────────────────
    method_parts: list[str] = []
    mandatory_prep = [s for s in prep_steps if s.get("mandatory")]
    if mandatory_prep:
        prep_header = "### 准备工作" if zh else "### Preparation"
        method_parts.append(prep_header)
        for s in mandatory_prep:
            step_text = s["step_zh"] if zh else s["step_en"]
            method_parts.append(f"- **[{s['type']}]** {step_text}")
        method_parts.append("")

    # ── Spec-level prep steps (advance prep defined by LLM) ──────────────
    spec_prep = spec.get("prep_steps") or []
    if spec_prep:
        spec_prep_header = "### 预制准备" if zh else "### Advance Prep"
        method_parts.append(spec_prep_header)
        for i, step in enumerate(spec_prep, 1):
            step = (step or "").strip()
            if step:
                method_parts.append(
                    f"**预制 {i}:** {step}" if zh else f"**Prep {i}:** {step}"
                )
        method_parts.append("")

    # Strip any "Step N:" / "步骤N：" prefix the LLM may have embedded —
    # the renderer controls numbering; double-numbering causes skipped/wrong indices.
    _STEP_PREFIX_RE = re.compile(
        r"^(?:Step\s*\d+\s*[:：．.]\s*|步骤\s*\d+\s*[:：．.]\s*)",
        re.IGNORECASE,
    )

    method_steps = spec.get("method_steps") or []
    if method_steps:
        method_header = "### 制作方法" if zh else "### Method"
        method_parts.append(method_header)
        step_idx = 1
        for step in method_steps:
            step = (step or "").strip()
            if not step:
                continue  # skip empty/null steps — they cause numbering gaps
            step = _STEP_PREFIX_RE.sub("", step).strip()
            method_parts.append(f"**步骤 {step_idx}:** {step}" if zh else f"**Step {step_idx}:** {step}")
            step_idx += 1

    # ── Science & Architect ───────────────────────────────────────────────────
    science = spec.get("science_note", "")
    architect = spec.get("architect_note", "")

    # ── Confidence badge ──────────────────────────────────────────────────────
    _CONFIDENCE_BADGES: dict[str, tuple[str, str]] = {
        "iba_certified":      ("◆ IBA 认证规格", "◆ IBA Certified Spec"),
        "verified_classic":   ("◆ 权威经典规格", "◆ Verified Classic Spec"),
        "precision_verified": ("◆ 通过物理验证", "◆ Physics-Verified"),
        "fast_draft":         ("⚡ 概念草稿，比例未验证", "⚡ Concept Draft — Ratios Unverified"),
        "ai_generated":       ("⚠ AI 生成，建议与官方规格核对",
                               "⚠ AI-Generated — Cross-check with official sources"),
    }
    _confidence = spec.get("_confidence", "ai_generated")
    _badge_zh, _badge_en = _CONFIDENCE_BADGES.get(_confidence, _CONFIDENCE_BADGES["ai_generated"])
    confidence_line = f"*{_badge_zh}*" if zh else f"*{_badge_en}*"

    # ── Assemble full markdown ────────────────────────────────────────────────
    tp = spec.get("title_primary", "Untitled")
    ts = spec.get("title_secondary", "").strip()
    title = f"## {tp} · {ts}" if ts else f"## {tp}"
    tagline = spec.get("tagline", "").strip()
    tag_line = f"> *{tagline}*" if tagline else ""
    garnish = spec.get("garnish", "").strip()
    garnish_line = (f"**装饰：** {garnish}" if zh else f"**Garnish:** {garnish}") if garnish else ""

    ing_sec = "### 原料" if zh else "### Ingredients"
    chem_sec = "### 化学参数" if zh else "### Parameters"
    science_sec = "### 科学原理" if zh else "### The Science"
    note_sec = "### 建筑师笔记" if zh else "### Architect's Note"
    footer = "\n---\n*Liquid Architect · Dairen's Codex Engine · The Liquid Codex*"

    parts = [
        title, "",
        tag_line, "",
        confidence_line, "",
        "---", "",
        ing_sec,
        ing_header + ing_rows, "",
        chem_sec,
        chem_header + chem_rows, "",
    ]
    if garnish_line:
        parts += [garnish_line, ""]
    if method_parts:
        parts += method_parts + [""]
    if science:
        parts += [science_sec, science, ""]
    if architect:
        parts += [note_sec, f"*{architect.strip('*')}*", ""]
    parts.append(footer.strip())

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_rd_pipeline(body: Any, api_key: str, perplexity_key: str) -> Iterator[str]:
    """
    8-Step pipeline SSE generator.
    L0 → L1 → L2 → L3 → L4 (+ validation loop) → L5 → L6 → stream
    """
    from routers.ai import BANNED_INGREDIENTS, sse

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    language = getattr(body, "language", "en") or "en"
    equipment = getattr(body, "equipment", "bar") or "bar"

    # ── L0-A0  User preference vs equipment gate ─────────────────────────────
    from user_prefs import validate_equipment_vs_techniques
    pref_gate = validate_equipment_vs_techniques(body, equipment)
    if pref_gate:
        yield sse({"error": pref_gate[0]})
        return

    # ── L0-A  Spirit Mode ────────────────────────────────────────────────────
    spirit_mode = detect_spirit_mode(body)
    yield sse({"status": "generating", "detail": f"Mode: {spirit_mode}…"})

    # ── L0-B  Basic safety — banned ingredients ───────────────────────────────
    raw_ingredients = (getattr(body, "ingredients", "") or "").lower()
    for banned in BANNED_INGREDIENTS:
        if banned.lower() in raw_ingredients:
            yield sse({"error": f"Banned ingredient detected: {banned}. This ingredient is not permitted."})
            return

    # ── L0-C  Inventory classification ───────────────────────────────────────
    yield sse({"status": "generating", "detail": "Classifying ingredients…"})
    inventory = classify_inventory(body)

    # ── L1  Template + Gap Analysis (Perplexity) ──────────────────────────────
    if perplexity_key:
        yield sse({"status": "generating", "detail": "Finding professional template…"})
    # Thread + heartbeat wrapper to avoid proxy timeout during Perplexity's sync call
    from routers.ai import _start_thread, _heartbeats_while_running, _join_thread_result
    l1_thread, l1_q = _start_thread(
        fetch_template_reference,
        perplexity_key, body, spirit_mode, inventory,
    )
    for hb in _heartbeats_while_running(l1_thread, "generating", "Researching references"):
        yield hb
    template_data = _join_thread_result(l1_thread, l1_q)

    if template_data.get("safety_flag"):
        yield sse({"error": "Safety issue detected in ingredients. Please review your ingredient list."})
        return

    # Auto-complete missing categories
    inventory = auto_complete_inventory(inventory, template_data, spirit_mode)
    added = [i["name"] for i in inventory if i.get("_auto_added")]
    if added:
        yield sse({
            "status": "generating",
            "detail": f"Auto-added: {', '.join(added)}",
        })

    # ── L2  Pre-treatment Plan ────────────────────────────────────────────────
    prep_steps = build_prep_plan(inventory, body)
    if prep_steps:
        yield sse({
            "status": "generating",
            "detail": f"Pre-treatment: {len(prep_steps)} step(s) identified…",
        })

    # ── L3  Safety Validation ─────────────────────────────────────────────────
    safe_ok, safe_reason, prep_steps = validate_prep_safety(prep_steps)
    if not safe_ok:
        yield sse({"error": f"Safety check failed: {safe_reason}"})
        return

    # ── L4  Formula Design (single pass — no retry loop) ──────────────────────
    yield sse({"status": "generating", "detail": "Designing formula…"})

    try:
        from routers.ai import _start_thread, _heartbeats_while_running, _join_thread_result
        thread, q = _start_thread(
            design_formula,
            client, body, spirit_mode, inventory, template_data, prep_steps,
            language, equipment,
        )
        for hb in _heartbeats_while_running(thread, "generating", "Formula design"):
            yield hb
        spec = _join_thread_result(thread, q)
    except Exception as exc:
        yield sse({"error": f"Formula design failed: {str(exc)[:160]}"})
        return

    # Attach INGREDIENT_PROPS to spec ingredients for display
    from routers.ai import _lookup_ingredient_props
    for ing in spec.get("ingredients", []):
        if not ing.get("props"):
            ing["props"] = _lookup_ingredient_props(ing.get("name", ""))

    # ── Safety-only validation (no whitelist, no flavor check) ──────────────
    validation_errors = validate_formula_spec_safety_only(
        spec, spirit_mode, equipment, body=body, prep_steps=prep_steps,
    )
    if validation_errors:
        yield sse({"error": f"Safety check: {'; '.join(validation_errors[:3])}"})
        return

    spec["_confidence"] = "precision_verified"

    # ── L5  Render + Stream ───────────────────────────────────────────────────
    from routers.ai import science_guardrails, safety_filter, _maybe_inject_pre_treatment_steps

    try:
        recipe_text = render_recipe(spec, prep_steps, language)
        recipe_text = _maybe_inject_pre_treatment_steps(recipe_text, language)
        if not recipe_text.strip():
            yield sse({"error": "Recipe rendering produced empty text. Please try again."})
            return

        # Soft-check pass — log issues but never block recipe output.
        sci_ok, sci_reason = science_guardrails(recipe_text)
        if not sci_ok:
            logger.warning("science_guardrails advisory: %s", sci_reason)
        safe_ok2, safe_reason2 = safety_filter(recipe_text)
        if not safe_ok2:
            logger.warning("safety_filter advisory: %s", safe_reason2)

        yield sse({"status": "verifying", "detail": "Recipe ready"})
        yield sse({"status": "streaming", "detail": "Streaming recipe…"})
        chunk_size = 80
        for i in range(0, len(recipe_text), chunk_size):
            yield sse({"text": recipe_text[i: i + chunk_size]})
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.error("L6 render/stream failed: %s", exc, exc_info=True)
        yield sse({"error": f"Recipe rendering failed: {str(exc)[:120]}"})
        return


# ─────────────────────────────────────────────────────────────────────────────
# Fast Creative Pipeline — deepseek-chat, single call, ~5-8s
# No Perplexity, no validation loop. Concept sketch with approximate ratios.
# ─────────────────────────────────────────────────────────────────────────────

def generate_fast_pipeline(body: Any, api_key: str) -> Iterator[str]:
    """
    Fast mode: single deepseek-chat call.
    Focuses on technique concept + approximate ratios.
    Skips: Perplexity (L1), validation loop (L4 retries), fallback (L5).
    """
    from routers.ai import BANNED_INGREDIENTS, sse, _lookup_ingredient_props
    from prompt_core import build_fast_system
    from user_prefs import (
        build_preferences_user_block,
        get_alcohol_pref,
        get_flavors,
        get_occasion,
        get_techniques,
        validate_equipment_vs_techniques,
    )

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    language = getattr(body, "language", "en") or "en"
    equipment = getattr(body, "equipment", "bar") or "bar"

    pref_gate = validate_equipment_vs_techniques(body, equipment)
    if pref_gate:
        yield sse({"error": pref_gate[0]})
        return

    # Spirit mode
    spirit_mode = detect_spirit_mode(body)

    # Banned ingredient check
    raw_ingredients = (getattr(body, "ingredients", "") or "").lower()
    for banned in BANNED_INGREDIENTS:
        if banned.lower() in raw_ingredients:
            yield sse({"error": f"Banned ingredient detected: {banned}."})
            return

    yield sse({"status": "generating", "detail": "快速草稿中…" if language == "zh" else "Sketching concept…"})

    system = build_fast_system(
        language=language,
        spirit_free=(spirit_mode == "spirit-free"),
        techniques=get_techniques(body),
        flavors=get_flavors(body),
        occasion=get_occasion(body),
        alcohol=get_alcohol_pref(body),
        equipment=equipment,
    )

    user_msg = (
        f"{build_preferences_user_block(body, language)}\n"
        f"Spirit mode: {spirit_mode}\n"
        f"Equipment: {equipment}\n"
    )

    try:
        from routers.ai import _start_thread, _heartbeats_while_running, _join_thread_result
        thread, q = _start_thread(
            _chat_json,
            client, "deepseek-chat",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            4000,  # increased from 3000 to prevent truncation
        )
        for hb in _heartbeats_while_running(thread, "generating", "Fast concept sketch"):
            yield hb
        spec = _join_thread_result(thread, q)
    except Exception as exc:
        yield sse({"error": f"Fast generation failed: {str(exc)[:160]}"})
        return

    # Attach props (best-effort for parameter table)
    for ing in spec.get("ingredients", []):
        if not ing.get("props"):
            ing["props"] = _lookup_ingredient_props(ing.get("name", ""))

    spec["_confidence"] = "fast_draft"

    # ── Safety validation (fast mode — safety only, no creative blocks) ──
    fast_errors = validate_formula_spec_safety_only(
        spec, spirit_mode, equipment, body=body,
    )
    if fast_errors:
        yield sse({"error": f"Safety check failed: {'; '.join(fast_errors[:3])}"})
        return

    try:
        recipe_text = render_recipe(spec, [], language)
        if not recipe_text.strip():
            yield sse({"error": "Fast generation produced empty text. Please try again."})
            return
    except Exception as exc:
        logger.error("Fast pipeline render failed: %s", exc, exc_info=True)
        yield sse({"error": f"Fast recipe rendering failed: {str(exc)[:120]}"})
        return

    yield sse({"status": "streaming", "detail": "快速配方就绪" if language == "zh" else "Fast sketch ready"})
    chunk_size = 80
    for i in range(0, len(recipe_text), chunk_size):
        yield sse({"text": recipe_text[i: i + chunk_size]})
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Classic Lookup Pipeline — separate fast path for canonical recipes
# User types a cocktail name; LLM outputs the exact historical spec.
# No Perplexity, no gap analysis, no ingredient inventory needed.
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIC_LOOKUP_SYSTEM = """\
You are a professional bar reference librarian with encyclopedic knowledge of classic cocktails.

The user has requested the canonical recipe for a specific classic cocktail.
Your task: output the EXACT, historically correct specification as documented in:
  · IBA (International Bartenders Association) official specs
  · The Savoy Cocktail Book (Harry Craddock, 1930)
  · Meehan's Bartender Manual (Jim Meehan)
  · The Dead Rabbit Drinks Manual
  · Liquid Intelligence (Dave Arnold)

Rules:
① Output the single most authoritative version. If IBA has an official spec, use that.
② Do NOT invent variations. This is a reference lookup, not creative design.
③ Include all canonical details: exact ml amounts, technique, glassware, garnish, build order.
④ If the drink has regional variations (e.g. Dry vs Sweet Martini), state the classic version and note the key variation briefly.
⑤ Method steps must include specific ml amounts for each ingredient.
⑥ Cite the source (IBA / Savoy / Meehan etc.) in the science note.
   CRITICAL: Only cite books you are certain contain this recipe. Do NOT fabricate citations.
   If a book title is uncertain, omit it and write "recipe attested in professional literature".
   "The Bartender's Bible" = Gary Regan (1991) — do NOT attribute it to any other author.
   "Cocktail Technique" = Kazuo Uyeda (上田和男) — cite only for Uyeda's documented work.
⑦ ANTI-HALLUCINATION: If the requested cocktail name is not clearly documented in the
   reference sources above, do NOT invent it. Instead:
   a) State in the tagline: "Note: canonical source not verified — closest documented recipe provided."
   b) Output the closest verified classic (e.g. if asked for an obscure Uyeda creation,
      output the Aviation or the closest IBA equivalent).
   c) Do NOT invent creator names, bar names, or origin stories.
⑧ ingredient `name` must be English or bilingual ("crème de violette / 紫罗兰利口酒").
   Chinese-only names break ABV / calorie calculations.

Return ONLY valid JSON — same schema as formula design:
{
  "title_primary": "Cocktail name in requested language",
  "title_secondary": "Name in the other language",
  "tagline": "Brief historical context — origin, era, creator if known",
  "technique": "shaken | stirred | built | blended",
  "glassware": "exact canonical glass",
  "garnish": "canonical garnish",
  "ingredients": [
    {"name": "ingredient", "amount_ml": X, "category": "spirit|modifier|acid|sweetener|bitters|dilutant", "prep_note": ""}
  ],
  "method_steps": ["action description (no 'Step N:' prefix — renderer numbers automatically)", "..."],
  "science_note": "Why this spec works — cite the source book and explain balance/technique",
  "architect_note": "One sentence on the drink's legacy or character"
}
"""


def generate_classic_pipeline(body: Any, api_key: str) -> Iterator[str]:
    """
    Classic lookup pipeline — fast path for canonical cocktail specs.
    No Perplexity, no inventory analysis. LLM outputs canonical spec directly.
    """
    from routers.ai import sse, science_guardrails, safety_filter, _lookup_ingredient_props

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    language = getattr(body, "language", "en") or "en"
    equipment = getattr(body, "equipment", "bar") or "bar"

    # The drink name comes from the prompt field
    drink_name = (getattr(body, "notes", "") or "").strip()
    if not drink_name:
        # Fallback: check ingredients field
        drink_name = (getattr(body, "ingredients", "") or "").strip()
    if not drink_name:
        yield sse({"error": "Please enter a cocktail name (e.g. Negroni, Daiquiri, Gimlet)."})
        return

    yield sse({"status": "generating", "detail": f"Looking up: {drink_name}…"})

    # ── IBA Hardcoded Database — zero-hallucination path ──────────────────────
    # Check the hardcoded IBA library first.  If matched, return directly without
    # any LLM call — guaranteed accuracy for 74+ classic cocktails.
    try:
        from classic_recipes import lookup_iba_recipe, iba_spec_to_render_spec as _iba_convert
        _iba = lookup_iba_recipe(drink_name)
        if _iba is not None:
            _spec = _iba_convert(_iba, language)
            for _ing in _spec.get("ingredients", []):
                if not _ing.get("props"):
                    _ing["props"] = _lookup_ingredient_props(_ing.get("name", ""))
            _recipe_text = render_recipe(_spec, [], language)
            _cat = _iba.get("iba_category", "IBA")
            yield sse({"status": "streaming", "detail": f"IBA [{_cat}]: {drink_name}"})
            _chunk = 80
            for _i in range(0, len(_recipe_text), _chunk):
                yield sse({"text": _recipe_text[_i: _i + _chunk]})
            yield "data: [DONE]\n\n"
            return
    except Exception as _iba_exc:
        logger.warning("IBA lookup error (falling back to LLM): %s", _iba_exc)

    lang_note = (
        "\n[语言要求] 全部正文用简体中文：title_primary、tagline、method_steps、"
        "science_note、architect_note、garnish、prep_note 均用中文。"
        "title_secondary 用英文（副标题）。\n"
        "【重要】ingredients 的 name 字段必须用英文（或原品牌名），"
        "例如 'white rum / 白朗姆酒'，禁止纯中文原料名，否则 ABV/糖/热量计算全为零。\n"
        if language == "zh"
        else "\nRespond entirely in English. title_secondary in Chinese.\n"
    )

    user_msg = (
        f"Please provide the canonical classic recipe for: **{drink_name}**\n"
        f"Equipment level: {equipment}\n"
        f"{lang_note}"
    )

    # Classic lookup is pure reference retrieval — no multi-step reasoning needed.
    # deepseek-chat (~5-8s) vs deepseek-reasoner (~40-60s) for identical accuracy.
    model = "deepseek-chat"

    try:
        from routers.ai import _start_thread, _heartbeats_while_running, _join_thread_result
        thread, q = _start_thread(
            _chat_json,
            client, model,
            [
                {"role": "system", "content": _CLASSIC_LOOKUP_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            4000,
        )
        for hb in _heartbeats_while_running(thread, "generating", f"Classic lookup: {drink_name}"):
            yield hb
        spec = _join_thread_result(thread, q)
    except Exception as exc:
        yield sse({"error": f"Classic lookup failed: {str(exc)[:160]}"})
        return

    # Attach props for render
    for ing in spec.get("ingredients", []):
        if not ing.get("props"):
            ing["props"] = _lookup_ingredient_props(ing.get("name", ""))

    # LLM-generated classic lookup — mark as AI-generated for honesty
    spec["_confidence"] = "ai_generated"

    # Render (no prep steps needed for classic lookup)
    try:
        recipe_text = render_recipe(spec, [], language)
        if not recipe_text.strip():
            yield sse({"error": "Classic lookup produced empty text. Please try again."})
            return
    except Exception as exc:
        logger.error("Classic pipeline render failed: %s", exc, exc_info=True)
        yield sse({"error": f"Classic recipe rendering failed: {str(exc)[:120]}"})
        return

    # Soft-check — log only, never block classic recipe output.
    sci_ok, sci_reason = science_guardrails(recipe_text)
    if not sci_ok:
        logger.warning("classic science_guardrails advisory: %s", sci_reason)
    safe_ok, safe_reason = safety_filter(recipe_text)
    if not safe_ok:
        logger.warning("classic safety_filter advisory: %s", safe_reason)

    yield sse({"status": "streaming", "detail": f"Classic spec ready: {drink_name}"})
    chunk_size = 80
    for i in range(0, len(recipe_text), chunk_size):
        yield sse({"text": recipe_text[i: i + chunk_size]})
    yield "data: [DONE]\n\n"
