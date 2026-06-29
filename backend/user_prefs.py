"""
User preference wiring — flavors, techniques, occasion, spirit.

Maps UI chip selections to prep steps, prompt blocks, and Python validation.
"""

from __future__ import annotations

import re
from typing import Any

from reference_canon import EQUIPMENT_TIERS, normalize_equipment_tier

# ── Occasion → serve guidance ────────────────────────────────────────────────

_OCCASION_GUIDANCE: dict[str, dict] = {
    "aperitivo": {
        "en": "Aperitivo / pre-dinner — bitter-leaning, moderate ABV, appetite-stimulating",
        "zh": "开胃酒——偏苦、中度酒精、晚餐前",
        "techniques": ("built", "stirred"),
        "glasses": ("rocks", "coupe", "cocktail glass", "wine glass", "nick & nora"),
    },
    "digestif": {
        "en": "Digestif — spirit-forward, after-dinner, rich or bitter finish",
        "zh": "餐后酒——偏烈、浓郁或苦甜收尾",
        "techniques": ("stirred",),
        "glasses": ("rocks", "coupe", "cocktail glass", "nick & nora"),
    },
    "highball": {
        "en": "Long drink / highball — tall serve, built in glass, carbonated top-up likely",
        "zh": "长饮——高球杯直调，常加气泡",
        "techniques": ("built",),
        "glasses": ("highball", "collins"),
    },
    "sour": {
        "en": "Sour / citrus-forward — shaken, balanced acid and sweetener",
        "zh": "酸酒——摇匀，酸甜平衡",
        "techniques": ("shaken",),
        "glasses": ("coupe", "cocktail glass", "nick & nora"),
    },
    "stirred": {
        "en": "Stirred / spirit-forward — low dilution, elegant short serve",
        "zh": "搅拌类——低稀释、短饮",
        "techniques": ("stirred",),
        "glasses": ("rocks", "coupe", "cocktail glass", "nick & nora"),
    },
    "tiki": {
        "en": "Tiki / complex tropical — layered rum, exotic fruit, crushed ice optional",
        "zh": "Tiki 热带——复杂果味、朗姆基酒",
        "techniques": ("shaken", "blended"),
        "glasses": ("tiki mug", "collins", "highball", "coupe"),
    },
    "non-alcoholic": {
        "en": "Non-alcoholic — ABV 0%, full flavor without spirits",
        "zh": "无酒精——ABV 0%",
        "techniques": ("shaken", "built", "stirred"),
        "glasses": ("highball", "coupe", "collins", "rocks"),
    },
}

# ── Flavor profile keywords for validation ───────────────────────────────────

_FLAVOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "citrus": (r"citrus", r"lemon", r"lime", r"orange", r"grapefruit", r"yuzu",
               r"柑橘", r"柠檬", r"青柠", r"橙", r"葡萄柚"),
    "tropical": (r"tropical", r"pineapple", r"mango", r"passion", r"guava", r"coconut", r"banana",
                 r"热带", r"菠萝", r"芒果", r"百香果", r"番石榴", r"椰子"),
    "berry": (r"berry", r"strawberry", r"raspberry", r"blueberry", r"blackberry", r"cranberry",
              r"浆果", r"草莓", r"覆盆子", r"蓝莓"),
    "floral": (r"floral", r"rose", r"elderflower", r"lavender", r"violet", r"hibiscus",
               r"花香", r"玫瑰", r"接骨木", r"薰衣草", r"紫罗兰"),
    "herbaceous": (r"herb", r"basil", r"mint", r"thyme", r"sage", r"cilantro", r"rosemary",
                   r"草本", r"罗勒", r"薄荷", r"百里香", r"迷迭香"),
    "earthy": (r"earthy", r"mushroom", r"truffle", r"beet", r"土", r"菌", r"松露"),
    "smoky": (r"smoky", r"smoke", r"mezcal", r"peat", r"烟熏", r"泥煤"),
    "spicy": (r"spicy", r"chili", r"pepper", r"ginger", r"cinnamon", r"辣", r"胡椒", r"姜"),
    "umami": (r"umami", r"miso", r"soy", r"seaweed", r"鲜", r"味噌", r"海藻"),
    "bitter": (r"bitter", r"campari", r"fernet", r"amaro", r"苦", r"金巴利"),
    "sweet": (r"sweet", r"syrup", r"honey", r"vanilla", r"甜", r"糖浆", r"蜂蜜"),
    "sour": (r"sour", r"acid", r"vinegar", r"shrub", r"酸"),
    "savoury": (r"savoury", r"savory", r"olive", r"tomato", r"咸", r"橄榄"),
    "fermented": (r"ferment", r"kombucha", r"kefir", r"发酵", r"康普茶"),
    "creamy": (r"cream", r"milk", r"coconut cream", r"egg", r"奶油", r"乳"),
    "mineral": (r"mineral", r"salty", r"saline", r"矿", r"盐"),
}

# ── Technique prep templates + validation keywords ───────────────────────────

_TECHNIQUE_SPECS: dict[str, dict] = {
    "maceration": {
        "keywords": (r"macerat", r"infus", r"steep", r"浸泡", r"浸渍", r"infusion"),
        "forbidden_tier": {},
        "step_en": (
            "Macerate {target} with chosen aromatics at 40–45% ABV, room temperature "
            "48–72 h. Strain through fine mesh. (Logsdon, Modernist Infusions)"
        ),
        "step_zh": (
            "将{target}与所选芳香原料按 40–45% ABV 浸渍，室温 48–72 小时，"
            "细网过滤。（Logsdon《现代浸泡》）"
        ),
    },
    "fat-wash": {
        "keywords": (r"fat.?wash", r"脂洗", r"freeze.?filter"),
        "forbidden_tier": {},
        "step_en": (
            "Fat-wash {target}: combine spirit with rendered fat at 5:1 ratio, "
            "infuse 2–4 h room temp, freeze −18 °C 24 h, strain through coffee filter."
        ),
        "step_zh": (
            "脂洗{target}：烈酒与脂肪 5:1 混合，室温 2–4 小时，"
            "−18 °C 冷冻 24 小时后咖啡滤纸过滤。"
        ),
    },
    "clarification": {
        "keywords": (r"clarif", r"milk.?wash", r"agar", r"filter", r"centrifug",
                      r"澄清", r"过滤", r"滤纸"),
        "forbidden_tier": {},
        "step_en": (
            "Clarify {target}: pass through coffee filter or milk-wash "
            "(add 20% whole milk by volume, curdle with acid, strain — yields crystal-clear liquid)."
        ),
        "step_zh": (
            "澄清{target}：咖啡滤纸过滤或奶洗法"
            "（加入 20% 全脂奶，以酸促凝，过滤得澄清液）。"
        ),
    },
    "fermentation": {
        "keywords": (r"ferment", r"kombucha", r"lacto", r"发酵"),
        "forbidden_tier": {},
        "step_en": (
            "Ferment {target}: lacto-ferment or kombucha-style ferment "
            "3–7 days at 20–25 °C until desired acidity; strain before use."
        ),
        "step_zh": (
            "发酵{target}：乳酸发酵或康普茶式发酵 3–7 天（20–25 °C），"
            "达目标酸度后过滤使用。"
        ),
    },
    "sous-vide": {
        "keywords": (r"sous.?vide", r"低温", r"vacuum", r"水浴", r"circulator", r"\d+\s*°C"),
        "forbidden_tier": {"home": "sous-vide requires Bar/Professional equipment tier"},
        "step_en": (
            "Sous-vide {target}: vacuum-seal with aromatics, "
            "55–65 °C water bath 2–4 h, ice-bath chill, strain."
        ),
        "step_zh": (
            "低温慢煮{target}：真空密封与芳香原料，55–65 °C 水浴 2–4 小时，"
            "冰浴降温后过滤。"
        ),
    },
    "carbonation": {
        "keywords": (r"carbonat", r"charge", r"siphon", r"isi", r"苏打", r"充气", r"气泡"),
        "forbidden_tier": {"home": "iSi siphon / carbonation requires Bar/Professional tier"},
        "step_en": (
            "Carbonate {target}: charge in iSi siphon with 2 CO₂ cartridges "
            "or top with 120 ml chilled soda water at service."
        ),
        "step_zh": (
            "充气{target}：iSi 瓶充 2 颗 CO₂ 弹，或出品时加 120 ml 冰镇苏打水。"
        ),
    },
    "spherification": {
        "keywords": (r"spherif", r"alginate", r"calcium", r"caviar", r"球化", r"海藻酸钠", r"乳酸钙"),
        "forbidden_tier": {"home": "spherification (sodium alginate) requires Bar/Professional tier"},
        "step_en": (
            "Spherification: prepare 2% sodium alginate bath and 5% calcium lactate setting bath; "
            "drop {target} alginate mixture into calcium bath, rinse, serve as garnish or component."
        ),
        "step_zh": (
            "球化：配制 2% 海藻酸钠浴与 5% 乳酸钙凝固浴；"
            "滴加{target}混合液，冲洗后作装饰或组分。"
        ),
    },
    "foam": {
        "keywords": (r"foam", r"lecithin", r"xanthan", r"aquafaba", r"泡沫"),
        "forbidden_tier": {},
        "step_en": (
            "Foam {target}: blend with 0.5% soy lecithin or 0.2% xanthan, "
            "charge in iSi siphon or hand-blend to stable foam."
        ),
        "step_zh": (
            "泡沫{target}：加入 0.5% 大豆卵磷脂或 0.2% 黄原胶，"
            "iSi 瓶或手持搅拌至稳定泡沫。"
        ),
    },
    "dehydration": {
        "keywords": (r"dehydrat", r"dry", r"oven", r"干燥", r"风干", r"脱水"),
        "forbidden_tier": {},
        "step_en": (
            "Dehydrate {target}: slice thin, dehydrate 55–60 °C for 6–8 h "
            "until crisp; use as garnish or powder."
        ),
        "step_zh": (
            "脱水{target}：切薄片，55–60 °C 脱水 6–8 小时至脆，"
            "作装饰或粉末。"
        ),
    },
    "rotovap": {
        "keywords": (r"rotovap", r"rotary", r"distill", r"旋转蒸发", r"蒸馏"),
        "forbidden_tier": {
            "home": "rotovap requires Bar/Professional tier",
            "bar": "rotovap is not available at this equipment tier — use clarification or maceration instead",
        },
        "step_en": (
            "Rotovap distill {target}: rotary evaporator at reduced pressure, "
            "40–50 °C bath — capture aromatic fraction."
        ),
        "step_zh": (
            "旋转蒸发{target}：减压蒸馏，浴温 40–50 °C，收集芳香馏分。"
        ),
    },
}

_SPIRIT_PATTERNS: dict[str, tuple[str, ...]] = {
    "whisky": (r"whisk", r"bourbon", r"rye", r"scotch", r"威士忌", r"波本"),
    "gin": (r"\bgin\b", r"金酒", r"琴酒"),
    "vodka": (r"vodka", r"伏特加"),
    "rum": (r"\brum\b", r"朗姆"),
    "tequila": (r"tequila", r"mezcal", r"龙舌兰", r"梅斯卡尔"),
    "cognac": (r"cognac", r"brandy", r"armagnac", r"干邑", r"白兰地"),
    "sake": (r"sake", r"shochu", r"soju", r"清酒", r"烧酒"),
    "pisco": (r"pisco", r"皮斯科"),
}


def get_techniques(body: Any) -> list[str]:
    raw = getattr(body, "techniques", None)
    if raw is not None:
        return [str(t).strip().lower() for t in raw if str(t).strip()]
    # Legacy: techniques merged into flavors before round 34
    _TECH_IDS = set(_TECHNIQUE_SPECS.keys())
    return [f for f in (getattr(body, "flavors", None) or []) if f.lower() in _TECH_IDS]


def get_flavors(body: Any) -> list[str]:
    _TECH_IDS = set(_TECHNIQUE_SPECS.keys())
    return [f for f in (getattr(body, "flavors", None) or []) if f.lower() not in _TECH_IDS]


def get_occasion(body: Any) -> str:
    return (getattr(body, "occasion", "") or "").strip().lower()


def get_alcohol_pref(body: Any) -> str:
    alc = (getattr(body, "alcohol", "") or "").strip().lower()
    if alc in ("", "let the codex decide", "由知识库决定"):
        return ""
    return alc


def occasion_guidance(occasion: str, language: str) -> str:
    if not occasion:
        return ""
    spec = _OCCASION_GUIDANCE.get(occasion, {})
    return spec.get("zh" if language == "zh" else "en", occasion)


def validate_equipment_vs_techniques(body: Any, equipment: str) -> list[str]:
    """Early gate — reject impossible technique + equipment combos."""
    tier = normalize_equipment_tier(equipment)
    errors: list[str] = []
    for tech in get_techniques(body):
        spec = _TECHNIQUE_SPECS.get(tech, {})
        msg = spec.get("forbidden_tier", {}).get(tier)
        if msg:
            errors.append(msg)
    # Home tier forbidden ingredients for spherification
    if tier == "home" and "spherification" in get_techniques(body):
        errors.append("Spherification (sodium alginate) requires Bar/Professional equipment tier.")
    return errors


def _pick_target(inventory: list[dict], prefer: str = "spirit") -> str:
    for ing in inventory:
        if ing.get("category") == prefer:
            return ing.get("name") or "base ingredient"
    if inventory:
        return inventory[0].get("name") or "base ingredient"
    return "base ingredient"


def build_technique_prep_steps(
    body: Any,
    inventory: list[dict],
    language: str,
) -> list[dict]:
    """Generate mandatory prep steps for each user-selected technique."""
    steps: list[dict] = []
    techniques = get_techniques(body)
    if not techniques:
        return steps

    for tech in techniques:
        spec = _TECHNIQUE_SPECS.get(tech)
        if not spec:
            continue
        if tech in ("fat-wash", "maceration", "sous-vide"):
            target = _pick_target(inventory, "spirit")
        elif tech in ("clarification", "fermentation", "carbonation", "foam"):
            target = _pick_target(inventory, "acid")
            if target == "base ingredient":
                target = _pick_target(inventory, "unknown")
        elif tech == "dehydration":
            target = _pick_target(inventory, "unknown")
        else:
            target = _pick_target(inventory, "spirit")

        step_en = spec["step_en"].format(target=target)
        step_zh = spec["step_zh"].format(target=target)
        steps.append({
            "ingredient": target,
            "type": tech,
            "mandatory": True,
            "step_en": step_en,
            "step_zh": step_zh,
        })
    return steps


def _blob_from_spec(spec: dict, prep_steps: list[dict] | None = None) -> str:
    parts = [
        spec.get("tagline") or "",
        spec.get("garnish") or "",
        " ".join(spec.get("method_steps") or []),
    ]
    for ing in spec.get("ingredients") or []:
        parts.append(ing.get("name") or "")
        parts.append(ing.get("prep_note") or "")
    for step in prep_steps or []:
        parts.append(step.get("step_en") or "")
        parts.append(step.get("step_zh") or "")
    return " ".join(parts).lower()


def validate_preferred_techniques(
    spec: dict,
    body: Any,
    prep_steps: list[dict] | None = None,  # noqa: ARG001 — kept for call-site compat
) -> list[str]:
    """Ensure every user-selected prep technique appears in recipe output."""
    techniques = get_techniques(body)
    if not techniques:
        return []

    blob = _blob_from_spec(spec)  # output only — not the L2 prep plan
    errors: list[str] = []
    for tech in techniques:
        spec_t = _TECHNIQUE_SPECS.get(tech)
        if not spec_t:
            continue
        if not any(re.search(kw, blob, re.IGNORECASE) for kw in spec_t["keywords"]):
            errors.append(
                f"User selected prep technique '{tech}' but it does not appear in "
                f"method_steps, prep_notes, or pre-treatment plan. "
                f"You MUST include a dedicated step for {tech}."
            )
    return errors


def validate_occasion_match(spec: dict, body: Any) -> list[str]:
    occasion = get_occasion(body)
    if not occasion or occasion not in _OCCASION_GUIDANCE:
        return []

    guidance = _OCCASION_GUIDANCE[occasion]
    technique = (spec.get("technique") or "").lower()
    glass = (spec.get("glassware") or "").lower()

    errors: list[str] = []
    if guidance["techniques"] and technique not in guidance["techniques"]:
        errors.append(
            f"Occasion '{occasion}' expects serve technique "
            f"{' or '.join(guidance['techniques'])}, got '{technique}'."
        )
    if guidance["glasses"]:
        glass_ok = any(g in glass for g in guidance["glasses"])
        if not glass_ok:
            errors.append(
                f"Occasion '{occasion}' expects glassware like "
                f"{', '.join(guidance['glasses'][:3])}, got '{spec.get('glassware')}'."
            )
    return errors


def validate_flavor_profiles(spec: dict, body: Any) -> list[str]:
    flavors = get_flavors(body)
    if not flavors:
        return []

    blob = _blob_from_spec(spec)
    errors: list[str] = []
    for flavor in flavors:
        keywords = _FLAVOR_KEYWORDS.get(flavor, (flavor,))
        if not any(re.search(kw, blob, re.IGNORECASE) for kw in keywords):
            errors.append(
                f"User selected flavor profile '{flavor}' but the recipe does not "
                f"clearly express it in ingredients, garnish, or method."
            )
    return errors


def validate_spirit_preference(spec: dict, body: Any) -> list[str]:
    pref = get_alcohol_pref(body)
    if not pref or pref in ("none", "no spirit", "spirit-free", "无酒精"):
        return []

    patterns = _SPIRIT_PATTERNS.get(pref, (pref,))
    blob = _blob_from_spec(spec)
    spirits = [
        i for i in (spec.get("ingredients") or [])
        if i.get("category") == "spirit"
    ]
    if not spirits:
        return [f"User requested base spirit '{pref}' but no spirit ingredient found."]

    spirit_blob = " ".join(
        (i.get("name") or "") + " " + (i.get("prep_note") or "") for i in spirits
    ).lower()
    if not any(re.search(p, spirit_blob, re.IGNORECASE) for p in patterns):
        return [
            f"User requested base spirit '{pref}' but primary spirit "
            f"({spirits[0].get('name')}) does not match."
        ]
    return []


def validate_all_user_prefs(
    spec: dict,
    body: Any,
    prep_steps: list[dict] | None = None,
) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_preferred_techniques(spec, body, prep_steps))
    errors.extend(validate_occasion_match(spec, body))
    errors.extend(validate_flavor_profiles(spec, body))
    errors.extend(validate_spirit_preference(spec, body))
    return errors


def build_preferences_user_block(body: Any, language: str) -> str:
    """Structured user-preference section for LLM user messages."""
    lines: list[str] = []
    concept = (getattr(body, "notes", "") or "").strip()
    if concept:
        lines.append(f"Concept / creative brief: {concept}")

    flavors = get_flavors(body)
    if flavors:
        lines.append(f"Flavor profiles (MUST express in the drink): {', '.join(flavors)}")

    techniques = get_techniques(body)
    if techniques:
        lines.append(
            "Preferred PREP techniques (MUST use ALL — each needs its own method_step "
            f"with time/temp): {', '.join(techniques)}"
        )
        lines.append(
            "Note: 'technique' in JSON = serve style (shaken/stirred/built/blended) ONLY. "
            "Prep techniques above go in method_steps and prep_note."
        )

    alc = get_alcohol_pref(body)
    if alc:
        lines.append(f"Preferred base spirit: {alc} (use as primary spirit)")

    occasion = get_occasion(body)
    if occasion:
        lines.append(f"Occasion / serve type: {occasion_guidance(occasion, language)}")

    ingredients = (getattr(body, "ingredients", "") or "").strip()
    if ingredients and ingredients.lower() not in ("no specific constraint", "无"):
        lines.append(f"Available ingredients (design using these): {ingredients}")

    return "\n".join(lines)
