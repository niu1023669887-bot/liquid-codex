"""
Professional reference canon — single source of truth for ratios, technique, and dilution.

Grounded in:
  · Dave Arnold, Liquid Intelligence (dilution, shake/stir kinetics, acid math)
  · Jim Meehan, Meehan's Bartender Manual (classic specs, serve discipline)
  · Sean Muldoon & Jack McGarry, Dead Rabbit Drinks Manual (Irish whiskey classics, build order)
  · Jason Logsdon, Modernist Infusions (time/temp for macerations)
  · The Liquid Codex (Darien) — layered on top as project-native extensions

Python balance engine and prompts MUST import from here — do not duplicate constants in ai.py.
"""

from __future__ import annotations

# ── Dilution by technique (Arnold: shake ~20–25%, stir ~15–18%, built ~10%) ──
# built dilution raised to 10% to account for ice melt in highball glasses.
DILUTION_BY_STYLE: dict[str, float] = {
    "shaken": 0.22,
    "blended": 0.25,
    "stirred": 0.17,
    "built": 0.10,
}

# ── Classic sour sweet:acid mass ratio (Meehan / IBA sours: 1.8–2.2) ──
SOUR_SWEET_ACID_RATIO_DEFAULT = 2.0
SOUR_SWEET_ACID_RANGE = (1.2, 2.2)

# ── Serve volumes (Meehan: coupe ~90–120 ml finished; highball 180–240) ──
DEFAULT_SINGLE_SERVE_ML = 120
HIGHBALL_SERVE_ML = 200
SPIRIT_FREE_SHAKEN_ML = 110

# ── Technique norms (Arnold + Dead Rabbit) ──
SHAKE_SECONDS = (10, 12)
STIR_SECONDS = (25, 35)
DOUBLE_STRAIN = "Hawthorne strainer + fine-mesh strainer"

# ── Infusion reference (Logsdon + Codex Vol.II) ──
INFUSION_ABV_SWEET_SPOT = (0.40, 0.50)
INFUSION_ROOM_TEMP_DAYS = (1, 7)
INFUSION_HEAT_MAX_SPIRIT_ABV = 0.25  # never co-heat ≥25% ABV spirit above 78°C

# ── Book citation snippets for prompts (short — full rules in hard_filter) ──
CANON_BLURB = """\
=== PROFESSIONAL REFERENCE CANON ===
Your recipes must align with these published standards (cite when relevant in 科学原理 / The Science):

· Dave Arnold · Liquid Intelligence — shake dilution ~22%, stir ~17%; ingredient amounts are LOCKED by the Python engine; you write technique and rationale only. Double-strain shaken drinks: Hawthorne + fine mesh.

· Jim Meehan · Bartender Manual — classic sours: sweetener:acid mass ratio 1.8–2.0; respect serve volume and glassware; build order: cheapest ingredient first when batching.

· Dead Rabbit · Drinks Manual — Irish whiskey sours and stirred spirit-forward drinks: precise ratios, no improvisation on locked quantities; method steps use ingredient NAMES not ml.

· Jason Logsdon · Infusions — room-temp maceration 40–50% ABV; time-bound extractions; fat-wash requires freeze-filter.

· The Liquid Codex (8 volumes) — material science, safety, and frontier technique layered above classics. When PRE-CALCULATED BALANCE exists, those numbers override all prose.

NON-NEGOTIABLE OUTPUT RULES:
① Do NOT output ingredient tables, chemical tables, recipe title, or tagline — those are pre-assembled by Python.
② Method steps: ingredient NAMES only (no ml/g in Method). Times (s), temperatures (°C), pH targets OK.
③ Spirit-free / mocktail: final ABV 0.0% — never imply Sidecar, Martini, or other spirit classics.
④ Fruit juice acid sources carry natural Brix — acknowledge in Science section when engine lists total_sugar_g > 0.
⑤ First prose section MUST be ### Equipment (or ### 设备 in Chinese).
"""

EQUIPMENT_TIERS: dict[str, dict] = {
    "home": {
        "labels": ("home", "kitchen", "家庭", "家用"),
        "forbidden_tools": (
            "centrifuge", "rotovap", "liquid nitrogen", "pacojet",
            "isi siphon", "refractometer", "ph meter", "sous-vide",
            "离心机", "旋转蒸发", "液氮", "均质机",
        ),
        "forbidden_ingredients": ("acid powder", "pectinex", "sodium alginate", "酸粉", "海藻酸钠"),
        "max_abv_pct": 22.0,
    },
    "bar": {
        "labels": ("bar", "professional", "pro", "专业", "酒吧", "实验室"),
        "forbidden_tools": ("rotovap", "liquid nitrogen", "pacojet", "centrifuge", "旋转蒸发", "液氮"),
        "forbidden_ingredients": (),
        "max_abv_pct": 28.0,
    },
}


def normalize_equipment_tier(equipment: str) -> str:
    lower = (equipment or "").lower()
    for tier, spec in EQUIPMENT_TIERS.items():
        if any(label in lower for label in spec["labels"]):
            return tier
    return "bar"


def equipment_constraint_block(equipment: str, language: str = "en") -> str:
    tier = normalize_equipment_tier(equipment)
    spec = EQUIPMENT_TIERS[tier]
    if language == "zh":
        forbidden = "、".join(spec["forbidden_tools"][:6])
        return (
            f"【设备档：{'家庭' if tier == 'home' else '专业'}】"
            f"禁止出现以下设备或技法：{forbidden}。"
            f"最终酒精度 ≤ {spec['max_abv_pct']:.0f}%。"
        )
    forbidden = ", ".join(spec["forbidden_tools"][:6])
    return (
        f"[Equipment tier: {'Home' if tier == 'home' else 'Bar/Pro'}] "
        f"FORBIDDEN tools/techniques: {forbidden}. "
        f"Max final ABV {spec['max_abv_pct']:.0f}%."
    )


# ── Fermentation parameters (Liquid Codex Vol.III — 003-cocktail-rules module fifteen) ──
LACTO_FERMENTATION_PH = (3.0, 3.5)
ACETIC_FERMENTATION_PH = (2.5, 3.0)
KOMBUCHA_FERMENTATION_PH = (2.5, 3.5)
KOMBUCHA_ABV_MAX = 0.015  # 1.5%
SHRUB_AGING_WEEKS = 2
FERMENTATION_TEMP_LACTO = (20, 24)  # °C
FERMENTATION_TEMP_KOMBUCHA = (24, 28)  # °C
FERMENTATION_DAYS_LACTO = (5, 14)
FERMENTATION_DAYS_KOMBUCHA = (7, 14)
SALT_PCT_LACTO = (1.5, 2.5)  # ±0.5% around 2%

# ── Molecular gastronomy ranges (Liquid Codex Vol.V — module seventeen) ──
AGAR_FLUID_GEL = (0.5, 2.0)  # g/L
SODIUM_ALGINATE_SPHERIFICATION = (0.5, 2.0)  # % w/v
XANTHAN_GUM_SUSPENSION = (0.1, 0.5)  # % w/v
LECITHIN_FOAM = (0.5, 2.0)  # % w/v
METHYLCELLULOSE_THERMAL_GEL = (0.5, 2.0)  # % w/v
GELLAN_GUM_FLUID_GEL = (0.3, 1.5)  # g/L
GUAR_GUM_THICKENER = (0.1, 0.5)  # % w/v

# ── Frontier technique temperatures (Liquid Codex Vol.VI — module eighteen) ──
ROTOVAP_TEMP = (40, 60)  # °C under vacuum
ROTOVAP_CHILLER_TEMP = (-10, None)  # °C; ≤ -10°C
LIQUID_NITROGEN_TEMP = -196  # °C
ISOMALT_SUGAR_WORK_TEMP = (160, 170)  # °C
SOUS_VIDE_GOLDEN_TEMP = (55, 65)  # °C; golden range
SOUS_VIDE_SAFE_HOURS_MAX = 8  # max hours at pH > 4.6

# ── Cocktail balance targets (Dave Arnold, Liquid Intelligence) ──
# Each type specifies (ABV_min%, ABV_max%, Brix_min g/100ml, Brix_max, Acid_min%, Acid_max%)
COCKTAIL_BALANCE_TARGETS: dict[str, tuple] = {
    "shaken":    (14, 17, 7, 11, 0.7, 1.1),
    "stirred":   (28, 33, 3, 5, 0.0, 0.09),
    "built":     (31, 33, 7, 8, 0.0, 0.1),
    "carbonated":(10, 12, 6, 8, 0.5, 0.8),
}

# ── Ice / technique validation constants ──
# Stirred must use large ice; shaken must use standard cubes; built must use large/rock cubes.
ICE_TYPE_BY_TECHNIQUE: dict[str, set[str]] = {
    "shaken":  {"standard_cube", "ice_cube", "standard"},
    "stirred": {"large_cube", "king_cube", "large ice cube", "large"},
    "built":   {"large_cube", "king_cube", "rock", "large ice cube", "rock ice", "large"},
}
SHAKE_MIN_SECONDS = 10  # minimum shake time for thermal equilibrium
STIR_MIN_SECONDS = 20   # minimum stir time
STIR_MAX_SECONDS = 60   # maximum stir time (over-dilution risk)

# ── Bitters / garnish usage limits ──
BITTERS_MAX_DASH = 5       # total dashes cap
BITTERS_MAX_TYPES = 3      # max different bitters in one drink
BITTERS_DASH_ML = 1.5      # 1 dash ≈ 1.5 ml (industry standard)

# ── Punch (潘切) five-element rule (Meehan's Bartender Manual) ──
PUNCH_ELEMENTS = {"spirit", "acid", "sweetener", "dilutant", "modifier"}
# spice element is handled via prep_note or keyword matching in validation


def dilution_for(style: str) -> float:
    return DILUTION_BY_STYLE.get(style, 0.20)
