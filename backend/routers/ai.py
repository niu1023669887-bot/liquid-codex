import os
import re
import json
import time
import queue
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

from data_codex import MATERIALS, EFSA_DATA
from lab_metrology import attach_metrology, classify_props_provenance, reconcile_measurements
from classic_anchor import (
    resolve_balance,
    match_classic_anchor,
    match_mocktail_anchor,
    selection_has_base_spirit,
    ANCHOR_THRESHOLD,
)
from prompt_core import build_generation_principles, build_surgeon_principles
from reference_canon import dilution_for, SPIRIT_FREE_SHAKEN_ML, DEFAULT_SINGLE_SERVE_ML
from ingredient_normalize import apply_synonym_hints, is_powder_ingredient

router = APIRouter(prefix="/api/ai", tags=["ai"])

# ── Quality Gate ────────────────────────────────────────────────────────────
# After all Surgeon passes, re-score with Codex Judge; loop until score ≥ floor.
# Set QUALITY_GATE_FLOOR=0 via env to disable entirely.
QUALITY_GATE_FLOOR: float = float(os.getenv("QUALITY_GATE_FLOOR", "7.5"))
QUALITY_GATE_MAX_PASSES: int = 2  # max extra Surgeon + Judge cycles

# ── Constants ──────────────────────────────────────────────────────────────
BANNED_INGREDIENTS = [
    "sassafras", "safrole", "黄樟树皮", "黄樟素",
    "pennyroyal", "胡薄荷", "胡薄荷油", "pulegone",
    "tansy oil", "艾菊油",
    "colchicine", "秋水仙碱",
    "cantharidin", "斑蝥素",
    "comfrey", "紫草",           # pyrrolizidine alkaloids
    "aristolochic", "马兜铃酸",
    "gyromitra", "鹿花菌", "gyromitrin",  # false morel — hydrazine; NOT safe even after cooking
]

# Foraged culinary ingredients — allowed as flavor sources; pipeline enforces heat protocols.
_FORAGED_CULINARY_OK = re.compile(
    r"见手青|红见手青|黄见手青|白见手青|jianshouqing|"
    r"boletus\b|porcini|cep\b|"
    r"松茸|牛肝菌|白牛肝菌|黄牛肝菌|红牛肝菌|黑牛肝菌|鸡枞|羊肚菌|"
    r"morel\b|chanterelle|matsutake|truffle|"
    r"elder\s*berr(?:y|ies)|接骨木(?:浆)?果",
    re.IGNORECASE,
)
_THERMOLABILE_MUSHROOM_RE = re.compile(
    r"见手青|红见手青|黄见手青|白见手青|jianshouqing|"
    r"牛肝菌|白牛肝菌|黄牛肝菌|红牛肝菌|黑牛肝菌|"
    r"boletus\b|porcini|cep\b|羊肚菌|morel\b",
    re.IGNORECASE,
)
_MUSHROOM_HEAT_TREATMENT_RE = re.compile(
    r"≥\s*100\s*°?\s*[Cc]|100\s*°?\s*[Cc]|沸水|沸腾|boiling\s+water|"
    r"\bblanch\b|焯水|煮沸|蒸煮|pre.?cook|parboil|热烫|灭活",
    re.IGNORECASE,
)
_ELDERBERRY_RAW_RE = re.compile(
    r"elder\s*berr(?:y|ies)(?!\s*(?:syrup|liqueur|cordial|juice|spirit|wine))|"
    r"接骨木(?:浆)?果(?!糖浆|果汁|酒)|黑接骨木果|sambucus\s+(?:nigra\s+)?berr",
    re.IGNORECASE,
)
_ELDERBERRY_TREATED_RE = re.compile(
    r"elder\s*berry\s+(?:syrup|liqueur|juice|cordial|spirit)|"
    r"cooked?\s+elder|simmer.*elder|boiled?\s+elder|elder.*simmer|"
    r"接骨木(?:糖浆|果汁|酒)|熟制接骨木",
    re.IGNORECASE,
)
_RAW_LEGUME_RE = re.compile(
    r"\braw\s+(?:kidney|runner|string|green|french|navy|cannellini|haricot|borlotti)\s+bean|"
    r"\buncooked\s+(?:kidney|runner|green|navy|french)\s+bean|"
    r"生(?:芸豆|四季豆|豆角|腰豆|菜豆)",
    re.IGNORECASE,
)
_LEGUME_TREATED_RE = re.compile(
    r"boiled?\s+(?:kidney|runner|green|navy|french|bean)|"
    r"cook(?:ed)?\s+(?:kidney|bean)|blanch.*bean|"
    r"熟(?:芸豆|四季豆|豆角|腰豆)|焯.*豆|煮.*豆",
    re.IGNORECASE,
)

# Registry: each entry = ingredient that requires deterministic pre-treatment injection
_PRE_TREATMENT_REGISTRY = [
    {
        "key": "thermolabile_mushroom",
        "detect_re": _THERMOLABILE_MUSHROOM_RE,
        "treated_re": _MUSHROOM_HEAT_TREATMENT_RE,
        "label_en": "Prep — Pre-cook wild mushrooms (food safety)",
        "label_zh": "准备 — 野生菌预处理（食品安全）",
        "step_en": (
            "Blanch wild mushrooms in vigorously boiling water (≥100 °C) for ≥5 min. "
            "Drain and cool; discard blanching liquid before further use. "
            "Destroys thermolabile proteins and reduces raw toxicity."
        ),
        "step_zh": (
            "将野生菌放入沸腾清水（≥100°C）中焯水 ≥5 min，捞出沥干冷却后方可使用。"
            "焯水液废弃不入饮品。此步骤可破坏热不稳定毒素蛋白。"
        ),
    },
    {
        "key": "raw_elderberry",
        "detect_re": _ELDERBERRY_RAW_RE,
        "treated_re": _ELDERBERRY_TREATED_RE,
        "label_en": "Prep — Pre-cook elderberries (cyanogenic glycosides)",
        "label_zh": "准备 — 接骨木浆果预处理（氰苷）",
        "step_en": (
            "Simmer elderberries in 200 ml water over medium heat for ≥15 min. "
            "Strain; discard solids — raw elderberries contain cyanogenic glycosides (sambunigrin). "
            "Use only the cooked liquid, or substitute commercial pre-cooked elderberry syrup/juice."
        ),
        "step_zh": (
            "将接骨木浆果加入 200 ml 清水中以中火熬煮 ≥15 min，过滤去除果渣与种子。"
            "生接骨木浆果含氰苷（sambunigrin），充分加热后方可安全使用。"
            "亦可直接用商业预熟接骨木糖浆替代生果。"
        ),
    },
    {
        "key": "raw_lectin_legume",
        "detect_re": _RAW_LEGUME_RE,
        "treated_re": _LEGUME_TREATED_RE,
        "label_en": "Prep — Pre-cook beans (lectin / PHA denaturation)",
        "label_zh": "准备 — 豆类预处理（植物凝集素灭活）",
        "step_en": (
            "Boil beans at a full rolling boil (≥100 °C) for ≥10 min. "
            "Do NOT use a slow cooker — temperatures below 100 °C activate rather than denature "
            "phytohaemagglutinin (PHA). Full boil completely destroys lectin activity."
        ),
        "step_zh": (
            "将豆类放入沸水（≥100°C）中大火滚煮 ≥10 min。"
            "禁止使用慢炖锅低温处理（低于 100°C 反而激活植物凝集素 PHA）。"
            "充分沸煮方可彻底灭活 PHA。"
        ),
    },
]

# ── Deterministic ingredient knowledge base ────────────────────────────────
INGREDIENT_PROPS: dict[str, dict] = {
    # key = lowercase English name (fuzzy-matched)
    # value = {abv_pct, ta_pct, brix, density, category}
    # category: spirit | acid | sweetener | dilutant | modifier | bitters

    # ── Base spirits ──
    "gin":               {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "london dry gin":    {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "hendricks":         {"abv_pct": 0.41, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "bombay sapphire":   {"abv_pct": 0.47, "ta_pct": 0.0, "brix": 0.0,  "density": 0.94, "category": "spirit"},
    "tanqueray":         {"abv_pct": 0.43, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "vodka":             {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "tequila":           {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "tequila blanco":    {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "tequila reposado":  {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "mezcal":            {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "rum":               {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "white rum":         {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "dark rum":          {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "aged rum":          {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "bourbon":           {"abv_pct": 0.43, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "rye whiskey":       {"abv_pct": 0.45, "ta_pct": 0.0, "brix": 0.0,  "density": 0.94, "category": "spirit"},
    "scotch":            {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "whisky":            {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "whiskey":           {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "irish whiskey":     {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    # ── Peated Islay malts (exact bottling ABV — LLM must NOT round to 40%) ──
    "ardbeg":            {"abv_pct": 0.46, "ta_pct": 0.0, "brix": 0.0,  "density": 0.945,"category": "spirit"},  # Ardbeg 10y 46%
    "ardbeg 10":         {"abv_pct": 0.46, "ta_pct": 0.0, "brix": 0.0,  "density": 0.945,"category": "spirit"},
    "ardbeg 10y":        {"abv_pct": 0.46, "ta_pct": 0.0, "brix": 0.0,  "density": 0.945,"category": "spirit"},
    "ardbeg uigeadail":  {"abv_pct": 0.546,"ta_pct": 0.0, "brix": 0.0,  "density": 0.938,"category": "spirit"},
    "ardbeg corryvreckan":{"abv_pct":0.574,"ta_pct": 0.0, "brix": 0.0,  "density": 0.936,"category": "spirit"},
    "laphroaig":         {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},  # Laphroaig 10y 40%
    "laphroaig 10":      {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "laphroaig quarter cask": {"abv_pct": 0.48, "ta_pct": 0.0, "brix": 0.0, "density": 0.943,"category": "spirit"},
    "lagavulin":         {"abv_pct": 0.43, "ta_pct": 0.0, "brix": 0.0,  "density": 0.948,"category": "spirit"},  # Lagavulin 16y 43%
    "lagavulin 16":      {"abv_pct": 0.43, "ta_pct": 0.0, "brix": 0.0,  "density": 0.948,"category": "spirit"},
    "bowmore":           {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},  # Bowmore 12y 40%
    "highland park":     {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},  # HP 12y 40%
    "peated scotch":     {"abv_pct": 0.46, "ta_pct": 0.0, "brix": 0.0,  "density": 0.945,"category": "spirit"},  # default peated to 46%
    "peated single malt":{"abv_pct": 0.46, "ta_pct": 0.0, "brix": 0.0,  "density": 0.945,"category": "spirit"},
    "islay whisky":      {"abv_pct": 0.46, "ta_pct": 0.0, "brix": 0.0,  "density": 0.945,"category": "spirit"},
    # ── Liqueurs / Amari (exact bottling ABV) ──
    "campari":           {"abv_pct": 0.25, "ta_pct": 0.3,  "brix": 28.0, "density": 1.06, "category": "modifier"},  # 25%; citric acid present
    "aperol":            {"abv_pct": 0.11, "ta_pct": 0.15, "brix": 26.0, "density": 1.07, "category": "modifier"},  # 11%; orange peel acidity
    "chartreuse":        {"abv_pct": 0.55, "ta_pct": 0.0,  "brix": 25.0, "density": 0.98, "category": "modifier"},  # Green Chartreuse 55%
    "green chartreuse":  {"abv_pct": 0.55, "ta_pct": 0.0,  "brix": 25.0, "density": 1.01, "category": "modifier"},
    "yellow chartreuse": {"abv_pct": 0.43, "ta_pct": 0.0,  "brix": 38.0, "density": 1.04, "category": "modifier"},
    "benedictine":       {"abv_pct": 0.40, "ta_pct": 0.1,  "brix": 35.0, "density": 1.10, "category": "modifier"},
    "averna":            {"abv_pct": 0.29, "ta_pct": 0.1,  "brix": 28.0, "density": 1.06, "category": "modifier"},
    "cynar":             {"abv_pct": 0.165,"ta_pct": 0.4,  "brix": 28.0, "density": 1.05, "category": "modifier"},
    "suze":              {"abv_pct": 0.15, "ta_pct": 0.2,  "brix": 18.0, "density": 1.02, "category": "modifier"},
    "drambuie":          {"abv_pct": 0.40, "ta_pct": 0.1,  "brix": 38.0, "density": 1.10, "category": "modifier"},
    "amontillado sherry":{"abv_pct": 0.185,"ta_pct": 0.5, "brix": 8.0,  "density": 1.02, "category": "modifier"},  # 18.5% typical
    "oloroso sherry":    {"abv_pct": 0.18, "ta_pct": 0.4, "brix": 6.0,  "density": 1.02, "category": "modifier"},
    "pedro ximenez":     {"abv_pct": 0.17, "ta_pct": 0.5, "brix": 35.0, "density": 1.15, "category": "sweetener"},  # PX very sweet
    "cognac":            {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "brandy":            {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "calvados":          {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "pisco":             {"abv_pct": 0.42, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "absinthe":          {"abv_pct": 0.68, "ta_pct": 0.0, "brix": 0.0,  "density": 0.91, "category": "spirit"},
    "grappa":            {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "soju":              {"abv_pct": 0.25, "ta_pct": 0.0, "brix": 0.0,  "density": 0.96, "category": "spirit"},
    "baijiu":            {"abv_pct": 0.53, "ta_pct": 0.0, "brix": 0.0,  "density": 0.93, "category": "spirit"},
    "sake":              {"abv_pct": 0.15, "ta_pct": 0.0, "brix": 4.0,  "density": 0.99, "category": "spirit"},
    "shochu":            {"abv_pct": 0.25, "ta_pct": 0.0, "brix": 0.0,  "density": 0.96, "category": "spirit"},

    # ── Liqueurs / Modifiers ──
    "cointreau":         {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 40.0, "density": 1.04, "category": "modifier"},
    "triple sec":        {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 40.0, "density": 1.04, "category": "modifier"},
    "grand marnier":     {"abv_pct": 0.40, "ta_pct": 0.0, "brix": 35.0, "density": 1.03, "category": "modifier"},
    "st-germain":        {"abv_pct": 0.20, "ta_pct": 0.0, "brix": 45.0, "density": 1.07, "category": "modifier"},
    "kahlua":            {"abv_pct": 0.20, "ta_pct": 0.0, "brix": 50.0, "density": 1.10, "category": "modifier"},
    "baileys":           {"abv_pct": 0.17, "ta_pct": 0.0, "brix": 30.0, "density": 1.06, "category": "modifier"},
    "maraschino":        {"abv_pct": 0.32, "ta_pct": 0.0, "brix": 30.0, "density": 1.04, "category": "modifier"},
    "creme de cassis":   {"abv_pct": 0.15, "ta_pct": 0.0, "brix": 55.0, "density": 1.15, "category": "modifier"},
    "falernum":          {"abv_pct": 0.06, "ta_pct": 0.3, "brix": 45.0, "density": 1.08, "category": "modifier"},
    "amaro":             {"abv_pct": 0.30, "ta_pct": 0.0, "brix": 25.0, "density": 1.05, "category": "modifier"},
    "fernet branca":     {"abv_pct": 0.39, "ta_pct": 0.0, "brix": 20.0, "density": 1.04, "category": "modifier"},
    # ── Fortified wines ──
    "sweet vermouth":    {"abv_pct": 0.16, "ta_pct": 0.5, "brix": 15.0, "density": 1.02, "category": "modifier"},
    "dry vermouth":      {"abv_pct": 0.18, "ta_pct": 0.5, "brix": 4.0,  "density": 1.01, "category": "modifier"},
    "lillet blanc":      {"abv_pct": 0.17, "ta_pct": 0.4, "brix": 8.0,  "density": 1.02, "category": "modifier"},
    "sherry":            {"abv_pct": 0.17, "ta_pct": 0.4, "brix": 6.0,  "density": 1.02, "category": "modifier"},
    "port":              {"abv_pct": 0.20, "ta_pct": 0.5, "brix": 10.0, "density": 1.05, "category": "modifier"},

    # ── Fresh juices (acid sources) ──
    "lime juice":        {"abv_pct": 0.0, "ta_pct": 5.5, "brix": 7.5,  "density": 1.03, "category": "acid"},
    "lemon juice":       {"abv_pct": 0.0, "ta_pct": 5.0, "brix": 8.0,  "density": 1.03, "category": "acid"},
    "grapefruit juice":  {"abv_pct": 0.0, "ta_pct": 1.5, "brix": 10.0, "density": 1.04, "category": "acid"},
    "orange juice":      {"abv_pct": 0.0, "ta_pct": 0.8, "brix": 11.0, "density": 1.04, "category": "acid"},
    "yuzu juice":        {"abv_pct": 0.0, "ta_pct": 4.2, "brix": 7.0,  "density": 1.03, "category": "acid"},
    "pineapple juice":   {"abv_pct": 0.0, "ta_pct": 0.7, "brix": 12.0, "density": 1.04, "category": "acid"},
    "passion fruit":     {"abv_pct": 0.0, "ta_pct": 3.0, "brix": 15.0, "density": 1.05, "category": "acid"},
    "apple juice":       {"abv_pct": 0.0, "ta_pct": 0.4, "brix": 12.0, "density": 1.04, "category": "acid"},
    "cranberry juice":   {"abv_pct": 0.0, "ta_pct": 1.5, "brix": 11.0, "density": 1.04, "category": "acid"},
    "verjuice":          {"abv_pct": 0.0, "ta_pct": 1.2, "brix": 7.0,  "density": 1.03, "category": "acid"},

    # ── Sweeteners ──
    "simple syrup":      {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 50.0, "density": 1.18, "category": "sweetener"},
    "1:1 simple syrup":  {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 50.0, "density": 1.18, "category": "sweetener"},
    "rich syrup":        {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 65.0, "density": 1.29, "category": "sweetener"},
    "2:1 syrup":         {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 65.0, "density": 1.29, "category": "sweetener"},
    "honey syrup":       {"abv_pct": 0.0, "ta_pct": 0.1, "brix": 55.0, "density": 1.22, "category": "sweetener"},
    "agave syrup":       {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 55.0, "density": 1.22, "category": "sweetener"},
    "grenadine":         {"abv_pct": 0.0, "ta_pct": 0.3, "brix": 60.0, "density": 1.24, "category": "sweetener"},
    "orgeat":            {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 60.0, "density": 1.24, "category": "sweetener"},
    "demerara syrup":    {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 65.0, "density": 1.29, "category": "sweetener"},
    "maple syrup":       {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 66.0, "density": 1.30, "category": "sweetener"},

    # ── Dilutants ──
    "soda water":        {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 0.0, "density": 1.00, "category": "dilutant"},
    "sparkling water":   {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 0.0, "density": 1.00, "category": "dilutant"},
    "coconut water":     {"abv_pct": 0.0, "ta_pct": 0.1, "brix": 4.0, "density": 1.02, "category": "dilutant"},
    "water":             {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 0.0, "density": 1.00, "category": "dilutant"},
    "cold brew":         {"abv_pct": 0.0, "ta_pct": 0.3, "brix": 2.0, "density": 1.01, "category": "dilutant"},
    "green tea":         {"abv_pct": 0.0, "ta_pct": 0.2, "brix": 1.0, "density": 1.00, "category": "dilutant"},

    # ── Dairy / Egg ──
    "heavy cream":       {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 0.0, "density": 1.01, "category": "modifier"},
    "egg white":         {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 0.0, "density": 1.03, "category": "modifier"},
    "whole egg":         {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 0.0, "density": 1.03, "category": "modifier"},
    "milk":              {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 4.5, "density": 1.03, "category": "modifier"},
    "coconut cream":     {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 12.0,"density": 1.04, "category": "modifier"},
    "coconut milk":      {"abv_pct": 0.0, "ta_pct": 0.0, "brix": 7.0, "density": 1.02, "category": "modifier"},

    # ── Bitters ──
    "angostura":         {"abv_pct": 0.447,"ta_pct": 0.0,"brix": 20.0,"density": 1.02, "category": "bitters"},
    "peychauds":         {"abv_pct": 0.35, "ta_pct": 0.0,"brix": 15.0,"density": 1.02, "category": "bitters"},
    "orange bitters":    {"abv_pct": 0.28, "ta_pct": 0.0,"brix": 15.0,"density": 1.02, "category": "bitters"},
    "bitters":           {"abv_pct": 0.40, "ta_pct": 0.0,"brix": 15.0,"density": 1.02, "category": "bitters"},

    # ── Enriched entries (Perplexity sonar, one-time offline run 2026-06-24) ──

    # ── Sparkling wines / low-ABV spirits ──
    "prosecco":              {"abv_pct": 0.11, "ta_pct": 0.6,  "brix": 1.8,  "density": 0.99, "category": "spirit"},
    "champagne":             {"abv_pct": 0.12, "ta_pct": 0.7,  "brix": 1.5,  "density": 0.99, "category": "spirit"},
    "cava":                  {"abv_pct": 0.11, "ta_pct": 0.65, "brix": 1.6,  "density": 0.99, "category": "spirit"},
    "sake nigori":           {"abv_pct": 0.155,"ta_pct": 0.4,  "brix": 12.0, "density": 1.01, "category": "spirit"},
    "baijiu sauce aroma":    {"abv_pct": 0.54, "ta_pct": 0.2,  "brix": 2.0,  "density": 0.94, "category": "spirit"},
    "beer":                  {"abv_pct": 0.05, "ta_pct": 0.2,  "brix": 3.5,  "density": 1.01, "category": "dilutant"},
    "stout":                 {"abv_pct": 0.055,"ta_pct": 0.25, "brix": 5.2,  "density": 1.02, "category": "dilutant"},

    # ── Additional liqueurs / modifiers ──
    # Crème de Violette — Rothman & Winter 20% ABV; Tempus Fugit 16%. Brix ~52 (very sweet floral).
    "creme de violette":     {"abv_pct": 0.20, "ta_pct": 0.1,  "brix": 52.0, "density": 1.07, "category": "modifier"},
    "crème de violette":     {"abv_pct": 0.20, "ta_pct": 0.1,  "brix": 52.0, "density": 1.07, "category": "modifier"},
    "violet liqueur":        {"abv_pct": 0.20, "ta_pct": 0.1,  "brix": 52.0, "density": 1.07, "category": "modifier"},
    "violette":              {"abv_pct": 0.20, "ta_pct": 0.1,  "brix": 52.0, "density": 1.07, "category": "modifier"},
    "紫罗兰利口酒":           {"abv_pct": 0.20, "ta_pct": 0.1,  "brix": 52.0, "density": 1.07, "category": "modifier"},
    # Crème de Mûre / Yvette / other floral-berry crèmes
    "creme de mure":         {"abv_pct": 0.18, "ta_pct": 1.5,  "brix": 50.0, "density": 1.06, "category": "modifier"},
    "creme de peche":        {"abv_pct": 0.18, "ta_pct": 0.5,  "brix": 48.0, "density": 1.06, "category": "modifier"},
    "creme de menthe":       {"abv_pct": 0.25, "ta_pct": 0.0,  "brix": 55.0, "density": 1.10, "category": "modifier"},
    "creme de cacao":        {"abv_pct": 0.25, "ta_pct": 0.0,  "brix": 50.0, "density": 1.08, "category": "modifier"},
    "parfait amour":         {"abv_pct": 0.25, "ta_pct": 0.5,  "brix": 48.0, "density": 1.06, "category": "modifier"},
    "umeshu":                {"abv_pct": 0.14, "ta_pct": 0.8,  "brix": 35.0, "density": 1.08, "category": "modifier"},
    "frangelico":            {"abv_pct": 0.20, "ta_pct": 0.1,  "brix": 40.0, "density": 1.13, "category": "modifier"},
    "chambord":              {"abv_pct": 0.165,"ta_pct": 2.5,  "brix": 45.0, "density": 1.05, "category": "modifier"},
    "midori":                {"abv_pct": 0.20, "ta_pct": 1.5,  "brix": 50.0, "density": 1.07, "category": "modifier"},
    "blue curacao":          {"abv_pct": 0.20, "ta_pct": 1.0,  "brix": 40.0, "density": 1.04, "category": "modifier"},
    "passoa":                {"abv_pct": 0.15, "ta_pct": 2.0,  "brix": 48.0, "density": 1.06, "category": "modifier"},
    "pimms":                 {"abv_pct": 0.25, "ta_pct": 0.2,  "brix": 20.0, "density": 1.02, "category": "modifier"},
    "malibu":                {"abv_pct": 0.13, "ta_pct": 0.5,  "brix": 35.0, "density": 1.03, "category": "modifier"},
    "disaronno":             {"abv_pct": 0.28, "ta_pct": 1.2,  "brix": 42.0, "density": 1.05, "category": "modifier"},
    "italicus":              {"abv_pct": 0.20, "ta_pct": 1.8,  "brix": 38.0, "density": 1.04, "category": "modifier"},
    "strega":                {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 35.0, "density": 0.95, "category": "modifier"},
    "galliano":              {"abv_pct": 0.30, "ta_pct": 0.0,  "brix": 40.0, "density": 0.98, "category": "modifier"},
    "licor 43":              {"abv_pct": 0.27, "ta_pct": 0.0,  "brix": 45.0, "density": 1.02, "category": "modifier"},
    "tia maria":             {"abv_pct": 0.20, "ta_pct": 0.0,  "brix": 38.0, "density": 1.01, "category": "modifier"},
    "montenegro":            {"abv_pct": 0.23, "ta_pct": 0.0,  "brix": 36.0, "density": 0.96, "category": "modifier"},
    "carpano antica":        {"abv_pct": 0.20, "ta_pct": 0.0,  "brix": 50.0, "density": 1.08, "category": "modifier"},
    "cocchi americano":      {"abv_pct": 0.16, "ta_pct": 0.0,  "brix": 48.0, "density": 1.06, "category": "modifier"},

    # ── Flavoured syrups ──
    "lavender syrup":        {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 65.0, "density": 1.25, "category": "sweetener"},
    "rose syrup":            {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 65.0, "density": 1.28, "category": "sweetener"},
    "elderflower syrup":     {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 64.0, "density": 1.27, "category": "sweetener"},
    "passion fruit syrup":   {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 62.0, "density": 1.26, "category": "sweetener"},
    "ginger syrup":          {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 66.0, "density": 1.29, "category": "sweetener"},
    "cinnamon syrup":        {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 65.0, "density": 1.28, "category": "sweetener"},
    "hibiscus syrup":        {"abv_pct": 0.0, "ta_pct": 1.5,  "brix": 63.0, "density": 1.26, "category": "sweetener"},
    "raspberry syrup":       {"abv_pct": 0.0, "ta_pct": 1.0,  "brix": 64.0, "density": 1.27, "category": "sweetener"},
    "blackberry syrup":      {"abv_pct": 0.0, "ta_pct": 0.9,  "brix": 64.0, "density": 1.27, "category": "sweetener"},
    "thai basil syrup":      {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 48.5, "density": 1.20, "category": "sweetener"},
    "honey":                 {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 80.0, "density": 1.40, "category": "sweetener"},
    "agave nectar":          {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 75.0, "density": 1.32, "category": "sweetener"},
    "invert sugar syrup":    {"abv_pct": 0.0, "ta_pct": 0.05, "brix": 70.0, "density": 1.31, "category": "sweetener"},

    # ── Acid powders / solutions ──
    "citric acid":           {"abv_pct": 0.0, "ta_pct": 8.0,  "brix": 0.0,  "density": 1.00, "category": "acid"},
    "citric acid solution":  {"abv_pct": 0.0, "ta_pct": 8.5,  "brix": 0.0,  "density": 1.05, "category": "acid"},
    "malic acid":            {"abv_pct": 0.0, "ta_pct": 7.5,  "brix": 0.0,  "density": 1.00, "category": "acid"},
    "malic acid solution":   {"abv_pct": 0.0, "ta_pct": 7.2,  "brix": 0.0,  "density": 1.04, "category": "acid"},
    "tartaric acid":         {"abv_pct": 0.0, "ta_pct": 7.0,  "brix": 0.0,  "density": 1.00, "category": "acid"},
    "tartaric acid solution":{"abv_pct": 0.0, "ta_pct": 9.0,  "brix": 0.0,  "density": 1.06, "category": "acid"},
    "lactic acid":           {"abv_pct": 0.0, "ta_pct": 6.5,  "brix": 0.0,  "density": 1.20, "category": "acid"},
    "lactic acid solution":  {"abv_pct": 0.0, "ta_pct": 6.5,  "brix": 0.0,  "density": 1.03, "category": "acid"},
    "succinic acid":         {"abv_pct": 0.0, "ta_pct": 6.0,  "brix": 0.0,  "density": 1.00, "category": "acid"},
    "phosphoric acid":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.33, "category": "acid"},
    "acetic acid":           {"abv_pct": 0.0, "ta_pct": 6.0,  "brix": 0.0,  "density": 1.05, "category": "acid"},
    "ascorbic acid":         {"abv_pct": 0.0, "ta_pct": 4.5,  "brix": 0.0,  "density": 1.00, "category": "acid"},

    # ── Fresh fruit (as acid/flavor source) ──
    "yellow lemon":          {"abv_pct": 0.0, "ta_pct": 5.0,  "brix": 9.0,  "density": 1.03, "category": "acid"},
    "lime":                  {"abv_pct": 0.0, "ta_pct": 5.0,  "brix": 8.0,  "density": 1.03, "category": "acid"},
    "yuzu":                  {"abv_pct": 0.0, "ta_pct": 4.5,  "brix": 10.0, "density": 1.04, "category": "acid"},
    "matcha":                {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.55, "category": "modifier"},
    "sudachi":               {"abv_pct": 0.0, "ta_pct": 4.5,  "brix": 9.0,  "density": 1.03, "category": "acid"},
    "calamansi":             {"abv_pct": 0.0, "ta_pct": 4.5,  "brix": 9.0,  "density": 1.03, "category": "acid"},
    "bergamot":              {"abv_pct": 0.0, "ta_pct": 5.8,  "brix": 8.2,  "density": 0.98, "category": "acid"},
    "finger lime":           {"abv_pct": 0.0, "ta_pct": 5.0,  "brix": 10.0, "density": 1.04, "category": "acid"},
    "grapefruit":            {"abv_pct": 0.0, "ta_pct": 2.5,  "brix": 10.0, "density": 1.04, "category": "acid"},
    "blood orange":          {"abv_pct": 0.0, "ta_pct": 2.0,  "brix": 11.0, "density": 1.05, "category": "acid"},
    "strawberry":            {"abv_pct": 0.0, "ta_pct": 0.7,  "brix": 6.5,  "density": 1.03, "category": "acid"},
    "raspberry":             {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 7.8,  "density": 1.03, "category": "acid"},
    "blackberry":            {"abv_pct": 0.0, "ta_pct": 0.9,  "brix": 10.2, "density": 1.03, "category": "acid"},
    "blueberry":             {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 10.5, "density": 1.02, "category": "acid"},
    "cranberry":             {"abv_pct": 0.0, "ta_pct": 2.0,  "brix": 8.5,  "density": 1.04, "category": "acid"},
    "black currant":         {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 13.5, "density": 1.05, "category": "acid"},
    "maqui berry":           {"abv_pct": 0.0, "ta_pct": 1.8,  "brix": 10.0, "density": 1.03, "category": "acid"},
    "mulberry":              {"abv_pct": 0.0, "ta_pct": 1.6,  "brix": 12.0, "density": 1.02, "category": "acid"},
    "roselle/hibiscus":      {"abv_pct": 0.0, "ta_pct": 2.4,  "brix": 5.5,  "density": 1.01, "category": "acid"},
    "wolfberry/goji":        {"abv_pct": 0.0, "ta_pct": 2.1,  "brix": 14.5, "density": 1.04, "category": "acid"},
    "green plum/ume":        {"abv_pct": 0.0, "ta_pct": 4.5,  "brix": 8.0,  "density": 1.05, "category": "acid"},
    "mango":                 {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 22.0, "density": 1.06, "category": "acid"},
    "mango juice":           {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 14.5, "density": 1.04, "category": "acid"},
    "lychee":                {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 18.0, "density": 1.05, "category": "acid"},
    "fig":                   {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 17.5, "density": 1.06, "category": "acid"},
    "sweet cherry":          {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 18.0, "density": 1.03, "category": "acid"},
    "plum":                  {"abv_pct": 0.0, "ta_pct": 1.0,  "brix": 16.0, "density": 1.02, "category": "acid"},
    "apricot":               {"abv_pct": 0.0, "ta_pct": 0.9,  "brix": 14.0, "density": 1.02, "category": "acid"},
    "white peach":           {"abv_pct": 0.0, "ta_pct": 0.7,  "brix": 12.5, "density": 1.01, "category": "acid"},
    "fuji apple":            {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 16.5, "density": 1.05, "category": "acid"},
    "green apple":           {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 14.0, "density": 1.04, "category": "acid"},
    "asian pear":            {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 12.5, "density": 1.04, "category": "acid"},
    "pineapple":             {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 13.5, "density": 1.04, "category": "acid"},
    "watermelon":            {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 8.0,  "density": 1.02, "category": "acid"},
    "tomato juice":          {"abv_pct": 0.0, "ta_pct": 0.85, "brix": 4.2,  "density": 1.03, "category": "acid"},
    "beet juice":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 7.7,  "density": 1.05, "category": "dilutant"},
    "carrot juice":          {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 6.5,  "density": 1.04, "category": "dilutant"},
    "watermelon juice":      {"abv_pct": 0.0, "ta_pct": 0.15, "brix": 5.8,  "density": 1.02, "category": "dilutant"},

    # ── More dilutants ──
    "tonic water":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 8.0,  "density": 1.03, "category": "dilutant"},
    "ginger beer":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 10.0, "density": 1.04, "category": "dilutant"},
    "kombucha":              {"abv_pct": 0.005,"ta_pct": 0.5, "brix": 3.0,  "density": 1.01, "category": "dilutant"},
    "oat milk":              {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 4.5,  "density": 1.03, "category": "modifier"},
    "almond milk":           {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 2.5,  "density": 1.02, "category": "modifier"},

    # ── Herbs, botanicals, spices (lookup props for Stage 0.6 awareness) ──
    "peppermint":            {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 0.5,  "density": 0.98, "category": "modifier"},
    "spearmint":             {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 0.5,  "density": 0.98, "category": "modifier"},
    "rosemary":              {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 0.6,  "density": 0.99, "category": "modifier"},
    "sweet basil":           {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 0.5,  "density": 0.97, "category": "modifier"},
    "kaffir lime leaf":      {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 0.4,  "density": 0.98, "category": "modifier"},
    "pandan":                {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.0,  "density": 0.99, "category": "modifier"},
    "lemongrass":            {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 1.5,  "density": 0.96, "category": "modifier"},
    "lemon verbena":         {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 1.8,  "density": 0.97, "category": "modifier"},
    "tarragon":              {"abv_pct": 0.0, "ta_pct": 0.7,  "brix": 2.3,  "density": 0.98, "category": "modifier"},
    "hyssop":                {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 2.5,  "density": 0.99, "category": "modifier"},
    "chamomile":             {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 2.0,  "density": 0.97, "category": "modifier"},
    "elderflower":           {"abv_pct": 0.0, "ta_pct": 2.0,  "brix": 14.0, "density": 1.04, "category": "modifier"},
    "lavender":              {"abv_pct": 0.0, "ta_pct": 3.2,  "brix": 10.0, "density": 1.01, "category": "modifier"},
    "red rose":              {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 0.3,  "density": 0.97, "category": "modifier"},
    "osmanthus":             {"abv_pct": 0.0, "ta_pct": 2.5,  "brix": 15.0, "density": 1.05, "category": "modifier"},
    "jasmine":               {"abv_pct": 0.0, "ta_pct": 1.8,  "brix": 12.0, "density": 1.02, "category": "modifier"},
    "butterfly pea flower":  {"abv_pct": 0.0, "ta_pct": 0.9,  "brix": 2.8,  "density": 0.99, "category": "modifier"},
    "ginger":                {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 5.0,  "density": 0.95, "category": "modifier"},
    "green cardamom":        {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 2.0,  "density": 0.92, "category": "modifier"},
    "cinnamon":              {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 0.0,  "density": 1.00, "category": "modifier"},
    "clove":                 {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 0.0,  "density": 1.01, "category": "modifier"},
    "star anise":            {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 0.0,  "density": 1.00, "category": "modifier"},
    "vanilla bean":          {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 0.0,  "density": 1.01, "category": "modifier"},
    "tonka bean":            {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 0.0,  "density": 1.00, "category": "modifier"},
    "nutmeg":                {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 12.0, "density": 0.90, "category": "modifier"},
    "saffron":               {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 0.0,  "density": 0.98, "category": "modifier"},
    "black pepper":          {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.0,  "density": 0.98, "category": "modifier"},
    "fennel seed":           {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 3.0,  "density": 0.94, "category": "modifier"},
    "coriander seed":        {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 2.5,  "density": 0.93, "category": "modifier"},
    "galangal":              {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 6.0,  "density": 0.96, "category": "modifier"},
    "aloe vera":             {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 6.0,  "density": 1.02, "category": "modifier"},
    "cucumber":              {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 2.5,  "density": 0.96, "category": "modifier"},

    # ── Teas ──
    "lapsang souchong":      {"abv_pct": 0.0, "ta_pct": 4.5,  "brix": 8.0,  "density": 0.99, "category": "modifier"},
    "assam":                 {"abv_pct": 0.0, "ta_pct": 3.8,  "brix": 9.0,  "density": 0.98, "category": "modifier"},
    "longjing green tea":    {"abv_pct": 0.0, "ta_pct": 2.2,  "brix": 7.0,  "density": 1.00, "category": "modifier"},
    "tieguanyin oolong":     {"abv_pct": 0.0, "ta_pct": 2.8,  "brix": 11.0, "density": 1.03, "category": "modifier"},
    "pu-erh":                {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 0.0,  "density": 1.01, "category": "modifier"},
    "rooibos":               {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 0.0,  "density": 1.00, "category": "modifier"},
    "yerba mate":            {"abv_pct": 0.0, "ta_pct": 1.5,  "brix": 0.0,  "density": 1.02, "category": "modifier"},

    # ── Molecular / texture agents ──
    "sodium alginate":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},
    "calcium chloride":      {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},
    "calcium lactate":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},
    "agar-agar":             {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.92, "category": "modifier"},
    "gelatin":               {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.35, "category": "modifier"},
    "xanthan gum":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.35, "category": "modifier"},
    "gum arabic":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.35, "category": "modifier"},
    "carrageenan":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},
    "lecithin":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},
    "glycerol":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.26, "category": "modifier"},
    "sorbitol":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},
    "steviol glycosides":    {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "sweetener"},
    "pectinex":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.91, "category": "modifier"},

    # ── Botanicals / bitters-adjacent ──
    "juniper berry":         {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 0.0,  "density": 0.95, "category": "modifier"},
    "grains of paradise":    {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 0.0,  "density": 0.92, "category": "modifier"},
    "cinchona bark":         {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 0.0,  "density": 0.88, "category": "modifier"},
    "gentian root":          {"abv_pct": 0.0, "ta_pct": 0.9,  "brix": 0.0,  "density": 0.90, "category": "modifier"},
    "wormwood":              {"abv_pct": 0.0, "ta_pct": 0.7,  "brix": 0.0,  "density": 0.89, "category": "modifier"},
    "orris root":            {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 0.0,  "density": 0.93, "category": "modifier"},
    "angelica root":         {"abv_pct": 0.0, "ta_pct": 0.85, "brix": 0.0,  "density": 0.91, "category": "modifier"},
    "licorice root":         {"abv_pct": 0.0, "ta_pct": 1.8,  "brix": 25.0, "density": 1.05, "category": "modifier"},
    "hops":                  {"abv_pct": 0.0, "ta_pct": 1.5,  "brix": 3.5,  "density": 1.03, "category": "modifier"},
    "quassia wood":          {"abv_pct": 0.0, "ta_pct": 2.5,  "brix": 15.0, "density": 1.10, "category": "modifier"},

    # ── Salts / saline ──
    "sea salt":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.00, "category": "modifier"},
    "himalayan pink salt":   {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.00, "category": "modifier"},
    "20% saline solution":   {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.15, "category": "modifier"},

    # ── EFSA key aliases — 51 entries added for complete EFSA→PROPS coverage ──
    # Organic acids (used as powders or concentrated solutions in bartending)
    "acetic_acid":           {"abv_pct": 0.0, "ta_pct": 5.0,  "brix": 0.0,  "density": 1.006,"category": "acid"},      # 5% food-grade solution
    "ascorbic_acid":         {"abv_pct": 0.0, "ta_pct": 50.0, "brix": 0.0,  "density": 1.65, "category": "modifier"},  # E300; antioxidant powder
    "citric_acid":           {"abv_pct": 0.0, "ta_pct": 99.5, "brix": 0.0,  "density": 1.665,"category": "acid"},      # anhydrous powder
    "lactic_acid":           {"abv_pct": 0.0, "ta_pct": 80.0, "brix": 0.0,  "density": 1.209,"category": "acid"},      # 80% food-grade solution
    "malic_acid":            {"abv_pct": 0.0, "ta_pct": 99.0, "brix": 0.0,  "density": 1.609,"category": "acid"},      # L-malic acid powder
    "phosphoric_acid":       {"abv_pct": 0.0, "ta_pct": 75.0, "brix": 0.0,  "density": 1.574,"category": "acid"},      # E338; 75% food-grade; restricted
    "succinic_acid":         {"abv_pct": 0.0, "ta_pct": 99.0, "brix": 0.0,  "density": 1.572,"category": "acid"},      # crystalline powder; umami + sour
    "tartaric_acid":         {"abv_pct": 0.0, "ta_pct": 99.0, "brix": 0.0,  "density": 1.788,"category": "acid"},      # E334; wine-derived; ADI 30 mg/kg

    # Mineral salts / molecular agents
    "black_salt":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 2.10, "category": "modifier"},  # kala namak; sulfurous aroma
    "calcium_chloride":      {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 2.15, "category": "modifier"},  # E509; spherification setting bath
    "calcium_lactate":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.77, "category": "modifier"},  # E327; reverse spherification core
    "epsom_salt":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.68, "category": "modifier"},  # MgSO4·7H2O; mineral bitterness
    "himalayan_salt":        {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 2.16, "category": "modifier"},  # trace minerals; salt finishing
    "saline_20":             {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.148,"category": "modifier"},  # 20% NaCl solution; salinity dosing
    "sea_salt":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 2.16, "category": "modifier"},  # unrefined; trace mineral modifier
    "baking_soda":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 2.20, "category": "modifier"},  # NaHCO3 E500(i); pH buffer / leavening

    # Sugars & sweeteners
    "white_sugar":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 50.0, "density": 1.23, "category": "sweetener"}, # 1:1 simple syrup baseline
    "invert_sugar":          {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 80.0, "density": 1.39, "category": "sweetener"}, # inverted syrup; prevents crystallisation
    "glucose":               {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 75.0, "density": 1.32, "category": "sweetener"}, # glucose syrup (DE42); low-sweetness body
    "erythritol":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 40.0, "density": 1.45, "category": "sweetener"}, # E968; 70% solution; 0 kcal
    "maple_syrup":           {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 66.0, "density": 1.33, "category": "sweetener"}, # Grade A amber; 66 °Brix
    "muscovado":             {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 65.0, "density": 1.36, "category": "sweetener"}, # unrefined cane; molasses aroma
    "palm_sugar":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 72.0, "density": 1.35, "category": "sweetener"}, # coconut/arenga palm; caramel notes
    "steviol_glycosides":    {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 5.0,  "density": 1.05, "category": "sweetener"}, # E960; 300× sweet; use at ~0.1 g/L

    # Gelling / stabilising agents
    "gum_arabic":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.35, "category": "modifier"},  # E414; emulsifier / texture

    # Fruits
    "black_currant":         {"abv_pct": 0.0, "ta_pct": 2.5,  "brix": 12.5, "density": 1.06, "category": "acid"},      # fresh juice; high vitamin C
    "blood_orange":          {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 11.0, "density": 1.05, "category": "acid"},      # fresh juice; anthocyanin pigment
    "green_apple":           {"abv_pct": 0.0, "ta_pct": 0.7,  "brix": 11.0, "density": 1.04, "category": "acid"},      # Granny Smith juice; malic-dominant
    "passion_fruit":         {"abv_pct": 0.0, "ta_pct": 3.5,  "brix": 13.5, "density": 1.07, "category": "acid"},      # pulp/juice; high citric + malic
    "red_apple":             {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 12.5, "density": 1.04, "category": "acid"},      # sweet cultivar juice; lower acidity

    # Novel food / colour
    "butterfly_pea":         {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 1.5,  "density": 1.00, "category": "modifier"},  # Clitoria ternatea; pH-sensitive colour

    # Botanicals & herbs (values represent typical aqueous infusion)
    "aloe_vera":             {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 1.0,  "density": 1.005,"category": "modifier"},  # aloin-free gel; restricted
    "angelica_root":         {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.0,  "density": 0.95, "category": "modifier"},  # Angelica archangelica; musk/bitter
    "bay_leaf":              {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.5,  "density": 0.92, "category": "modifier"},  # Laurus nobilis; eucalyptol/eugenol
    "black_pepper":          {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.5,  "density": 1.01, "category": "modifier"},  # Piper nigrum; piperine heat
    "coriander_seed":        {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.5,  "density": 0.97, "category": "modifier"},  # linalool-dominant citrus/floral
    "fennel_seed":           {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.0,  "density": 0.95, "category": "modifier"},  # trans-anethole; anise character
    "grains_of_paradise":    {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.5,  "density": 0.97, "category": "modifier"},  # Aframomum melegueta; pepper/ginger
    "lemon_verbena":         {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 1.0,  "density": 0.98, "category": "modifier"},  # Aloysia citrodora; lemony aldehyde
    "licorice_root":         {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 5.0,  "density": 1.05, "category": "modifier"},  # glycyrrhizin; ADI 100 mg/day (restricted)
    "orris_root":            {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.5,  "density": 1.00, "category": "modifier"},  # Iris pallida; violet/woody fixative
    "star_anise":            {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 1.0,  "density": 1.00, "category": "modifier"},  # trans-anethole; shikimic acid source

    # Teas (brewed infusion values at standard 3 g/200 ml)
    "lapsang_souchong":      {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 3.5,  "density": 1.005,"category": "modifier"},  # pine-smoked black tea
    "puer":                  {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 3.0,  "density": 1.005,"category": "modifier"},  # post-fermented; earthy/umami
    "yerba_mate":            {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 4.0,  "density": 1.005,"category": "modifier"},  # caffeine + chlorogenic acids

    # ── Wild / foraged umami (thermolabile — heat treatment mandatory in recipe) ──
    "见手青":                 {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.5,  "density": 1.05, "category": "modifier"},  # Boletus spp.; MUST blanch ≥100°C
    "红见手青":               {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.5,  "density": 1.05, "category": "modifier"},
    "黄见手青":               {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.5,  "density": 1.05, "category": "modifier"},
    "jian shou qing":        {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.5,  "density": 1.05, "category": "modifier"},
    "boletus speciosus":     {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.5,  "density": 1.05, "category": "modifier"},
    "matsutake":             {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 2.0,  "density": 1.04, "category": "modifier"},
    "porcini":               {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 2.0,  "density": 1.04, "category": "modifier"},
    "morel":                 {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 2.0,  "density": 1.04, "category": "modifier"},

    # Restricted / banned (kept for completeness; BANNED_INGREDIENTS blocks pipeline use)
    "calamus":               {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.5,  "density": 0.96, "category": "modifier"},  # β-asarone; restricted EU 1334/2008
    "pennyroyal":            {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 0.5,  "density": 0.94, "category": "modifier"},  # pulegone; hepatotoxic; BANNED
    "sassafras":             {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.5,  "density": 1.00, "category": "modifier"},  # safrole; carcinogenic; BANNED
    "st_johns_wort":         {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 1.0,  "density": 1.00, "category": "modifier"},  # hypericin; CYP3A4 inducer; restricted
    "tonka_bean":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 1.0,  "density": 1.10, "category": "modifier"},  # coumarin; ADI 0.1 mg/kg; restricted
    "bitter_almond":         {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 8.0,  "density": 1.05, "category": "modifier"},  # HCN precursor; restricted EU 1334/2008

    # ── Additional liqueurs / amari (Liquid Codex Vol.III + 003-cocktail-rules module eleven) ──
    "sambuca":               {"abv_pct": 0.38, "ta_pct": 0.0,  "brix": 38.0, "density": 1.06, "category": "modifier"},
    "ouzo":                  {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 10.0, "density": 0.98, "category": "spirit"},
    "raki":                  {"abv_pct": 0.45, "ta_pct": 0.0,  "brix": 8.0,  "density": 0.97, "category": "spirit"},
    "jagermeister":          {"abv_pct": 0.35, "ta_pct": 0.3,  "brix": 30.0, "density": 1.04, "category": "modifier"},
    "amer picon":            {"abv_pct": 0.21, "ta_pct": 0.8,  "brix": 38.0, "density": 1.08, "category": "modifier"},
    "unicum":                {"abv_pct": 0.40, "ta_pct": 0.2,  "brix": 22.0, "density": 1.03, "category": "modifier"},
    "becherovka":            {"abv_pct": 0.38, "ta_pct": 0.2,  "brix": 20.0, "density": 1.02, "category": "modifier"},
    "ramazzotti":            {"abv_pct": 0.30, "ta_pct": 0.5,  "brix": 32.0, "density": 1.06, "category": "modifier"},
    "braulio":               {"abv_pct": 0.21, "ta_pct": 0.6,  "brix": 22.0, "density": 1.04, "category": "modifier"},
    "zucca":                 {"abv_pct": 0.30, "ta_pct": 0.4,  "brix": 28.0, "density": 1.05, "category": "modifier"},
    "nonino":                {"abv_pct": 0.35, "ta_pct": 0.3,  "brix": 26.0, "density": 1.04, "category": "modifier"},
    "limoncello":            {"abv_pct": 0.26, "ta_pct": 0.7,  "brix": 32.0, "density": 1.05, "category": "modifier"},
    "advocaat":              {"abv_pct": 0.15, "ta_pct": 0.2,  "brix": 28.0, "density": 1.08, "category": "modifier"},
    "sloe gin":              {"abv_pct": 0.27, "ta_pct": 0.8,  "brix": 38.0, "density": 1.05, "category": "modifier"},
    "dubonnet":              {"abv_pct": 0.19, "ta_pct": 0.6,  "brix": 16.0, "density": 1.04, "category": "modifier"},
    "byrrh":                 {"abv_pct": 0.18, "ta_pct": 0.5,  "brix": 18.0, "density": 1.03, "category": "modifier"},
    "lillet rose":           {"abv_pct": 0.17, "ta_pct": 0.4,  "brix": 10.0, "density": 1.02, "category": "modifier"},
    "madeira":               {"abv_pct": 0.19, "ta_pct": 0.6,  "brix": 8.0,  "density": 1.03, "category": "modifier"},
    "marsala":               {"abv_pct": 0.18, "ta_pct": 0.5,  "brix": 12.0, "density": 1.03, "category": "modifier"},
    "amaretto":              {"abv_pct": 0.24, "ta_pct": 0.2,  "brix": 42.0, "density": 1.06, "category": "modifier"},
    "noilly prat dry":       {"abv_pct": 0.18, "ta_pct": 0.5,  "brix": 4.0,  "density": 1.01, "category": "modifier"},
    "cocchi rosa":           {"abv_pct": 0.175,"ta_pct": 1.2,  "brix": 28.0, "density": 1.04, "category": "modifier"},
    "pineau des charentes":  {"abv_pct": 0.18, "ta_pct": 0.6,  "brix": 16.0, "density": 1.03, "category": "modifier"},

    # ── More spirits (cachaca, batavia arrack, apple brandy, ginjinha) ──
    "cachaca":               {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "batavia arrack":        {"abv_pct": 0.50, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.93, "category": "spirit"},
    "apple brandy":          {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.95, "category": "spirit"},
    "kirsch":                {"abv_pct": 0.43, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.94, "category": "spirit"},
    "ginjinha":              {"abv_pct": 0.20, "ta_pct": 0.6,  "brix": 30.0, "density": 1.06, "category": "spirit"},

    # ── More bitters varieties ──
    "chocolate bitters":     {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 12.0, "density": 1.02, "category": "bitters"},
    "mole bitters":          {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 14.0, "density": 1.03, "category": "bitters"},
    "grapefruit bitters":    {"abv_pct": 0.40, "ta_pct": 0.2,  "brix": 16.0, "density": 1.02, "category": "bitters"},
    "lavender bitters":      {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 14.0, "density": 1.02, "category": "bitters"},
    "celery bitters":        {"abv_pct": 0.40, "ta_pct": 0.0,  "brix": 10.0, "density": 1.01, "category": "bitters"},
    "absinthe bitters":      {"abv_pct": 0.68, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.91, "category": "bitters"},
    "rhubarb bitters":       {"abv_pct": 0.40, "ta_pct": 0.3,  "brix": 18.0, "density": 1.03, "category": "bitters"},

    # ── Fermentation / cultured ingredients (Liquid Codex Vol.III — module fifteen) ──
    "shrub":                 {"abv_pct": 0.0, "ta_pct": 3.5,  "brix": 25.0, "density": 1.06, "category": "acid"},      # drinking vinegar; 1:1:1 fruit:sugar:vinegar
    "apple cider vinegar":   {"abv_pct": 0.0, "ta_pct": 5.0,  "brix": 0.5,  "density": 1.01, "category": "acid"},
    "rice vinegar":          {"abv_pct": 0.0, "ta_pct": 4.5,  "brix": 0.5,  "density": 1.01, "category": "acid"},
    "balsamic vinegar":      {"abv_pct": 0.0, "ta_pct": 6.0,  "brix": 30.0, "density": 1.12, "category": "acid"},
    "sherry vinegar":        {"abv_pct": 0.0, "ta_pct": 7.0,  "brix": 2.0,  "density": 1.02, "category": "acid"},
    "lacto-fermented juice": {"abv_pct": 0.005,"ta_pct": 3.5,  "brix": 8.0,  "density": 1.03, "category": "acid"},
    "kefir":                 {"abv_pct": 0.005,"ta_pct": 0.8,  "brix": 3.5,  "density": 1.02, "category": "modifier"},
    "yogurt":                {"abv_pct": 0.0, "ta_pct": 0.7,  "brix": 4.5,  "density": 1.03, "category": "modifier"},
    "creme fraiche":         {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 3.0,  "density": 1.02, "category": "modifier"},
    "buttermilk":            {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 4.0,  "density": 1.02, "category": "modifier"},
    "aquafaba":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 1.0,  "density": 1.01, "category": "modifier"},  # chickpea brine; vegan egg-white sub
    "koji kin":              {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 3.0,  "density": 1.00, "category": "modifier"},
    "koji rice":             {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 5.0,  "density": 1.02, "category": "modifier"},
    "tempeh":                {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 1.0,  "density": 1.01, "category": "modifier"},
    "miso":                  {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 18.0, "density": 1.08, "category": "modifier"},
    "gochujang":             {"abv_pct": 0.0, "ta_pct": 1.5,  "brix": 25.0, "density": 1.12, "category": "modifier"},

    # ── Molecular / texture agents — expanded (Liquid Codex Vol.V — module seventeen) ──
    "isomalt":               {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.50, "category": "sweetener"},  # sugar work; 160-170°C
    "methylcellulose":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.03, "category": "modifier"},  # E461; thermal gelling; hot-set foam
    "gellan gum":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},  # E418; fluid gel; thermo-reversible
    "locust bean gum":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.35, "category": "modifier"},  # E410; synergistic with carrageenan
    "guar gum":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.35, "category": "modifier"},  # E412; cold-soluble thickener
    "konjac gum":            {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},  # E425; glucomannan; high viscosity
    "tara gum":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.35, "category": "modifier"},  # E417; stabiliser
    "tapioca maltodextrin":  {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 5.0,  "density": 0.45, "category": "modifier"},  # oil-to-powder conversion; N-Zorbit
    "maltodextrin":          {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 10.0, "density": 0.50, "category": "modifier"},
    "dextrose":              {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 70.0, "density": 1.35, "category": "sweetener"},  # glucose monohydrate
    "pectin":                {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.02, "category": "modifier"},  # E440; fruit pectin; thermo-reversible gel

    # ── More vegetables / savoury ingredients ──
    "celery":                {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 2.5,  "density": 0.95, "category": "modifier"},
    "celery juice":          {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 2.0,  "density": 1.01, "category": "dilutant"},
    "bell pepper":           {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 5.0,  "density": 0.96, "category": "modifier"},
    "jalapeno":              {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 4.0,  "density": 0.95, "category": "modifier"},
    "habanero":              {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 5.0,  "density": 0.94, "category": "modifier"},
    "serrano":               {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 4.0,  "density": 0.95, "category": "modifier"},
    "thyme":                 {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 2.0,  "density": 0.96, "category": "modifier"},
    "sage":                  {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 2.5,  "density": 0.97, "category": "modifier"},
    "dill":                  {"abv_pct": 0.0, "ta_pct": 0.4,  "brix": 2.0,  "density": 0.95, "category": "modifier"},
    "cilantro":              {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 2.0,  "density": 0.95, "category": "modifier"},
    "shiso":                 {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 2.5,  "density": 0.96, "category": "modifier"},
    "turmeric":              {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 5.0,  "density": 0.94, "category": "modifier"},
    "horseradish":           {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 6.0,  "density": 0.95, "category": "modifier"},
    "wasabi":                {"abv_pct": 0.0, "ta_pct": 0.6,  "brix": 5.0,  "density": 0.96, "category": "modifier"},
    "truffle oil":           {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.0,  "density": 0.92, "category": "modifier"},
    "sesame oil":            {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.0,  "density": 0.92, "category": "modifier"},
    "olive oil":             {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 0.0,  "density": 0.92, "category": "modifier"},

    # ── More fruits (tropical / exotic) ──
    "guava":                 {"abv_pct": 0.0, "ta_pct": 1.0,  "brix": 12.0, "density": 1.04, "category": "acid"},
    "guava juice":           {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 11.0, "density": 1.04, "category": "acid"},
    "papaya":                {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 14.0, "density": 1.03, "category": "acid"},
    "dragon fruit":          {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 10.0, "density": 1.02, "category": "acid"},
    "durian":                {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 16.0, "density": 1.05, "category": "modifier"},
    "jackfruit":             {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 15.0, "density": 1.04, "category": "modifier"},
    "persimmon":             {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 18.0, "density": 1.05, "category": "modifier"},
    "quince":                {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 14.0, "density": 1.04, "category": "acid"},
    "rhubarb":               {"abv_pct": 0.0, "ta_pct": 1.5,  "brix": 4.0,  "density": 1.02, "category": "acid"},
    "tamarind":              {"abv_pct": 0.0, "ta_pct": 6.0,  "brix": 45.0, "density": 1.20, "category": "acid"},
    "pomegranate":           {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 14.0, "density": 1.05, "category": "acid"},
    "pomegranate juice":     {"abv_pct": 0.0, "ta_pct": 1.5,  "brix": 15.0, "density": 1.05, "category": "acid"},
    "cherry juice":          {"abv_pct": 0.0, "ta_pct": 1.0,  "brix": 14.0, "density": 1.04, "category": "acid"},
    "pear juice":            {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 13.0, "density": 1.04, "category": "acid"},
    "grape juice":           {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 16.0, "density": 1.05, "category": "acid"},
    "coconut syrup":         {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 60.0, "density": 1.24, "category": "sweetener"},
    "vanilla syrup":         {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 66.0, "density": 1.28, "category": "sweetener"},

    # ── Teas — expanded ──
    "genmaicha":             {"abv_pct": 0.0, "ta_pct": 2.0,  "brix": 6.0,  "density": 1.00, "category": "modifier"},
    "hojicha":               {"abv_pct": 0.0, "ta_pct": 1.8,  "brix": 5.0,  "density": 1.00, "category": "modifier"},
    "hibiscus tea":          {"abv_pct": 0.0, "ta_pct": 2.5,  "brix": 4.0,  "density": 1.01, "category": "modifier"},
    "chai":                  {"abv_pct": 0.0, "ta_pct": 1.0,  "brix": 8.0,  "density": 1.02, "category": "modifier"},

    # ── Dairy / alternatives — expanded ──
    "half and half":         {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 4.5,  "density": 1.02, "category": "modifier"},
    "soy milk":              {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 3.5,  "density": 1.02, "category": "modifier"},
    "coconut yogurt":        {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 6.0,  "density": 1.03, "category": "modifier"},
    "evaporated milk":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 12.0, "density": 1.06, "category": "modifier"},
    "condensed milk":        {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 60.0, "density": 1.30, "category": "sweetener"},

    # ── Frontier / safety-related (Liquid Codex Vol.VI — module eighteen) ──
    "liquid nitrogen":       {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 0.81, "category": "modifier"},  # -196°C; cryo-muddling; safety-critical
    "carbon dioxide":        {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 1.56, "category": "modifier"},  # CO2 from carbonation equipment

    # ── More powders / culinary bases ──
    "mushroom powder":       {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 4.0,  "density": 0.60, "category": "modifier"},
    "matcha powder":         {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 3.0,  "density": 0.55, "category": "modifier"},
    "cacao nibs":            {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 2.0,  "density": 0.55, "category": "modifier"},
    "coffee bean":           {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 1.0,  "density": 0.60, "category": "modifier"},
    "pandan leaf":           {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 1.5,  "density": 0.97, "category": "modifier"},
    "banana":                {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 18.0, "density": 1.04, "category": "acid"},
    "avocado":               {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 2.0,  "density": 1.02, "category": "modifier"},
    "honeycomb":             {"abv_pct": 0.0, "ta_pct": 0.2,  "brix": 75.0, "density": 1.35, "category": "sweetener"},
    "sorghum syrup":         {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 68.0, "density": 1.31, "category": "sweetener"},
    "molasses":              {"abv_pct": 0.0, "ta_pct": 1.0,  "brix": 70.0, "density": 1.35, "category": "sweetener"},
    "brown sugar syrup":     {"abv_pct": 0.0, "ta_pct": 0.1,  "brix": 66.0, "density": 1.30, "category": "sweetener"},
    "ginger juice":          {"abv_pct": 0.0, "ta_pct": 0.8,  "brix": 8.0,  "density": 1.02, "category": "modifier"},
    "cucumber juice":        {"abv_pct": 0.0, "ta_pct": 0.3,  "brix": 2.0,  "density": 1.00, "category": "dilutant"},
    "celery salt":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 2.00, "category": "modifier"},
    "smoked salt":           {"abv_pct": 0.0, "ta_pct": 0.0,  "brix": 0.0,  "density": 2.10, "category": "modifier"},
    "msg":                   {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 0.0,  "density": 1.60, "category": "modifier"},  # monosodium glutamate; umami
    "nutritional yeast":     {"abv_pct": 0.0, "ta_pct": 0.5,  "brix": 5.0,  "density": 0.45, "category": "modifier"},
    "soy sauce":             {"abv_pct": 0.0, "ta_pct": 1.2,  "brix": 10.0, "density": 1.06, "category": "modifier"},
    "fish sauce":            {"abv_pct": 0.0, "ta_pct": 1.0,  "brix": 5.0,  "density": 1.08, "category": "modifier"},
}


# Runtime in-memory cache for cold-start Perplexity lookups.
# Persists for process lifetime; reset on Railway restart.
_PROPS_RUNTIME_CACHE: dict[str, dict] = {}

# ── Persistent data directory (survives process restarts within same deployment) ──
_DATA_DIR = Path(__file__).parent.parent / "data"
_EXTRA_PROPS_PATH = _DATA_DIR / "ingredient_props_extra.json"
_AUDIT_LOG_PATH = _DATA_DIR / "error_audit.jsonl"
_WRITE_LOCK = threading.Lock()


def _load_extra_props() -> None:
    """Merge previously Perplexity-discovered ingredient props from disk into runtime cache."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if _EXTRA_PROPS_PATH.exists():
            with open(_EXTRA_PROPS_PATH, "r", encoding="utf-8") as f:
                extra: dict = json.load(f)
            _PROPS_RUNTIME_CACHE.update(extra)
    except Exception:
        pass


_load_extra_props()


def _persist_extra_props(name: str, props: dict) -> None:
    """Thread-safe write of a newly discovered ingredient to the extra-props JSON file."""
    with _WRITE_LOCK:
        try:
            existing: dict = {}
            if _EXTRA_PROPS_PATH.exists():
                with open(_EXTRA_PROPS_PATH, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing[name] = props
            with open(_EXTRA_PROPS_PATH, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


_COLD_START_SYSTEM = """\
You are a food-science expert. Return ONLY valid JSON — no markdown, no explanation.
abv_pct is decimal (40% = 0.40). ta_pct is total acidity % by weight (citric equiv).
brix is °Brix. density is g/ml at 20°C.
category: spirit | acid | sweetener | dilutant | modifier | bitters
"""


def _append_error_audit(ingredients: list[str], issues: list[str]) -> None:
    """Thread-safe append of a Stage-4 correction event to error_audit.jsonl.

    Each line is a JSON record: {ts, ingredients, issues}.
    Periodically review this file to promote recurring patterns into science_guardrails.
    """
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ingredients": ingredients,
            "issues": issues,
        }
        with _WRITE_LOCK:
            with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _lookup_ingredient_props(name: str) -> "dict | None":
    """
    Fuzzy-match ingredient physical properties.
    Exact → word-boundary substring → cache → None (fail-open).
    Uses word-boundary matching to prevent 'gin' hitting 'ginger beer', etc.
    """
    n = name.lower().strip()
    if n in INGREDIENT_PROPS:
        return INGREDIENT_PROPS[n]
    if n in _PROPS_RUNTIME_CACHE:
        return _PROPS_RUNTIME_CACHE[n]
    best, best_score = None, 0
    for key, props in INGREDIENT_PROPS.items():
        matched = False
        if key in n:
            # key is a substring of n — only accept as a complete word/phrase
            if re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", n):
                matched = True
        elif n in key:
            # n is a substring of key — only accept as a complete word/phrase
            if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", key):
                matched = True
        if matched:
            score = len(key)
            if score > best_score:
                best_score, best = score, props

    # Word-level fallback: try matching individual words
    if best is None:
        name_words = set(re.sub(r"[^a-z0-9\s]", "", n).split())
        for key, props in INGREDIENT_PROPS.items():
            key_words = set(re.sub(r"[^a-z0-9\s]", "", key).split())
            overlap = name_words & key_words
            if overlap:
                score = sum(len(w) for w in overlap)
                if score > best_score:
                    best_score, best = score, props

    return best


def _fetch_props_cold_start(name: str, perplexity_key: str) -> "dict | None":
    """
    Cold-start: query Perplexity sonar for an unknown ingredient's physical constants.
    Result cached in _PROPS_RUNTIME_CACHE for process lifetime (no filesystem write).
    Returns None on any failure (fail-open).
    """
    n = name.lower().strip()
    if n in _PROPS_RUNTIME_CACHE:
        return _PROPS_RUNTIME_CACHE[n]
    if not perplexity_key:
        return None
    try:
        client_pplx = OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai")
        prompt = (
            f'What are the standard physical constants for "{name}" used in bartending/food science?\n'
            "Return ONLY this JSON object (no markdown):\n"
            '{"abv_pct": <0.0-1.0>, "ta_pct": <float>, "brix": <float>, '
            '"density": <g/ml float>, "category": "spirit|acid|sweetener|dilutant|modifier|bitters"}'
        )
        resp = client_pplx.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": _COLD_START_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            stream=False, max_tokens=200, temperature=0.0, timeout=12,
        )
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            (resp.choices[0].message.content or "").strip(),
            flags=re.MULTILINE,
        ).strip()
        props = json.loads(raw)
        required = {"abv_pct", "ta_pct", "brix", "density", "category"}
        if not required.issubset(props.keys()):
            return None
        abv = float(props["abv_pct"])
        if not (0.0 <= abv <= 1.0):
            return None
        if props["category"] not in {"spirit", "acid", "sweetener", "dilutant", "modifier", "bitters"}:
            return None
        clean = {
            "abv_pct":  round(abv, 3),
            "ta_pct":   round(float(props["ta_pct"]), 2),
            "brix":     round(float(props["brix"]), 1),
            "density":  round(float(props["density"]), 3),
            "category": props["category"],
        }
        _PROPS_RUNTIME_CACHE[n] = clean
        # Persist to disk: survives process restarts within the same deployment.
        # Periodically commit backend/data/ingredient_props_extra.json to Git for full persistence.
        threading.Thread(target=_persist_extra_props, args=(n, clean), daemon=True).start()
        return clean
    except Exception:
        return None


def _estimate_ph(total_acid_g: float, total_vol_ml: float) -> float:
    """
    Citric-acid-equivalent concentration → estimated cocktail pH.

    Two-segment empirical model calibrated against measured sour cocktails:
      • High-acid zone (conc ≥ 8 g/L, e.g. Daiquiri/Sour with 20-25 ml lemon juice):
          pH ≈ 2.55 − 0.22 × log10(conc/8)   → range ~2.3–2.8
      • Low-acid zone  (conc < 8 g/L, e.g. light spritz or built drinks):
          pH ≈ 2.90 + 0.50 × log10(8/conc)   → range ~3.0–4.5

    Accuracy: ±0.2 pH vs. measured sour cocktails (better than prior ±0.3 power law).
    Not a true Henderson-Hasselbalch derivation; citrus juice contains multiple acids
    (citric/malic/ascorbic) and buffers that defeat a pure weak-acid calculation.
    """
    import math
    if total_vol_ml <= 0 or total_acid_g <= 0:
        return 4.2
    conc = total_acid_g / (total_vol_ml / 1000)  # g/L citric-acid equiv.
    if conc >= 8.0:
        ph = 2.55 - 0.22 * math.log10(conc / 8.0)
    else:
        ph = 2.90 + 0.50 * math.log10(8.0 / max(conc, 0.1))
    return round(max(2.3, min(5.5, ph)), 1)

VERIFY_SYSTEM = """\
You are a combined food-safety and culinary-technique expert for cocktail recipes.
You have built-in web-search capability. You will be given a cocktail recipe. Your task has TWO parts:

PART A — EU EFSA Safety:
1. Identify every ingredient and its stated quantity.
2. Cross-check against EU EFSA maximum limits (use training data AND web search).
3. "safety_verdict" is "FAIL" only if an ingredient clearly exceeds EU safety limits or is an outright banned substance.
   Missing quantities or novel-food notes = "PASS" with a warning in safety_issues.
   Thermolabile wild mushrooms (见手青 / Boletus spp., morels): NOT an automatic FAIL.
   These are ALLOWED when the recipe uses only pre-cooked extract/syrup/tincture with documented
   heat denaturation (≥100°C blanch/boil ≥5 min, or sous-vide ≥100°C × ≥10 min on mushroom
   material BEFORE spirit or beverage contact). FAIL only if raw, undercooked, or fresh garnish
   consumption is implied. Alcohol maceration of raw 见手青 without prior blanch = FAIL.
4. Raw animal products in beverages (CRITICAL): raw bone marrow, raw marrow fat, raw beef/pork tallow, or any
   unpasteurized animal fat MUST be fully rendered/pasteurized in a dedicated pre-step BEFORE contact with
   spirit or any drink component. Fat-washing at 55–65°C does NOT pasteurize meat/marrow — pathogens
   (E. coli, Salmonella, Listeria) survive and contaminate the final beverage. Missing pre-render/pasteurize
   step = safety FAIL. Acceptable: oven-roast marrow bones until core ≥74°C then harvest fat; use commercial
   pasteurized tallow; or sous-vide marrow ≥63°C × ≥15 min BEFORE fat-wash infusion.

PART B — Culinary / Technique Reasonableness:
4. Check that every technique is correctly and completely described:
   - Fat infusion (sous-vide / maceration with meat, bacon, butter, nuts, coconut): MUST include freeze at −18°C ≥2 h + skimming + straining. Missing = culinary FAIL.
   - Raw animal fat/marrow: bone marrow, beef tallow, or any raw meat fat MUST be rendered/pasteurized BEFORE spirit contact — fat-wash at 60°C alone is NOT pasteurization. Missing pre-cook/render step = culinary FAIL.
   - Fermentation: pH target < 4.6 within 72 h must be stated. Missing = culinary FAIL.
   - Sous-vide with protein: pasteurisation temperature and hold time must be stated. Missing = culinary FAIL.
   - Ingredient quantities must be internally consistent and proportionally logical for a cocktail (e.g. total volume, ABV, acid balance).
   - Steps must be in a logical physical sequence (cannot stir before combining, cannot serve before chilling, etc.).
   - Coffee filter / paper filter in the final shake-and-pour step is WRONG: if the recipe uses a coffee filter at the final serve step (after all ingredients are combined and shaken), that is a technique error. Coffee filter = pre-prep only (fat-washing, clarification). The correct final strainer is Hawthorne strainer + fine-mesh strainer.
   - Yeast autolysis: if the recipe heats kombucha, beer, or any live-yeast liquid above 50°C without a prior centrifuge or microfiltration step, that is a critical technique error.
   - Tannin over-extraction: cocoa/walnut/chestnut/tea in ≥40% ABV spirit at ≥60°C for >45 min without skin removal = technique error (excessive astringency).
   - CO₂ layering: if the recipe places a spirit layer (ABV ≥10%) BELOW a sparkling water layer and claims it is stable, that is a density physics error — spirit is lighter than water.
   - Data alignment: if the Ingredients table quantities differ from quantities calculated in the Balance or Method section, that is an error.
   - Quantity contradiction: if any ingredient entry lists two different quantities in the same cell (e.g., "25 g (约5克)"), that is an error.
   - Ouzo effect: if anethole-bearing ingredients (star anise, fennel, tarragon, pastis) are extracted in high-ABV spirit and the final drink ABV is <20%, the drink will turn milky — claiming clarity is an error.
   - Gelatin iSi: gelatin concentration for iSi siphon use at 4°C must not exceed 1.0% at 200 Bloom — higher values produce solid gel that jams the valve.
   - Proteolytic enzyme destruction: if the recipe combines raw papaya juice, raw pineapple juice, raw fig, or raw kiwi with gelatin OR dairy cream/milk/egg-white foam without a prior dedicated heat-inactivation step (≥85°C, ≥5 min on the fruit juice alone), the gelatin/foam WILL liquefy — this is a critical technique failure.
   - Spirit evaporation: heating high-ABV spirit (gin, vodka, whisky, rum, brandy, tequila, etc.) together with a liquid to ≥79°C causes ethanol boil-off (bp 78.37°C at 1 atm) — technique error. Spirit must be added only after the hot phase has cooled below 60°C.
   - Agar/gel freeze-thaw clarification: the thaw phase MUST occur at 4°C in a refrigerator (slow thaw 4–12 h). Room-temperature thaw (20–22°C / ambient) is a technique error — uncontrolled syneresis releases trapped solids.
   - Reverse spherification pH: if sodium alginate spherification uses apple juice or any liquid with pH <3.6, sodium citrate buffer to pH 4.0–4.5 is MANDATORY — otherwise alginic acid precipitation prevents gelation.
   - Reverse spherification alcohol: core/filling liquid ABV must be ≤20% (ideally 15–20%) before calcium lactate mixing. Using 55% Chartreuse or any ≥25% ABV spirit undiluted in the calcium-lactate core is a critical technique failure.
   - Spherification direction: reverse spherification = calcium lactate IN core + alginate EXTERNAL bath. Alginate IN core + CaCl₂ bath = direct spherification — mislabeling is a technique error.
   - Gelatin iSi concentration >1.0% (200 Bloom) or milk-clarification gelatin >0.5% causes solid gel / valve jam — technique error.
   - Bentonite hydration: bentonite slurry MUST be prepared in hot water (≥60°C); room-temperature hydration forms non-functional gravelly lumps.
5. "culinary_verdict" is "FAIL" only if there is a concrete procedural error or a mandatory step is missing.
   Style preferences or alternative techniques are NOT failures.

Return ONLY a JSON object — no markdown, no explanation outside the JSON:
{
  "safety_verdict": "PASS" or "FAIL",
  "safety_issues": ["<issue 1>", ...],
  "culinary_verdict": "PASS" or "FAIL",
  "culinary_issues": ["<issue 1>", ...]
}
"""

_CODEX_JUDGE_SYSTEM = """\
You are The Liquid Codex compliance judge. Your sole task: evaluate a generated cocktail recipe against the 8-volume Codex knowledge base below and return a structured JSON score.

{vol_context}

=== SCORING RULES ===
• Score only volumes whose topics APPEAR in this recipe. Volumes not touched by the recipe score 10 (N/A = no penalty).
• Scale: 0–3 = critical violation | 4–6 = has issues | 7–8 = good | 9–10 = excellent or N/A.
• violations: list only ACTUAL problems you see (not theoretical risks). Each entry must cite the Codex volume and specific rule.
  Example: "[Vol.III] Recipe calls for pouring carbonated water from height, causing immediate CO₂ loss."
• highlights: list genuine strengths (correct technique, good science, proper ratios).
• Keep violations concise: one sentence each, max 20 entries.

Return ONLY this JSON — no markdown fences, no prose:
{{
  "overall_score": <float 0.0–10.0>,
  "vol_scores": {{"vol1":<0-10>,"vol2":<0-10>,"vol3":<0-10>,"vol4":<0-10>,"vol5":<0-10>,"vol6":<0-10>,"vol7":<0-10>,"vol8":<0-10>}},
  "violations": ["..."],
  "highlights": ["..."]
}}
"""


def codex_compliance_judge(recipe_text: str, client: OpenAI) -> dict:
    """
    Stage 2.5: Evaluate recipe against all 8 Liquid Codex volumes.
    Uses deepseek-chat (fast, cheap). Returns dict with score + violations.
    Fail-open: any error returns a perfect score so the pipeline continues.
    """
    try:
        try:
            from routers.qa import VOL_CONTEXTS
        except ImportError:
            from qa import VOL_CONTEXTS
        vol_text = "\n\n".join(VOL_CONTEXTS.values())
    except Exception:
        return {"overall_score": 10.0, "vol_scores": {}, "violations": [], "highlights": []}

    system = _CODEX_JUDGE_SYSTEM.replace("{vol_context}", vol_text)
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Evaluate this recipe:\n\n{recipe_text[:4000]}"},
            ],
            max_tokens=900,
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```\s*$", "", raw)
        result = json.loads(raw)
        # Validate expected structure
        if "overall_score" not in result:
            result["overall_score"] = 10.0
        result.setdefault("violations", [])
        result.setdefault("highlights", [])
        result.setdefault("vol_scores", {})
        return result
    except Exception:
        return {"overall_score": 10.0, "vol_scores": {}, "violations": [], "highlights": []}


RECIPE_SURGEON_SYSTEM = """\
ABSOLUTE OUTPUT RULE: Your response MUST begin with `## ` on the very first character — the recipe title. NEVER write any preamble, greeting, "好的", "指挥官", "任务已解析", "液态法典推导", changelog, "here are the changes", or ANY text before `## `. Output the corrected recipe and nothing else.

You are the final editorial pass for a cocktail recipe. You receive a FLAWED recipe plus a numbered audit failure list and the user's original flavor intent. Output ONE COMPLETE replacement recipe that fixes EVERY audit issue while preserving the sensory concept, emotional story, and ingredient spirit of the original.

SURGERY RULES (non-negotiable):
1. Output the FULL recipe only — same Codex section order:
   ### Ingredients / ### Equipment / ### Chemical Alignment (if clarification, fat-wash, fermentation,
   spherification, or pH-critical technique) / ### Method / ### The Science / ### Architect's Note
2. Equipment bullets: ONE tool per line, ≤80 characters, purpose only — NEVER embed method steps or filters.
   VALID: "- Boston shaker — dry/wet shake" · "- Jigger — 45 ml / 22 ml measures" · "- Fine strainer — double-strain"
   INVALID: "- Step 1 — add 45 ml gin" · "- Pour 30 ml lemon into shaker and shake 12 s" · any numbered step in Equipment
3. Method steps: **Step N —** or **步骤 N —** sequential 1..N. Include temperature (°C), time (min/s), pH, concentration (%) where relevant. Do NOT write ingredient ml/g amounts in steps — the Python engine injects locked amounts automatically; refer to ingredients by name only.
4. LOCKED TABLES (when provided in user message): COPY the ### Ingredients table VERBATIM — do NOT
   change any ml amount or ingredient name. The ### Chemical Alignment table MAY be revised ONLY when
   an audit violation explicitly targets a chemical parameter (e.g. ABV too high, pH unsafe). If no
   audit item targets chemistry, reproduce Chemical Alignment unchanged. If audit cites an ingredient
   amount mismatch, restore the locked ml values from the pre-filled table.
5. Fix physics/biochemistry errors with correct technique — do not delete the flavor concept:
   - Ouzo/anethole: if star anise/fennel in high-ABV extract + final ABV ≤21% → raise ABV to ≥22% OR reduce
     anethole load OR shorten extraction — never claim guaranteed clarity at borderline ABV.
   - Fat wash: freeze −18°C ≥2 h → skim solid fat → strain (coffee filter OK for pre-prep only).
   - 见手青 / Boletus spp.: NEVER raw. Insert ≥100°C blanch ≥5 min BEFORE extract/spirit — do NOT delete the ingredient.
   - Raw marrow/meat fat: MUST add a pre-render/pasteurize step (e.g. roast marrow bones ≥74°C core, or
     sous-vide ≥63°C × ≥15 min) BEFORE any spirit fat-wash — 60°C infusion alone is unsafe.
   - Spirit heat: never co-heat ≥25% ABV spirit above 79°C.
6. Keep the Architect's Note poetic — it may acknowledge what was corrected scientifically.
7. NO preamble, NO changelog, NO "here are the changes" — only the finished recipe markdown.
"""

# Slim core for R1 — full safety enforced by hard_filter + science_guardrails
CODEX_GENERATION_CORE = """\
=== LOCKED NUMBERS & FORMAT (engine-enforced) ===
PRE-CALCULATED BALANCE (when present): Chemical Alignment MUST quote locked ABV, acid g, sugar g, pH, volume exactly.
Method: ingredient names only — no ml/g per ingredient in steps (times, °C, pH OK).
Ingredients table: one quantity per cell; must match locked balance.
Shaken serves: final strain = Hawthorne + fine-mesh (Liquid Intelligence / Dead Rabbit).
Per-serving: ≤0.45 g pure acid from powders/≥20% solutions; ≤3 drops 20% saline; fruit juice natural TA 1–2 g is normal.
Plain still water forbidden as ingredient — dilution from ice melt only (soda/tonic/coconut water OK as dilutant).
"""

INPUT_VALIDATE_SYSTEM = """\
You are an input validator for a cocktail recipe AI. The user fills in two fields:
1. Flavour concept / intention
2. Available ingredients / constraints

Determine if the combined input is a genuine, interpretable cocktail or beverage request.
Rules:
- VALID: any input in any language that could plausibly be interpreted as a flavour idea, ingredient, food item, occasion, or drink concept — even if vague or very short (e.g. "mint", "bourbon and chocolate", "夏日清爽", "我手边有一包牛肉干", "见手青").
- VALID: wild foraged culinary ingredients (见手青, truffles, wild mushrooms, regional fungi) even if toxic when raw — do NOT reject on food-safety grounds; the recipe pipeline enforces heat-treatment protocols.
- INVALID: random keyboard mashing (e.g. "asdfgh"), meaningless repeated characters, purely off-topic text (e.g. "fix my printer"), or inputs with zero culinary information.
- Be lenient — err strongly on the side of VALID.

Return ONLY a JSON object:
{"valid": true}
  OR
{"valid": false, "reason": "<one sentence in the same language as the input telling the user what to specify>"}
"""

CODEX_PRINCIPLES = """\
=== THE LIQUID CODEX — SCIENTIFIC CORE (8 Volumes) ===

OUTPUT RULE (ABSOLUTE — overrides everything): Your response MUST begin with `## ` on the very first character. NEVER write any opener, greeting, self-introduction, roleplay, task summary, or analysis block before the recipe title. Forbidden openers include but are not limited to: "好的", "指挥官", "我是", "任务已解析", "液态法典推导", "Here is", "I'll create", "Let me". The first two characters of your output must be `##`.

FUNCTION: This is the Darien Liquid Architect AI recipe engine, powered by The Liquid Codex — an 8-volume scientific treatise on bartending from classical kitchen techniques to extreme frontier physics. Task: translate flavor intent + available ingredients into a scientifically grounded, emotionally resonant drink recipe.

── EQUIPMENT LEVELS ──
• Home Kitchen: Vol.I-II only. Tools: saucepan, mason jars, fine strainer, knife.
• Enthusiast Workshop: Vol.I-IV. Adds: precision scale (0.1g), vacuum bags, sous-vide, pH strips, iSi siphon.
• Professional Lab: All 8 volumes. Centrifuge, refractometer, Rotovap, Pacojet, liquid nitrogen.

── VOL.I · MATERIALS ──
Citrus = "head note" + high-frequency acid. NEVER heat fresh citrus juice.
Berry cell walls are fragile; cold press or Pectinex enzyme clarification, NOT hot cooking.
Acid powders (citric/malic/tartaric/lactic/succinic) provide stable, oxidation-free acidity.
Salt rule: 20% saline drops suppress bitterness receptors, amplify sweetness/aroma.
Fixatives (orris root, angelica root) anchor light aromatic compounds.
Thermolabile wild mushrooms (见手青 Boletus spp., morels): contain heat-labile toxins when raw.
ALLOWED only as pre-cooked extract, syrup, or tincture. MANDATORY before any spirit/beverage use:
fresh material blanched in boiling water ≥100°C for ≥5 min (or sous-vide ≥100°C × ≥10 min).
FORBIDDEN: raw maceration, raw garnish, or serving uncooked slices. Alcohol alone does NOT denature toxins.

── VOL.II · CLASSICAL ERA ──
Cold Maceration: 40–50% ABV is ideal solvent.
Oleo-Saccharum: Sugar hygroscopy pulls citrus terpenes via osmotic pressure, 12–24h.
Fat-Washing: ANY spirit infused with a fatty ingredient (meat, bacon, butter, nuts, coconut cream, dairy fat, animal protein) MUST follow this exact sequence — (1) infuse at target temp, (2) cool to room temperature, (3) freeze at −18°C for ≥2 hours until fat solidifies, (4) skim or lift off solid fat layer, (5) strain through fine-mesh strainer lined with coffee filter or cheesecloth. OMITTING the freeze-filter step leaves turbid, oil-slick spirit and is a critical technique failure.
Raw animal fat pasteurization (CRITICAL — FOOD SAFETY): bone marrow, beef tallow, suet, raw bacon fat, or any unpasteurized meat-derived fat MUST be fully rendered/pasteurized BEFORE any spirit contact. Fat-washing at 55–65°C does NOT kill pathogens inside marrow or on raw fat surfaces — E. coli O157:H7, Salmonella, and Listeria can survive and contaminate the entire batch. MANDATORY pre-step (choose one): (a) roast marrow bones in oven at ≥180°C until marrow core reaches ≥74°C, harvest only the cooked/rendered fat; (b) sous-vide raw marrow/fat at ≥63°C for ≥15 minutes before rendering; (c) use commercially pasteurized/rendered tallow. FORBIDDEN: placing raw marrow fat directly into spirit for sous-vide fat-wash without prior cook/render.
Milk Clarification: ALWAYS pour spirit/acid INTO milk. Casein precipitates at pH 4.6. Gelatin for milk clarification: ≤0.5% w/w — higher concentrations set the entire batch into a solid jelly, not a pourable clarified spirit.
Bentonite fining (CRITICAL): bentonite clay platelets MUST be hydrated in hot water (≥60°C, ideally boiling water poured over powder with high-shear mixing). Room-temperature or cold hydration prevents layer exfoliation → gravelly lumps, zero adsorption capacity, and permanent earthy off-flavor in the finished drink.
Agar freeze-thaw clarification (CRITICAL): hydrate agar in NON-ALCOHOLIC liquid (juice, water, syrup) at 85–90°C; cool; freeze at −18°C ≥4 h; thaw SLOWLY in a refrigerator at 4°C for 4–12 h — NEVER at room temperature (20–22°C), which causes uncontrolled syneresis and re-turbidity.
Spirit heat limit (CRITICAL): ethanol boils at 78.37°C at 1 atm. NEVER heat undiluted or high-ABV spirit (≥25% ABV gin, vodka, whisky, rum, brandy, etc.) to ≥79°C in the same step — alcohol and volatile botanicals evaporate completely. Correct sequence: hydrate gums/clarifiers in juice or water → heat to 85–90°C if needed → cool below 60°C → THEN add spirit.

── VOL.III · ICE & KINETICS ──
Large ice sphere = minimal surface area = slow dilution → spirit-forward.
Shake = turbulent flow → rapid cooling + dilution + emulsification + aeration.
Stir = laminar flow → gentle dilution, NO aeration, preserves silky texture.
Double-strain (standard serve): Hawthorne strainer on shaker + fine-mesh strainer underneath. This is the ONLY correct straining method for the final shake-and-pour step.
Coffee filter = PRE-PREP TOOL ONLY (fat-washing, enzyme clarification, cold infusion). NEVER use a coffee filter at the final serve step — it blocks flow, strips aroma, and defeats the purpose of prior clarification. If earlier steps already clarified the liquid, final serve needs only Hawthorne + fine mesh.
Carbonated liquid density: soda water / sparkling water ≈ 0.998 g/ml; spirit at 15% ABV ≈ 0.978 g/ml; spirit at 40% ABV ≈ 0.942 g/ml. RULE: spirit is ALWAYS lighter than sparkling water. In any layered carbonated drink, the spirit layer naturally floats on top of soda water — NEVER below it. Claiming the reverse is a physics violation.
CO₂ pour height: ALWAYS pour carbonated water LOW (≤2 cm above liquid surface) and SLOW along the glass wall. Pouring from height (>5 cm) causes turbulent nucleation → immediate mass CO₂ release → flat drink in seconds. FORBIDDEN claim: "high-pour extends carbonation". More nucleation sites = FASTER CO₂ depletion, never slower.

── VOL.IV · LIVING ALCHEMY ──
Lacto-fermentation: 2% salt by total weight. pH MUST drop below 4.6 within 48–72h.
Koji (Aspergillus oryzae): 30°C/85% humidity, 48h. 60°C = enzyme optimum AND safety threshold.
SCOBY kombucha: sweet tea + SCOBY at 24–28°C for 7–14 days.
Yeast autolysis prevention (CRITICAL): Any live-yeast liquid (kombucha, beer, wine must, kefir, active yeast ferment) heated above 50°C WITHOUT prior yeast removal will undergo autolysis — yeast cell walls rupture, releasing hydrogen sulfide (H₂S) and mercaptans → rotten egg / sulfur / rubber odor that CANNOT be masked. MANDATORY before any heat treatment: remove yeast cells first by (a) centrifuge ≥4000 RPM for 10 min, OR (b) filter through ≤0.45 μm membrane. Only clarified, yeast-free liquid may be heat-pasteurised directly.

── VOL.V · CULINARY SCIENCE ──
Sous-vide: 55–65°C sweet spot — softens cell walls + releases aroma.
Tannin extraction time limit: High-tannin plant materials (cocoa nibs/husks, walnut skins, chestnut skin, tea leaves, grape seeds/skins) in high-ABV spirit (≥40%) at ≥60°C: HARD LIMIT 45 minutes. Beyond 45 min, short-chain tannin monomers (maximum astringency — sandpaper-on-tongue sensation) accumulate faster than they polymerize into gentler long-chain forms. CRITICAL ERROR: claiming "longer extraction reduces astringency" — it is the opposite. Mitigation if >45 min is needed: (a) remove tannin-bearing skin/husk/membrane before extraction, OR (b) reduce temperature to ≤50°C.
Proteolytic enzyme inactivation — CRITICAL: several common fruits contain powerful proteolytic enzymes that DESTROY gelatin networks and degrade dairy proteins (casein, whey):
  • Raw papaya → Papain: peak proteolytic activity 50–65°C. At this range papain cleaves gelatin peptide bonds within minutes — gelatin foam or jelly WILL liquify completely.
  • Raw pineapple → Bromelain: peak proteolytic activity 50–60°C. Same destruction mechanism.
  • Figs, kiwi, ginger (raw, high quantity), mango (unripe) also contain proteases with similar risk.
  MANDATORY inactivation rule: any recipe combining raw papaya juice / pineapple juice with gelatin, cream, milk proteins, or egg whites MUST include a dedicated prior step — heat the fruit juice ALONE to ≥85°C and hold for ≥5 minutes — BEFORE mixing with any protein or gelatin. This temperature irreversibly denatures the enzyme. Skipping this step produces a liquefied, unsettable foam / broken cream and is a critical technique failure.
  If the recipe heats the juice together WITH gelatin or cream, the enzyme is active during the heat window (50–65°C) before inactivation is reached — causing partial or total destruction. The heat-inactivation MUST be a separate, prior step on the juice alone.
iSi Rapid Infusion: N₂O pressurizes plant tissue; 2 min = equivalent of 24h cold maceration.
Carbonation: CO₂ most soluble near 0°C at 30–40 PSI.

── VOL.VI · MODERN LAB ──
Acid adjustment: citric = bright lemon; malic = green apple; tartaric = dry wine; lactic = dairy.
Pectinex Ultra SP-L: pectin lyase destroys fruit particle suspension → clarity.
Centrifuge 4000+ RPM: optically clear supernatant. Balance ±0.1g maximum.
Ouzo Effect / Spontaneous emulsification (CRITICAL): Hydrophobic aromatic compounds — primarily trans-anethole (star anise, fennel, pastis, absinthe), estragole, limonene, and most essential-oil terpenes — are fully soluble in high-ABV spirit (≥25% ABV) but precipitate as sub-micron droplets when diluted below ≈15–20% ABV, causing instant opaque white/milky turbidity (the "louche"). This phase transition is IRREVERSIBLE by filtration, clarification, or any post-dilution process because the emulsification happens after dilution. RULE: if a recipe extracts anethole-bearing ingredients (star anise, fennel seed, anise hyssop, tarragon, pastis) in high-ABV spirit then dilutes the final drink below 20% ABV, the drink WILL turn milky — claiming optical clarity is a physics violation. Solution: either (a) keep final ABV ≥20%, or (b) use rotovap distillation to isolate aroma without the heavy terpenes, or (c) explicitly design the louche as a deliberate aesthetic feature.

── VOL.VII · EXTREME PHYSICS ──
Rotovap: vacuum lowers boiling point to 20–30°C → captures heat-sensitive "head notes".
Pacojet: ultra-high-speed blade on frozen mass. Alcohol/sugar suppress freezing point.
Direct spherification: sodium alginate IN flavored liquid (0.5% w/w) + CaCl₂ bath (0.5% w/w) → thin membrane; limited to pH-neutral liquids.
Reverse spherification (CRITICAL): calcium lactate gluconate (0.5% w/w) mixed INTO the flavored core liquid; sodium alginate (0.5% w/w) in the EXTERNAL bath only. Dispense core with syringe/pipette into alginate bath → Ca²⁺/alginate membrane forms at interface; thicker membrane, longer shelf life (up to 24 h refrigerated).
Alginate pH window (CRITICAL): sodium alginate requires pH ≥3.6 to remain soluble and gel. Below pH 3.6, alginate protonates to insoluble alginic acid and coagulates — NO sphere forms. Apple juice (pH ~2.8, malic acid) REQUIRES sodium citrate buffer to raise core pH to 4.0–4.5 BEFORE mixing with calcium lactate. State citrate mass/volume, mixing sequence, and final measured pH in a chemical alignment block.
Alcohol in reverse-sphere core (CRITICAL): ethanol >20% ABV disrupts alginate-calcium ionic crosslinking — spheres dissolve on contact with bath or fail to gel ("melts on sight"). High-ABV spirits (e.g. 55% Green Chartreuse) MUST be diluted with juice/syrup/water so the calcium-lactate CORE liquid is 15–20% ABV max. Calculate and state core ABV, Brix, and density before spherification.
Sphere density / osmosis: if sphere must rest on glass bottom, core density must exceed serving liquid density — adjust syrup Brix and dilution so sphere sinks; state osmotic/density rationale.
Gelatin in iSi siphon — flowability hard limits (CRITICAL): gelatin must remain pourable at refrigerator temperature (4°C) to pass through the iSi valve. Maximum safe concentrations at 4°C: 200 Bloom ≤ 1.0%; 150 Bloom ≤ 1.3%; 100 Bloom ≤ 1.8%. Exceeding these limits produces a solid set gel that jams the siphon valve — pressurised gas cannot displace solid gelatin and the valve blocks. FORBIDDEN: gelatin concentration >1.0% for 200 Bloom in any iSi application. If recipe specifies higher concentration, reduce it or replace with a thixotropic agent (xanthan, low-acyl gellan) that retains flowability under pressure.

── VOL.VIII · GARNISH & SENSORY ──
Citrus twist: spray essential oil over glass rim = "olfactory handshake".
Dehydration: 50–60°C, 4–8h → flavor concentration + geometric visual beauty.
Isomalt: moisture-resistant sugar glass for structural garnish.

── QUANTITATIVE MANDATE (NON-NEGOTIABLE) ──
Every Method step MUST contain at least one specific numeric value where relevant.
Acceptable: temperature (°C), time (min/h), pH, concentration (%), ratio, RPM.
INGREDIENT AMOUNTS: Do NOT write ingredient ml/g amounts in Method steps — use ingredient names only
("Add the bourbon", not "Add 52 ml bourbon"). The Python engine injects exact locked amounts.
FORBIDDEN phrasings: "heat until warm", "add a little", "season to taste", "approximately", "some".
If a step truly has no measurable quantity, combine it with an adjacent step that does.
Data alignment (NON-NEGOTIABLE): The Ingredients table is the single source of truth. If the Balance Calculation or any Method step revises a quantity, the Ingredients table MUST be updated to match BEFORE finalising output. FORBIDDEN: Ingredients table showing different values from what the Method or Balance section calculates.
Quantity consistency (NON-NEGOTIABLE): Each ingredient entry must state exactly ONE quantity. FORBIDDEN: contradictory quantities in the same cell (e.g., "25 g herb (约5克)" — pick one and state it precisely).

── SAFETY PRE-CHECK (evaluate silently before writing) ──
Run through each gate before generating the Method:
① Lacto-fermentation: final pH target < 4.6. If substrate natural pH > 5.5, state the exact salt % (≥ 2% w/w) and the 48–72h checkpoint. REJECT recipe if no acid control is specified.
② Liquid nitrogen: ALWAYS include the line "serve only after all visible fog has dissipated and vessel wall is no longer cold to the touch." NEVER suggest adding LN₂ directly to a closed container.
③ Sous-vide with protein (dairy/egg/meat): state hold time at pasteurisation temp (e.g., 63°C × 30 min for egg yolk; 72°C × 15 s for dairy).
③-b Fat infusion (ANY technique — sous-vide, maceration, warm infusion — using meat, bacon, butter, nuts, coconut cream, or any ingredient containing visible fat): MANDATORY freeze-filter sequence is REQUIRED before the spirit is used in a cocktail: cool → freeze at −18°C ≥2 h → remove solidified fat layer → strain through coffee filter or cheesecloth. If this sequence is absent from the Method, the recipe is INCOMPLETE — insert it as a dedicated numbered step. Skipping this step produces turbid, oily spirit and is a critical error.
③-b2 Raw animal fat pasteurization (FOOD SAFETY — evaluate BEFORE fat-wash): if recipe uses bone marrow, marrow fat, beef/pork tallow, suet, or raw bacon fat — a dedicated pre-render/pasteurize step is MANDATORY before any spirit contact. Acceptable: oven-roast marrow ≥74°C core; sous-vide marrow ≥63°C × ≥15 min; commercial pasteurized tallow. FORBIDDEN: raw marrow fat placed directly into spirit at 55–65°C for fat-wash without prior cook — this does NOT achieve pasteurization.
③-c Live fermented liquids (kombucha, beer, kefir, active wine must, any liquid with visible yeast or SCOBY residue): before heat treatment above 50°C, a yeast-removal step is MANDATORY — centrifuge ≥4000 RPM 10 min OR filter through ≤0.45 μm membrane. If this step is absent and the recipe heats the unfiltered live liquid, append a ⚠ Autolysis Warning and insert the removal step before heating.
③-d High-tannin extraction: if recipe uses cocoa nibs/husks, walnut/chestnut skins, tea leaves, or grape seeds/skins in ≥40% ABV spirit at ≥60°C, the infusion time MUST be ≤45 min. If the recipe exceeds this without prior skin/membrane removal, flag it as a technique error and add the time cap or the pre-treatment step.
③-e Ouzo effect check: if recipe extracts from star anise, fennel, tarragon, anise hyssop, pastis, or any anethole-bearing ingredient in ≥25% ABV spirit AND the final drink ABV is <20%: the drink will spontaneously emulsify milky-white (Ouzo effect). Claiming the drink will be "crystal clear" or "optically transparent" at <20% ABV with anethole present is a physics violation — correct the claim or redesign.
③-f Gelatin iSi check: if recipe uses gelatin in an iSi siphon at ≥200 Bloom, the concentration MUST be ≤1.0%. At ≥150 Bloom, ≤1.3%. Exceeding these values at 4°C produces a solid gel that blocks the valve — a critical equipment failure.
③-g Proteolytic enzyme + protein/gelatin coexistence check: if the recipe uses raw papaya juice, raw pineapple juice, raw fig juice, or raw kiwi juice AND the recipe also contains gelatin, heavy cream, milk, egg white, or any protein-based foam agent — a dedicated heat-inactivation step for the fruit juice (≥85°C, ≥5 minutes, juice ALONE, before mixing) is MANDATORY. If this step is absent, the proteolytic enzymes will destroy the gelatin or cream protein structure and the foam/jelly will completely liquefy. Insert the inactivation step and add a ⚠ Enzyme Note explaining the hazard.
③-h Spirit heat check: if any Method step heats gin, vodka, whisky, rum, brandy, tequila, or other ≥25% ABV spirit in the same vessel to ≥79°C, the recipe is WRONG — insert a corrected sequence: hydrate clarifiers in juice/water only → heat → cool below 60°C → then blend in spirit. Never co-heat spirit and gum/clarifier slurry above 78°C.
③-i Agar freeze-thaw thaw check: if the recipe uses agar freeze-thaw clarification, the thaw step MUST state 4°C refrigerator thaw (4–12 h). If thaw is described at room temperature (20–22°C) or ambient, correct it to 4°C slow thaw before finalising.
③-j Alginate pH check: if recipe uses sodium alginate for spherification AND core liquid includes apple juice, citrus juice, or any stated pH <3.6 — sodium citrate buffer to final core pH 4.0–4.5 is MANDATORY before calcium lactate addition. If absent, insert buffering step with calculated citrate dose and pH measurement.
③-k Reverse spherification ABV check: if recipe uses reverse spherification with calcium lactate in core AND includes ≥25% ABV spirit (e.g. 55% Chartreuse) — core liquid MUST be diluted to 15–20% ABV before lactate mixing. Calculate dilution volumes; FORBIDDEN: undiluted high-proof spirit in calcium-lactate core.
③-l Spherification direction check: reverse = calcium lactate IN core + sodium alginate in EXTERNAL bath only. FORBIDDEN: labeling as "reverse" when alginate is mixed into the core and CaCl₂ is the bath — that is direct spherification and will fail with acidic/high-ABV cores.
③-m Bentonite hydration check: if recipe uses bentonite, slurry preparation MUST use hot water ≥60°C with shear mixing. Room-temperature bentonite slurry is a critical technique error.
③-n Chemical Alignment section: any recipe using spherification, clarification (milk/gelatin/agar/bentonite), fermentation, or isoelectric precipitation MUST include a ### 化学参数对齐 / ### Chemical Alignment table with pH, ABV, Brix, and/or density targets before Method steps.
③-o Pre-treatment mandatory ingredients — if any of these appear WITHOUT documented treatment, INSERT the required step BEFORE extract/spirit contact; do NOT remove the ingredient:
   • Wild mushrooms (见手青, 牛肝菌, boletus, porcini, morel, 羊肚菌 etc.): blanch ≥100°C for ≥5 min; discard liquid.
   • Raw elderberries (接骨木浆果, sambucus berries): simmer ≥15 min; strain solids (cyanogenic glycosides).
   • Raw kidney/runner/green beans (生芸豆, 四季豆, 豆角): full rolling boil ≥100°C for ≥10 min (low heat activates PHA).
⑦ Data alignment check: scan the Ingredients table and the Balance Calculation / Method steps for the same ingredient. If any quantity differs between table and calculation, the table value is WRONG — correct it to match the calculated value.
④ Centrifuge: state RPM and balance tolerance ±0.1 g. If user equipment is "Home Kitchen" or "Enthusiast Workshop", OMIT centrifuge steps entirely.
⑤ Pressure carbonation > 60 PSI: mark as "Professional lab — rated pressure vessel required" and include burst-pressure spec.
⑥ Final serve straining: if the Method ends with shaking or stirring into a glass, the straining tool MUST be "Hawthorne strainer + fine-mesh strainer" (double-strain). A coffee filter at this step is FORBIDDEN — coffee filters are pre-prep only. If you wrote "coffee filter" in the final pour step, remove it and replace with fine-mesh strainer.
⑥ Unknown industrial chemicals or unlicensed reagents: REFUSE and explain why.
   Documented foraged food ingredients (见手青, truffles, wild fungi, regional mushrooms) with proper
   heat protocols are ALLOWED — do NOT refuse the concept; enforce preparation steps instead.
If ANY gate fails, append a ⚠ Safety Note section explaining the hazard and the corrected parameter before the Architect's Note.

── SAFETY ABSOLUTES ──
Never serve still-vaporizing liquid nitrogen.
Botulism prevention: fermented/sous-vide products must reach pH < 4.6.
Centrifuge balance ±0.1g max.

── BALANCE MANDATE (NON-NEGOTIABLE) ──
Every cocktail recipe MUST satisfy all three balance gates simultaneously:
① pH gate: final drink pH 3.0–4.5. If pH calculation is uncertain, specify acid adjustments to achieve this range.
② Sweet-acid ratio: total sweetener Brix-equivalent 1.2× to 2.0× total acid TA% in the pour. State the ratio.
③ ABV gate: final beverage ABV ≤ 22% (home/enthusiast), ≤ 28% (bar/professional). Calculate and state final ABV.
If any gate cannot be satisfied with the given ingredients, state the constraint violation explicitly and propose a correction.

── EU EFSA SAFETY MANDATE ──
Before finalizing any recipe, cross-check every ingredient against EU EFSA constraints:
• RESTRICTED ingredients: enforce maximum levels — thujone ≤35 mg/kg (bitters), quinine ≤100 mg/L (tonic), coumarin ≤2 mg/kg (beverages), tartaric acid ADI 30 mg/kg bw/day, phosphoric acid ADI 70 mg/kg bw/day total phosphorus.
• Tonka bean: verify coumarin in final product ≤2 mg/kg; state calculated coumarin contribution.
• Cinnamon (Cassia): prefer Ceylon; if using Cassia state coumarin contribution ≤2 mg/kg.
• Elderflower: flowers only — explicitly exclude berries/leaves/bark.
• Novel food (finger lime, wolfberry, maqui): may be used; note EU novel food status.
• If any ingredient exceeds EU limits: REFUSE to include it at that dose and propose a compliant alternative.

── REALISM MANDATE (NON-NEGOTIABLE) ──
Every recipe MUST be immediately executable by the stated equipment level.
FORBIDDEN: recipes that list equipment not available at the stated level.

If Equipment level = "Home Kitchen":
  • Tools allowed: chef's knife, saucepan, mason jars, fine-mesh strainer, cocktail shaker, jigger, mixing glass, bar spoon, muddler, basic digital scale (1g precision), refrigerator/freezer.
  • Ingredients MUST be purchasable at a supermarket, Asian grocery, or mainstream online retailer. No laboratory reagents, no specialty chemical suppliers.
  • Techniques allowed: cold maceration, oleo-saccharum, fat-washing, simple syrups, basic carbonation with soda syphon, hand-juicing, muddling, shaking, stirring.
  • FORBIDDEN at Home: centrifuge, rotovap, iSi siphon, sous-vide circulator, Pacojet, liquid nitrogen, pH meter, refractometer, acid powder titration.
  • Single-serve format: base spirit 45–60 ml, total beverage volume 90–150 ml, served in standard glassware (rocks glass, coupe, highball).

If Equipment level = "Bar / Professional":
  • Tools allowed: all Home tools PLUS precision scale (0.1g), sous-vide circulator, iSi whipping siphon, pH strips or digital pH pen, refractometer, commercial carbonation equipment, dehydrator (50–60°C), vacuum sealer.
  • Ingredients: may include specialty syrups, acid powders (citric, malic, tartaric), enzyme preparations (Pectinex Ultra SP-L), food-grade additives available from bar-supply or food-science retailers.
  • Techniques: all Home techniques PLUS sous-vide infusion, iSi rapid infusion, enzyme clarification, acid adjustment, foaming agents (lecithin/xanthan), dehydration, basic and reverse spherification (sodium alginate + calcium lactate).
  • FORBIDDEN at Bar: rotovap, Pacojet, liquid nitrogen (unless venue has certified LN₂ protocol), high-speed centrifuge.
  • Batch-or-single-serve format: clearly state if the recipe is a batch (yields 6–10 servings) or single-serve; include per-serve volume.

ALWAYS specify:
① Exact gram or ml quantities for every ingredient — no "a splash", "to taste", or "some".
② Commercially available product names or clear substitutes (e.g., "Fever-Tree tonic" or "any dry tonic water").
③ Any ingredient that may require advance preparation time (e.g., "oleo-saccharum: prepare 12 h ahead").
④ Realistic shelf life for any batch component (e.g., "refrigerate; use within 5 days").

── CHEMICAL ALIGNMENT TRUTH MANDATE (NON-NEGOTIABLE) ──
If a PRE-CALCULATED BALANCE block is present in this system prompt, the Chemical Alignment table MUST
directly quote those locked values as authoritative.
FORBIDDEN: calculating your own ABV, TA, Brix, or pH independently when a PRE-CALCULATED BALANCE exists.
FORBIDDEN: showing different ABV, TA, or Brix values in the Chemical Alignment table vs. the PRE-CALCULATED BALANCE block.
Brand ABV must use the exact bottling strength (e.g., Ardbeg 10y = 46%, not 40%; Fernet-Branca = 39%).
FORBIDDEN: silently rounding or reassigning a named spirit's ABV to simplify your calculation.

── PER-SERVING SAFETY HARD LIMITS (ABSOLUTE — violations cause Stage 2 rejection) ──
① Concentrated/crystalline acid safety limit (applies ONLY to acid powders and ≥20% acid solutions):
   NEVER exceed 0.45 g pure anhydrous acid per serving from powders or high-concentration solutions (≥20%).
   Calculation: 10 ml × 20% citric acid solution = 2.0 g — OVER LIMIT, causes gastric injury.
   Safe dose: ≤2 ml of 20% acid solution (0.4 g).
   IMPORTANT — THIS LIMIT DOES NOT APPLY TO FRESH JUICE: fresh lemon/lime juice at natural TA
   (5–6% w/v) is safe up to 30 ml per serve. 30 ml × 5.5% TA = 1.65 g total acid — normal and
   desirable. The Chemical Alignment "Total acid" figure reflects natural juice TA and will typically
   read 1.2–1.8 g in classic sour-style recipes — do NOT apply the 0.45 g limit to that number.
② Sodium chloride (salt) load: NEVER exceed 0.12 g NaCl per serving.
   20% saline solution: maximum 3 drops (~0.15 ml total) as a finishing agent.
   FORBIDDEN: using 20% saline solution as a volume-filling dilutant (e.g., 20 ml saline = 4 g NaCl — lethal dose for a cocktail).
   Replace any volume deficit with soda water, tonic water, or another flavored/carbonated dilutant. NEVER use plain still water — dilution in cocktails comes from ice melt (already accounted for in the balance engine).
③ Insoluble powder (charcoal, activated carbon, bamboo charcoal, squid ink powder) MUST be measured in grams (g), never ml.
   Maximum per serving: 0.05 g. Exceeding this produces gritty sandpaper texture.
   Charcoal bulk density ≈ 0.3–0.5 g/ml: 4 ml ≈ 1.5 g = 30× the safe limit.

── SOLID ABSORPTION MANDATE ──
When recipe uses porous solids (bread, toast, crackers, dried botanicals ≥3 g per serve) in direct-contact
spirit fat-wash or infusion:
① Do NOT specify per-serve ingredient quantities for the infusion step — specify batch-scale ratios
   (e.g., 750 ml spirit + 100 g sourdough bread, yields approximately 600 ml after losses).
② Acknowledge absorption: sourdough bread absorbs ~3–5 ml liquid per gram dry weight.
   8 g bread in 54 ml spirit → expect only ~14–22 ml yield — state this in the recipe.
③ FORBIDDEN: stating "(preserving losses)" or "约50 ml" after a porous solid infusion without
   specifying the actual expected yield accounting for absorption.

── HIGH-VELOCITY SERVICE MANDATE ──
If recipe notes or scene context indicate high-volume service (晚高峰, rush hour, high-velocity,
pre-batch, volume bar, ≥40 covers/hour,爆款), ALL the following actions are FORBIDDEN at service time:
• Fresh juicing / pressing of any ingredient per order
• Current-order grinding, grating, or cold-pressing of solids
• Per-serve gravity filtering, double-straining, or any filtration step that takes >5 seconds
• Sous-vide or iSi extraction initiated per order
REQUIRED: every complex component must be assigned to a labeled "Batch Prep / 日间预制" stage
with batch yield noted (e.g., "makes 20 servings"). Service steps must total ≤30 seconds:
measure pre-batched components → shake/stir with ice → pour. State this explicitly in the recipe.
"""


def build_material_context() -> str:
    lines = [f"=== MATERIAL DATABASE ({len(MATERIALS)} entries, with EU EFSA safety data) ==="]
    for m in MATERIALS:
        tip = (m.get("tip") or "")[:60]
        eu = EFSA_DATA.get(m["id"], {})
        eu_status = eu.get("eu_status", "natural")
        eu_e = eu.get("eu_e_number") or ""
        eu_adi = eu.get("eu_adi") or ""
        eu_notes_short = (eu.get("eu_notes") or "")[:120]

        eu_str = f"EU:{eu_status}"
        if eu_e:
            eu_str += f"/{eu_e}"
        if eu_adi and eu_adi != "no_limit":
            eu_str += f"/ADI={eu_adi}mg/kg"
        if eu_notes_short:
            eu_str += f" | {eu_notes_short}"

        lines.append(
            f"• {m['name_en']} / {m['name']} | {m['category']} | "
            f"pH={m['ph']}, Brix={m['brix']} | [{eu_str}] | {tip}"
        )
    return "\n".join(lines)


def build_material_context_targeted(selected_names: list) -> str:
    """Inject EFSA data only for selected ingredients (vs all 125 entries)."""
    selected_lower = {n.lower() for n in selected_names}
    relevant = [
        m for m in MATERIALS
        if any(
            m["name_en"].lower() in n or n in m["name_en"].lower() or
            m["name"].lower() in n or n in m["name"].lower()
            for n in selected_lower
        )
    ]
    if not relevant:
        return "=== MATERIAL DATA: No specific EFSA restrictions identified for selected ingredients ==="
    lines = [f"=== RELEVANT MATERIAL DATA ({len(relevant)} entries, EU EFSA) ==="]
    for m in relevant:
        eu = EFSA_DATA.get(m["id"], {})
        eu_status = eu.get("eu_status", "natural")
        eu_notes  = (eu.get("eu_notes") or "")[:120]
        lines.append(
            f"• {m['name_en']} / {m['name']} | pH={m['ph']}, Brix={m['brix']} "
            f"| EU:{eu_status} | {eu_notes}"
        )
    return "\n".join(lines)


def build_system_prompt(
    language: str = "en",
    selected_names: "list | None" = None,
    balance: "dict | None" = None,
    equipment: str = "",
    spirit_free: bool = False,
) -> str:
    prefilled = balance is not None
    if language == "zh":
        if prefilled:
            ing_block = (
                "### 原料\n"
                "【⚠ 已预填 — 将系统提示 PRE-FILLED TABLES 中的原料表原文复制至此，禁止修改任何数量】"
            )
            chem_block = (
                "### 化学参数对齐\n"
                "【⚠ 已预填 — 将系统提示 PRE-FILLED TABLES 中的化学参数表原文复制至此，禁止修改任何数值】"
            )
        else:
            ing_block = (
                "### 原料\n"
                "| 用量 | 原料 | 角色 |\n"
                "|------|------|------|\n"
                "| [amount] | [ingredient] | [note] |"
            )
            chem_block = (
                "### 化学参数对齐（分子/澄清/发酵类配方必填 — 不可省略）\n"
                "| 参数 | 目标值 | 计算依据 |\n"
                "|------|--------|----------|\n"
                "| [pH / ABV / Brix / 密度等] | [数值] | [1 句依据] |"
            )
        output_format = f"""

=== OUTPUT FORMAT ===
Generate a complete drink recipe structured EXACTLY as below.
Section headers MUST use these Chinese titles exactly.

## [配方名] · [English Name]

> *[1–2 句诗意概念]*

---

{ing_block}

### 设备
- [设备名称] — [用途]
（每行一件，用短横线列表；禁止用表格）

{chem_block}

### 制作方法
**步骤 1 — [名称]:** [含具体数值的操作] · *[1 句科学原理]*
**步骤 2 — [名称]:** [含具体数值的操作] · *[科学原理]*
（继续 **步骤 3**、**步骤 4** … 必须连续编号，禁止用 "2." 普通列表代替步骤标题）

### 科学原理
[2–3 段 Codex 科学阐述]

### 建筑师笔记
*[诗意结语]*

---
*Liquid Architect · Darien's Codex Engine · The Liquid Codex*
"""
    else:
        if prefilled:
            ing_block = (
                "### Ingredients\n"
                "[⚠ PRE-FILLED — copy the ingredient table from the PRE-FILLED TABLES block "
                "in this system prompt VERBATIM here. Do NOT modify any amount.]"
            )
            chem_block = (
                "### Chemical Alignment\n"
                "[⚠ PRE-FILLED — copy the chemical parameters table from the PRE-FILLED TABLES "
                "block in this system prompt VERBATIM here. Do NOT modify any value. "
                "This section is ALWAYS required when balance is pre-computed.]"
            )
        else:
            ing_block = (
                "### Ingredients\n"
                "| Amount | Ingredient | Role |\n"
                "|--------|-----------|------|\n"
                "| [amount] | [ingredient] | [flavor/function note] |"
            )
            chem_block = (
                "### Chemical Alignment (REQUIRED for spherification / clarification / fermentation)\n"
                "| Parameter | Target | Rationale |\n"
                "|-----------|--------|-----------|\n"
                "| [pH / ABV / Brix / density] | [value] | [1-sentence basis] |"
            )
        output_format = f"""

=== OUTPUT FORMAT ===
Generate a complete drink recipe structured EXACTLY as below.

## [Recipe Name] · [English Name]

> *[1–2 sentence poetic concept — the "why" and emotional essence]*

---

{ing_block}

### Equipment
- [Equipment name] — [purpose]
(one item per line, bullet list only; NO table format)

{chem_block}

### Method
**Step 1 — [Name]:** [Action using ingredient NAME only — no ml amounts; include °C / pH / time where relevant] · *[1 sentence Codex science rationale]*
**Step 2 — [Name]:** [Action with technique parameter (°C / time / pH)] · *[rationale]*
(Continue **Step 3**, **Step 4** … sequential numbering required; NEVER use plain "2." lists instead of **Step N —** headers)

### The Science
[2–3 paragraphs connecting this recipe to specific Codex principles.]

### Architect's Note
*[Short, poetic closing thought about what makes this drink special]*

---
*Liquid Architect · Darien's Codex Engine · The Liquid Codex*
"""
    material_ctx = (
        build_material_context_targeted(selected_names)
        if selected_names
        else build_material_context()
    )
    principles = build_generation_principles(language, equipment, spirit_free)
    return principles + "\n\n" + CODEX_GENERATION_CORE + "\n\n" + material_ctx + output_format


# ── Pipeline helpers ───────────────────────────────────────────────────────

_SECTION_INGREDIENTS = re.compile(
    r"^#{2,4}\s*\*{0,2}(?:(?:Recipe\s+|Cocktail\s+)?Ingredients|原料|配料|成分|配方原料)\*{0,2}\s*[：:]?",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_METHOD = re.compile(
    r"^#{2,4}\s*\*{0,2}(?:Method|How\s+to\s+(?:Make|Prepare)|Preparation|Instructions|"
    r"制作方法|方法|制作(?:步骤|方法)?|步骤|制作流程|调制方法)\*{0,2}\s*[：:]?",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_SCIENCE = re.compile(
    r"^#{2,4}\s*\*{0,2}(?:The Science|科学(?:原理|解析)?|原理)\*{0,2}\s*[：:]?",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_CHEM_ALIGNMENT = re.compile(
    r"^#{2,4}\s*\*{0,2}(?:Chemical Alignment|化学参数对齐)\*{0,2}\s*[：:]?",
    re.IGNORECASE | re.MULTILINE,
)
_STEP_MARKERS = re.compile(r"\*\*(?:Step|步骤)\s*(\d+)", re.IGNORECASE)
_STEP_LINE = re.compile(r"^\s*\*\*(?:Step|步骤)\s+\d+", re.IGNORECASE)


def _find_section(pattern: re.Pattern, text: str) -> int:
    m = pattern.search(text)
    return m.start() if m else -1


_KNOWN_SECTION_RE = re.compile(
    r"^#{2,4}\s*\*{0,2}(?:"
    r"(?:Recipe\s+|Cocktail\s+)?Ingredients|原料|配料|成分|配方原料|"
    r"Equipment|设备|"
    r"Method|How\s+to\s+(?:Make|Prepare)|Preparation|Instructions|"
    r"制作方法|方法|步骤|制作流程|调制方法|"
    r"The Science|科学原理|科学|原理|"
    r"Chemical Alignment|化学参数对齐|"
    r"Architect|建筑师"
    r")\*{0,2}\s*[：:]?",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_recipe_text(text: str) -> str:
    """Strip fences/preamble and normalize section headers before structural filter."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:markdown)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    # Drop R1 preamble before the recipe title (## level) if present
    title_m = re.search(r"^##\s+\S", text, re.MULTILINE)
    if title_m and title_m.start() > 0:
        text = text[title_m.start():]
    elif not title_m:
        # No ## title found — strip everything before the first known section header
        section_m = _KNOWN_SECTION_RE.search(text)
        if section_m and section_m.start() > 0:
            text = text[section_m.start():]
    # #### 原料 → ### 原料 ; ### **原料** → ### 原料
    text = re.sub(r"^#{4,}\s*", "### ", text, flags=re.MULTILINE)
    text = re.sub(
        r"^#{2,3}\s*\*\*(Ingredients|原料|配料|成分|Method|制作方法|方法|步骤|"
        r"Equipment|设备|The Science|科学原理|科学|原理|Chemical Alignment|化学参数对齐)\*\*",
        r"### \1",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return text.strip()


_EQ_SECTION = re.compile(
    r"^#{2,3}\s*(?:Equipment|设备)\b.*?\n(.*?)(?=^#{2,3}|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_SPIRIT_RE = re.compile(
    r"\b(gin|vodka|rum|whisky|whiskey|bourbon|tequila|brandy|cognac|mezcal)\b|"
    r"(伦敦干金酒|金酒|伏特加|威士忌|朗姆酒?|干邑|白兰地|龙舌兰|梅斯卡尔)",
    re.IGNORECASE,
)
_HEAT_RE = re.compile(
    r"加热|隔水|升温|煮沸|炖煮|heat(?:ed|ing)?|simmer|boil",
    re.IGNORECASE,
)
_HIGH_TEMP_RE = re.compile(
    r"(?:^|[^\d.])(7[89]|[89]\d|1[0-4]\d)\s*°?\s*C",
    re.IGNORECASE,
)
_AGAR_CTX_RE = re.compile(
    r"冻融|freeze[- ]?thaw|琼脂|agar[- ]?agar",
    re.IGNORECASE,
)
_THAW_RE = re.compile(r"解冻|thaw", re.IGNORECASE)
_ROOM_THAW_RE = re.compile(
    r"室温|room\s*temperature|room\s*temp|ambient(?:\s*temp)?|20\s*[–\-~～to至]+\s*22",
    re.IGNORECASE,
)
_COLD_THAW_RE = re.compile(r"4\s*°?\s*C|冷藏|冰箱|refriger", re.IGNORECASE)
_SPHERE_CTX_RE = re.compile(
    r"球化|spherif|海藻酸钠|sodium alginate|乳酸钙|calcium lactate",
    re.IGNORECASE,
)
_LOW_PH_JUICE_RE = re.compile(
    r"苹果汁|apple juice|pH\s*[：:≈~～]?\s*2\.[0-9]|pH\s*[：:≈~～]?\s*3\.[0-5]|极酸",
    re.IGNORECASE,
)
_CITRATE_BUFFER_RE = re.compile(r"柠檬酸钠|sodium citrate", re.IGNORECASE)
_SAFE_SPHERE_PH_RE = re.compile(
    r"pH\s*4\.[0-5]|4\.0\s*[–\-~～to至]+\s*4\.5|修正.*pH|缓冲.*pH|buffer.*pH",
    re.IGNORECASE,
)
_REVERSE_SPHERE_RE = re.compile(r"反向球化|reverse spherif", re.IGNORECASE)
_HIGH_ABV_CORE_RE = re.compile(
    r"55\s*%|chartreuse.*55|绿沙特.*55|"
    r"核心.*ABV.*2[1-9]|核心液.*ABV.*2[1-9]|"
    r"fill(?:ing|er).*ABV.*2[1-9]",
    re.IGNORECASE,
)
_DILUTION_MARK_RE = re.compile(
    r"稀释|dilut|core.*ABV.*1[0-9]\s*%|"
    r"核心.*ABV.*1[0-9]|ABV.*1[5-9]\s*%|ABV.*≤\s*20|"
    r"15\s*[–\-~～]\s*20\s*%",
    re.IGNORECASE,
)


def _procedure_blocks(text: str) -> str:
    """Equipment + Method body text — used for science guardrails (excludes Science section)."""
    parts: list[str] = []
    eq_match = _EQ_SECTION.search(text)
    if eq_match:
        parts.append(eq_match.group(1))
    method_start = _find_section(_SECTION_METHOD, text)
    if method_start != -1:
        science_start = _find_section(_SECTION_SCIENCE, text)
        method_end = science_start if science_start != -1 else len(text)
        parts.append(text[method_start:method_end])
    return "\n".join(parts)


def _check_spirit_heat_violation(text: str) -> tuple[bool, str]:
    block = _procedure_blocks(text)
    if not block.strip():
        return True, "ok"
    for line in block.splitlines():
        if _SPIRIT_RE.search(line) and _HEAT_RE.search(line) and _HIGH_TEMP_RE.search(line):
            return False, (
                "Spirit heat violation: high-ABV spirit must not be heated to ≥79°C "
                "(ethanol bp 78.37°C). Hydrate clarifiers in juice/water first; "
                "add spirit only after the hot phase cools below 60°C."
            )
    return True, "ok"


def _check_agar_thaw_violation(text: str) -> tuple[bool, str]:
    if not _AGAR_CTX_RE.search(text):
        return True, "ok"
    block = _procedure_blocks(text)
    for line in block.splitlines():
        if not _THAW_RE.search(line):
            continue
        if _ROOM_THAW_RE.search(line) and not _COLD_THAW_RE.search(line):
            return False, (
                "Agar freeze-thaw violation: gel thaw must occur at 4°C in a refrigerator "
                "(slow thaw 4–12 h). Room-temperature thaw causes uncontrolled syneresis."
            )
    return True, "ok"


def _check_spherification_violation(text: str) -> tuple[bool, str]:
    if not _SPHERE_CTX_RE.search(text):
        return True, "ok"

    if re.search(r"海藻酸钠|sodium alginate", text, re.IGNORECASE):
        if _LOW_PH_JUICE_RE.search(text):
            if not _CITRATE_BUFFER_RE.search(text):
                return False, (
                    "Alginate acid precipitation: juice pH <3.6 converts alginate to "
                    "insoluble alginic acid. Sodium citrate buffer to pH 4.0–4.5 is MANDATORY."
                )
            if not _SAFE_SPHERE_PH_RE.search(text):
                return False, (
                    "Alginate spherification must state buffered core pH 4.0–4.5 "
                    "after sodium citrate addition."
                )

    is_reverse = _REVERSE_SPHERE_RE.search(text) or re.search(
        r"乳酸钙|calcium lactate", text, re.IGNORECASE
    )
    if is_reverse and re.search(r"chartreuse|绿沙特|55\s*%", text, re.IGNORECASE):
        block = _procedure_blocks(text)
        undiluted = re.search(
            r"55\s*%[^.\n]{0,100}(?:乳酸钙|calcium lactate|核心)|"
            r"绿沙特[^.\n]{0,60}(?:直接|未稀释|undiluted)|"
            r"(?:乳酸钙|calcium lactate)[^.\n]{0,100}55\s*%",
            block,
            re.IGNORECASE,
        )
        if undiluted and not _DILUTION_MARK_RE.search(text):
            return False, (
                "Reverse spherification alcohol swelling: core ABV must be diluted to "
                "15–20% max before calcium lactate mixing. 55% Chartreuse cannot be "
                "used undiluted in the core."
            )
        if _HIGH_ABV_CORE_RE.search(text) and not _DILUTION_MARK_RE.search(text):
            return False, (
                "Reverse spherification core ABV exceeds safe 15–20% window for "
                "alginate-calcium crosslinking — dilution calculation required."
            )

    return True, "ok"


_ANETHOLE_RE = re.compile(
    r"八角|star\s*anise|茴香|fennel|anethole|pastis|absinthe|tarragon|龙蒿",
    re.IGNORECASE,
)
_CLARITY_CLAIM_RE = re.compile(
    r"水晶|清澈|澄清透明|透明|optically clear|crystal\s*clear|transparent",
    re.IGNORECASE,
)
_FINAL_ABV_RE = re.compile(
    r"(?:最终|final)\s*ABV[^0-9]{0,20}(\d+\.?\d*)\s*%",
    re.IGNORECASE,
)
_LAB_TECHNIQUE_RE = re.compile(
    r"球化|澄清|发酵|下胶|等电点|spherif|clarif|ferment|bentonite|膨润土|"
    r"agar|琼脂|下胶|milk\s*clarif|奶澄清",
    re.IGNORECASE,
)
_CHEM_SECTION_RE = re.compile(r"化学参数对齐|Chemical Alignment", re.IGNORECASE)
_ISI_RE = re.compile(r"iSi|奶油枪|siphon|发泡器", re.IGNORECASE)
_GELATIN_CONC_RE = re.compile(
    r"(\d+\.?\d*)\s*%\s*(?:200\s*Bloom\s*)?(?:明胶|gelatin)|"
    r"(?:明胶|gelatin)[^.\n]{0,40}(\d+\.?\d*)\s*%",
    re.IGNORECASE,
)


def _check_ouzo_violation(text: str) -> tuple[bool, str]:
    if not _ANETHOLE_RE.search(text):
        return True, "ok"
    abv_vals: list[float] = []
    for m in _FINAL_ABV_RE.finditer(text):
        try:
            abv_vals.append(float(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r"ABV[^0-9|]{0,30}(\d+\.?\d*)\s*%", text, re.IGNORECASE):
        try:
            abv_vals.append(float(m.group(1)))
        except ValueError:
            pass
    for abv in abv_vals:
        if abv <= 21.0 and (
            _CLARITY_CLAIM_RE.search(text)
            or re.search(r"无乳白|不浑浊|不会触发|澄清度", text, re.IGNORECASE)
        ):
            return False, (
                f"Ouzo effect violation: anethole-bearing extraction with ABV {abv}% "
                f"≤21% cannot be guaranteed optically clear — raise final ABV to ≥22%, "
                "reduce star anise/anethole load, or redesign as intentional louche."
            )
    return True, "ok"


def _check_gelatin_rheology_violation(text: str) -> tuple[bool, str]:
    for m in _GELATIN_CONC_RE.finditer(text):
        pct = float(m.group(1) or m.group(2))
        if _ISI_RE.search(text) and pct > 1.0:
            return False, (
                f"Gelatin iSi rheology violation: {pct}% gelatin at 4°C jams the "
                "siphon valve — 200 Bloom maximum is 1.0%."
            )
        if re.search(r"milk\s*clarif|奶澄清|下胶|casein", text, re.IGNORECASE) and pct > 0.5:
            return False, (
                f"Milk clarification gelatin violation: {pct}% sets the batch into "
                "a solid jelly — maximum ~0.5% for pourable clarified spirit."
            )
    return True, "ok"


def _check_sphere_direction_violation(text: str) -> tuple[bool, str]:
    if not _REVERSE_SPHERE_RE.search(text):
        return True, "ok"
    block = _procedure_blocks(text)
    alginate_in_core = re.search(
        r"海藻酸钠.{0,50}(?:核心|内液|fill|core|风味液|内核)|"
        r"(?:核心|内液|内核|core|fill).{0,50}海藻酸钠|"
        r"sodium alginate.{0,40}(?:into|in|混入).{0,30}(?:core|filling)|"
        r"混入.{0,30}海藻酸钠",
        block,
        re.IGNORECASE,
    )
    cacl2_bath = re.search(
        r"氯化钙.{0,30}(?:浴|bath)|CaCl[₂2].{0,30}(?:浴|bath)",
        text,
        re.IGNORECASE,
    )
    if alginate_in_core and cacl2_bath:
        return False, (
            "Spherification direction error: alginate in core + CaCl₂ bath is "
            "DIRECT spherification, not reverse. Reverse requires calcium lactate "
            "IN core and sodium alginate in the EXTERNAL bath only."
        )
    return True, "ok"


def _check_bentonite_hydration_violation(text: str) -> tuple[bool, str]:
    if not re.search(r"bentonite|膨润土", text, re.IGNORECASE):
        return True, "ok"
    block = _procedure_blocks(text)
    hot_hydration = re.search(
        r"(?:≥|>=|>|沸水|boiling)\s*(?:60|70|80|90|100)\s*°?\s*C|"
        r"hot\s+water|热水|开水",
        block,
        re.IGNORECASE,
    )
    if not hot_hydration:
        return False, (
            "Bentonite hydration violation: bentonite MUST be slurried in hot water "
            "(≥60°C) with shear mixing. Room-temperature hydration forms gravelly "
            "lumps and earthy off-flavor."
        )
    return True, "ok"


def _check_chem_alignment_required(text: str) -> tuple[bool, str]:
    if _LAB_TECHNIQUE_RE.search(text) and not _CHEM_SECTION_RE.search(text):
        return False, (
            "Missing required Chemical Alignment section (### 化学参数对齐 / "
            "### Chemical Alignment) for spherification/clarification/fermentation recipe."
        )
    return True, "ok"


# ── Stage-2 locked-value validator ────────────────────────────────────────────

_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ml", re.IGNORECASE)
_ABV_IN_TABLE_RE = re.compile(
    r"(?:ABV|最终\s*ABV|Final\s*ABV)[^\d|]*\|?\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_ACID_IN_TABLE_RE = re.compile(
    r"(?:Total\s*acid|总酸量)[^\d|]*\|?\s*(\d+(?:\.\d+)?)\s*g",
    re.IGNORECASE,
)


def _check_locked_values(text: str, locked_balance: "dict | None") -> tuple[bool, str]:
    """
    Stage-2 guard: verify R1 reproduced the pre-filled ingredient amounts.

    Only checks ml-based ingredient amounts (within ±1 ml).
    ABV / total acid / sugar are intentionally NOT enforced here — Surgeon and
    Quality Gate are allowed to adjust chemical parameters freely.
    """
    if not locked_balance:
        return True, "ok"

    locked_ings = locked_balance.get("ingredients", [])
    numeric_ings = [
        i for i in locked_ings
        if re.search(r"\d+\s*ml", (i.get("amount") or ""), re.IGNORECASE)
    ]
    if numeric_ings:
        missing = []
        for ing in numeric_ings:
            amt_str = ing.get("amount", "")
            m = re.match(r"(\d+(?:\.\d+)?)", amt_str)
            if not m:
                continue
            locked_val = float(m.group(1))
            all_detected = [float(fm.group(1)) for fm in _AMOUNT_RE.finditer(text)]
            found_close = any(abs(v - locked_val) <= 1.0 for v in all_detected)
            if not found_close:
                # Include closest detected value so Surgeon knows the discrepancy
                closest_hint = ""
                if all_detected:
                    closest = min(all_detected, key=lambda v: abs(v - locked_val))
                    if abs(closest - locked_val) < 50:
                        closest_hint = f" — your output has {closest:.0f} ml instead"
                missing.append(
                    f"⚠️ LOCKED VALUE MISMATCH: expected {amt_str} {ing.get('name', '')}"
                    f"{closest_hint}. Restore from pre-filled table VERBATIM."
                )
        if missing:
            return False, " | ".join(missing)
    return True, "ok"


_GHOST_ML_RE = re.compile(
    r"(?:^|[\s(])(\d+(?:\.\d+)?)\s*ml\b(?:\s+of)?\s+"
    r"((?:[A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff\-]{1,20})"
    r"(?:\s+[A-Za-z\u4e00-\u9fff\-]{2,20})?)",
    re.IGNORECASE | re.MULTILINE,
)
_GHOST_STOPWORDS = frozenset([
    "the", "a", "an", "of", "all", "each", "this", "that", "fresh", "cold",
    "room", "ice", "chilled", "frozen", "hot", "warm", "into", "to", "from",
    "with", "and", "or", "in", "on", "at", "total", "final", "diluted",
    "mixture", "mix", "liquid", "solution", "output", "serving",
])


def _check_ingredient_ghost(text: str, locked_balance: "dict | None") -> tuple[bool, str]:
    """
    Gear 1 assertion: detect if R1 introduced measurable quantities (≥5 ml) of an
    ingredient not present in the locked table. Scans Method section only to avoid
    false positives from the Ingredients table itself.

    Does NOT fire when locked_balance is None (no locked table = no contract to violate).
    """
    if not locked_balance:
        return True, "ok"
    locked_names = [
        i.get("name", "").lower().strip()
        for i in locked_balance.get("ingredients", [])
        if i.get("name")
    ]
    if not locked_names:
        return True, "ok"

    # Scope to Method section only
    method_m = _SECTION_METHOD.search(text)
    if not method_m:
        return True, "ok"
    science_m = _SECTION_SCIENCE.search(text)
    end = (
        science_m.start()
        if science_m and science_m.start() > method_m.start()
        else len(text)
    )
    method_text = text[method_m.start(): end]

    ghosts: list[str] = []
    for m in _GHOST_ML_RE.finditer(method_text):
        vol = float(m.group(1))
        if vol < 5:
            continue  # ignore drops / dashes
        candidate = m.group(2).strip().lower()
        if candidate in _GHOST_STOPWORDS or len(candidate) < 3:
            continue
        first_word = candidate.split()[0]
        matched = any(
            re.search(r"(?<!\w)" + re.escape(first_word) + r"(?!\w)", lk)
            or re.search(r"(?<!\w)" + re.escape(lk.split()[0]) + r"(?!\w)", candidate)
            for lk in locked_names
        )
        if not matched:
            ghosts.append(f"{vol:.0f} ml {m.group(2).strip()}")

    if ghosts:
        locked_list = ", ".join(locked_names[:6])
        return False, (
            f"⚠️ GHOST INGREDIENT(S): {ghosts[:3]} appear in Method but are NOT in the locked "
            f"### Ingredients table [{locked_list}]. "
            "Do NOT introduce any ingredient or quantity not in the pre-filled table."
        )
    return True, "ok"


_RAW_ANIMAL_FAT_RE = re.compile(
    r"bone\s+marrow|marrow\s+fat|marrow\b|牛骨髓|骨髓|"
    r"beef\s+tallow|牛(?:油|脂)|tallow|suet|schmaltz|"
    r"bone\s+fat|raw\s+beef\s+fat|raw\s+pork\s+fat",
    re.IGNORECASE,
)
_FAT_WASH_CTX_RE = re.compile(
    r"fat.?wash|脂肪洗|fat\s+infusion|温浸|低温浸|慢煮浸|sous.?vide",
    re.IGNORECASE,
)
_PRE_RENDER_RE = re.compile(
    r"预熟|先烤|预烤|render|roast|rendered|pasteur|熟化|"
    r"180\s*°?\s*C|200\s*°?\s*C|烤箱|烘烤|炙烤|"
    r"≥?\s*63\s*°?\s*C[^.\n]{0,40}(?:15|≥15|\×\s*15)|"
    r"≥?\s*74\s*°?\s*C|74°C|core\s+temperature|"
    r"commercial(?:ly)?\s+pasteurized|pasteurized\s+tallow|"
    r"高纯度熟牛油|熟牛油|精炼牛脂|预.*熬.*油",
    re.IGNORECASE,
)


def _check_raw_animal_fat_violation(text: str) -> tuple[bool, str]:
    """Raw marrow/meat fat must be rendered/pasteurized before spirit fat-wash."""
    if not _RAW_ANIMAL_FAT_RE.search(text):
        return True, "ok"
    if not _FAT_WASH_CTX_RE.search(text):
        return True, "ok"
    if _PRE_RENDER_RE.search(text):
        return True, "ok"
    return False, (
        "Food safety violation: raw bone marrow or unpasteurized animal fat must be "
        "rendered/pasteurized BEFORE spirit fat-wash (e.g. roast marrow core ≥74°C, "
        "sous-vide ≥63°C × ≥15 min, or use pasteurized tallow). "
        "Fat-wash at 55–65°C does NOT pasteurize meat/marrow."
    )


def _check_thermolabile_mushroom_heat(text: str) -> tuple[bool, str]:
    """见手青 / Boletus spp. require ≥100°C pre-cook before beverage use."""
    if not _THERMOLABILE_MUSHROOM_RE.search(text):
        return True, "ok"
    if _MUSHROOM_HEAT_TREATMENT_RE.search(text):
        return True, "ok"
    _raw_mushroom = re.compile(r"生[^。\n]{0,8}见手青|raw[^.\n]{0,20}见手青|见手青[^。\n]{0,20}生", re.I)
    if _raw_mushroom.search(text):
        return False, (
            "见手青 must not be used raw. Insert a dedicated pre-cook step: "
            "blanch in boiling water ≥100°C for ≥5 min BEFORE extract, syrup, or spirit contact."
        )
    # Mentioned but no documented heat step — require surgeon to add protocol, not delete ingredient.
    return False, (
        "见手青 / thermolabile Boletus requires documented heat treatment "
        "(≥100°C blanch/boil ≥5 min on mushroom material before beverage use). "
        "Insert the pre-cook step — do not remove the ingredient."
    )


_METHOD_SECTION_RE = re.compile(
    r"(###\s*(?:制作方法|Method|Steps?)\s*\n)",
    re.IGNORECASE,
)


def _maybe_inject_pre_treatment_steps(text: str, language: str = "en") -> str:
    """
    For every pre-treatment-required ingredient found in the recipe without
    documented treatment, deterministically prepend numbered Step 0 blocks
    at the start of the Method section.  Applied as a final post-process so
    no filter can block output on these rules.
    """
    needed = [
        entry for entry in _PRE_TREATMENT_REGISTRY
        if entry["detect_re"].search(text) and not entry["treated_re"].search(text)
    ]
    if not needed:
        return text

    zh = language == "zh"
    multi = len(needed) > 1
    parts = []
    for i, entry in enumerate(needed):
        suffix = f" ({chr(ord('a') + i)})" if multi else ""
        if zh:
            parts.append(
                f"**{entry['label_zh']}{suffix}：** {entry['step_zh']}"
            )
        else:
            parts.append(
                f"**{entry['label_en']}{suffix}:** {entry['step_en']}"
            )

    inject = "\n" + "\n\n".join(parts) + "\n"
    if _METHOD_SECTION_RE.search(text):
        return _METHOD_SECTION_RE.sub(r"\1" + inject, text, count=1)
    return text


_CONC_ACID_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*ml[^,\n|]{0,20}?"
    r"(?:20%|30%|40%|50%|60%|70%|80%|90%|99%)[^,\n|]{0,30}?"
    r"(?:citric|malic|tartaric|lactic|succinic|acetic|phosphoric|柠檬酸|苹果酸|酒石酸|乳酸|琥珀酸|乙酸|磷酸)"
    r"|(?:20%|30%|40%|50%|60%|70%|80%|90%|99%)[^,\n|]{0,30}?"
    r"(?:citric|malic|tartaric|lactic|succinic|acetic|phosphoric|柠檬酸|苹果酸|酒石酸|乳酸|琥珀酸|乙酸|磷酸)"
    r"[^,\n|]{0,20}?(\d+(?:\.\d+)?)\s*ml)",
    re.IGNORECASE,
)
_HIGH_CONC_ACID_THRESHOLD_ML = 6.0  # >6 ml of ≥20% acid solution = >1.2 g acid — approaching danger zone

_SALINE_LARGE_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*ml[^,\n|]{0,20}?"
    r"(?:20%|25%|30%)[^,\n|]{0,20}?(?:saline|salt water|brine|盐水|食盐水)"
    r"|(?:20%|25%|30%)[^,\n|]{0,20}?(?:saline|salt water|brine|盐水|食盐水)"
    r"[^,\n|]{0,20}?(\d+(?:\.\d+)?)\s*ml)",
    re.IGNORECASE,
)
_SALINE_VOLUME_LIMIT_ML = 5.0  # >5 ml of 20% saline = >1 g NaCl — crosses palatable limit

_POWDER_ML_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*ml[^,\n|]{0,30}?"
    r"(?:charcoal|activated carbon|bamboo charcoal|squid ink powder|carbon black|"
    r"炭黑|活性炭|竹炭|植物炭|碳粉|黑粉)",
    re.IGNORECASE,
)


def _check_acid_overload(text: str) -> tuple[bool, str]:
    """Per-serving concentrated acid safety gate: >6 ml of ≥20% acid solution risks gastric injury."""
    for m in _CONC_ACID_RE.finditer(text):
        vol = float(m.group(1) or m.group(2) or 0)
        if vol > _HIGH_CONC_ACID_THRESHOLD_ML:
            return False, (
                f"Per-serving acid overload: {vol:.0f} ml of high-concentration acid solution "
                f"(≥20%) delivers >{vol*0.2:.1f} g pure acid — gastric injury risk. "
                f"Maximum safe dose: ≤6 ml at 20% concentration (1.2 g acid) or use fresh juice "
                f"at natural concentration (citrus ~6% TA). Reformulate: reduce concentration or volume."
            )
    return True, "ok"


def _check_salt_overload(text: str) -> tuple[bool, str]:
    """Per-serving saline safety gate: >5 ml of 20% NaCl solution crosses palatability/safety limit."""
    for m in _SALINE_LARGE_RE.finditer(text):
        vol = float(m.group(1) or m.group(2) or 0)
        if vol > _SALINE_VOLUME_LIMIT_ML:
            return False, (
                f"Per-serving sodium overload: {vol:.0f} ml of 20% saline solution "
                f"= {vol*0.20:.1f} g NaCl — far exceeds palatability and triggers nausea/vomiting reflex. "
                f"Maximum finishing dose: 2–3 drops (~0.1–0.15 ml at 20% = 0.02–0.03 g NaCl). "
                 f"Replace volume-filling use of saline with soda water or another appropriate flavored dilutant (never plain water — dilution comes from ice melt)."
            )
    return True, "ok"


def _check_powder_volume_unit(text: str) -> tuple[bool, str]:
    """Insoluble powder unit check: charcoal/carbon must be weighed in grams, never measured in ml."""
    m = _POWDER_ML_RE.search(text)
    if m:
        vol = float(m.group(1))
        return False, (
            f"Unit error: insoluble powder (charcoal/activated carbon) measured as {vol} ml — "
            "powders must be weighed in grams (g). "
            "Bamboo charcoal bulk density ≈ 0.3–0.5 g/ml: "
            f"{vol} ml ≈ {vol*0.4:.1f} g — far exceeds the 0.05 g per-serve palatable maximum. "
            "Correct: use ≤0.05 g weighed on a 0.01 g-precision scale."
        )
    return True, "ok"


# ── Physics Audit Triad (mass conservation · dilution · container geometry) ──
# Parses Chemical Alignment table rows in LLM output and compares against
# the locked_balance dict produced by the deterministic algebraic engine.

_CHEM_ALIGN_TOTAL_VOLUME_RE = re.compile(
    r"\|\s*(?:Total\s+volume|总体积|成杯体积|Volume)\s*\|\s*(\d+(?:\.\d+)?)\s*ml",
    re.IGNORECASE,
)
_CHEM_ALIGN_TOTAL_SUGAR_RE = re.compile(
    r"\|\s*(?:Total\s+sugar|总糖量|糖量|Sugar)\s*\|\s*(\d+(?:\.\d+)?)\s*g",
    re.IGNORECASE,
)

# Glass capacity table (ml): used by the container geometry audit
_GLASS_CAPACITIES: dict[str, int] = {
    "highball":              300,
    "collins":               300,
    "tall glass":            300,
    "rocks":                 200,
    "old fashioned":         200,
    "old-fashioned":         200,
    "lowball":               200,
    "double rocks":          250,
    "coupe":                 120,
    "martini":               120,
    "nick and nora":         100,
    "nick & nora":           100,
    "wine glass":            250,
    "white wine":            250,
    "red wine":              350,
    "champagne flute":       150,
    "flute":                 150,
    "shot glass":            60,
    "shot":                  60,
    "copper mug":            350,
    "moscow mule mug":       350,
    "tiki mug":              350,
    "高球杯":                300,
    "古典杯":                200,
    "碟形杯":                120,
    "平底杯":                200,
}

_GLASS_NAME_RE = re.compile(
    r"\b(highball|collins|tall\s+glass|rocks?(?:\s+glass)?|old[-\s]fashioned|"
    r"lowball|double\s+rocks?|coupe|martini|nick\s*(?:and|&)\s*nora|"
    r"wine\s+glass|white\s+wine|red\s+wine|champagne\s+flute|flute|"
    r"shot\s+glass|shot|copper\s+mug|moscow\s+mule\s+mug|tiki\s+mug|"
    r"高球杯|古典杯|碟形杯|平底杯)\b",
    re.IGNORECASE,
)

_GLASS_MIN_FILL = 0.60   # below this = commercial "half-cup" failure
_GLASS_MAX_FILL = 1.10   # above this = overflow risk
_SUGAR_TOLERANCE_RELATIVE = 0.60  # ±60% relative before we flag (LLM rounding is coarse)


def _check_sugar_mass_conservation(
    text: str, locked_balance: "dict | None"
) -> tuple[bool, str]:
    """
    Mass conservation audit (Audit #1).

    Brix × density × volume_ml / 100 = sugar_grams.
    If the LLM wrote a 'Total sugar' row in its Chemical Alignment table that
    deviates >60% from the deterministic engine's locked value, flag it.

    Example catch: engine says 9.2 g (15 ml × 1:1 syrup × 1.23 × 0.50),
    LLM writes "2.0 g" — 5× undercount.
    """
    if not locked_balance:
        return True, "ok"
    locked_sugar = locked_balance.get("total_sugar_g")
    if locked_sugar is None or locked_sugar <= 0:
        return True, "ok"
    m = _CHEM_ALIGN_TOTAL_SUGAR_RE.search(text)
    if not m:
        return True, "ok"  # section absent — only check when row is present
    text_sugar = float(m.group(1))
    ratio = abs(text_sugar - locked_sugar) / max(locked_sugar, 0.1)
    if ratio > _SUGAR_TOLERANCE_RELATIVE:
        return False, (
            f"Sugar mass conservation violation: Chemical Alignment states {text_sugar:.1f} g sugar "
            f"but deterministic engine (Brix × density × volume) calculates {locked_sugar:.1f} g "
            f"(deviation {ratio*100:.0f}%). Update 'Total sugar' to {locked_sugar:.1f} g."
        )
    return True, "ok"


def _check_dilution_volume_consistency(
    text: str, locked_balance: "dict | None"
) -> tuple[bool, str]:
    """
    Thermodynamics / ice-melt audit (Audit #2).

    Ice shaken 12–15 s adds ~22% water by mass (17% stirred / 25% blended).
    If the LLM's Chemical Alignment 'Total volume' row equals only the raw
    ingredient sum (no dilution water), the stated volume is physically wrong.

    Catches: 'shake vigorously 15 s → total volume 107 ml' when physics gives 130 ml.
    """
    if not locked_balance:
        return True, "ok"
    locked_vol = locked_balance.get("total_volume_ml")
    if locked_vol is None or locked_vol <= 0:
        return True, "ok"
    m = _CHEM_ALIGN_TOTAL_VOLUME_RE.search(text)
    if not m:
        return True, "ok"  # row absent — skip
    text_vol = float(m.group(1))
    # Recover dilution fraction from balance_notes (e.g. "dil=22%")
    notes = locked_balance.get("balance_notes", "")
    dil_match = re.search(r"dil=(\d+)%", notes)
    dil_pct = int(dil_match.group(1)) / 100 if dil_match else 0.20
    # If stated volume is <85% of the locked total the ice-melt water was simply omitted
    if text_vol < locked_vol * 0.85:
        pre_dilution_vol = locked_vol / (1 + dil_pct)
        return False, (
            f"Thermodynamic dilution inconsistency: Chemical Alignment states {text_vol:.0f} ml "
            f"(≈ pre-dilution ingredient sum {pre_dilution_vol:.0f} ml; "
            f"{dil_pct*100:.0f}% ice-melt water not counted). "
            f"Shaking/stirring adds {locked_vol - pre_dilution_vol:.0f} ml; "
            f"correct 'Total volume' is {locked_vol:.0f} ml."
        )
    return True, "ok"


def _check_glass_fill_ratio(
    text: str, locked_balance: "dict | None"
) -> tuple[bool, str]:
    """
    Container geometry audit (Audit #3).

    A cocktail must fill 60–110% of its stated glassware.
    Below 60% = commercial 'half-cup' failure (customer feels cheated).
    Above 110% = overflow / spillage on service.
    """
    if not locked_balance:
        return True, "ok"
    locked_vol = locked_balance.get("total_volume_ml")
    if locked_vol is None or locked_vol <= 0:
        return True, "ok"
    glass_m = _GLASS_NAME_RE.search(text)
    if not glass_m:
        return True, "ok"  # no glass name found — skip
    glass_raw = glass_m.group(0).lower().strip()
    cap: int | None = None
    for key, capacity in _GLASS_CAPACITIES.items():
        if key in glass_raw or glass_raw in key:
            cap = capacity
            break
    if cap is None:
        # Partial word match fallback
        for key, capacity in _GLASS_CAPACITIES.items():
            if any(w in glass_raw for w in key.split() if len(w) > 3):
                cap = capacity
                break
    if cap is None:
        return True, "ok"
    fill_ratio = locked_vol / cap
    if fill_ratio < _GLASS_MIN_FILL:
        return False, (
            f"Container geometry violation — 'half-cup' failure: "
            f"{locked_vol:.0f} ml in a {cap} ml {glass_raw} = {fill_ratio*100:.0f}% fill "
            f"(minimum commercial fill = 60%). Either use a smaller glass "
            f"(≤{int(locked_vol / 0.75)} ml) or increase recipe volume to "
            f"≥{int(cap * _GLASS_MIN_FILL)} ml."
        )
    if fill_ratio > _GLASS_MAX_FILL:
        return False, (
            f"Container geometry violation — overflow: "
            f"{locked_vol:.0f} ml exceeds {cap} ml {glass_raw} capacity "
            f"({fill_ratio*100:.0f}% fill, maximum = 110%). "
            f"Use a ≥{int(locked_vol / 1.0)} ml glass or reduce volume to "
            f"≤{int(cap * _GLASS_MAX_FILL)} ml."
        )
    return True, "ok"


def science_guardrails(text: str) -> tuple[bool, str]:
    """Deterministic physics/biochemistry checks — precision, classic, and sketch with full sections."""
    # _check_acid_overload and _check_salt_overload removed: quantity limits are now
    # handled by validate_formula_spec (sweet/acid ratio) + Reasoner professional knowledge.
    # The old regex threshold (>6 ml ≥20% solution) was set for unconstrained LLMs and
    # produced false positives on legitimate craft cocktail specs.
    for check in (
        _check_spirit_heat_violation,
        _check_agar_thaw_violation,
        _check_spherification_violation,
        _check_sphere_direction_violation,
        _check_ouzo_violation,
        _check_gelatin_rheology_violation,
        _check_bentonite_hydration_violation,
        _check_raw_animal_fat_violation,
        _check_powder_volume_unit,
    ):
        ok, reason = check(text)
        if not ok:
            return False, reason
    return True, "ok"


def structural_filter(text: str) -> tuple[bool, str]:
    """
    Section structure, step continuity, equipment bleed, ingredients table integrity.
    Used by precision pipeline (via hard_filter) and fast mode when output has full recipe sections.
    """
    if not text or not text.strip():
        return False, "Empty recipe output"

    checks = [
        (_SECTION_INGREDIENTS, "Ingredients / 原料"),
        (_SECTION_METHOD, "Method / 方法"),
        (_SECTION_SCIENCE, "The Science / 科学"),
    ]
    for pattern, label in checks:
        if not pattern.search(text):
            return False, f"Missing required section: {label}"

    method_start = _find_section(_SECTION_METHOD, text)
    science_start = _find_section(_SECTION_SCIENCE, text)
    method_end = science_start if science_start != -1 else len(text)
    if method_start != -1:
        method_block = text[method_start:method_end]
        step_lines = [ln for ln in method_block.splitlines() if _STEP_LINE.match(ln)]
        if not step_lines:
            return False, "Method section has no numbered steps (Step N / 步骤 N)"
        step_nums = [int(n) for n in _STEP_MARKERS.findall(method_block)]
        if step_nums:
            expected = list(range(1, max(step_nums) + 1))
            found = sorted(set(step_nums))
            if found != expected:
                missing = sorted(set(expected) - set(found))
                return False, f"Method steps must be sequential 1..N; missing: {missing}"
        for ln in step_lines:
            if not re.search(r"\d", ln):
                return False, f"Method step missing numeric value: {ln[:100]}"

    eq_match = _EQ_SECTION.search(text)
    if eq_match:
        eq_body = eq_match.group(1)
        # Line-level bleed detection — avoid whole-section regex false positives
        # (e.g. "Jigger — 45 ml spirit / 22 ml citrus" is valid equipment, not method).
        _eq_imperative = re.compile(
            r"(?:"
            r"(?<![量计])取\s*\d+\s*ml|加入\s*\d+\s*ml|将\s*\d+\s*ml[^。\n]{0,30}(?:倒入|注入|加入)|"
            r"移液管取\s*\d|过滤至\s*\d|搅拌至\s*\d|摇荡\s*\d+\s*(?:秒|s)|"
            r"(?:shake|stir|pour)\s+\d+\s*ml\s+(?:of\s+)?(?:the\s+)?(?:spirit|base|mix)|"
            r"add\s+\d+\s*ml\s+(?:of\s+)?(?:the\s+)?(?:spirit|gin|vodka|bourbon|rum)|"
            r"\*\*(?:步骤|Step)\s*\d|"
            r"^\s*[-*]\s*(?:步骤|Step)\s*\d"
            r")",
            re.IGNORECASE | re.MULTILINE,
        )
        for eq_line in eq_body.splitlines():
            stripped = eq_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _eq_imperative.search(stripped):
                return False, "Equipment section contains method action text (content bleed between sections)"
            if stripped.startswith(("-", "*")) and len(stripped) > 200:
                return False, (
                    f"Equipment bullet line too long (>{len(stripped)} chars) "
                    "— likely method step bleed"
                )

    ingr_m = _SECTION_INGREDIENTS.search(text)
    if ingr_m:
        rest = text[ingr_m.end():]
        next_m = re.compile(
            r"^#{2,3}\s*(?:Equipment|设备|Method|制作方法|方法|步骤|The Science|科学)",
            re.IGNORECASE | re.MULTILINE,
        ).search(rest)
        if next_m:
            ingr_block = rest[: next_m.start()]
            for ln in ingr_block.splitlines():
                stripped = ln.strip()
                if stripped.startswith("|") and not stripped.endswith("|"):
                    return False, (
                        f"Ingredients table row is missing closing pipe (truncation): {stripped[:80]}"
                    )

    return True, "ok"


def _is_full_recipe_output(text: str) -> bool:
    """True when fast/sketch output accidentally uses the full precision recipe template."""
    return bool(
        _SECTION_METHOD.search(text)
        and (_SECTION_INGREDIENTS.search(text) or _EQ_SECTION.search(text))
    )


def safety_filter(text: str) -> tuple[bool, str]:
    """
    Lightweight safety-only pass: banned ingredients, dangerous patterns, fat-wash check.
    Does NOT check structural sections — safe to use on fast/classic outputs.
    """
    if not text or not text.strip():
        return False, "Empty output"
    text_lower = text.lower()
    for kw in BANNED_INGREDIENTS:
        if kw.lower() in text_lower:
            return False, f"Banned ingredient detected: '{kw}'"
    if re.search(r"ln[₂2].*clos|clos.*ln[₂2]|nitrogen.*seal|seal.*nitrogen", text_lower):
        return False, "Unsafe pattern: liquid nitrogen in closed/sealed container"
    FAT_INFUSION = re.compile(
        r"sous.?vide|low.?temp\s+infus|fat.?wash|脂肪浸|低温浸|慢煮浸",
        re.IGNORECASE,
    )
    FAT_INGR = re.compile(
        r"\b(beef|pork|bacon|duck\s+fat|schmaltz|lard|tallow|suet|lamb\s+fat|"
        r"butter|ghee|coconut\s+(?:cream|fat|oil)|cream\s+of\s+coconut|"
        r"heavy\s+cream|double\s+cream|milk\s+fat|cheese|chorizo|pancetta|"
        r"prosciutto|foie\s+gras|marrow|bone\s+fat|nut\s+(?:oil|butter)|"
        r"peanut\s+butter|almond\s+(?:butter|oil)|sesame\s+(?:oil|paste)|"
        r"tahini|olive\s+oil|avocado\s+oil|"
        r"牛肉|猪肉|培根|黄油|猪油|椰浆|椰油|椰子油|奶油|奶酪|芝士|"
        r"鹅肝|骨髓|花生酱|芝麻酱|坚果油|肉干|jerky|五花肉|腊肉|肥肉)\b",
        re.IGNORECASE,
    )
    FREEZE_FILTER = re.compile(
        r"freez|−18|冷冻|撇脂|过滤|filter|strain|skim",
        re.IGNORECASE,
    )
    if FAT_INFUSION.search(text) and FAT_INGR.search(text):
        if not FREEZE_FILTER.search(text):
            return False, "Fat infusion detected but mandatory freeze-filter step is missing"
    raw_ok, raw_reason = _check_raw_animal_fat_violation(text)
    if not raw_ok:
        return False, raw_reason
    return True, "ok"


def _check_protease_violation(text: str) -> tuple[bool, str]:
    PROTEASE_INGR = re.compile(
        r"\b(raw\s+papaya|papaya\s+juice|生木瓜|木瓜汁|"
        r"raw\s+pineapple|pineapple\s+juice|菠萝汁|生菠萝|"
        r"raw\s+fig|生无花果|raw\s+kiwi|生猕猴桃)\b",
        re.IGNORECASE,
    )
    PROTEASE_TARGET = re.compile(
        r"\b(gelatin|明胶|heavy\s+cream|double\s+cream|鲜奶油|淡奶油|"
        r"egg\s+white|蛋清|蛋白|milk\b|牛奶|cream\b|奶油|lecithin\s+foam|"
        r"protein\s+foam|foam\s+base)\b",
        re.IGNORECASE,
    )
    INACTIVATE = re.compile(
        r"(85|90|95|100)\s*°?C|灭活|inactivat|denatur|热处理.*木瓜|木瓜.*热处理|"
        r"heat.*papaya|papaya.*heat|热处理.*菠萝|菠萝.*热处理|heat.*pineapple|pineapple.*heat",
        re.IGNORECASE,
    )
    if PROTEASE_INGR.search(text) and PROTEASE_TARGET.search(text):
        if not INACTIVATE.search(text):
            return False, (
                "Proteolytic enzyme hazard: recipe combines raw papaya/pineapple with "
                "gelatin or dairy protein but lacks a prior heat-inactivation step "
                "(≥85°C ≥5 min on the juice alone). Gelatin/foam will completely liquefy."
            )
    return True, "ok"


def _surgeon_failopen_allowed(issues: list[str]) -> bool:
    """Allow streaming best-effort Surgeon output when only non-safety issues remain."""
    _blocked = ("safety", "protease", "banned")
    return bool(issues) and not any(
        b in issue.lower() for issue in issues for b in _blocked
    )


def _check_method_number_leak(
    text: str, locked_balance: "dict | None"
) -> tuple[bool, str]:
    """
    Audit: detect when R1 wrote a different ml amount for an ingredient than locked_balance.
    Fires ONLY when a number in Method text disagrees with the locked amount by >1 ml.
    Correct amounts (written by _inject_method_amounts) are tolerated.
    """
    if not locked_balance:
        return True, ""
    meth_m = _SECTION_METHOD.search(text)
    if not meth_m:
        return True, ""
    meth_body_start = meth_m.end()
    rest = text[meth_body_start:]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    meth_body_end = meth_body_start + (next_hdr.start() if next_hdr else len(rest))
    method_body = text[meth_body_start:meth_body_end]

    violations: list[str] = []
    for ing in locked_balance.get("ingredients", []):
        name = ing.get("name", "").strip()
        amount_str = ing.get("amount", "").strip()
        if not name or not amount_str:
            continue
        locked_m = re.search(r"(\d+(?:\.\d+)?)\s*ml", amount_str, re.IGNORECASE)
        if not locked_m:
            continue
        locked_ml = float(locked_m.group(1))

        # Find "N ml [of] ingredient" or "ingredient … N ml" patterns in Method
        pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*ml\s+(?:of\s+)?" + re.escape(name) + r"\b"
            r"|" + re.escape(name) + r"\b[^.\n]{0,30}?(\d+(?:\.\d+)?)\s*ml",
            re.IGNORECASE,
        )
        for m_obj in pattern.finditer(method_body):
            found_ml = float((m_obj.group(1) or m_obj.group(2)))
            if abs(found_ml - locked_ml) > 1.0:
                violations.append(
                    f"{name}: Method says {found_ml:.0f} ml but locked is {locked_ml:.0f} ml"
                )
    if violations:
        return False, "Method amount mismatch — " + "; ".join(violations[:3])
    return True, ""


def collect_audit_issues(
    text: str,
    locked_balance: "dict | None" = None,
    flavors: "list[str] | None" = None,
    ingredients_input: str = "",
) -> list[str]:
    """Run every deterministic audit; return all failure reasons (empty = clean)."""
    issues: list[str] = []
    checks = [
        ("structure", structural_filter),
        ("chemical alignment", _check_chem_alignment_required),
        ("science", science_guardrails),
        ("protease", _check_protease_violation),
        ("safety", safety_filter),
    ]
    for label, fn in checks:
        ok, reason = fn(text)
        if not ok:
            issues.append(f"[{label}] {reason}")
    # Check locked balance values if a pre-filled balance is available
    locked_ok, locked_reason = _check_locked_values(text, locked_balance)
    if not locked_ok:
        issues.append(f"[locked values] {locked_reason}")
    # Check for ghost ingredients (R1 inventing unlocked ingredients)
    ghost_ok, ghost_reason = _check_ingredient_ghost(text, locked_balance)
    if not ghost_ok:
        issues.append(f"[ghost ingredient] {ghost_reason}")
    # Physics audit triad: mass conservation · ice-melt dilution · container geometry
    for phys_label, phys_fn in (
        ("sugar mass conservation", _check_sugar_mass_conservation),
        ("dilution volume",         _check_dilution_volume_consistency),
        ("glass fill ratio",        _check_glass_fill_ratio),
    ):
        phys_ok, phys_reason = phys_fn(text, locked_balance)
        if not phys_ok:
            issues.append(f"[{phys_label}] {phys_reason}")
    # Method number leak: R1 wrote a different ml amount than locked_balance
    leak_ok, leak_reason = _check_method_number_leak(text, locked_balance)
    if not leak_ok:
        issues.append(f"[method number leak] {leak_reason}")
    # Flavor intent: stated targets must be in Ingredients table, not only garnish
    intent_ok, intent_reason = _check_intent_in_ingredients_table(
        text, flavors, ingredients_input
    )
    if not intent_ok:
        issues.append(f"[flavor intent] {intent_reason}")
    # Domain rules: no plain water as dilutant; carbonated = top-up only
    water_ok, water_reason = _check_direct_water_dilution(text, locked_balance)
    if not water_ok:
        issues.append(f"[water dilution] {water_reason}")
    carb_ok, carb_reason = _check_carbonation_technique(text, locked_balance)
    if not carb_ok:
        issues.append(f"[carbonation] {carb_reason}")
    bev_ok, bev_reason = _check_method_unlisted_beverages(text, locked_balance)
    if not bev_ok:
        issues.append(f"[method beverage ghost] {bev_reason}")
    shaken_ok, shaken_reason = _check_shaken_requires_body(locked_balance)
    if not shaken_ok:
        issues.append(f"[shaken structure] {shaken_reason}")
    return issues


# ── Flavour-intent coverage audit ─────────────────────────────────────────

_GARNISH_CONTEXT_RE = re.compile(
    r"garnish|decoration|装饰|twist|peel|rim|dust|sprinkle|挤压|zest|squeeze",
    re.IGNORECASE,
)

def _check_intent_in_ingredients_table(
    text: str,
    flavors: "list[str] | None",
    ingredients_input: str,
) -> tuple[bool, str]:
    """
    Audit: explicit flavor targets and user-listed ingredients must appear in the
    Ingredients TABLE — not only as garnish/decoration in Method steps.
    """
    raw_keywords: list[str] = list(flavors or [])
    for chunk in re.split(r"[,;、/]+", ingredients_input or ""):
        t = chunk.strip()
        if t:
            raw_keywords.append(t)
    if not raw_keywords:
        return True, ""

    ing_m = _SECTION_INGREDIENTS.search(text)
    if not ing_m:
        return True, ""
    rest = text[ing_m.end():]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    ing_table = (rest[:next_hdr.start()] if next_hdr else rest).lower()

    meth_m = _SECTION_METHOD.search(text)
    meth_body = ""
    if meth_m:
        rest2 = text[meth_m.end():]
        next_hdr2 = _KNOWN_SECTION_RE.search(rest2)
        meth_body = (rest2[:next_hdr2.start()] if next_hdr2 else rest2).lower()

    STOP = {
        "the", "and", "or", "with", "no", "none", "some", "any", "all",
        "fresh", "low", "high", "medium", "sweet", "sour", "bitter", "dry",
        "cold", "hot", "light", "dark", "strong", "weak", "classic",
        "好", "坏", "多", "少", "无", "有", "用", "和", "的", "不", "要",
        "cocktail", "drink", "recipe", "鸡尾酒", "配方",
    }

    flagged: list[str] = []
    for kw in raw_keywords:
        kw_lower = kw.lower().strip()
        if len(kw_lower) < 2 or kw_lower in STOP:
            continue
        if kw_lower in ing_table:
            continue
        # In Method but missing from Ingredients table — fail regardless of garnish context
        if kw_lower in meth_body:
            flagged.append(kw)
            continue
        # Substring match for compound names (e.g. 椰子 in 椰子水)
        if any(kw_lower in cell or cell in kw_lower for cell in re.findall(r"[^\|]+", ing_table)):
            continue
        if any(kw_lower in tok for tok in re.split(r"[\s,，、;；]+", meth_body) if len(tok) >= 2):
            flagged.append(kw)

    if flagged:
        return False, (
            f"Flavor intent coverage failure — {flagged[:3]} appear in concept/method "
            f"but are NOT measured ingredients in the Ingredients table. "
            f"Add each to the locked Ingredients table with ml/g amounts."
        )
    return True, ""


# Beverage ingredients that must appear in the locked table if named in Method
_METHOD_BEVERAGE_TOKENS = re.compile(
    r"\b(?:coconut\s+water|lime\s+juice|lemon\s+juice|yuzu\s+juice|grapefruit\s+juice|"
    r"orange\s+juice|simple\s+syrup|honey\s+syrup|agave|soda\s+water|tonic|ginger\s+beer|"
    r"椰子水|青柠汁?|柠檬汁?|柚子汁?|糖浆|苏打水|汤力水)\b",
    re.IGNORECASE,
)


def _check_method_unlisted_beverages(
    text: str, locked_balance: "dict | None"
) -> tuple[bool, str]:
    """Method names a beverage ingredient by name but it is absent from locked table."""
    if not locked_balance:
        return True, ""
    meth_m = _SECTION_METHOD.search(text)
    if not meth_m:
        return True, ""
    rest = text[meth_m.end():]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    meth_body = rest[: next_hdr.start()] if next_hdr else rest

    locked_blob = " ".join(
        i.get("name", "").lower() for i in locked_balance.get("ingredients", [])
    )
    missing: list[str] = []
    for m in _METHOD_BEVERAGE_TOKENS.finditer(meth_body):
        token = m.group(0).lower()
        # fuzzy: any locked name shares a significant substring
        if not any(
            token in lk or lk in token
            or token.split()[0] in lk
            or (len(token) >= 2 and token[:2] in lk)
            for lk in locked_blob.split()
        ):
            # also check full locked names not split by space
            if not any(
                token in i.get("name", "").lower() or i.get("name", "").lower() in token
                for i in locked_balance.get("ingredients", [])
            ):
                missing.append(m.group(0))

    if missing:
        return False, (
            f"Method references {missing[:3]} but they are NOT in the locked Ingredients table. "
            "Every beverage ingredient in Method must be in Stage 0.5 selection and the locked table."
        )
    return True, ""


def _check_shaken_requires_body(
    locked_balance: "dict | None",
) -> tuple[bool, str]:
    """Shaken drinks need acid, sweetener, or dilutant — not spirit-only."""
    if not locked_balance:
        return True, ""
    if locked_balance.get("serve_style") != "shaken":
        return True, ""
    roles = {i.get("role", "") for i in locked_balance.get("ingredients", [])}
    if roles & {"acid", "sweetener", "dilutant"}:
        return True, ""
    return False, (
        "Shaken serve with spirit/modifier only — no acid, sweetener, or dilutant in locked table. "
        "Spirit-forward drinks must use stirred; sours/highballs need acid or dilutant."
    )


# ── Direct water dilution audit ────────────────────────────────────────────
# Cocktail dilution comes from ice melt — plain water as a recipe ingredient is wrong.

_SAFE_WATER_VARIANTS_RE = re.compile(
    r"\b(?:soda\s+water|sparkling\s+water|tonic\s+(?:water)?|coconut\s+water|"
    r"ginger\s+(?:ale|beer)|cold\s+water|ice\s+water|kombucha|champagne|prosecco|cava|"
    r"club\s+soda|fever.?tree|san\s+pellegrino|perrier|"
    r"苏打水|气泡水|汤力水|椰子水|姜汁啤酒|姜汁汽水|香槟|起泡酒)\b",
    re.IGNORECASE,
)
_PLAIN_WATER_RE = re.compile(
    r"\b(?:water|still\s+water|tap\s+water|mineral\s+water|spring\s+water|"
    r"distilled\s+water|纯水|矿泉水|自来水|温水|热水)\b",
    re.IGNORECASE,
)


def _check_direct_water_dilution(
    text: str, balance: "dict | None"
) -> tuple[bool, str]:
    """
    Audit: plain still water must not appear as a measured ml ingredient.
    Dilution is provided by ice melt (accounted for in balance engine).
    Safe: soda water, tonic water, sparkling water, coconut water, ginger beer, etc.
    """
    ing_m = _SECTION_INGREDIENTS.search(text)
    if not ing_m:
        return True, ""
    rest = text[ing_m.end():]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    ing_table = rest[:next_hdr.start()] if next_hdr else rest

    for row in ing_table.splitlines():
        if not re.search(r"\|\s*\d+\s*ml", row):
            continue
        if _SAFE_WATER_VARIANTS_RE.search(row):
            continue
        if _PLAIN_WATER_RE.search(row):
            return False, (
                f"Direct water dilution — plain water appears as a measured ingredient "
                f"({row.strip()[:80]}). "
                f"Cocktail dilution comes from ice melt (already in balance engine). "
                f"Replace with soda water / tonic / coconut water if dilutant is needed."
            )
    return True, ""


# ── Carbonation technique audit ────────────────────────────────────────────

_CARBONATED_INGREDIENT_RE = re.compile(
    r"\b(?:soda\s+water|sparkling\s+water|tonic\s+(?:water)?|ginger\s+(?:ale|beer)|"
    r"club\s+soda|champagne|prosecco|cava|kombucha|fever.?tree|"
    r"苏打水|气泡水|汤力水|姜汁啤酒|姜汁汽水|香槟|起泡酒)\b",
    re.IGNORECASE,
)
_SHAKE_BLEND_ACTION_RE = re.compile(
    r"\b(?:shake|shaken|摇[制荡晃]+|摇荡|blend(?:ed)?|搅拌机|Vitamix|blender)\b",
    re.IGNORECASE,
)


def _check_carbonation_technique(
    text: str, balance: "dict | None"
) -> tuple[bool, str]:
    """
    Audit: carbonated ingredients (soda, tonic, sparkling water, ginger beer,
    champagne) must NOT be shaken or blended — CO2 escapes and the drink goes flat.
    They must be top-up additions in the final Method step.
    """
    ing_m = _SECTION_INGREDIENTS.search(text)
    if not ing_m:
        return True, ""
    rest = text[ing_m.end():]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    ing_table = rest[:next_hdr.start()] if next_hdr else rest

    carb_m = _CARBONATED_INGREDIENT_RE.search(ing_table)
    if not carb_m:
        return True, ""
    carb_name = carb_m.group(0)

    meth_m = _SECTION_METHOD.search(text)
    if not meth_m:
        return True, ""
    rest2 = text[meth_m.end():]
    next_hdr2 = _KNOWN_SECTION_RE.search(rest2)
    method_body = rest2[:next_hdr2.start()] if next_hdr2 else rest2

    if _SHAKE_BLEND_ACTION_RE.search(method_body):
        return False, (
            f"Carbonation technique violation — '{carb_name}' is carbonated but method "
            f"includes shaking/blending, which destroys CO2. "
            f"Use stirred or built technique; add carbonated ingredient as final top-up."
        )

    # Check carbonated ingredient appears only in the last step
    step_data: list[tuple[str, int]] = []
    for ln in method_body.splitlines():
        nums = re.findall(r"\*\*(?:Step|步骤)\s*(\d+)", ln, re.IGNORECASE)
        for n in nums:
            step_data.append((ln, int(n)))
    if step_data:
        max_step = max(s for _, s in step_data)
        carb_steps = [s for ln, s in step_data if _CARBONATED_INGREDIENT_RE.search(ln)]
        if carb_steps and min(carb_steps) < max_step:
            return False, (
                f"Carbonation sequence violation — '{carb_name}' appears in Step "
                f"{min(carb_steps)} but there are {max_step} steps. "
                f"Carbonated ingredients must be the final top-up (last step)."
            )
    return True, ""


# ── Chemical Alignment table completeness (hard filter) ────────────────────

def _check_chem_table_completeness(
    text: str, balance: "dict | None"
) -> tuple[bool, str]:
    """
    Hard filter: when balance was pre-computed, the Chemical Alignment section
    must exist and contain at least the ABV row (truncation detector).
    """
    if not balance or balance.get("final_abv_pct") is None:
        return True, ""
    chem_m = _SECTION_CHEM_ALIGNMENT.search(text)
    if not chem_m:
        return False, (
            "Chemical Alignment section missing — required when balance is pre-computed. "
            "Copy the PRE-FILLED TABLE from system prompt verbatim."
        )
    rest = text[chem_m.end():]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    chem_body = rest[:next_hdr.start()] if next_hdr else rest
    if not re.search(r"ABV|最终\s*ABV", chem_body, re.IGNORECASE):
        return False, (
            "Chemical Alignment table truncated — ABV row missing. "
            "Reproduce the full PRE-FILLED TABLE from system prompt."
        )
    return True, ""


# ── Stage 0.5 ingredient coverage verification ────────────────────────────

def _verify_ingredient_coverage(body, selection: dict) -> list[str]:
    """
    Check that every ingredient the user listed (body.ingredients) was selected
    by Stage 0.5. Returns list of ingredient names that were omitted.
    Also scans notes + flavors for explicit material terms (椰子, lime, etc.).
    """
    if not selection:
        return []
    selected_lower = {
        i.get("name", "").lower().strip()
        for i in selection.get("ingredients", [])
    }
    missing: list[str] = []

    concept_blob = " ".join([
        getattr(body, "ingredients", "") or "",
        getattr(body, "notes", "") or "",
        " ".join(getattr(body, "flavors", []) or []),
    ])
    _CONCEPT_TERMS = (
        "椰子", "coconut", "青柠", "lime", "柠檬", "lemon", "柚子", "yuzu",
        "蜂蜜", "honey", "糖浆", "syrup", "苏打", "soda", "汤力", "tonic",
        "姜汁", "ginger", "菠萝", "pineapple", "芒果", "mango", "薄荷", "mint",
    )
    chunks: list[str] = []
    for chunk in re.split(r"[,;、/]+", concept_blob):
        t = chunk.strip()
        if len(t) >= 2:
            chunks.append(t)
    for term in _CONCEPT_TERMS:
        if term.lower() in concept_blob.lower():
            chunks.append(term)

    seen: set[str] = set()
    for item in chunks:
        item_lower = item.lower().strip()
        if len(item_lower) < 2 or item_lower in seen:
            continue
        seen.add(item_lower)
        if not any(item_lower in sn or sn in item_lower for sn in selected_lower):
            missing.append(item)
    return missing


def surgeon_revise_recipe(
    client: OpenAI,
    model: str,
    user_msg: str,
    recipe_text: str,
    issues: list[str],
    codex_system: str,
    locked_balance: "dict | None" = None,
    language: str = "en",
) -> str:
    """Full recipe rewrite fixing all audit issues while preserving flavor intent."""
    issues_text = "\n".join(f"{i + 1}. {issue}" for i, issue in enumerate(issues))
    user_content = (
        f"ORIGINAL USER REQUEST:\n{user_msg}\n\n"
        f"FLAWED RECIPE:\n{recipe_text}\n\n"
    )
    if locked_balance:
        user_content += (
            "LOCKED PRE-FILLED TABLES (copy ### Ingredients + ### Chemical Alignment VERBATIM; "
            "do NOT recalculate ABV, acid, sugar, pH, or ml amounts):\n"
            f"{format_balance_injection(locked_balance, language)}\n\n"
        )
    user_content += (
        f"AUDIT FAILURES (fix ALL {len(issues)} item(s)):\n{issues_text}\n\n"
        "Rewrite the complete corrected recipe now."
    )
    messages = [
        {
            "role": "system",
            "content": build_surgeon_principles(language) + "\n\n" + RECIPE_SURGEON_SYSTEM,
        },
        {"role": "user", "content": user_content},
    ]
    return generate_once(client, model, messages)


def _check_markdown_table_bleed(text: str) -> tuple[bool, str]:
    """
    Hard filter: detect section headers or method steps swallowed into a table row.
    Caused when a markdown table is not followed by a blank line before the next section.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.match(r"^\|[\s\-:|]+\|$", stripped):
            continue
        if re.search(r"#{2,4}\s", stripped):
            return False, (
                "Markdown table bleed — section header found inside a table row. "
                "Tables must be followed by a blank line before the next ### section."
            )
        if re.search(r"\*\*(?:Step|步骤)\s*\d+", stripped, re.IGNORECASE):
            return False, (
                "Markdown table bleed — method step found inside a table row. "
                "Tables must be followed by a blank line before ### Method."
            )
        if re.search(r"(?:步骤|Step)\s*\d+\s*[—\-–:]", stripped, re.IGNORECASE):
            return False, "Markdown table bleed — numbered method step inside a table row."

    # Method steps exist but no Method section header → likely swallowed into a table cell
    if _STEP_MARKERS.search(text) and not _SECTION_METHOD.search(text):
        return False, (
            "Method steps present but ### Method / ### 制作方法 header missing — "
            "likely table formatting bleed."
        )
    return True, ""


def hard_filter(
    text: str,
    locked_balance: "dict | None" = None,
) -> tuple[bool, str]:
    """Returns (pass, reason). pass=True means the recipe passed all checks."""
    struct_ok, struct_reason = structural_filter(text)
    if not struct_ok:
        return False, struct_reason

    chem_ok, chem_reason = _check_chem_alignment_required(text)
    if not chem_ok:
        return False, chem_reason

    sci_ok, sci_reason = science_guardrails(text)
    if not sci_ok:
        return False, sci_reason

    protease_ok, protease_reason = _check_protease_violation(text)
    if not protease_ok:
        return False, protease_reason

    safety_ok, safety_reason = safety_filter(text)
    if not safety_ok:
        return False, safety_reason

    locked_ok, locked_reason = _check_locked_values(text, locked_balance)
    if not locked_ok:
        return False, locked_reason

    ghost_ok, ghost_reason = _check_ingredient_ghost(text, locked_balance)
    if not ghost_ok:
        return False, ghost_reason

    bev_ok, bev_reason = _check_method_unlisted_beverages(text, locked_balance)
    if not bev_ok:
        return False, bev_reason

    shaken_ok, shaken_reason = _check_shaken_requires_body(locked_balance)
    if not shaken_ok:
        return False, shaken_reason

    chem_ok, chem_reason = _check_chem_table_completeness(text, locked_balance)
    if not chem_ok:
        return False, chem_reason

    bleed_ok, bleed_reason = _check_markdown_table_bleed(text)
    if not bleed_ok:
        return False, bleed_reason

    return True, "ok"


def _max_tokens_for(model: str) -> int:
    # reasoner spends tokens on chain-of-thought before the final recipe
    if "reasoner" in model:
        return 16384
    return 3000


def generate_once(client: OpenAI, model: str, messages: list) -> str:
    """Non-streaming generation — returns final recipe text."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=_max_tokens_for(model),
        temperature=0.35,
        timeout=300,
    )
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if "reasoner" in model:
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        if reasoning:
            if not text:
                text = reasoning
            elif not _SECTION_INGREDIENTS.search(text) and _SECTION_INGREDIENTS.search(reasoning):
                text = reasoning
    text = normalize_recipe_text(text)
    if not text:
        finish = getattr(resp.choices[0].message, "finish_reason", "") or getattr(
            resp.choices[0], "finish_reason", ""
        )
        raise RuntimeError(
            f"Model returned empty content (finish_reason={finish}). "
            "R1 may have exhausted the token budget during reasoning."
        )
    return text


def _start_thread(fn, *args, **kwargs) -> tuple[threading.Thread, queue.Queue]:
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result_q.put(("ok", fn(*args, **kwargs)))
        except Exception as exc:
            result_q.put(("err", str(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, result_q


def _heartbeats_while_running(thread, status: str, detail_prefix: str, interval: int = 5):
    elapsed = 0
    # Immediate ping so proxies don't idle-timeout before first R1 token
    yield sse({"status": status, "detail": f"{detail_prefix}… (0s)"})
    while thread.is_alive():
        time.sleep(interval)
        elapsed += interval
        yield sse({"status": status, "detail": f"{detail_prefix}… ({elapsed}s)"})
        yield ": keepalive\n\n"


def _join_thread_result(thread, result_q):
    thread.join(timeout=5)
    flag, payload = result_q.get()
    if flag == "err":
        raise RuntimeError(payload)
    return payload


def verify_recipe(recipe_text: str, perplexity_key: str) -> dict:
    """
    Uses Perplexity sonar to verify EU food safety AND culinary technique reasonableness.
    Returns {
        "safety_verdict": "PASS"|"FAIL", "safety_issues": [...],
        "culinary_verdict": "PASS"|"FAIL", "culinary_issues": [...]
    }
    Falls back to all-PASS on error.
    """
    _safe_default = {
        "safety_verdict": "PASS", "safety_issues": [],
        "culinary_verdict": "PASS", "culinary_issues": [],
    }
    try:
        client_pplx = OpenAI(
            api_key=perplexity_key,
            base_url="https://api.perplexity.ai",
        )
        user_msg = (
            "Review this cocktail recipe for both EU EFSA food safety compliance "
            "AND culinary/technique correctness.\n\n"
            f"RECIPE:\n{recipe_text}\n\n"
            "Return ONLY the JSON object described in your instructions. No markdown, no preamble."
        )
        resp = client_pplx.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            stream=False,
            max_tokens=1024,
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        # Normalise keys — also accept legacy "verdict"/"issues" from old model responses
        if "verdict" in result and "safety_verdict" not in result:
            result["safety_verdict"] = result.pop("verdict")
        if "issues" in result and "safety_issues" not in result:
            result["safety_issues"] = result.pop("issues")
        for k, default in _safe_default.items():
            result.setdefault(k, default)
        return result
    except Exception as exc:
        default = dict(_safe_default)
        default["safety_issues"] = [f"Verification skipped: {exc}"]
        return default


def validate_input_with_ai(client: OpenAI, ingredients: str, notes: str, language: str) -> dict:
    """
    Quick deepseek-chat call to detect gibberish / off-topic input.
    Returns {"valid": True} or {"valid": False, "reason": "<user-facing message>"}
    Always returns valid=True on API error (fail open).
    """
    combined = f"{notes or ''} {ingredients or ''}".strip()

    # Known foraged culinary inputs — always valid; pipeline enforces heat protocols.
    if _FORAGED_CULINARY_OK.search(combined):
        return {"valid": True}

    # Fast heuristics — no API call needed
    letters = re.sub(r'\s', '', combined)
    if len(letters) < 2:
        lang_msg = "请描述您的风味概念或填写可用的材料。" if language == "zh" \
            else "Please describe your flavour concept or specify available ingredients."
        return {"valid": False, "reason": lang_msg}

    # Keyboard-mash detection: if > 70 % of chars are the same char, likely gibberish
    if letters and max(letters.count(c) for c in set(letters)) / len(letters) > 0.7:
        lang_msg = "输入内容无法识别，请填写具体的风味描述或材料名称。" if language == "zh" \
            else "Input looks unrecognisable. Please enter a flavour concept or ingredient name."
        return {"valid": False, "reason": lang_msg}

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": INPUT_VALIDATE_SYSTEM},
                {"role": "user", "content": f"Concept: {notes or '(none)'}\nIngredients: {ingredients or '(none)'}"},
            ],
            stream=False,
            max_tokens=120,
            temperature=0.0,
            timeout=15,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        result.setdefault("valid", True)
        return result
    except Exception:
        return {"valid": True}


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _lang_instruction(language: str) -> str:
    lang_map = {"zh": "Please respond entirely in Chinese.", "en": "Please respond entirely in English."}
    return lang_map.get(language, lang_map["en"])


INGREDIENT_SELECTOR_SYSTEM = """\
You are a cocktail ingredient selector. Given the flavor concept and available ingredients,
identify which ingredients to use and categorize them.

Return ONLY valid JSON — no markdown, no preamble:
{
  "ingredients": [
    {"name": "<exact ingredient name from user input>", "category": "spirit|acid|sweetener|dilutant|modifier|bitters", "note": "<1-word role>"}
  ],
  "target_abv_pct": <number 0.0-28.0>,
  "target_volume_ml": <number 60-220>,
  "sweet_acid_ratio": <number 1.0-3.5>,
  "serve_style": "shaken|stirred|built|blended"
}

Category rules:
- spirit: base alcohol >=25% ABV
- acid: primary acid source (citrus juice, acid powders)
- sweetener: sugar source (syrups, honey)
- modifier: flavor liqueur, vermouth, Campari, Cointreau, OR foraged umami (见手青, truffles, wild mushrooms)
- dilutant: water-based extender (soda, coconut water)
- bitters: dash-only agents

CRITICAL: Use ONLY ingredients from the user's input. Do NOT invent new ingredients.
CRITICAL: Include EVERY ingredient the user explicitly listed — do not silently drop any.
If a user-listed ingredient is unsuitable, include it anyway and set its role appropriately.

DOMAIN RULES (non-negotiable):
- NEVER select plain water (still/mineral/tap) as a dilutant — cocktail dilution comes from
  ice melt during shaking/stirring (already handled by the balance engine).
  Valid dilutants: soda water, tonic water, sparkling water, ginger beer, coconut water, juice.
- Carbonated dilutants (soda water, tonic, ginger beer, champagne) are top-up only —
  they must NOT be shaken or blended. Mark their note as "top-up".

ABV target: read the user's Alcohol pref field FIRST — if it states an explicit range (e.g. "8-10%")
or keyword (low/aperitivo/日间/session/低度), honour it precisely.
SPIRIT-FREE (CRITICAL): If alcohol preference is none / 无酒精 / mocktail / 0% / spirit-free:
  target_abv_pct MUST be 0.0
  shaken mocktail sours: target_volume_ml 90-120; built highballs: 160-220
  NEVER assign sour-style ABV 22-26% when no spirit is selected.
Home/Enthusiast max ≤22%, Bar/Professional max ≤28% (0.0 allowed).
The target_abv_pct field must respect these ceilings regardless of style defaults.
Do NOT compute any ml quantities — the math engine handles that.

ABV style reference (use when user gives no explicit ABV):
- Sour-style (citrus juice present, shaken): 22-26%  ← default 24%
- Spirit-forward (no citrus, stirred: Old Fashioned, Negroni, Manhattan): 25-30%  ← default 28%
- Highball / long drink (dilutant or top-up soda present): 8-14%  ← default 10%
- Aperitivo / spritz / low-ABV requested: 8-12%  ← default 9%
- Dessert / after-dinner (cream, coffee, chocolate): 15-20%  ← default 17%
sweet_acid_ratio note: this is sugar_GRAMS / acid_GRAMS (mass ratio, NOT volume).
For classic sours (Daiquiri, Whiskey Sour, Gold Rush): set 1.8-2.2.
For sweeter tropical drinks: set 2.5-3.5. For tart/dry cocktails: set 1.2-1.5.
"""


# ── ABV constraint extractor (deterministic — runs after LLM selection) ──────

_ABV_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*[-–~～至到]\s*(\d+(?:\.\d+)?)\s*%")
_ABV_SINGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:abv|ABV)?")
_LOW_ABV_KEYWORDS = frozenset([
    "低度", "low abv", "low alcohol", "aperitivo", "日间", "session",
    "daytime", "lunch", "brunch", "morning", "早午", "开胃",
])
_MID_ABV_KEYWORDS = frozenset([
    "中度", "medium abv", "moderate", "mid strength", "mid-strength",
])
_SPIRIT_FREE_KEYWORDS = frozenset([
    "无酒精", "零酒精", "不含酒精", "不要酒精", "无酒", "去酒精",
    "无醇", "脱醇", "mocktail", "spirit-free", "spirit free", "alcohol-free",
    "alcohol free", "non-alcoholic", "non alcoholic", "zero abv", "0 abv",
    "no alcohol", "without alcohol", "sans alcool",
])


def _extract_abv_constraint(alcohol: str, notes: str) -> "float | None":
    """
    Deterministically extract an explicit ABV target from the user's alcohol preference
    and concept notes.  Returns the midpoint of a detected range, or None if unspecified.
    This value OVERRIDES the LLM's target_abv_pct to prevent scene-context blindness.
    """
    combined = f"{alcohol} {notes or ''}"
    lower = combined.lower()
    if any(k in lower for k in _SPIRIT_FREE_KEYWORDS):
        return 0.0
    # Explicit range: "8-10%", "12%–15%", "8.0～10.0"
    m = _ABV_RANGE_RE.search(combined)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if 1.0 <= lo < hi <= 55.0:
            return round((lo + hi) / 2.0, 1)
    # Single target: "10% ABV", "18%"
    m = _ABV_SINGLE_RE.search(combined)
    if m:
        val = float(m.group(1))
        if 1.0 <= val <= 55.0:
            return val
    # Keyword mapping
    if any(k in lower for k in _LOW_ABV_KEYWORDS):
        return 9.0   # aperitivo / day-drinking midpoint
    if any(k in lower for k in _MID_ABV_KEYWORDS):
        return 13.0  # moderate sipping midpoint
    return None


# ── Stage 0.5 helpers: ingredient selection ─────────────────────────────

def select_ingredients(
    client: OpenAI,
    body,
    perplexity_key: str = "",
    extra_constraint: str = "",
) -> "dict | None":
    """
    Stage 0.5 redesign: LLM selects + categorizes ingredients only, no ml quantities.
    For any ingredient not found in INGREDIENT_PROPS, triggers cold-start Perplexity
    lookup and caches the result in _PROPS_RUNTIME_CACHE for this process lifetime.
    extra_constraint: injected at end of prompt for coverage-retry forcing.
    """
    _strict_line = (
        "STRICT MODE: select ONLY the ingredients listed above — do NOT add any other ingredient.\n"
        if getattr(body, "strict_ingredients", False) else ""
    )
    prompt = (
        f"Equipment: {body.equipment}\n"
        f"Flavor targets: {', '.join(body.flavors) if body.flavors else 'None'}\n"
        f"Alcohol pref: {body.alcohol}\n"
        f"Ingredients: {body.ingredients}\n"
        f"Concept: {body.notes or 'None'}\n\n"
        f"{_strict_line}"
        f"{extra_constraint}"
        "Select and categorize ingredients. NO quantities."
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": INGREDIENT_SELECTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            stream=False, max_tokens=500, temperature=0.0, timeout=20,
        )
        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            (resp.choices[0].message.content or "").strip(),
            flags=re.MULTILINE,
        ).strip()
        result = json.loads(raw)
        if not isinstance(result.get("ingredients"), list) or not result["ingredients"]:
            return None
        for ing in result["ingredients"]:
            name = ing.get("name", "")
            n = name.lower().strip()
            props = _lookup_ingredient_props(name)
            from_perplexity = False
            if props is None and perplexity_key:
                props = _fetch_props_cold_start(name, perplexity_key)
                from_perplexity = True
            if props is None:
                logger.warning(
                    "props=None for ingredient %r — balance engine will use category defaults; "
                    "add to INGREDIENT_PROPS or ensure Perplexity cold-start succeeds.",
                    name or "?",
                )
            ing["props"] = props
            if n in INGREDIENT_PROPS:
                ing["_props_meta"] = classify_props_provenance(
                    name, props, from_codex_exact=True,
                )
            elif from_perplexity:
                ing["_props_meta"] = classify_props_provenance(
                    name, props, from_perplexity=True,
                )
            elif props is not None:
                ing["_props_meta"] = {"source": "codex_db_fuzzy", "confidence": "medium"}
            else:
                ing["_props_meta"] = classify_props_provenance(name, None)
            # B15 fix: INGREDIENT_PROPS category is authoritative for known ingredients.
            # Override LLM classification when they disagree (LLM at temp=0 is still stochastic
            # for liqueurs like Chartreuse, Aperol, or Absinthe that straddle spirit/modifier).
            if props is not None:
                props_cat = props.get("category")
                llm_cat   = ing.get("category")
                if props_cat and props_cat != llm_cat:
                    ing["category"] = props_cat
                    logger.debug(
                        "B15 category override %r: LLM=%s → PROPS=%s",
                        ing.get("name", "?"), llm_cat, props_cat,
                    )
        # Deterministic ABV override: user's explicit scene constraint beats LLM estimate.
        abv_override = _extract_abv_constraint(
            getattr(body, "alcohol", ""),
            getattr(body, "notes", "") or "",
        )
        if abv_override is not None:
            result["target_abv_pct"] = abv_override
        elif not any(i.get("category") == "spirit" for i in result["ingredients"]) and not any(
            (i.get("props") or {}).get("abv_pct", 0) > 0.2
            for i in result["ingredients"]
            if i.get("category") == "modifier"
        ):
            result["target_abv_pct"] = 0.0
            if float(result.get("target_volume_ml", 90)) > 150:
                style = result.get("serve_style", "shaken")
                result["target_volume_ml"] = (
                    SPIRIT_FREE_SHAKEN_ML if style in ("shaken", "blended") else 200
                )
        result["ingredients"] = apply_synonym_hints(result["ingredients"])
        return result
    except Exception:
        return None


def _calculate_balance_spirit_free(
    ingredients: list, serve_style: str, target_vol: float,
    sweet_acid: float, dilution: float,
) -> dict:
    """
    Spirit-free / mocktail branch of Stage 0.6.
    No ABV equation — only volume + sweet-acid balance.
    Called when there are no spirits and no high-ABV (>20%) modifiers.
    """
    acids      = [i for i in ingredients if i.get("category") == "acid"]
    sweeteners = [i for i in ingredients if i.get("category") == "sweetener"]
    dilutants  = [i for i in ingredients if i.get("category") == "dilutant"]
    bitters    = [i for i in ingredients if i.get("category") == "bitters"]
    modifiers  = [i for i in ingredients if i.get("category") == "modifier"]

    # Ice melt: shaken/blended always pre-dilutes; built highballs use explicit dilutant.
    _has_dilutant = bool(dilutants)
    _shake_dilute = serve_style in ("shaken", "blended")
    if _has_dilutant and _shake_dilute:
        v_liquid = target_vol / (1 + dilution)
    elif _shake_dilute:
        v_liquid = target_vol / (1 + dilution)
    else:
        v_liquid = target_vol

    # Two structural paths:
    #   A) with dilutant   → acid+sweet capped to 25% of total (typical highball/spritz)
    #   B) without dilutant → acid+sweet fill pre-shake liquid budget
    if dilutants:
        # Mocktail highball: acid+sweet limited to 25% of vol (hard cap: 30–60 ml)
        _acid_sweet_budget = max(min(0.25 * v_liquid, 60.0), 30.0)
        v_mod = min(0.15 * v_liquid * len(modifiers), 0.30 * v_liquid) if modifiers else 0.0
        v_remain2 = _acid_sweet_budget
    elif modifiers and acids and not sweeteners:
        # Acid-forward mocktail sour (e.g. yuzu + matcha): balanced pre-shake split
        v_mod = v_liquid * 0.45
        v_remain2 = v_liquid - v_mod
    else:
        v_mod = min(0.30 * v_liquid * len(modifiers), 0.50 * v_liquid) if modifiers else 0.0
        v_remain2 = max(v_liquid - v_mod, 0.0)

    # Sweet-acid balance within budget
    if acids and sweeteners:
        a_props = acids[0].get("props") or {}
        s_props = sweeteners[0].get("props") or {}
        a_ta   = a_props.get("ta_pct", 5.0)
        a_dens = a_props.get("density", 1.03)
        s_brix = s_props.get("brix", 50.0)
        s_dens = s_props.get("density", 1.18)
        if a_ta > 0 and s_brix > 0:
            k = sweet_acid * a_ta * a_dens / (s_brix * s_dens)
            v_acid  = v_remain2 / (1 + k) if v_remain2 > 0 else 15.0
            v_sweet = v_remain2 - v_acid  if v_remain2 > 0 else 10.0
        else:
            v_acid, v_sweet = v_remain2 * 0.55, v_remain2 * 0.45
        # Sensory floors
        v_acid  = max(v_acid, 10.0)
        v_sweet = max(v_sweet, 8.0)
    elif acids:
        v_acid, v_sweet = min(v_remain2, 45.0) if dilutants else v_remain2, 0.0
    elif sweeteners:
        v_acid, v_sweet = 0.0, min(v_remain2, 35.0) if dilutants else v_remain2
    else:
        v_acid, v_sweet = 0.0, 0.0

    # Dilutant fills remainder to target volume
    if dilutants:
        other = v_mod + v_acid + v_sweet
        v_dilutant = max(target_vol - other, 30.0)
    else:
        v_dilutant = 0.0

    def _split(total: float, items: list, mn: int = 5) -> list[int]:
        if not items:
            return []
        n = len(items)
        each = max(round(total / n), mn)
        amts = [each] * n
        diff = round(total) - sum(amts)
        amts[0] = max(amts[0] + diff, mn)
        return amts

    result_ings: list[dict] = []
    for mi, amt in zip(modifiers, _split(v_mod, modifiers, 10)):
        if is_powder_ingredient(mi.get("name", "")):
            g_amt = max(4, min(8, round(amt / 5) or 6))
            result_ings.append({"amount": f"{g_amt} g", "name": mi["name"], "role": "modifier"})
        else:
            result_ings.append({"amount": f"{amt} ml", "name": mi["name"], "role": "modifier"})
    for a, amt in zip(acids, _split(v_acid, acids, 5)):
        result_ings.append({"amount": f"{amt} ml", "name": a["name"], "role": "acid"})
    for sw, amt in zip(sweeteners, _split(v_sweet, sweeteners, 5)):
        result_ings.append({"amount": f"{amt} ml", "name": sw["name"], "role": "sweetener"})
    for d, amt in zip(dilutants, _split(v_dilutant, dilutants, 30)):
        result_ings.append({"amount": f"{amt} ml", "name": d["name"], "role": "dilutant"})
    for b in bitters:
        result_ings.append({"amount": "2 dashes", "name": b["name"], "role": "bitters"})

    total_vol = sum(
        float(i["amount"].split()[0]) for i in result_ings if "ml" in i["amount"]
    )
    if _shake_dilute and not dilutants and total_vol > 0:
        total_vol = total_vol * (1 + dilution)
    a_props2 = (acids[0].get("props") or {}) if acids else {}
    s_props2 = (sweeteners[0].get("props") or {}) if sweeteners else {}
    total_acid_g = (
        v_acid * a_props2.get("ta_pct", 5.0) * a_props2.get("density", 1.03) / 100
        if acids and v_acid > 0 else 0.0
    )
    total_sugar_g = (
        v_sweet * s_props2.get("brix", 50.0) * s_props2.get("density", 1.18) / 100
        if sweeteners and v_sweet > 0 else 0.0
    )
    if not sweeteners and acids and v_acid > 0:
        for a in acids:
            ap = a.get("props") or {}
            brix = ap.get("brix", 0.0)
            if brix > 0:
                share = v_acid / len(acids)
                total_sugar_g += share * brix * ap.get("density", 1.03) / 100
    result = {
        "ingredients":     result_ings,
        "final_abv_pct":   0.0,
        "total_acid_g":    round(total_acid_g, 2),
        "total_sugar_g":   round(total_sugar_g, 1),
        "ph_estimate":     _estimate_ph(total_acid_g, total_vol or target_vol),
        "total_volume_ml": round(total_vol or target_vol),
        "balance_notes":   f"[DETERMINISTIC ENGINE · {serve_style}·spirit-free] No-ABV balance",
        "serve_style":     serve_style,
        "_engine":         "deterministic-spirit-free",
    }
    return attach_metrology(result, {"serve_style": serve_style, "ingredients": ingredients})


def calculate_balance_deterministic(selection: dict) -> "dict | None":
    """
    Stage 0.6: Pure Python algebraic balance engine — no LLM hallucination.
    Serve-style–aware dilution, spirit-forward modifier scaling, Sour ratio guard.

    Equations:
      ① (V_spirit × ABV_spirit + V_mod × ABV_mod) / V_liquid ≈ target_ABV × (1 + dilution)
      ② sugar_g / acid_g = sweet_acid_ratio   (Sour mode)
      ③ V_spirit + V_mod + V_acid + V_sweet + V_dilutant ≈ V_liquid

    Fail-open: returns None if unsolvable — caller must abort (no LLM number fallback).
    """
    # ── Dilution by serve style (technique-accurate) ───────────────────────
    serve_style = selection.get("serve_style", "shaken")
    DILUTION = dilution_for(serve_style)

    ingredients = selection.get("ingredients", [])
    target_abv  = selection.get("target_abv_pct", 18.0) / 100
    target_vol  = float(selection.get("target_volume_ml", 90))
    sweet_acid  = float(selection.get("sweet_acid_ratio", 1.5))

    spirits    = [i for i in ingredients if i.get("category") == "spirit"]
    modifiers  = [i for i in ingredients if i.get("category") == "modifier"]
    acids      = [i for i in ingredients if i.get("category") == "acid"]
    sweeteners = [i for i in ingredients if i.get("category") == "sweetener"]
    dilutants  = [i for i in ingredients if i.get("category") == "dilutant"]
    bitters    = [i for i in ingredients if i.get("category") == "bitters"]

    # ── Hard check: carbonated dilutant cannot be shaken ──────────────────
    _CARBONATED = frozenset({"soda water", "tonic water", "ginger beer", "prosecco",
                              "sparkling water", "champagne", "cava", "苏打水", "汤力水",
                              "气泡水", "姜汁汽水", "香槟", "起泡酒"})
    if serve_style == "shaken" and dilutants:
        for d in dilutants:
            dname = (d.get("name") or "").lower().strip()
            if any(c in dname for c in _CARBONATED) or dname in _CARBONATED:
                serve_style = "built"
                DILUTION = dilution_for("built")
                break

    # ── Equipment ABV hard cap (from reference_canon) ─────────────────────
    from reference_canon import normalize_equipment_tier, EQUIPMENT_TIERS
    _tier = normalize_equipment_tier(selection.get("equipment", "bar"))
    _max_abv = EQUIPMENT_TIERS[_tier]["max_abv_pct"] / 100.0
    if target_abv > _max_abv:
        target_abv = _max_abv

    if not spirits and not any(
        (i.get("props") or {}).get("abv_pct", 0) > 0.2 for i in modifiers
    ):
        # Spirit-free: delegate to dedicated mocktail solver (no LLM fallback)
        return _calculate_balance_spirit_free(
            ingredients, serve_style, target_vol, sweet_acid, DILUTION
        )

    # Style flags
    is_spirit_forward = not acids and not sweeteners and not dilutants
    is_sour           = bool(acids and sweeteners and serve_style in ("shaken", "blended"))

    v_liquid = target_vol / (1 + DILUTION)

    primary = spirits[0] if spirits else modifiers[0]
    p_props = primary.get("props") or {}
    p_abv   = p_props.get("abv_pct", 0.40)
    if p_abv <= 0:
        return None

    # ── B13 fix: primary-modifier-as-spirit detection ──────────────────────
    # When spirits=[], the first high-ABV modifier (e.g. Green Chartreuse 55%,
    # Aperol 11%) acts as the de-facto base spirit. It must be sized by the ABV
    # equation (v_spirit), not by the accent-modifier estimate (v_mod_est).
    # Without this, chartreuse/lime/agave would list chartreuse at only ~15 ml
    # (v_mod_est) while the ABV numerator counted 52 ml — a 3× mass-balance fraud.
    _primary_is_modifier = not spirits and bool(modifiers)
    _accent_mods = modifiers[1:] if _primary_is_modifier else modifiers

    # ── Modifier volume estimate for ABV solve (accent modifiers only) ─────
    # Spirit-forward (Negroni/Manhattan): equal-parts split among spirit + accents.
    # Sour/Highball: accents are ~15% total, capped at 25%.
    if _accent_mods:
        if is_spirit_forward:
            n_parts = 1 + len(_accent_mods)
            v_mod_est = v_liquid * len(_accent_mods) / n_parts
        else:
            v_mod_est = min(0.15 * v_liquid * len(_accent_mods), 0.25 * v_liquid)
    else:
        v_mod_est = 0.0

    # Only accent modifiers feed the pre-estimate; primary modifier IS the spirit
    mod_abv_contrib = sum(
        (v_mod_est / len(_accent_mods)) * (m.get("props") or {}).get("abv_pct", 0)
        for m in _accent_mods
    ) if _accent_mods else 0.0

    # ── Spirit volume (ABV equation) ───────────────────────────────────────
    v_spirit = (target_vol * target_abv - mod_abv_contrib) / p_abv
    spirit_cap = 0.55 if is_spirit_forward else 0.68
    v_spirit = max(min(v_spirit, v_liquid * spirit_cap), 10.0)

    # ── Final accent modifier volume (after knowing v_spirit) ─────────────
    if _accent_mods:
        if is_spirit_forward:
            v_mod = max(v_liquid - v_spirit, 10.0 * len(_accent_mods))
        else:
            v_mod = v_mod_est
    else:
        v_mod = 0.0

    # ── Acid / Sweet allocation ────────────────────────────────────────────
    v_remain = max(v_liquid - v_spirit - v_mod, 0.0)

    if acids and sweeteners:
        a_props = acids[0].get("props") or {}
        s_props = sweeteners[0].get("props") or {}
        a_ta   = a_props.get("ta_pct", 5.0)
        a_dens = a_props.get("density", 1.03)
        s_brix = s_props.get("brix", 50.0)
        s_dens = s_props.get("density", 1.18)
        if a_ta > 0 and s_brix > 0:
            k = sweet_acid * a_ta * a_dens / (s_brix * s_dens)
            v_acid  = v_remain / (1 + k) if v_remain > 0 else 15.0
            v_sweet = v_remain - v_acid  if v_remain > 0 else 15.0
        else:
            v_acid, v_sweet = v_remain * 0.55, v_remain * 0.45
        # Sweetener sensory floor (concentrated syrups can algebraically → <5 ml)
        _MIN_SWEET_ML = 8.0
        if v_sweet < _MIN_SWEET_ML and v_remain > _MIN_SWEET_ML + 5:
            v_sweet = _MIN_SWEET_ML
            v_acid  = v_remain - v_sweet
        # Sour commercial ratio guard: spirit:acid ≥ 2:1 (classic 60:22 ≈ 2.7:1)
        # Enforce minimum acid ≥ spirit / 3.0 to avoid watery-sour / under-acidic output.
        if is_sour and v_spirit > 0:
            min_acid = max(v_spirit / 3.0, 15.0)
            if v_acid < min_acid:
                shortfall = min_acid - v_acid
                v_acid    = min_acid
                v_sweet   = max(v_sweet - shortfall * 0.4, _MIN_SWEET_ML)
    elif acids:
        v_acid, v_sweet = v_remain, 0.0
    elif sweeteners:
        v_acid, v_sweet = 0.0, v_remain
    else:
        # Spirit-forward (Negroni/Manhattan) — no ghost volume (bug-fix)
        v_acid, v_sweet = 0.0, 0.0

    # ── Bitters: dashes scale with serve style (1 ml ≈ 1 dash) ──────────
    # spirit-forward = 3 dashes · sour = 2 dashes · highball/built = 1 dash
    _BITTERS_DASHES = 3 if is_spirit_forward else (2 if is_sour else 1)
    _BITTERS_DASH_ML = 1.0  # 1 dash ≈ 1 ml (standard jigger dash)
    bitters_abv_contrib = sum(
        _BITTERS_DASHES * _BITTERS_DASH_ML * (b.get("props") or {}).get("abv_pct", 0.44)
        for b in bitters
    )
    bitters_vol = _BITTERS_DASHES * _BITTERS_DASH_ML * len(bitters)

    # ── Highball dilutant: quantify soda/water volume (bug-fix) ──────────
    v_dilutant = 0.0
    if dilutants:
        other_vol = v_spirit + v_mod + v_acid + v_sweet + bitters_vol
        # Soda fills to target_vol; minimum 30 ml for perceptible effervescence
        v_dilutant = max(target_vol - other_vol, 30.0)

    # ── Final stats ───────────────────────────────────────────────────────
    total_abv_numerator = v_spirit * p_abv + mod_abv_contrib + bitters_abv_contrib
    total_liquid        = v_spirit + v_mod + v_acid + v_sweet + bitters_vol + v_dilutant
    # Built drinks with explicit dilutant: the dilutant IS the volume expansion.
    # Do NOT divide by (1+DILUTION) again — that causes double-dilution on ABV
    # and inflates total volume by ~10% (G&T 180ml → 198ml bug).
    _is_built_with_dilutant = serve_style == "built" and bool(dilutants)
    if total_liquid > 0:
        if _is_built_with_dilutant:
            actual_abv_final = total_abv_numerator / total_liquid
        else:
            actual_abv_final = (total_abv_numerator / total_liquid) / (1 + DILUTION)
    else:
        actual_abv_final = target_abv

    a_props2 = (acids[0].get("props") or {}) if acids else {}
    s_props2 = (sweeteners[0].get("props") or {}) if sweeteners else {}
    total_acid_g = (
        v_acid * a_props2.get("ta_pct", 5.0) * a_props2.get("density", 1.03) / 100
        if acids and v_acid > 0 else 0
    )
    total_sugar_g = (
        v_sweet * s_props2.get("brix", 50.0) * s_props2.get("density", 1.18) / 100
        if sweeteners and v_sweet > 0 else 0
    )

    def _split_volume(total: float, items: list, min_ml: int = 5) -> list[int]:
        """Distribute total ml evenly across items; add rounding remainder to first item."""
        if not items:
            return []
        n = len(items)
        each = max(round(total / n), min_ml)
        amounts = [each] * n
        # Correct rounding error: adjust first element so sum == round(total)
        target = round(total)
        diff = target - sum(amounts)
        amounts[0] = max(amounts[0] + diff, min_ml)
        return amounts

    # ── bitters dashes: scale with serve style ──────────────────────────────
    if is_spirit_forward:
        _bitters_dashes = 3   # Manhattan, Old Fashioned: bold
    elif is_sour:
        _bitters_dashes = 2   # Whiskey Sour: classic 2 dashes
    else:
        _bitters_dashes = 1   # Highball / built: light accent

    result_ings: list[dict] = []
    if spirits:
        for s, amt in zip(spirits, _split_volume(v_spirit, spirits, 5)):
            result_ings.append({"amount": f"{amt} ml", "name": s["name"], "role": "base"})
    elif _primary_is_modifier:
        # High-ABV modifier acting as base spirit: use algebraically solved v_spirit volume
        result_ings.append({
            "amount": f"{round(v_spirit)} ml",
            "name": modifiers[0]["name"],
            "role": "modifier",
        })
    if _accent_mods:
        for m, amt in zip(_accent_mods, _split_volume(v_mod, _accent_mods, 5)):
            result_ings.append({"amount": f"{amt} ml", "name": m["name"], "role": "modifier"})
    if acids:
        for a, amt in zip(acids, _split_volume(v_acid, acids, 5)):
            result_ings.append({"amount": f"{amt} ml", "name": a["name"], "role": "acid"})
    if sweeteners:
        for sw, amt in zip(sweeteners, _split_volume(v_sweet, sweeteners, 5)):
            result_ings.append({"amount": f"{amt} ml", "name": sw["name"], "role": "sweetener"})
    if dilutants:
        for d, amt in zip(dilutants, _split_volume(v_dilutant, dilutants, 30)):
            result_ings.append({"amount": f"{amt} ml", "name": d["name"], "role": "dilutant"})
    if bitters:
        for b in bitters:
            result_ings.append({"amount": f"{_bitters_dashes} dashes", "name": b["name"], "role": "bitters"})

    style_tag = f"{serve_style}·dil={round(DILUTION * 100)}%"
    # Built+dilutant: final volume = liquid sum (no extra expansion)
    # Shaken/stirred: final volume = liquid * (1 + dilution_factor)
    _final_vol = total_liquid if _is_built_with_dilutant else total_liquid * (1 + DILUTION)
    balance_out = {
        "ingredients":     result_ings,
        "final_abv_pct":   round(actual_abv_final * 100, 1),
        "total_acid_g":    round(total_acid_g, 2),
        "total_sugar_g":   round(total_sugar_g, 2),
        "ph_estimate":     _estimate_ph(total_acid_g, _final_vol),
        "total_volume_ml": round(_final_vol),
        "balance_notes": (
            f"[DETERMINISTIC ENGINE · {style_tag}] "
            f"Spirit {round(v_spirit)} ml · Mod {round(v_mod)} ml · "
            f"Acid {round(v_acid)} ml · Sweet {round(v_sweet)} ml"
            + (f" · Dilutant {round(v_dilutant)} ml" if dilutants else "")
            + (f" · Bitters {round(bitters_vol)} ml" if bitters else "")
        ),
        "serve_style":     serve_style,
        "_engine": "deterministic",
    }
    return attach_metrology(balance_out, selection)


def _build_locked_table_markdown(balance: dict, language: str = "en") -> tuple[str, str]:
    """Return (ingredients_table_md, chemical_table_md) without section headers."""
    zh = (language == "zh")
    if zh:
        ing_header, ing_sep = "| 用量 | 原料 | 角色 |", "|------|------|------|"
        chem_header, chem_sep = "| 参数 | 目标值 | 计算依据 |", "|------|--------|----------|"
    else:
        ing_header, ing_sep = "| Amount | Ingredient | Role |", "|--------|-----------|------|"
        chem_header, chem_sep = "| Parameter | Target | Rationale |", "|-----------|--------|-----------|"

    ing_rows = [
        f"| {ing.get('amount', '')} | {ing.get('name', '')} | {ing.get('role', '')} |"
        for ing in balance.get("ingredients", [])
    ]
    ing_table = "\n".join([ing_header, ing_sep] + ing_rows)

    chem_rows = []
    if balance.get("final_abv_pct") is not None:
        label = "最终 ABV" if zh else "ABV"
        chem_rows.append(
            f"| {label} | {balance['final_abv_pct']:.1f}% | 确定性引擎代数求解 |" if zh
            else f"| {label} | {balance['final_abv_pct']:.1f}% | Deterministic algebraic engine |"
        )
    if balance.get("total_acid_g") is not None:
        label = "总酸量" if zh else "Total acid"
        chem_rows.append(
            f"| {label} | {balance['total_acid_g']:.2f} g | 柠檬酸当量 |" if zh
            else f"| {label} | {balance['total_acid_g']:.2f} g | citric acid equiv. |"
        )
    if balance.get("total_sugar_g") is not None:
        label = "总糖量" if zh else "Total sugar"
        chem_rows.append(
            f"| {label} | {balance['total_sugar_g']:.1f} g | 糖度(Brix)换算 |" if zh
            else f"| {label} | {balance['total_sugar_g']:.1f} g | from sweetener Brix |"
        )
    if balance.get("ph_estimate") is not None:
        label = "pH 估算" if zh else "pH estimate"
        chem_rows.append(f"| {label} | {balance['ph_estimate']:.1f} | Henderson-Hasselbalch |")
    if balance.get("total_volume_ml") is not None:
        label = "成杯体积" if zh else "Final volume"
        chem_rows.append(
            f"| {label} | {balance['total_volume_ml']:.0f} ml | 含稀释量 |" if zh
            else f"| {label} | {balance['total_volume_ml']:.0f} ml | incl. dilution |"
        )
    if balance.get("gelatin_pct") is not None:
        label = "明胶浓度" if zh else "Gelatin conc."
        chem_rows.append(
            f"| {label} | {balance['gelatin_pct']:.2f}% | iSi 200-Bloom 上限 ≤1.0% |" if zh
            else f"| {label} | {balance['gelatin_pct']:.2f}% | iSi 200-Bloom limit ≤1.0% |"
        )
    if balance.get("balance_notes"):
        label = "引擎备注" if zh else "Engine note"
        chem_rows.append(f"| {label} | — | {balance['balance_notes']} |")

    chem_table = "\n".join([chem_header, chem_sep] + chem_rows)
    return ing_table, chem_table


def _replace_section_table(
    text: str, header_re: re.Pattern, table_md: str,
    fallback_header: str = "",
) -> str:
    """
    Replace the body of a markdown ### section with a locked table.
    If the section header is not found and fallback_header is provided,
    the table is appended at the start of the document rather than silently dropped.
    """
    m = header_re.search(text)
    if not m:
        if fallback_header:
            # Section missing entirely — prepend it so downstream audits can find it
            return f"{fallback_header}\n{table_md.strip()}\n\n" + text
        return text
    body_start = m.end()
    rest = text[body_start:]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    body_end = body_start + (next_hdr.start() if next_hdr else len(rest))
    # Two newlines ensure a blank line between the table and the next section header.
    # Without it, some markdown renderers treat the header as a table cell continuation.
    return text[:body_start] + "\n" + table_md.strip() + "\n\n" + text[body_end:]


def _inject_method_amounts(text: str, balance: "dict | None") -> str:
    """
    Inject locked ingredient amounts into Method steps (first mention per ingredient).

    R1 is instructed NOT to write ml amounts in Method prose; this function
    deterministically adds them back so every step references an exact quantity.
    Only injects when the ingredient name appears WITHOUT a preceding numeric amount.
    """
    if not balance or not text.strip():
        return text
    meth_m = _SECTION_METHOD.search(text)
    if not meth_m:
        return text
    meth_body_start = meth_m.end()
    rest = text[meth_body_start:]
    next_hdr = _KNOWN_SECTION_RE.search(rest)
    meth_body_end = meth_body_start + (next_hdr.start() if next_hdr else len(rest))
    body = text[meth_body_start:meth_body_end]

    for ing in balance.get("ingredients", []):
        name = ing.get("name", "").strip()
        amount = ing.get("amount", "").strip()
        if not name or not amount:
            continue

        # Build alias candidates: full name first, then significant tokens (>3 chars).
        # This handles R1 abbreviating "Bourbon Whiskey" → "bourbon", or
        # "Fresh Lime Juice" → "lime juice".
        tokens = [t for t in name.split() if len(t) > 3]
        candidates: list[str] = [name] + [t for t in tokens if t.lower() != name.lower()]

        # Skip if any candidate already has a nearby numeric amount — already injected.
        already = any(
            re.search(
                r"\d+\s*(?:ml|g|dash)\b[^.\n]{0,15}?" + re.escape(c) + r"\b"
                r"|\b" + re.escape(c) + r"[^.\n]{0,15}?\d+\s*(?:ml|g|dash)\b",
                body, re.IGNORECASE,
            )
            for c in candidates
        )
        if already:
            continue

        # Inject on first match of any candidate (full name preferred).
        for candidate in candidates:
            pat = re.compile(r"\b" + re.escape(candidate) + r"\b", re.IGNORECASE)
            new_body = pat.sub(
                lambda m_obj, _amt=amount: f"**{_amt}** {m_obj.group(0)}",
                body, count=1,
            )
            if new_body != body:
                body = new_body
                break

    return text[:meth_body_start] + body + text[meth_body_end:]


def _enforce_locked_tables(text: str, balance: "dict | None", language: str = "en") -> str:
    """
    Deterministically restore Stage 0.6 locked data into the recipe text:
      1. Ingredients table  — _replace_section_table
      2. Chemical Alignment table — _replace_section_table
      3. Method step amounts — _inject_method_amounts (new: first-mention injection)
    """
    if not balance or not text.strip():
        return text
    zh = (language == "zh")
    ing_table, chem_table = _build_locked_table_markdown(balance, language)
    # Fallback headers: if R1 omitted the section entirely, prepend rather than skip
    ing_fb  = "### 原料"            if zh else "### Ingredients"
    chem_fb = "### 化学参数对齐"     if zh else "### Chemical Alignment"
    text = _replace_section_table(text, _SECTION_INGREDIENTS, ing_table, fallback_header=ing_fb)
    text = _replace_section_table(text, _SECTION_CHEM_ALIGNMENT, chem_table, fallback_header=chem_fb)
    text = _inject_method_amounts(text, balance)
    return text


def format_balance_injection(balance: dict, language: str = "en") -> str:
    """
    Produce two pre-filled Markdown tables (ingredient table + chemical parameters table)
    to be injected verbatim into the R1 system prompt.

    Strategy: R1 is told to COPY these tables as-is into the appropriate ### sections.
    This eliminates LLM number hallucination at the source — numbers never touch R1.
    """
    zh = (language == "zh")
    ing_table, chem_table = _build_locked_table_markdown(balance, language)

    # ── Inject block with copy-verbatim instructions ──────────────────────────
    ing_sec  = "### 原料"            if zh else "### Ingredients"
    chem_sec = "### 化学参数对齐"     if zh else "### Chemical Alignment"
    eq_sec   = "### 设备"            if zh else "### Equipment"
    meth_sec = "### 制作方法"         if zh else "### Method"
    sci_sec  = "### 科学原理"         if zh else "### The Science"
    arch_sec = "### 建筑师笔记"       if zh else "### Architect's Note"

    lines = [
        "",
        "════ PRE-FILLED TABLES — DETERMINISTIC ENGINE (NON-NEGOTIABLE) ════════════",
        "The two tables below were computed by a Python algebraic engine using EFSA",
        "physical constants BEFORE recipe generation. Every number is locked.",
        "",
        f"▶ RULE 1 — Copy the {ing_sec} table below VERBATIM into your response.",
        "  Do NOT change any amount, name, or role.",
        "  You MAY append extra garnish/salt-rim rows at the bottom.",
        "",
        ing_sec,
        ing_table,
        "",
        f"▶ RULE 2 — Copy the {chem_sec} table below VERBATIM into your response.",
        "  Do NOT alter any value. This section is ALWAYS required.",
        "",
        chem_sec,
        chem_table,
        "",
        "▶ RULE 3 — Write ONLY these sections from scratch:",
        f"  • Recipe title + 1–2 sentence poetic concept",
        f"  • {eq_sec}  (list tools/equipment — no ml amounts needed)",
        f"  • {meth_sec}  ← CRITICAL: write steps as ACTIONS ONLY.",
        "    ✗ DO NOT write any ml/g/% amounts for ingredients in Method steps.",
        "    ✓ Refer to ingredients by NAME only: 'Add the bourbon', not 'Add 52 ml bourbon'.",
        "    ✓ Temperature (°C), time (min), technique parameters are fine.",
        "    → The Python engine will inject exact amounts automatically.",
        f"  • {sci_sec}",
        f"  • {arch_sec}",
        "",
        "VIOLATION = any ingredient ml/g amount in Method prose, OR any amount/ABV/acid/",
        "sugar/pH/volume in the tables that differs from the locked values above.",
        "Stage-2 audit WILL reject the recipe.",
        "════ END PRE-FILLED TABLES ══════════════════════════════════════════════════",
        "",
    ]
    return "\n".join(lines)


# ── Single pipeline entry (Full Precision only) ───────────────────────────

def dispatch_generator(body, api_key: str, perplexity_key: str):
    from rd_pipeline import generate_rd_pipeline, generate_classic_pipeline, generate_fast_pipeline
    recipe_mode = getattr(body, "recipe_mode", "creative")
    generation_mode = getattr(body, "mode", "precision")
    if recipe_mode == "classic":
        yield from generate_classic_pipeline(body, api_key)
    elif generation_mode == "fast":
        yield from generate_fast_pipeline(body, api_key)
    else:
        yield from generate_rd_pipeline(body, api_key, perplexity_key)


# ── Precision pipeline (full R1 + Perplexity) ─────────────────────────────

def generate_with_pipeline(body, api_key: str, perplexity_key: str):
    """
    Pipeline yielding SSE events:
      ① Generate  (deepseek-reasoner, non-stream)
      ② Hard filter (up to 2 retries)
      ③ Perplexity verify (sonar, built-in web search)
      ④ Recipe Surgeon — full rewrite fixing ALL audit issues (up to 2 attempts)
      ⑤ Stream final text to client
    Status events: {"status": "generating"|"verifying"|"correcting"|"streaming", "detail": "..."}
    """
    model = os.getenv("AI_MODEL", "deepseek-chat")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    lang_map = {"zh": "Please respond entirely in Chinese.", "en": "Please respond entirely in English."}
    lang_instruction = _lang_instruction(body.language)

    _strict_constraint = (
        "\n╔══════════════════════════════════════════════════════════════╗\n"
        "║  STRICT INGREDIENT CONSTRAINT — validation will REJECT      ║\n"
        "║  any ingredient not in the available list.                  ║\n"
        "║  Do NOT invent ingredients. Do NOT add extra ingredients.   ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

    _dilution_example = (
        "\n[DILUTION MODEL — reference only]\n"
        "  shaken  ≈ 22% dilution  →  60ml base + 30ml modifiers → ~110ml finished\n"
        "  stirred ≈ 17% dilution  →  60ml base + 20ml modifiers → ~94ml finished\n"
        "  built   ≈ 10% dilution  →  no additional ice melt factor\n"
        "  blended ≈ 25% dilution  →  60ml base + 90ml fruit → ~188ml finished\n"
        "Example: shaken 60ml rum + 30ml lime + 15ml syrup = 105ml → ~128ml final ABV.\n"
    )
    user_msg = f"""{lang_instruction}

Equipment level: {body.equipment}
Flavor profile targets: {', '.join(body.flavors) if body.flavors else 'No preference'}
Available ingredients: {body.ingredients}
Alcohol preference: {body.alcohol}
Additional requirements: {body.notes or 'None'}
{_strict_constraint}
{_dilution_example}
Design a complete recipe grounded in Liquid Intelligence, Meehan's Manual, Dead Rabbit, and The Liquid Codex — appropriate to this equipment level.
"""
    system_prompt = build_system_prompt(body.language, equipment=body.equipment)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    # ── Stage 0 · Input validation ────────────────────────────────────────
    yield sse({"status": "validating", "detail": "Checking input…"})
    validation = validate_input_with_ai(client, body.ingredients, body.notes, body.language)
    if not validation.get("valid", True):
        yield sse({"status": "invalid_input", "message": validation.get("reason", "Please specify clear ingredients or a flavour concept.")})
        return

    # ── Stage 0.5 · Ingredient selection (LLM categorize only) ──────────────
    yield sse({"status": "generating", "detail": "Selecting ingredients…"})
    selection = select_ingredients(client, body, perplexity_key=perplexity_key)

    # A2: Coverage verification — every user-listed ingredient must be selected
    if selection:
        missing_coverage = _verify_ingredient_coverage(body, selection)
        if missing_coverage:
            yield sse({
                "status": "generating",
                "detail": f"Coverage retry — must include: {', '.join(missing_coverage[:3])}…",
            })
            coverage_note = (
                f"REQUIRED — you MUST include ALL of these ingredients in your selection: "
                f"{', '.join(missing_coverage)}. "
                f"Do not omit any ingredient the user listed.\n"
            )
            selection_retry = select_ingredients(
                client, body,
                perplexity_key=perplexity_key,
                extra_constraint=coverage_note,
            )
            if selection_retry:
                selection = selection_retry

    if not selection:
        yield sse({
            "error": "Ingredient selection failed — please list clearer ingredients or constraints.",
        })
        return

    # ── Stage 0.55 + 0.6 · Classic / mocktail anchor → deterministic balance ─────
    _spirit_free = not selection_has_base_spirit(selection)
    yield sse({
        "status": "generating",
        "detail": "Mocktail anchor matching…" if _spirit_free else "Classic anchor matching…",
    })
    if _spirit_free:
        _anchor_preview, _anchor_score = match_mocktail_anchor(selection)
    else:
        _anchor_preview, _anchor_score = match_classic_anchor(selection)
    balance = resolve_balance(selection, calculate_balance_deterministic)
    if balance and balance.get("_engine") in ("classic-anchor", "mocktail-anchor"):
        _an = balance.get("_anchor", {})
        yield sse({
            "status": "generating",
            "detail": (
                f"Anchored to {_an.get('name', 'classic')} "
                f"(score {_an.get('match_score', _anchor_score):.2f})…"
            ),
        })
    else:
        yield sse({"status": "generating", "detail": "Deterministic balance engine…"})

    if balance is None:
        yield sse({
            "error": (
                "Balance engine could not solve this ingredient combination. "
                "Try adjusting ingredients, ABV target, or serve style."
            ),
        })
        return

    # Rebuild system prompt with targeted EFSA data for selected ingredients
    selected_names = [i["name"] for i in selection.get("ingredients", [])]
    system_prompt = build_system_prompt(
        body.language,
        selected_names or None,
        balance,
        equipment=body.equipment,
        spirit_free=_spirit_free or float(selection.get("target_abv_pct", 99)) < 1.0,
    )
    balance_str = format_balance_injection(balance, language=body.language)
    system_prompt_locked = system_prompt + balance_str
    messages = [
        {"role": "system", "content": system_prompt_locked},
        {"role": "user", "content": user_msg},
    ]
    engine_tag = balance.get("_engine", "deterministic")
    yield sse({
        "status": "generating",
        "detail": f"Balance locked [{engine_tag}] — starting R1 generation…",
    })
    if balance.get("_metrology"):
        yield sse({"metrology": balance["_metrology"]})

    # ── Stage ① Generate ──────────────────────────────────────────────────
    yield sse({"status": "generating", "detail": f"R1 reasoning ({model})…"})

    recipe_text = ""
    filter_ok = False
    filter_reason = ""

    for attempt in range(3):  # initial attempt + up to 2 retries
        if attempt > 0:
            retry_msg = (
                f"Your previous attempt failed the hard filter: {filter_reason}. "
                "Regenerate the COMPLETE recipe from scratch using EXACTLY these section headers:\n"
                "### 原料 (or ### Ingredients) — copy the PRE-FILLED TABLE from system prompt VERBATIM\n"
                "### 设备 (or ### Equipment)\n"
                "### 化学参数对齐 (or ### Chemical Alignment) — copy the PRE-FILLED TABLE from system prompt VERBATIM; ALWAYS required when balance is pre-computed\n"
                "### 制作方法 (or ### Method)\n"
                "### 科学原理 (or ### The Science)\n"
                "### 建筑师笔记 (or ### Architect's Note)\n"
                "The PRE-FILLED TABLES in the system prompt contain the exact ingredient amounts and chemical parameters — copy them verbatim, do NOT invent different numbers.\n"
                "Each step MUST use **Step N —** or **步骤 N —** format (sequential, no gaps). "
                "NEVER use plain '2.' numbered lists for method steps. "
                "CRITICAL: do NOT write ingredient ml/g amounts in Method steps — use ingredient "
                "names only ('Add the bourbon', not 'Add 52 ml bourbon'). "
                "Temperature (°C) and time (min) in steps are fine. "
                "Equipment section: one short bullet per tool — NEVER put method steps inside Equipment. "
                "NEVER heat gin/vodka/whisky together with liquid to ≥79°C — hydrate in juice/water first, "
                "cool below 60°C, then add spirit. "
                "Agar freeze-thaw: thaw ONLY at 4°C in a refrigerator, never at room temperature. "
                "Reverse spherification: buffer low-pH juice with sodium citrate to pH 4.0–4.5; "
                "dilute high-ABV core (e.g. 55% Chartreuse) to 15–20% ABV before calcium lactate."
            )
            messages = [
                {"role": "system", "content": system_prompt_locked},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": recipe_text},
                {"role": "user", "content": retry_msg},
            ]
            yield sse({"status": "generating", "detail": f"Retry {attempt}/2 — fixing structure…"})

        try:
            label = f"R1 reasoning ({model})" if attempt == 0 else f"Retry {attempt}/2"
            thread, result_q = _start_thread(generate_once, client, model, messages)
            for hb in _heartbeats_while_running(thread, "generating", label):
                yield hb
            recipe_text = _join_thread_result(thread, result_q)
            recipe_text = normalize_recipe_text(recipe_text)
            # B23: deterministically restore pre-filled tables if R1 corrupted them
            recipe_text = _enforce_locked_tables(recipe_text, balance, body.language)
        except Exception as exc:
            exc_diag = str(exc)[:220]
            logger.error("Stage ① generation exception attempt=%d: %s", attempt, exc_diag)
            _append_error_audit(
                [i.get("name", "") for i in (selection or {}).get("ingredients", [])],
                [f"[generation_exception] {exc_diag}"],
            )
            if attempt >= 2:
                yield sse({"error": f"Generation failed after retries: {exc_diag[:100]}"})
                return
            # Gear 2: exception becomes a diagnostic that feeds the NEXT retry
            filter_reason = (
                f"generation error on attempt {attempt + 1}: {exc_diag[:150]}. "
                "Shorten your response — aim for ≤1200 words. Omit extended reasoning chains."
            )
            yield sse({
                "status": "generating",
                "detail": f"Exception on attempt {attempt + 1} — diagnostic retry…",
            })
            continue  # do NOT break; next iteration rebuilds messages with filter_reason

        # ── Stage ② Hard filter ───────────────────────────────────────────
        filter_ok, filter_reason = hard_filter(recipe_text, locked_balance=balance)
        if filter_ok:
            break
        yield sse({"status": "generating", "detail": f"Filter failed ({filter_reason[:60]}), retrying…"})

    if not filter_ok:
        yield sse({
            "status": "generating",
            "detail": "Structure fallback — deepseek-chat strict format pass…",
        })
        fmt_hint = (
            "\n\nCRITICAL OUTPUT FORMAT: You MUST include ALL sections with ### headers:\n"
            "### 原料 (pipe table) · ### 设备 · ### 化学参数对齐 (if lab technique) · "
            "### 制作方法 · ### 科学原理 · ### 建筑师笔记"
            if body.language == "zh"
            else "\n\nCRITICAL: Include ### Ingredients (pipe table), ### Equipment, "
            "### Chemical Alignment (if needed), ### Method, ### The Science, ### Architect's Note"
        )
        try:
            fallback_messages = [
                {"role": "system", "content": messages[0]["content"]},
                {"role": "user", "content": user_msg + fmt_hint},
            ]
            recipe_text = generate_once(client, "deepseek-chat", fallback_messages)
            filter_ok, filter_reason = hard_filter(recipe_text, locked_balance=balance)
        except Exception:
            pass

    if not filter_ok:
        yield sse({
            "error": (
                "Recipe structure failed quality checks after retries. "
                f"{filter_reason[:200]}"
            ),
        })
        return

    # ── Stage 2.5 + Stage ③ · Parallel audit ────────────────────────────────
    # Both the Codex Judge (deepseek-chat, 8 volumes) and Perplexity sonar
    # are I/O-bound. Fire them simultaneously; single heartbeat loop waits
    # for both — saves 3-6 s vs sequential execution.
    yield sse({"status": "verifying", "detail": "Parallel audit: Codex judge (Vol. I–VIII) + Perplexity sonar…"})
    judge_thread, judge_q = _start_thread(codex_compliance_judge, recipe_text, client)
    if perplexity_key:
        v_thread, v_q = _start_thread(verify_recipe, recipe_text, perplexity_key)
    else:
        v_thread, v_q = None, None

    # Shared heartbeat loop until BOTH threads finish
    elapsed_hb = 0
    yield sse({"status": "verifying", "detail": "Parallel audit running… (0s)"})
    while judge_thread.is_alive() or (v_thread and v_thread.is_alive()):
        time.sleep(5)
        elapsed_hb += 5
        active = []
        if judge_thread.is_alive():
            active.append("Codex judge")
        if v_thread and v_thread.is_alive():
            active.append("Perplexity")
        yield sse({"status": "verifying", "detail": f"Parallel audit: {' + '.join(active)} ({elapsed_hb}s)…"})
        yield ": keepalive\n\n"

    judge_result = _join_thread_result(judge_thread, judge_q)
    judge_score = float(judge_result.get("overall_score", 10.0))
    judge_violations = [v for v in judge_result.get("violations", []) if isinstance(v, str) and v.strip()]
    judge_highlights = judge_result.get("highlights", [])
    vol_scores = judge_result.get("vol_scores", {})
    low_vols = [k for k, v in vol_scores.items() if isinstance(v, (int, float)) and v < 7]
    vol_labels = {"vol1":"I","vol2":"II","vol3":"III","vol4":"IV","vol5":"V","vol6":"VI","vol7":"VII","vol8":"VIII"}
    detail_msg = f"Codex judge: {judge_score:.1f}/10"
    if judge_violations:
        detail_msg += f" — {len(judge_violations)} issue(s) (weak: Vol.{', '.join(vol_labels.get(v,'?') for v in low_vols)})" if low_vols else f" — {len(judge_violations)} issue(s)"
    elif judge_highlights:
        detail_msg += " ✓"
    yield sse({"status": "verifying", "detail": detail_msg,
               "judge_score": judge_score,
               "judge_vol_scores": vol_scores,
               "judge_highlights": judge_highlights[:3]})

    if perplexity_key:
        verification = _join_thread_result(v_thread, v_q)

        # Surface Perplexity errors — if key is invalid/quota-exceeded the fallback
        # silently returns all-PASS; make that visible so it is not confused with a real review.
        pplx_skipped = any(
            "Verification skipped" in str(i) for i in verification.get("safety_issues", [])
        )
        if pplx_skipped:
            yield sse({
                "status": "verifying",
                "detail": "⚠ Perplexity verification failed (key invalid / quota exceeded) — local audit only.",
            })

        safety_fail = verification.get("safety_verdict") == "FAIL" and verification.get("safety_issues")
        culinary_fail = verification.get("culinary_verdict") == "FAIL" and verification.get("culinary_issues")

        # ── Stage ④ Recipe Surgeon — merge Judge + Perplexity + local audit ─────
        all_issues: list[str] = []
        # Codex judge violations (Stage 2.5) — only include if score < 8
        if judge_violations and judge_score < 8.0:
            all_issues += [f"[Codex judge/Vol.] {v}" for v in judge_violations]
        if safety_fail:
            all_issues += [f"[Perplexity/safety] {i}" for i in verification["safety_issues"]]
        if culinary_fail:
            all_issues += [f"[Perplexity/technique] {i}" for i in verification["culinary_issues"]]
        all_issues.extend(collect_audit_issues(
            recipe_text, locked_balance=balance,
            flavors=body.flavors, ingredients_input=body.ingredients,
        ))

        if all_issues:
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_issues = []
            for item in all_issues:
                if item not in seen:
                    seen.add(item)
                    unique_issues.append(item)

            # Fire-and-forget: log this correction event so recurring patterns can be
            # promoted into science_guardrails manually (zero-cost knowledge backflow).
            _ingredients_snapshot = [
                i.get("name", "") for i in (selection or {}).get("ingredients", [])
            ]
            threading.Thread(
                target=_append_error_audit,
                args=(_ingredients_snapshot, unique_issues),
                daemon=True,
            ).start()

            surgeon_model = model if "reasoner" in model else os.getenv("AI_MODEL", "deepseek-reasoner")
            if "reasoner" not in surgeon_model:
                surgeon_model = "deepseek-reasoner"

            revised_ok = False
            for surg_attempt in range(2):
                yield sse({
                    "status": "correcting",
                    "detail": (
                        f"Recipe Surgeon rewriting ({len(unique_issues)} issue(s))"
                        + (f" — attempt {surg_attempt + 1}/2" if surg_attempt else "")
                        + "…"
                    ),
                })
                try:
                    s_thread, s_q = _start_thread(
                        surgeon_revise_recipe,
                        client,
                        surgeon_model,
                        user_msg,
                        recipe_text,
                        unique_issues,
                        system_prompt,
                        balance,
                        body.language,
                    )
                    for hb in _heartbeats_while_running(s_thread, "correcting", "Recipe Surgeon"):
                        yield hb
                    revised = _join_thread_result(s_thread, s_q)
                    revised = _enforce_locked_tables(revised, balance, body.language)
                except Exception as exc:
                    yield sse({"status": "correcting", "detail": f"Surgeon pass failed ({exc})."})
                    break

                if not revised.strip():
                    continue

                corr_ok, corr_reason = hard_filter(revised, locked_balance=balance)
                if corr_ok:
                    recipe_text = revised
                    revised_ok = True
                    yield sse({"status": "correcting", "detail": "Recipe Surgeon — all audits passed."})
                    break

                unique_issues = collect_audit_issues(
                    revised, locked_balance=balance,
                    flavors=body.flavors, ingredients_input=body.ingredients,
                ) or [f"[filter] {corr_reason}"]
                recipe_text = revised
                yield sse({
                    "status": "correcting",
                    "detail": f"Surgeon draft still has issues ({corr_reason[:50]}), retrying…",
                })

            if not revised_ok and recipe_text.strip():
                yield sse({
                    "status": "correcting",
                    "detail": (
                        f"Surgeon streaming best effort. "
                        f"({unique_issues[0][:80] if unique_issues else 'ok'})"
                    ),
                })
    else:
        yield sse({"status": "verifying", "detail": "No PERPLEXITY_API_KEY — skipping safety & technique check."})
        # Even without Perplexity, run Surgeon if Codex judge found issues
        if judge_violations and judge_score < 8.0:
            local_issues = [f"[Codex judge] {v}" for v in judge_violations]
            local_issues.extend(collect_audit_issues(
                recipe_text, locked_balance=balance,
                flavors=body.flavors, ingredients_input=body.ingredients,
            ))
            if local_issues:
                seen2: set[str] = set()
                unique_local: list[str] = []
                for item in local_issues:
                    if item not in seen2:
                        seen2.add(item)
                        unique_local.append(item)
                surgeon_model2 = model if "reasoner" in model else "deepseek-reasoner"
                yield sse({"status": "correcting",
                           "detail": f"Codex Surgeon — fixing {len(unique_local)} issue(s)…"})
                try:
                    s2_thread, s2_q = _start_thread(
                        surgeon_revise_recipe, client, surgeon_model2,
                        user_msg, recipe_text, unique_local, system_prompt,
                        balance, body.language,
                    )
                    for hb in _heartbeats_while_running(s2_thread, "correcting", "Codex Surgeon"):
                        yield hb
                    revised2 = _join_thread_result(s2_thread, s2_q)
                    revised2 = _enforce_locked_tables(revised2, balance, body.language)
                    if revised2.strip():
                        rev_ok, _ = hard_filter(revised2, locked_balance=balance)
                        if rev_ok:
                            recipe_text = revised2
                            yield sse({"status": "correcting", "detail": "Codex Surgeon — passed."})
                except Exception:
                    pass

    # ── Quality Gate ──────────────────────────────────────────────────────
    # If initial Codex Judge score was below the floor, keep looping:
    #   re-score current recipe_text → if still low, run targeted Surgeon pass → repeat.
    # Max QUALITY_GATE_MAX_PASSES extra cycles. Fail-open: any exception breaks the loop.
    if QUALITY_GATE_FLOOR > 0 and judge_score < QUALITY_GATE_FLOOR:
        _qg_score = judge_score
        for _qg_pass in range(QUALITY_GATE_MAX_PASSES):
            # ── Re-score ────────────────────────────────────────────────
            yield sse({
                "status": "correcting",
                "detail": (
                    f"Quality gate — re-scoring pass {_qg_pass + 1}/{QUALITY_GATE_MAX_PASSES} "
                    f"(current {_qg_score:.1f}/10, target {QUALITY_GATE_FLOOR:.0f}.0)…"
                ),
            })
            try:
                _qg_jt, _qg_jq = _start_thread(codex_compliance_judge, recipe_text, client)
                for _hb in _heartbeats_while_running(_qg_jt, "correcting",
                                                     f"Quality gate {_qg_pass + 1}"):
                    yield _hb
                _qg_result = _join_thread_result(_qg_jt, _qg_jq)
            except Exception:
                break  # fail-open
            _qg_score = float(_qg_result.get("overall_score", 10.0))
            _qg_viols = [v for v in _qg_result.get("violations", [])
                         if isinstance(v, str) and v.strip()]

            _pass_label = "✓ passed" if _qg_score >= QUALITY_GATE_FLOOR else f"below {QUALITY_GATE_FLOOR:.0f}.0 — rewriting…"
            yield sse({
                "status": "correcting",
                "detail": f"Quality gate {_qg_pass + 1}: {_qg_score:.1f}/10 {_pass_label}",
                "judge_score": _qg_score,
                "judge_vol_scores": _qg_result.get("vol_scores", {}),
            })

            if _qg_score >= QUALITY_GATE_FLOOR:
                break

            # ── Targeted Surgeon rewrite ─────────────────────────────
            _qg_issues: list[str] = (
                [f"[Quality Gate] {v}" for v in _qg_viols]
                if _qg_viols
                else [
                    f"Overall Codex score {_qg_score:.1f}/10 must reach {QUALITY_GATE_FLOOR:.0f}.0. "
                    "Deepen chemical rationale (Vol. III-IV), enrich sensory narrative (Vol. I-II), "
                    "sharpen technique precision (Vol. V-VI), elevate Architect's Note (Vol. VII-VIII)."
                ]
            )
            _qg_issues.extend(collect_audit_issues(
                recipe_text, locked_balance=balance,
                flavors=body.flavors, ingredients_input=body.ingredients,
            ))

            try:
                _qg_st, _qg_sq = _start_thread(
                    surgeon_revise_recipe, client, "deepseek-reasoner",
                    user_msg, recipe_text, _qg_issues, system_prompt,
                    balance, body.language,
                )
                for _hb in _heartbeats_while_running(_qg_st, "correcting",
                                                     f"Quality Surgeon {_qg_pass + 1}"):
                    yield _hb
                _qg_revised = _join_thread_result(_qg_st, _qg_sq)
                _qg_revised = _enforce_locked_tables(_qg_revised, balance, body.language)
                if _qg_revised.strip():
                    _qg_fok, _ = hard_filter(_qg_revised, locked_balance=balance)
                    if _qg_fok:
                        recipe_text = _qg_revised
            except Exception as _qg_exc:
                yield sse({"status": "correcting",
                           "detail": f"Quality Surgeon {_qg_pass + 1} error — proceeding with current draft."})
                break

    # ── Stage ⑤ Stream final text ─────────────────────────────────────────
    try:
        recipe_text = _maybe_inject_pre_treatment_steps(recipe_text, body.language)
        if not recipe_text.strip():
            yield sse({"error": "Recipe rendering produced empty text. Please try again."})
            return
        final_ok, final_reason = hard_filter(recipe_text, locked_balance=balance)
        if not final_ok:
            yield sse({
                "error": (
                    "Recipe did not pass final audit — not streamed. "
                    f"{final_reason[:220]}"
                ),
            })
            return
        yield sse({"status": "streaming", "detail": "Streaming recipe…"})

        chunk_size = 80
        for i in range(0, len(recipe_text), chunk_size):
            yield sse({"text": recipe_text[i : i + chunk_size]})

        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.error("Stage 5 stream failed: %s", exc, exc_info=True)
        yield sse({"error": f"Recipe streaming failed: {str(exc)[:120]}"})
        return


# ── Models ────────────────────────────────────────────────────────────────

class RecipeRequest(BaseModel):
    flavors: list[str]
    techniques: list[str] = []
    equipment: str
    ingredients: str
    alcohol: str
    language: str
    notes: Optional[str] = ""
    occasion: Optional[str] = ""
    mode: str = "precision"  # legacy field — always Full Precision pipeline
    recipe_mode: str = "creative"  # "creative" | "classic"


class MetrologyReconcileRequest(BaseModel):
    """Bench measurement backfill — no UI required; for lab workflow / future clients."""
    balance: dict
    measurements: dict[str, float]  # slot_id → measured value


# ── Route ─────────────────────────────────────────────────────────────────

@router.post("/metrology/reconcile")
def metrology_reconcile(body: MetrologyReconcileRequest):
    """
    Apply measured bench values to a locked balance and return deltas.
    Does not mutate stored recipes — stateless reconciliation for lab workflows.
    """
    if not body.balance:
        raise HTTPException(status_code=400, detail="balance object required")
    updated = reconcile_measurements(dict(body.balance), body.measurements)
    return {
        "balance": updated,
        "metrology": updated.get("_metrology", {}),
    }


@router.post("/stream")
def stream_recipe(body: RecipeRequest, x_api_key: Optional[str] = Header(default=None)):
    # Railway server key takes priority (MVP shared-key mode)
    api_key = os.getenv("DEEPSEEK_API_KEY", "") or (x_api_key or "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI engine not configured — provide your DeepSeek API key or contact the administrator",
        )
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")

    return StreamingResponse(
        dispatch_generator(body, api_key, perplexity_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
