"""
Liquid Architect — One-Time Ingredient Enrichment Script
=========================================================
Run once (offline) to expand INGREDIENT_PROPS with Perplexity-verified data.

Usage:
    cd E:/a bar/mixologist
    set PERPLEXITY_API_KEY=pplx-...
    python backend/scripts/enrich_ingredients.py

Output:
    backend/scripts/enriched_props.json   — raw results (review before merging)
    backend/scripts/ingredient_props_patch.py — Python snippet ready to paste
"""

import os
import re
import sys
import json
import time
import math
import pathlib

# Allow imports from backend/
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from openai import OpenAI
from data_codex import MATERIALS

# ── Current INGREDIENT_PROPS keys (copied from ai.py) ─────────────────────
EXISTING_KEYS = {
    "gin", "london dry gin", "hendricks", "bombay sapphire", "tanqueray",
    "vodka", "tequila", "tequila blanco", "tequila reposado", "mezcal",
    "rum", "white rum", "dark rum", "aged rum", "bourbon", "rye whiskey",
    "scotch", "whisky", "whiskey", "irish whiskey", "cognac", "brandy",
    "calvados", "pisco", "absinthe", "grappa", "soju", "baijiu", "sake",
    "shochu", "cointreau", "triple sec", "grand marnier", "campari",
    "aperol", "green chartreuse", "yellow chartreuse", "st-germain",
    "kahlua", "baileys", "maraschino", "creme de cassis", "falernum",
    "amaro", "fernet branca", "averna", "sweet vermouth", "dry vermouth",
    "lillet blanc", "sherry", "port", "lime juice", "lemon juice",
    "grapefruit juice", "orange juice", "yuzu juice", "pineapple juice",
    "passion fruit", "apple juice", "cranberry juice", "verjuice",
    "simple syrup", "1:1 simple syrup", "rich syrup", "2:1 syrup",
    "honey syrup", "agave syrup", "grenadine", "orgeat", "demerara syrup",
    "maple syrup", "soda water", "sparkling water", "coconut water",
    "water", "cold brew", "green tea", "heavy cream", "egg white",
    "whole egg", "milk", "coconut cream", "coconut milk",
    "angostura", "peychauds", "orange bitters", "bitters",
}

# ── Extra cocktail ingredients beyond the 125 MATERIALS list ──────────────
EXTRA_INGREDIENTS = [
    # Additional spirits
    ("prosecco", "spirit"),
    ("champagne", "spirit"),
    ("cava", "spirit"),
    ("beer", "dilutant"),
    ("stout", "dilutant"),
    ("sake nigori", "spirit"),
    ("umeshu", "modifier"),
    ("baijiu sauce aroma", "spirit"),
    # Modifiers
    ("cynar", "modifier"),
    ("suze", "modifier"),
    ("benedictine", "modifier"),
    ("drambuie", "modifier"),
    ("frangelico", "modifier"),
    ("chambord", "modifier"),
    ("midori", "modifier"),
    ("blue curacao", "modifier"),
    ("passoã", "modifier"),
    ("pimms", "modifier"),
    ("malibu", "modifier"),
    ("disaronno", "modifier"),
    ("italicus", "modifier"),
    ("strega", "modifier"),
    ("galliano", "modifier"),
    ("licor 43", "modifier"),
    ("tia maria", "modifier"),
    ("montenegro", "modifier"),
    ("carpano antica", "modifier"),
    ("cocchi americano", "modifier"),
    # Syrups
    ("lavender syrup", "sweetener"),
    ("rose syrup", "sweetener"),
    ("elderflower syrup", "sweetener"),
    ("passion fruit syrup", "sweetener"),
    ("ginger syrup", "sweetener"),
    ("cinnamon syrup", "sweetener"),
    ("hibiscus syrup", "sweetener"),
    ("raspberry syrup", "sweetener"),
    ("blackberry syrup", "sweetener"),
    ("Thai basil syrup", "sweetener"),
    # Acid powders
    ("citric acid solution", "acid"),
    ("malic acid solution", "acid"),
    ("tartaric acid solution", "acid"),
    ("lactic acid solution", "acid"),
    # Fresh juices
    ("mango juice", "acid"),
    ("watermelon juice", "dilutant"),
    ("tomato juice", "acid"),
    ("beet juice", "acid"),
    ("carrot juice", "dilutant"),
    # Other
    ("tonic water", "dilutant"),
    ("ginger beer", "dilutant"),
    ("kombucha", "dilutant"),
    ("oat milk", "modifier"),
    ("almond milk", "modifier"),
]

# ── CATEGORY HINT MAP: MATERIALS category → INGREDIENT_PROPS category ─────
CATEGORY_MAP = {
    "蒸馏酒及基酒": "spirit",
    "调味酒精饮料": "modifier",
    "果酒葡萄酒": "modifier",
    "糖浆甜味剂": "sweetener",
    "柑橘类": "acid",
    "浆果类": "acid",
    "果汁类": "acid",
    "碳酸饮料": "dilutant",
    "草本植物类": "modifier",
    "香料类": "modifier",
    "辛香料类": "modifier",
}

PERPLEXITY_SYSTEM = """\
You are a food-science and bartending expert with access to current industry databases.
Return ONLY valid JSON — no markdown, no explanation outside the JSON.
All numeric values must be standard industry midpoint values (not ranges).
abv_pct must be in decimal form: 40% ABV = 0.40. Non-alcoholic = 0.0.
ta_pct (total acidity as citric acid equivalent) is a percentage by weight.
brix is °Brix sugar content.
density is g/ml at 20°C.
category must be exactly one of: spirit | acid | sweetener | dilutant | modifier | bitters
"""


def build_batch_prompt(batch: list[tuple[str, str]]) -> str:
    """Build a single Perplexity prompt for multiple ingredients."""
    items = "\n".join(f'  "{name}" (hint category: {cat})' for name, cat in batch)
    return (
        f"Look up the standard physical constants for these {len(batch)} ingredients used in bartending:\n"
        f"{items}\n\n"
        "Return a JSON object mapping each ingredient name to its properties:\n"
        "{\n"
        '  "<ingredient_name>": {\n'
        '    "abv_pct": <0.0–1.0>,\n'
        '    "ta_pct": <0.0–10.0>,\n'
        '    "brix": <0.0–80.0>,\n'
        '    "density": <0.85–1.35>,\n'
        '    "category": "spirit|acid|sweetener|dilutant|modifier|bitters"\n'
        "  },\n"
        "  ...\n"
        "}\n\n"
        "Use EXACTLY the ingredient names as keys (same spelling as input).\n"
        "Non-alcoholic ingredients: abv_pct = 0.0\n"
        "Pure water / soda: brix = 0.0, ta_pct = 0.0\n"
        "Return only the JSON object, nothing else."
    )


def validate_props(props: dict) -> bool:
    """Sanity-check a single props dict."""
    required = {"abv_pct", "ta_pct", "brix", "density", "category"}
    if not required.issubset(props.keys()):
        return False
    try:
        abv = float(props["abv_pct"])
        ta  = float(props["ta_pct"])
        bx  = float(props["brix"])
        dn  = float(props["density"])
        if not (0.0 <= abv <= 1.0): return False
        if not (0.0 <= ta <= 15.0): return False
        if not (0.0 <= bx <= 85.0): return False
        if not (0.80 <= dn <= 1.40): return False
        if props["category"] not in {"spirit","acid","sweetener","dilutant","modifier","bitters"}:
            return False
    except (TypeError, ValueError):
        return False
    return True


def round_props(props: dict) -> dict:
    """Normalise float precision."""
    return {
        "abv_pct":  round(float(props["abv_pct"]), 3),
        "ta_pct":   round(float(props["ta_pct"]), 2),
        "brix":     round(float(props["brix"]), 1),
        "density":  round(float(props["density"]), 3),
        "category": props["category"],
    }


def query_perplexity_batch(
    client: OpenAI, batch: list[tuple[str, str]], retries: int = 2
) -> dict[str, dict]:
    """Query Perplexity for a batch of ingredients. Returns {name: props}."""
    prompt = build_batch_prompt(batch)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="sonar",  # cheaper sonar for enrichment (not sonar-pro)
                messages=[
                    {"role": "system", "content": PERPLEXITY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=800,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            result = {}
            for name, props in data.items():
                if validate_props(props):
                    result[name.lower().strip()] = round_props(props)
                else:
                    print(f"  ⚠ Invalid props for '{name}': {props}")
            return result
        except Exception as e:
            if attempt < retries:
                print(f"  Retry {attempt+1}/{retries} after error: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"  ✗ Batch failed: {e}")
    return {}


def materials_to_candidates() -> list[tuple[str, str]]:
    """Extract ingredients from MATERIALS that aren't in INGREDIENT_PROPS."""
    candidates = []
    for m in MATERIALS:
        name_en = m["name_en"].lower().strip()
        # Simplify name (remove parenthetical brand info)
        simple = re.sub(r"\s*\(.*?\)", "", name_en).strip()
        if simple and simple not in EXISTING_KEYS:
            cat = CATEGORY_MAP.get(m.get("category", ""), "modifier")
            candidates.append((simple, cat))
    return candidates


def extra_candidates() -> list[tuple[str, str]]:
    """Extra ingredients beyond MATERIALS not in INGREDIENT_PROPS."""
    return [
        (name.lower(), cat)
        for name, cat in EXTRA_INGREDIENTS
        if name.lower() not in EXISTING_KEYS
    ]


def props_to_python(name: str, props: dict, source: str = "") -> str:
    src = f"  # source: {source}" if source else ""
    return (
        f'    "{name}":{" " * max(1, 25 - len(name))}'
        f'{{"abv_pct": {props["abv_pct"]}, "ta_pct": {props["ta_pct"]}, '
        f'"brix": {props["brix"]}, "density": {props["density"]}, '
        f'"category": "{props["category"]}"}},{src}'
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enrich INGREDIENT_PROPS via Perplexity sonar")
    parser.add_argument("--key", "-k", help="Perplexity API key (overrides env var)")
    parser.add_argument("--limit", "-n", type=int, default=0,
                        help="Limit number of candidates (0 = all, useful for testing)")
    args = parser.parse_args()

    api_key = args.key or os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        print("ERROR: PERPLEXITY_API_KEY not set.")
        print("  Option 1: set PERPLEXITY_API_KEY=pplx-... (Windows PowerShell)")
        print("  Option 2: python enrich_ingredients.py --key pplx-...")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    # Build candidate list
    candidates = materials_to_candidates() + extra_candidates()
    # Deduplicate
    seen = set()
    unique = []
    for name, cat in candidates:
        if name not in seen:
            seen.add(name)
            unique.append((name, cat))

    if args.limit and args.limit > 0:
        unique = unique[:args.limit]
        print(f"⚡ Test mode: limited to first {args.limit} candidates")

    print(f"📋 Total candidates to enrich: {len(unique)}")
    print(f"   (skipping {len(EXISTING_KEYS)} already in INGREDIENT_PROPS)")

    BATCH_SIZE = 8
    SLEEP_BETWEEN = 1.5  # seconds between Perplexity calls

    all_results: dict[str, dict] = {}
    batches = [unique[i:i+BATCH_SIZE] for i in range(0, len(unique), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        names = [n for n, _ in batch]
        print(f"\n[{i+1}/{len(batches)}] Querying: {names}")
        results = query_perplexity_batch(client, batch)
        all_results.update(results)
        print(f"  ✓ Got {len(results)}/{len(batch)} valid results")
        if i < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN)

    # Save raw JSON
    out_dir = pathlib.Path(__file__).parent
    json_path = out_dir / "enriched_props.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(all_results)} results → {json_path}")

    # Generate Python patch code
    py_lines = ["# ── ENRICHED INGREDIENT PROPS PATCH ──────────────────────────────────────\n"]
    py_lines.append("# Paste the entries below into INGREDIENT_PROPS in backend/routers/ai.py\n")
    py_lines.append("# All values verified by Perplexity sonar (offline enrichment run)\n\n")

    # Group by category
    by_cat: dict[str, list] = {}
    for name, props in sorted(all_results.items()):
        cat = props.get("category", "modifier")
        by_cat.setdefault(cat, []).append((name, props))

    cat_labels = {
        "spirit": "Base spirits (enriched)",
        "modifier": "Liqueurs / modifiers (enriched)",
        "acid": "Acid sources (enriched)",
        "sweetener": "Sweeteners (enriched)",
        "dilutant": "Dilutants (enriched)",
        "bitters": "Bitters (enriched)",
    }
    for cat in ["spirit", "modifier", "acid", "sweetener", "dilutant", "bitters"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        py_lines.append(f"    # ── {cat_labels.get(cat, cat)} ──\n")
        for name, props in items:
            py_lines.append(props_to_python(name, props, "perplexity_sonar") + "\n")
        py_lines.append("\n")

    py_path = out_dir / "ingredient_props_patch.py"
    with open(py_path, "w", encoding="utf-8") as f:
        f.writelines(py_lines)
    print(f"✅ Python patch → {py_path}")

    # Summary stats
    total = len(all_results)
    attempted = len(unique)
    failed = attempted - total
    print(f"\n📊 Summary: {total}/{attempted} enriched ({failed} failed/invalid)")
    if failed > 0:
        missing = [n for n, _ in unique if n not in all_results]
        print(f"   Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    print("\n🔍 Next step:")
    print("   1. Review backend/scripts/enriched_props.json")
    print("   2. Copy entries from ingredient_props_patch.py into INGREDIENT_PROPS in ai.py")
    print("   3. Run: git add backend/routers/ai.py && git commit -m 'feat(ai): enrich INGREDIENT_PROPS'")


if __name__ == "__main__":
    main()
