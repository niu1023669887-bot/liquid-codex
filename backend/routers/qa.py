import os
import json
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/qa", tags=["qa"])

# ── Per-volume knowledge contexts ──
VOL_CONTEXTS = {
    "vol1": """\
=== VOL. I · MATERIAL ATLAS ===
Domain: Natural bartending materials — chemistry, pH, Brix, density, TA%, and application.
Key facts:
• Citrus = "head note" + high-frequency acid. NEVER heat fresh citrus juice.
• Berry cell walls are fragile; cold press or Pectinex enzyme clarification, NOT hot cooking.
• Acid powders (citric/malic/tartaric/lactic/succinic) provide stable, oxidation-free acidity.
• 20% saline solution (20g NaCl per 80g water) suppresses bitterness, amplifies sweetness/aroma.
• Fixatives (orris root, angelica root) anchor light aromatic compounds in 40–50% ABV spirits.
• pH of common acids: citric ~2.2, malic ~2.3, tartaric ~2.2, lactic ~2.4 at 1% solution.
• Brix measures dissolved sugars: ripe orange ~11–12°Bx, lemon juice ~3–4°Bx, simple syrup ~66°Bx.
• 91 catalogued materials across categories: citrus, berries, botanicals, acids, salts, clarifiers, fermentables.
Answer scope: Material properties, flavor chemistry, ingredient substitutions, pairing science.""",

    "vol2": """\
=== VOL. II · CLASSICAL ERA ===
Domain: Traditional extraction techniques — maceration, oleo-saccharum, fat-washing, milk clarification, aging.
Key facts:
• Cold Maceration: 40–50% ABV is the ideal solvent window; duration 24h–4 weeks depending on material.
• Oleo-Saccharum: sugar hygroscopy extracts citrus terpenes via osmotic pressure over 12–24h at room temp.
• Fat-Washing: maillard-reacted fat added to spirit at room temp → freeze at −18°C for 2–4h → strain solid fat.
  Fat-to-spirit ratio: 50–100ml fat per 700ml spirit is typical.
• Milk Clarification: ALWAYS pour spirit/acid mixture INTO milk (not milk into spirit). Casein precipitates at pH 4.6.
  Ratio: 100–200ml whole milk per 500ml cocktail base.
• Barrel aging: minimum 2 weeks in 1–5L oak barrel for perceptible flavor change; optimal 4–8 weeks.
• Orgeat (almond syrup): blanch almonds, grind, press; 1:1 sugar ratio; 1 tsp orange flower water per 500ml.
Answer scope: Classical technique execution, ratios, timing, troubleshooting.""",

    "vol3": """\
=== VOL. III · ICE & KINETICS ===
Domain: Ice engineering, dilution curves, temperature, shake vs. stir thermodynamics.
Key facts:
• Large ice sphere (6–7cm diameter): minimal surface area → dilution rate ~1ml/min when stirring.
• Standard shake: 12–15 seconds drops liquid from room temp to ~0°C; adds ~20–30ml dilution per 60ml spirit.
• Stir: laminar flow, 30–40 rotations → gentle dilution ~15ml per 60ml spirit; NO aeration.
• Directional freezing: freeze from one side → pushes impurities to one end; produces crystal-clear ice.
  Method: insulated cooler open-top in freezer, 18–24h.
• Supercooled liquid: pure water can stay liquid to −4°C without nucleation; tap to trigger instant freeze.
• Dilution target: most stirred cocktails optimal at 20–25% dilution by final volume.
• Shake vs. stir: shaking emulsifies (egg white, cream), stirring preserves clarity and texture.
Answer scope: Ice selection, dilution calculation, temperature management, technique choice.""",

    "vol4": """\
=== VOL. IV · LIVING ALCHEMY ===
Domain: Fermentation science — lacto-fermentation, koji, SCOBY kombucha, pH control.
Key facts:
• Lacto-fermentation: 2% salt by total weight (e.g., 20g salt per 1000g substrate + water). pH MUST drop below 4.6 within 48–72h to prevent pathogen growth.
• Brine concentration: 2% = safe fermentation; 3–5% = slower, more controlled; <1.5% = DANGEROUS.
• Koji (Aspergillus oryzae): 30°C / 85% humidity, 48h cultivation. 60°C = enzyme optimum AND pasteurisation threshold.
• SCOBY kombucha: sweet tea (1L water + 60–80g sugar + 4g loose tea) + SCOBY at 24–28°C, 7–14 days. Target final pH: 2.5–3.5.
• Water kefir: 3% sugar solution + grains at 22–26°C, 24–48h; secondary fermentation 12–24h for carbonation.
• pH monitoring: check at 24h, 48h, 72h intervals. If pH has not dropped below 4.6 by 72h, discard batch.
• All fermented products must reach pH < 4.6 for botulism safety.
Answer scope: Fermentation protocols, safety thresholds, troubleshooting, salt ratios.""",

    "vol5": """\
=== VOL. V · CULINARY SCIENCE ===
Domain: Kitchen-tool precision extraction — sous-vide, iSi rapid infusion, carbonation, espumas.
Key facts:
• Sous-vide extraction: 55–65°C is the sweet spot for aromatic release without cooking off volatiles.
  Citrus peel + spirit at 55°C × 1h = equivalent of 2-week cold maceration.
  Dairy/egg pasteurisation: 63°C × 30 min (egg yolk); 72°C × 15 sec (dairy).
• iSi Rapid Infusion: charge N₂O at 2 bars → pressurizes plant tissue → vent slowly. 2 min = 24h cold maceration equivalent.
  Ratio: 100g herb/fruit per 500ml spirit; fill canister to ⅔ maximum.
• Carbonation: CO₂ most soluble near 0°C at 30–40 PSI (2–2.8 bar). Henry's Law: solubility doubles per 2-bar increase.
• Espumas (foam): lecithin 0.3–0.5% by weight for light foam; xanthan gum 0.1% for stable foam.
  Whip with immersion blender at surface.
• Vacuum concentration: reduce 500ml juice to ~200ml at 45°C under vacuum preserves aroma.
Answer scope: Equipment techniques, parameters, ratios, home vs. bar applicability.""",

    "vol6": """\
=== VOL. VI · MODERN LABORATORY ===
Domain: Precision acid adjustment, enzyme clarification, centrifuge use, refractometry.
Key facts:
• Acid adjustment: citric = bright lemon note; malic = green apple; tartaric = dry wine/grape; lactic = creamy dairy.
  Typical working concentration: 10% stock solution (100g acid per 900ml water). Add 5–10ml per 1L cocktail to adjust pH by ~0.3–0.5 units.
• Target pH ranges: sours 2.8–3.2; highballs 3.5–4.0; spirit-forward 4.0–5.0.
• Pectinex Ultra SP-L (pectin lyase): 0.5ml per litre of juice; 45°C × 2h → destroys pectin → optical clarity.
  Do NOT exceed 60°C (enzyme denatures).
• Centrifuge 4000+ RPM for 10–15 min: produces optically clear supernatant. Balance ±0.1g maximum.
• Refractometer: Brix reading at 20°C is accurate; correct for alcohol using: adjusted Brix = raw Brix − (ABV% × 0.6).
• Sodium alginate spherification: 0.5% alginate in flavored liquid + 0.5% CaCl₂ bath.
Answer scope: Precise measurements, pH targets, enzyme protocols, lab calculations.""",

    "vol7": """\
=== VOL. VII · EXTREME PHYSICS ===
Domain: Rotovap, Pacojet, spherification, electro-aging, high-end distillation.
Key facts:
• Rotovap (rotary evaporator): vacuum lowers boiling point to 20–30°C → captures heat-sensitive "head note" volatiles without thermal damage. Typical rotation: 60–80 RPM; bath temp: 35–40°C.
• Pacojet: ultra-high-speed blade (2000 RPM) on fully frozen (−18°C minimum) mass. Alcohol and sugar suppress freezing point: 10% ABV lowers FP by ~5°C; 20% Brix lowers FP by ~2°C.
• Basic spherification: 0.5g sodium alginate per 100g flavored liquid + 2.5g CaCl₂ per 500ml water bath. Gel membrane forms in 60–90 seconds.
• Reverse spherification: 0.5% CaCl₂ in flavored liquid + 0.5% sodium alginate bath → thicker membrane, longer shelf life.
• Liquid nitrogen: boiling point −196°C. ALWAYS serve only after all visible fog has dissipated and vessel wall is no longer cold to the touch. NEVER in closed containers.
• Electro-aging: applies 30–40V electric field to spirit for 20–30 min → accelerates ester formation equivalent to ~1 month barrel aging.
Answer scope: Advanced equipment protocols, freezing point calculation, spherification troubleshooting.""",

    "vol8": """\
=== VOL. VIII · GARNISH & ZERO WASTE ===
Domain: Sensory garnish engineering, dehydration, sugar art, zero-waste systems.
Key facts:
• Citrus twist: express essential oil over glass rim = "olfactory handshake" before first sip. Bergamot, lemon, orange each have distinct terpene profiles.
• Dehydration: 50–60°C, 4–8h in oven or dehydrator. At 55°C citrus wheels lose ~80% moisture; slice thickness: 3–5mm for uniform drying.
• Isomalt: melt at 160–175°C (caramelises above 180°C); pour/blow/pull when at 130–140°C. Moisture-resistant at <50% RH; suitable for structural garnishes stored up to 24h.
• Fruit leather: purée fruit, strain; spread 3–4mm on silicone; 60°C × 6–8h. Pectin content determines final texture.
• Zero-waste: citrus peels → oleo-saccharum or dehydrated zest; spent botanicals → compost or re-infuse in water; sugar syrup off-cuts → ferment.
• Smoke garnish: wood chips (applewood, cherry) in smoking gun; dose: 3–5 seconds into covered glass. Phenol compounds from smoke bind to spirit esters.
• Carbonated garnish: soak cut fruit in CO₂ chamber at 60 PSI for 30 min at 4°C.
Answer scope: Garnish construction, dehydration parameters, sugar work, sensory layering, sustainability.""",
}

VOLUME_NAMES = {
    "vol1": "Vol. I · Material Atlas",
    "vol2": "Vol. II · Classical Era",
    "vol3": "Vol. III · Ice & Kinetics",
    "vol4": "Vol. IV · Living Alchemy",
    "vol5": "Vol. V · Culinary Science",
    "vol6": "Vol. VI · Modern Laboratory",
    "vol7": "Vol. VII · Extreme Physics",
    "vol8": "Vol. VIII · Garnish & Zero Waste",
}


def build_qa_system_prompt(volume: str) -> str:
    ctx = VOL_CONTEXTS.get(volume, "")
    vol_name = VOLUME_NAMES.get(volume, volume)
    return f"""\
You are the Codex Intelligence for "{vol_name}" — Darien's scientific bartending knowledge base.

{ctx}

── ANSWER RULES (NON-NEGOTIABLE) ──
① Every answer MUST include at least one specific numeric value (temperature °C, time min/h, pH, concentration %, mass g, volume ml, ratio, RPM, PSI, Brix).
② Stay strictly within the knowledge domain of this volume. If a question falls outside this volume's scope, briefly acknowledge it and redirect to the relevant volume.
③ Be concise but complete: 2–4 paragraphs maximum. Use bullet points for multi-step procedures.
④ Always end practical answers with a "Safety note" if the technique has any hazard.
⑤ FORBIDDEN phrasings: "heat until warm", "add some", "approximately", "to taste", "a little".
⑥ Source every claim to a specific Codex principle from this volume's context above.
"""


class QARequest(BaseModel):
    volume: str
    question: str
    language: Optional[str] = "en"


class RecipeQARequest(BaseModel):
    recipe_text: str
    question: str
    language: Optional[str] = "en"


@router.post("/ask")
def ask_question(body: QARequest, x_api_key: Optional[str] = Header(default=None)):
    if body.volume not in VOL_CONTEXTS:
        raise HTTPException(status_code=400, detail=f"Unknown volume: {body.volume}. Valid values: vol1–vol8.")

    api_key = x_api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="AI engine not configured — provide your DeepSeek API key.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    lang_instruction = "Please respond entirely in Chinese." if body.language == "zh" else "Please respond entirely in English."

    user_msg = f"{lang_instruction}\n\nQuestion: {body.question.strip()}"

    def event_generator():
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": build_qa_system_prompt(body.volume)},
                    {"role": "user", "content": user_msg},
                ],
                stream=True,
                max_tokens=900,
                temperature=0.55,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/recipe")
def ask_recipe(body: RecipeQARequest, x_api_key: Optional[str] = Header(default=None)):
    """Answer follow-up questions about a specific generated recipe."""
    api_key = x_api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="AI engine not configured — provide your DeepSeek API key.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    lang_instruction = "Please respond entirely in Chinese." if body.language == "zh" else "Please respond entirely in English."

    system_prompt = f"""\
You are the Liquid Architect recipe consultant — Darien's personal assistant for clarifying, adapting, and troubleshooting cocktail recipes.

The user has just received the following recipe from the Codex AI:

=== CURRENT RECIPE ===
{body.recipe_text[:3000]}
=== END RECIPE ===

Your role: answer the user's follow-up question about THIS specific recipe.

ANSWER RULES:
① Always reference specific steps, ingredients, or quantities from the recipe above.
② Every practical answer must include at least one numeric value (g, ml, °C, pH, time, ratio).
③ For substitution questions: confirm if it works, explain the flavour/chemistry impact, give the exact substitute ratio.
④ For technique questions: give a step-by-step clarification with measured parameters.
⑤ Keep answers concise: 2–3 paragraphs or a short bullet list.
⑥ If the question is unrelated to this recipe, politely redirect.
"""

    user_msg = f"{lang_instruction}\n\nQuestion about the recipe: {body.question.strip()}"

    def event_generator():
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                stream=True,
                max_tokens=700,
                temperature=0.45,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
