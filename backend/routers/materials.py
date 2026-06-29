from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from typing import Optional

from data_codex import MATERIALS, all_categories, by_category, search as search_materials, enrich_with_efsa, MATERIAL_COUNT

router = APIRouter(prefix="/api/materials", tags=["materials"])

_NO_CACHE = {"Cache-Control": "no-store, max-age=0"}


@router.get("")
def get_materials(
    category: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
):
    if q:
        data = [enrich_with_efsa(m) for m in search_materials(q)]
    elif category:
        data = [enrich_with_efsa(m) for m in by_category(category)]
    else:
        data = [enrich_with_efsa(m) for m in MATERIALS]
    return JSONResponse(content=data, headers=_NO_CACHE)


@router.get("/count")
def get_count():
    """返回材料总数（轻量，无需传输完整列表）"""
    return JSONResponse(content={"count": MATERIAL_COUNT}, headers=_NO_CACHE)


@router.get("/categories")
def get_categories():
    return JSONResponse(content=all_categories(), headers=_NO_CACHE)


@router.get("/{material_id}")
def get_material(material_id: str):
    result = next((m for m in MATERIALS if m["id"] == material_id), None)
    if not result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Material not found")
    return JSONResponse(content=enrich_with_efsa(result), headers=_NO_CACHE)
