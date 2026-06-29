import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import auth, materials, ai, qa

app = FastAPI(title="Liquid Architect API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(materials.router)
app.include_router(ai.router)
app.include_router(qa.router)


@app.get("/")
def root():
    return {"name": "Liquid Architect API", "version": "2.0.0", "status": "operational"}


@app.get("/health")
def health():
    return {"status": "ok"}
