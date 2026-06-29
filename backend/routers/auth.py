import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, status
from jose import jwt
from pydantic import BaseModel

router = APIRouter(prefix="/api/auth", tags=["auth"])

SECRET_KEY = os.getenv("JWT_SECRET") or "dev-only-no-prod"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


class LoginRequest(BaseModel):
    password: str


class TokenResponse(BaseModel):
    token: str


def create_token() -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"exp": expire, "sub": "codex"}, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> bool:
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return True
    except Exception:
        return False


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    correct = os.getenv("PASSWORD") or "dev-only"
    if body.password != correct:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")
    return {"token": create_token()}


@router.get("/verify")
def verify(token: str):
    if not verify_token(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return {"valid": True}
