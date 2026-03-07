import os
import logging
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

logger = logging.getLogger("auth")

_bearer_scheme = HTTPBearer()


def verify_supabase_token(credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme)) -> dict:
    """
    FastAPI dependency that verifies a Supabase-issued JWT.

    Reads SUPABASE_JWT_SECRET from the environment.
    Returns the decoded token payload on success.
    Raises HTTP 401 on any failure.
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret:
        logger.error("SUPABASE_JWT_SECRET is not set — cannot verify tokens.")
        raise HTTPException(status_code=500, detail="Authentication not configured.")

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            # Supabase sets audience to "authenticated" for logged-in users
            options={"verify_aud": False},
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
