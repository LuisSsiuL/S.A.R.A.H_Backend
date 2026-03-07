import os
import logging
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt import PyJWKClient

logger = logging.getLogger("auth")

_bearer_scheme = HTTPBearer()

# Singleton JWKS client — fetches and caches public keys from Supabase
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_url:
            raise RuntimeError("SUPABASE_URL is not set — cannot build JWKS endpoint.")
        jwks_url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True)
        logger.info(f"JWKS client initialized: {jwks_url}")
    return _jwks_client


def verify_supabase_token(credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme)) -> dict:
    """
    FastAPI dependency that verifies a Supabase-issued JWT using JWKS.

    Fetches the public signing key from the project's JWKS endpoint
    and verifies the token signature (RS256 / ES256).
    Returns the decoded token payload on success.
    Raises HTTP 401 on any failure.
    """
    token = credentials.credentials
    try:
        client = _get_jwks_client()
        signing_key = client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"verify_aud": False},
        )
        return payload
    except Exception as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
