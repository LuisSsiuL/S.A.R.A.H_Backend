import os
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from database import init_db_pool, close_db_pool, update_feedback, fetch_table_data
from llm_pipeline import process_user_query
from auth import verify_supabase_token

# Allowed table names — prevents IDOR via arbitrary table enumeration
ALLOWED_TABLES = frozenset({
    "orders", "order_items", "products", "categories",
    "customers", "employees", "stock", "warehouses",
    "suppliers", "purchase_orders", "purchase_order_items",
})

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI integration...")
    await init_db_pool()
    yield
    logger.info("Shutting down FastAPI integration...")
    await close_db_pool()


app = FastAPI(title="S.A.R.A.H Backend", lifespan=lifespan)

# CORS — restrict to known frontend origins
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    message: str
    role: str = "Sales"


class FeedbackRequest(BaseModel):
    cache_id: int
    feedback_value: int

    @field_validator("feedback_value")
    @classmethod
    def must_be_vote(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("feedback_value must be 1 or -1")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/query")
async def query_ai(
    request: QueryRequest,
    _user: Annotated[dict, Depends(verify_supabase_token)],
):
    """
    Accepts a natural-language question and streams SSE events.
    Requires a valid Supabase JWT in the Authorization header.
    """
    try:
        from sse_starlette.sse import EventSourceResponse
        return EventSourceResponse(process_user_query(request.message, request.role))
    except Exception as e:
        logger.error(f"Error initializing query stream: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while initializing stream.")


@app.get("/api/data/{table_name}")
async def get_table_data(
    table_name: str,
    _user: Annotated[dict, Depends(verify_supabase_token)],
):
    """
    Fetches raw table data via asyncpg connection pool.
    Only tables in ALLOWED_TABLES may be queried.
    """
    if table_name not in ALLOWED_TABLES:
        raise HTTPException(status_code=404, detail="Table not found.")

    try:
        data = await fetch_table_data(table_name)
        return data
    except Exception as e:
        logger.error(f"Error fetching table {table_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch table data.")


@app.post("/api/feedback")
async def give_feedback(
    request: FeedbackRequest,
    _user: Annotated[dict, Depends(verify_supabase_token)],
):
    """
    Records thumbs-up (+1) or thumbs-down (-1) for a cached query.
    """
    try:
        success = await update_feedback(request.cache_id, request.feedback_value)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update feedback. Invalid cache ID.")
        return {"status": "success", "message": "Feedback recorded."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing feedback.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
