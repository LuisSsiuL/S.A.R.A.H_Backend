import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from supabase import create_client, Client
from database import init_db_pool, close_db_pool
from llm_pipeline import process_user_query

# Load Environment Variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Global Supabase Client
supabase_client: Client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    # Startup actions
    logger.info("Starting up FastAPI integration...")
    
    # Initialize the official Supabase Client for standard operations
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        logger.info("Successfully initialized official Supabase client.")
    else:
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY missing. Supabase client not initialized.")
        
    # Initialize asyncpg connection pool strictly for AI execution
    await init_db_pool()
    yield
    # Shutdown actions
    logger.info("Shutting down FastAPI integration...")
    await close_db_pool()


app = FastAPI(title="PJM AI Assistant v2 Backend", lifespan=lifespan)

# Configure CORS for React SPA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to the Cloudflare pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class QueryRequest(BaseModel):
    message: str
    role: str = "Sales"

class QueryResponse(BaseModel):
    text: str
    table: str
    sql: str
    database: str

@app.get("/api/health")
async def health_check():
    """Healthcheck endpoint."""
    return {"status": "healthy"}

@app.post("/api/query", response_model=QueryResponse)
async def query_ai(request: QueryRequest):
    """
    Accepts user question and role, processes via LLM Pipeline against the DB,
    and returns explanation + data.
    """
    try:
        result = await process_user_query(request.message, request.role)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing query.")

if __name__ == "__main__":
    import uvicorn
    # Optional logic to run local server automatically on python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
