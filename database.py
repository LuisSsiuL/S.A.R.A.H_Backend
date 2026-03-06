import os
import asyncpg
import logging

logger = logging.getLogger("database")

# Global variables for the connection pool
pool = None

async def init_db_pool():
    """Initializes the asyncpg connection pool."""
    global pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL is not set. Database integration will fail.")
        return

    try:
        pool = await asyncpg.create_pool(
            dsn=db_url,
            min_size=1,
            max_size=10,
            command_timeout=60
        )
        logger.info("Successfully established PostgreSQL connection pool.")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise e

async def close_db_pool():
    """Closes the asyncpg connection pool."""
    global pool
    if pool:
        await pool.close()
        logger.info("PostgreSQL connection pool closed.")

async def execute_query(sql_query: str) -> list[dict]:
    """
    Executes a read-only SQL query against the Supabase database using asyncpg.
    Returns the mapped dictionary of results.
    """
    global pool
    
    if "insert" in sql_query.lower() or "update" in sql_query.lower() or "delete" in sql_query.lower() or "drop" in sql_query.lower():
        raise Exception("Security Error: Only SELECT queries are permitted.")

    if not pool:
        raise Exception("Database pool is not initialized.")

    try:
        async with pool.acquire() as connection:
            records = await connection.fetch(sql_query)
            # asyncpg.Record -> dict
            return [dict(record) for record in records]
    except asyncpg.PostgresError as e:
        logger.error(f"PostgreSQL query failed: {e} | Query: {sql_query}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected execution error: {e}")
        raise e
