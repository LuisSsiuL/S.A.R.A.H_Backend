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
    import re
    
    # Use word boundaries so that 'last_updated' doesn't trigger 'update'
    forbidden_pattern = re.compile(r'\b(insert|update|delete|drop|alter|truncate|grant|revoke)\b', re.IGNORECASE)
    
    if forbidden_pattern.search(sql_query):
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

from typing import Optional

async def get_cached_sql(embedding: list[float], threshold: float = 0.95) -> Optional[tuple[int, str]]:
    """
    Searches the query_cache table using pgvector cosine distance.
    Returns (id, cached_sql_string) if the similarity (1 - distance) is > threshold.
    """
    global pool
    if not pool:
        logger.warning("Database pool not initialized. Skipping cache check.")
        return None

    # asyncpg expects the vector as a stringified array
    vec_str = f"[{','.join(map(str, embedding))}]"
    
    query = """
    SELECT id, generated_sql 
    FROM query_cache 
    WHERE 1 - (message_embedding <=> $1::vector) > $2 
    ORDER BY message_embedding <=> $1::vector 
    LIMIT 1
    """
    
    try:
        async with pool.acquire() as connection:
            result = await connection.fetchrow(query, vec_str, threshold)
            if result:
                logger.info("🎯 Cache HIT: Found semantically similar query.")
                return (result['id'], result['generated_sql'])
            logger.info("❌ Cache MISS: No similar query found.")
            return None
    except Exception as e:
        logger.error(f"Cache check failed: {e}")
        return None


async def save_to_cache(user_message: str, embedding: list[float], sql: str) -> Optional[int]:
    """Saves a successfully generated SQL query and its embedding to the cache and returns the new ID."""
    global pool
    if not pool:
        return None

    vec_str = f"[{','.join(map(str, embedding))}]"
    
    query = """
    INSERT INTO query_cache (user_message, message_embedding, generated_sql)
    VALUES ($1, $2::vector, $3)
    RETURNING id
    """
    
    try:
        async with pool.acquire() as connection:
            row_id = await connection.fetchval(query, user_message, vec_str, sql)
            logger.info("💾 Saved query to semantic cache.")
            return row_id
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}")
        return None


async def update_feedback(cache_id: int, feedback_value: int) -> bool:
    """Updates the feedback score for a cached query (+1/-1)."""
    global pool
    if not pool:
        return False
        
    query = """
    UPDATE query_cache 
    SET feedback = feedback + $2
    WHERE id = $1
    """
    
    try:
        async with pool.acquire() as connection:
            await connection.execute(query, cache_id, feedback_value)
            logger.info(f"Feedback updated for cache ID {cache_id}.")
            return True
    except Exception as e:
        logger.error(f"Failed to update feedback: {e}")
        return False
