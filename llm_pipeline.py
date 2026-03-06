import os
import json
import logging
from openai import AsyncOpenAI
from database import execute_query
from prompts import (
    INTENT_SYSTEM_PROMPT,
    SQL_GENERATION_SYSTEM_PROMPT,
    EXPLAINER_SYSTEM_PROMPT,
    SCHEMA_MAPPING
)

logger = logging.getLogger("llm_pipeline")

def get_llm_client():
    """Initialize the AsyncOpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)

def clean_json_response(raw_text: str) -> dict:
    """Helper to try to clean up Markdown-wrapped JSON from LLMs."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    if text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    return json.loads(text.strip())


async def stage_1_intent_classification(client: AsyncOpenAI, user_message: str) -> list[str]:
    """Stage 1: Intent & Schema Selection."""
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        domains = json.loads(content)
        return [d.lower() for d in domains if d.lower() in SCHEMA_MAPPING]
    except Exception as e:
        logger.error(f"Intent Classification failed: {e}")
        # Default fallback to all domains if intent classification fails
        return ["sales", "inventory", "procurement"]


async def stage_2_sql_generation(client: AsyncOpenAI, schema_context: str, user_message: str, user_role: str, error_feedback: str = None) -> dict:
    """Stage 2: SQL Generation (handles retries by accepting error_feedback)."""
    
    prompt_context = f"Schema Context:\n{schema_context}\n\nUser Role: {user_role}\n\nUser Request: {user_message}"
    
    if error_feedback:
        prompt_context += f"\n\nPREVIOUS ERROR (Fix this): {error_feedback}"

    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_context}
        ],
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    try:
        return clean_json_response(content)
    except Exception as e:
        logger.error(f"Failed to parse SQL generation response: {e}")
        raise ValueError("LLM returned malformed JSON during SQL generation.")


async def stage_3_sql_execution(sql: str) -> list[dict]:
    """Stage 3: SQL Execution."""
    return await execute_query(sql)


async def stage_4_explainer(client: AsyncOpenAI, user_message: str, data: list[dict], sql: str) -> dict:
    """Stage 4: Explainer (Markdown & Table)."""
    prompt = (
        f"Original Request: {user_message}\n"
        f"Executed SQL: {sql}\n"
        f"Query Results: {json.dumps(data, default=str)}\n"
    )

    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    content = response.choices[0].message.content
    try:
        return clean_json_response(content)
    except Exception as e:
        logger.error(f"Failed to parse Explainer response: {e}")
        # Graceful degradation if LLM fails JSON adherence here
        return {
            "text": "Terdapat masalah teknis dalam menerjemahkan output data.",
            "table": "Error rendering table."
        }


async def process_user_query(message: str, role: str):
    """
    Orchestrates the Text-to-SQL logic using Semantic Caching and SSE Streaming.
    Yields Server-Sent Events directly.
    """
    logger.info("process_user_query started")
    client = get_llm_client()
    
    # 0. Generate Embedding for Semantic Cache
    logger.info("Generating embedding for user message...")
    try:
        embed_res = await client.embeddings.create(
            model="text-embedding-3-small", 
            input=message
        )
        embedding = embed_res.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        yield json.dumps({'type': 'text', 'content': 'Gagal membuat vektor pencarian.'})
        return

    # 1. Semantic Cache Check
    logger.info("Checking semantic cache...")
    from database import get_cached_sql, save_to_cache
    cached_sql = await get_cached_sql(embedding, threshold=0.95)
    
    generated_sql = ""
    if cached_sql:
        generated_sql = cached_sql
        # Provide schema context for Explainer
        schema_context = "CACHED_QUERY" 
    else:
        # --- STAGE 1: INTENT & SCHEMA ---
        logger.info("Stage 1 started")
        domains = await stage_1_intent_classification(client, message)
        logger.info(f"Stage 1 finished: {domains}")
        schema_context = "\n\n".join([SCHEMA_MAPPING.get(d, "") for d in domains])
        if not schema_context.strip():
            schema_context = SCHEMA_MAPPING["sales"] # Fallback

        # --- STAGE 2: SQL GENERATION (WITH RETRIES) ---
        max_retries = 2
        attempt = 0
        error_feedback = None
        
        while attempt <= max_retries:
            try:
                # Stage 2
                logger.info(f"Stage 2 started (Attempt {attempt})")
                sql_response = await stage_2_sql_generation(client, schema_context, message, role, error_feedback)
                logger.info("Stage 2 finished")
                generated_sql = sql_response.get("sql", "")
                
                # Sub-check for warehouse admin restriction hit
                if "Akses ditolak" in generated_sql:
                    yield json.dumps({'type': 'data', 'table': '', 'sql': 'BLOCKED'})
                    yield json.dumps({'type': 'text', 'content': generated_sql})
                    return

                # Assuming valid syntax, check against DB natively
                logger.info(f"Stage 3 started (Dry Run / Test)")
                test_results = await stage_3_sql_execution(generated_sql)
                
                # If we get here, execution succeeded
                await save_to_cache(message, embedding, generated_sql)
                break 
                
            except Exception as e:
                error_feedback = str(e)
                logger.warning(f"SQL Execution failed on attempt {attempt}: {error_feedback}. Retrying...")
                attempt += 1

        if attempt > max_retries and error_feedback:
            # We failed after retrying
            yield json.dumps({'type': 'data', 'table': '', 'sql': generated_sql})
            yield json.dumps({'type': 'text', 'content': f'Maaf, saya gagal menjalankan query setelah 3 percobaan. Error terakhir: {error_feedback}'})
            return

    # --- STAGE 3: DATA RETRIEVAL ---
    logger.info("Executing final SQL...")
    query_results = []
    try:
        query_results = await stage_3_sql_execution(generated_sql)
    except Exception as e:
        logger.error(f"Final execution failed (perhaps cached query broke): {e}")
        yield json.dumps({'type': 'text', 'content': 'Error saat menjalankan query dari cache.'})
        return

    # Yield Data & SQL Chunk First
    data_chunk = {
        "type": "data",
        "table": "", # Frontend will handle raw JSON mapping later if we implement it, for now we let Explainer make the markdown
        "sql": generated_sql
    }
    yield json.dumps(data_chunk)

    # --- STAGE 4: EXPLAINER (STREAMING) ---
    logger.info("Stage 4 (Explainer Streaming) started")
    prompt = (
        f"Original Request: {message}\n"
        f"Executed SQL: {generated_sql}\n"
        f"Query Results: {json.dumps(query_results, default=str)}\n"
    )

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text_chunk = {"type": "text", "content": chunk.choices[0].delta.content}
                yield json.dumps(text_chunk)
                
    except Exception as e:
        logger.error(f"Streaming explainer failed: {e}")
        yield json.dumps({'type': 'text', 'content': 'Terdapat masalah teknis dalam menerjemahkan output data.'})
        
    logger.info("process_user_query stream finished")
