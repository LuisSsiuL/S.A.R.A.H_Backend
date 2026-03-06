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


async def process_user_query(message: str, role: str) -> dict:
    """Orchestrates the 4-stage Text-to-SQL logic."""
    logger.info("process_user_query started")
    client = get_llm_client()
    
    # --- STAGE 1: INTENT & SCHEMA ---
    logger.info("Stage 1 started")
    domains = await stage_1_intent_classification(client, message)
    logger.info(f"Stage 1 finished: {domains}")
    schema_context = "\n\n".join([SCHEMA_MAPPING.get(d, "") for d in domains])
    if not schema_context.strip():
        schema_context = SCHEMA_MAPPING["sales"] # Fallback

    # --- STAGE 2 & 3: GENERATION & EXECUTION (WITH RETRIES) ---
    max_retries = 2
    attempt = 0
    generated_sql = ""
    error_feedback = None
    query_results = []
    
    while attempt <= max_retries:
        try:
            # Stage 2
            logger.info(f"Stage 2 started (Attempt {attempt})")
            sql_response = await stage_2_sql_generation(client, schema_context, message, role, error_feedback)
            logger.info("Stage 2 finished")
            generated_sql = sql_response.get("sql", "")
            
            # Sub-check for warehouse admin restriction hit
            if "Akses ditolak" in generated_sql:
                return {
                    "text": generated_sql,
                    "table": "",
                    "sql": "BLOCKED",
                    "database": "FURNITURE_MOCK"
                }

            # Stage 3
            logger.info(f"Stage 3 started")
            query_results = await stage_3_sql_execution(generated_sql)
            logger.info(f"Stage 3 finished (Records: {len(query_results)})")
            
            # If we get here, execution succeeded
            break 
            
        except Exception as e:
            error_feedback = str(e)
            logger.warning(f"SQL Execution failed on attempt {attempt}: {error_feedback}. Retrying...")
            attempt += 1

    if attempt > max_retries and error_feedback:
        # We failed after retrying
        return {
            "text": f"Maaf, saya gagal menjalankan query setelah 3 percobaan. Error terakhir: {error_feedback}",
            "table": "",
            "sql": generated_sql,
            "database": "FURNITURE_MOCK"
        }

    # --- STAGE 4: EXPLAINER ---
    logger.info(f"Stage 4 started")
    final_output = await stage_4_explainer(client, message, query_results, generated_sql)
    logger.info(f"Stage 4 finished, preparing return payload")
    
    return {
        "text": final_output.get("text", ""),
        "table": final_output.get("table", ""),
        "sql": generated_sql,
        "database": "FURNITURE_MOCK"
    }
