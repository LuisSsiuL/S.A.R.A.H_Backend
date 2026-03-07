import os
import re
import json
import logging
from openai import AsyncOpenAI
from database import execute_query, get_cached_sql, save_to_cache
from prompts import (
    SQL_GENERATION_SYSTEM_PROMPT,
    EXPLAINER_SYSTEM_PROMPT,
    ALL_SCHEMAS,
    SCHEMA_MAPPING
)

logger = logging.getLogger("llm_pipeline")

_DOMAIN_KEYWORDS = {
    "sales": {"order", "customer", "sale", "revenue", "employee", "deliver", "ship", "discount", "invoice", "country", "retail"},
    "inventory": {"product", "stock", "warehouse", "category", "inventory", "sku", "qty", "reorder", "material", "weight", "shelf", "active"},
    "procurement": {"purchase", "supplier", "vendor", "po", "procurement", "lead time", "received"},
}

def _select_schema_context(message: str) -> str:
    """Return only schema domains relevant to this query to reduce prompt tokens."""
    msg = message.lower()
    selected = [domain for domain, kws in _DOMAIN_KEYWORDS.items() if any(kw in msg for kw in kws)]
    if not selected:
        return ALL_SCHEMAS
    return "\n".join(SCHEMA_MAPPING[d] for d in selected)

# Financial columns that Warehouse Admins must never see — enforced at app level
_FINANCIAL_COLUMNS = re.compile(
    r'\b(retail_price|cost_price|unit_price|unit_cost|shipping_cost|discount_pct)\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Singleton AI clients — created once at import time, reused across requests
# ---------------------------------------------------------------------------

def _make_deepseek_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _make_openai_client() -> AsyncOpenAI | None:
    api_key = os.getenv("STANDARD_OPENAI_API_KEY")
    if not api_key:
        logger.warning("STANDARD_OPENAI_API_KEY not found. Embeddings disabled.")
        return None
    return AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1")


deepseek_client: AsyncOpenAI = _make_deepseek_client()
openai_client: AsyncOpenAI | None = _make_openai_client()


def clean_json_response(raw_text: str) -> dict:
    """Strip Markdown fences and parse JSON from an LLM response."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    if text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    return json.loads(text.strip())


async def stage_2_sql_generation(
    schema_context: str,
    user_message: str,
    user_role: str,
    error_feedback: str = None
) -> dict:
    """Stage 2: SQL Generation (handles retries by accepting error_feedback)."""
    prompt_context = (
        f"Schema Context:\n{schema_context}\n\n"
        f"User Role: {user_role}\n\n"
        f"User Request: {user_message}"
    )
    if error_feedback:
        prompt_context += f"\n\nPREVIOUS ERROR (Fix this): {error_feedback}"

    response = await deepseek_client.chat.completions.create(
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


async def process_user_query(message: str, role: str):
    """
    Orchestrates the Text-to-SQL pipeline with semantic caching and SSE streaming.
    Yields Server-Sent Events.
    """
    logger.info("process_user_query started")

    # --- STAGE 1: SEMANTIC CACHE (EMBEDDING) ---
    logger.info("Generating embedding for user message...")
    embedding = None
    if openai_client:
        try:
            embed_res = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=message
            )
            embedding = embed_res.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed. Falling back to LLM generation. {e}")
    else:
        logger.info("OpenAI client not initialized. Skipping semantic cache.")

    logger.info("Checking semantic cache...")
    cached_sql = None
    cache_id = None
    if embedding is not None:
        cache_result = await get_cached_sql(embedding, threshold=0.95)
        if cache_result:
            cache_id, cached_sql = cache_result

    generated_sql = ""
    query_results = []

    if cached_sql:
        generated_sql = cached_sql
    else:
        # --- STAGE 2: SQL GENERATION (WITH RETRIES) ---
        schema_context = _select_schema_context(message)
        logger.info(f"Schema domains selected for: '{message[:60]}'  -> {schema_context[:80]}...")
        max_retries = 2
        attempt = 0
        error_feedback = None

        while attempt <= max_retries:
            try:
                logger.info(f"Stage 2 started (Attempt {attempt})")
                sql_response = await stage_2_sql_generation(
                    schema_context, message, role, error_feedback
                )
                logger.info("Stage 2 finished")
                generated_sql = sql_response.get("sql", "")

                # LLM-level access denial (Warehouse Admin financial data)
                if "Akses ditolak" in generated_sql:
                    yield json.dumps({"type": "data", "table": "", "sql": "BLOCKED"})
                    yield json.dumps({"type": "text", "content": generated_sql})
                    return

                # App-level enforcement: block financial columns for Warehouse Admin
                if role.lower() in ("warehouse_admin", "warehouse admin") and _FINANCIAL_COLUMNS.search(generated_sql):
                    yield json.dumps({"type": "data", "table": "", "sql": "BLOCKED"})
                    yield json.dumps({"type": "text", "content": "Akses ditolak: Admin Gudang tidak dapat melihat data finansial."})
                    return

                yield json.dumps({"type": "data", "sql": generated_sql})

                logger.info("Stage 3 started")
                # Execute once — reuse results for the data response below
                query_results = await execute_query(generated_sql)

                if embedding is not None:
                    cache_id = await save_to_cache(message, embedding, generated_sql)
                break

            except Exception as e:
                error_feedback = str(e)
                logger.warning(f"SQL Execution failed on attempt {attempt}: {error_feedback}. Retrying...")
                attempt += 1

        if attempt > max_retries and error_feedback:
            yield json.dumps({"type": "data", "table": "", "sql": generated_sql})
            yield json.dumps({
                "type": "text",
                "content": f"Maaf, saya gagal menjalankan query setelah 3 percobaan. Error terakhir: {error_feedback}"
            })
            return

    # --- STAGE 3: DATA RETRIEVAL (only runs for cache hits) ---
    if cached_sql:
        logger.info("Executing cached SQL...")
        try:
            query_results = await execute_query(generated_sql)
        except Exception as e:
            logger.error(f"Cached query execution failed: {e}")
            yield json.dumps({"type": "text", "content": "Error saat menjalankan query dari cache."})
            return

    yield json.dumps({
        "type": "data",
        "full_data": query_results,
        "cache_id": cache_id
    }, default=str)

    # --- STAGE 4: EXPLAINER (STREAMING) ---
    logger.info("Stage 4 (Explainer Streaming) started")

    # Build data context: include all rows, but cap JSON size to avoid huge prompts
    MAX_DATA_CHARS = 12000
    full_json = json.dumps(query_results, default=str)
    if len(full_json) <= MAX_DATA_CHARS:
        data_context = full_json
        data_note = f"[{len(query_results)} rows total — full dataset]"
    else:
        # Trim rows until it fits
        trimmed = query_results[:]
        while trimmed and len(json.dumps(trimmed, default=str)) > MAX_DATA_CHARS:
            trimmed = trimmed[:-10]
        data_context = json.dumps(trimmed, default=str)
        data_note = f"[Showing {len(trimmed)} of {len(query_results)} rows due to size]"

    prompt = (
        f"Original Request: {message}\n"
        f"Executed SQL: {generated_sql}\n"
        f"Query Results {data_note}:\n{data_context}\n"
    )

    try:
        response = await deepseek_client.chat.completions.create(
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
                yield json.dumps({"type": "text", "content": chunk.choices[0].delta.content})

    except Exception as e:
        logger.error(f"Streaming explainer failed: {e}")
        yield json.dumps({"type": "text", "content": "Terdapat masalah teknis dalam menerjemahkan output data."})

    logger.info("process_user_query stream finished")
