import asyncio
from database import init_db_pool, close_db_pool
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    pool = await asyncpg.create_pool(dsn=os.getenv("DATABASE_URL"))
    async with pool.acquire() as conn:
        try:
            await conn.execute("ALTER TABLE query_cache ADD COLUMN IF NOT EXISTS feedback INT DEFAULT 0;")
            print("Successfully added feedback column.")
        except Exception as e:
            print(f"Error: {e}")
    await pool.close()

asyncio.run(main())
