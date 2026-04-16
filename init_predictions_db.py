import asyncio
import os
import asyncpg
from dotenv import load_dotenv

# load from the correct directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

async def init_db():
    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
        statement_cache_size=0
    )
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            zone_name VARCHAR(100),
            latitude FLOAT,
            longitude FLOAT,
            input_data JSONB NOT NULL,
            risk_probability FLOAT NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            key_factors JSONB,
            recommendations JSONB,
            ai_reasoning TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    ''')
    print('✅ Table created successfully')
    await conn.close()

if __name__ == "__main__":
    asyncio.run(init_db())
