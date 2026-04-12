import os
import asyncpg
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# REST Client for simple CRUD and Auth
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# asyncpg for heavy operations / atomic versioning
async def get_db_connection():
    return await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
        statement_cache_size=0
    )

async def execute_query(query, *args):
    conn = await get_db_connection()
    try:
        return await conn.execute(query, *args)
    finally:
        await conn.close()

async def fetch_row(query, *args):
    conn = await get_db_connection()
    try:
        return await conn.fetchrow(query, *args)
    finally:
        await conn.close()

async def fetch_all(query, *args):
    conn = await get_db_connection()
    try:
        return await conn.fetch(query, *args)
    finally:
        await conn.close()
