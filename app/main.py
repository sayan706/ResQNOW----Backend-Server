from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth, projects, predictions
from celery import Celery  # <-- 1. Import Celery
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="ResQNow API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. NEW CELERY SETUP ---
# Grab the Upstash Redis URL from your .env or Render environment variables
redis_url = os.getenv("REDIS_URL")

# Create the distinct Celery instance
celery_app = Celery(
    "resqnow_worker",
    broker=redis_url,
    backend=redis_url
)

# Upstash rediss:// URLs require this config to avoid SSL connection errors
celery_app.conf.broker_use_ssl = {
    'ssl_cert_reqs': 'none'
}
celery_app.conf.redis_backend_use_ssl = {
    'ssl_cert_reqs': 'none'
}
# ---------------------------

# Routes
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(predictions.router)

# --- 3. UPDATED ROOT ROUTE ---
# Added @app.head("/") so Render's internal scanner doesn't trigger a 405 error
@app.get("/")
@app.head("/")
async def root():
    return {"message": "Welcome to ResQNow Backend API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)