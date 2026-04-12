from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth, projects
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

# Routes
app.include_router(auth.router)
app.include_router(projects.router)

@app.get("/")
async def root():
    return {"message": "Welcome to ResQNow Backend API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
