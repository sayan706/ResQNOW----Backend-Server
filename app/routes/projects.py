from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Optional
import uuid
import json
import os
from app.services.db.supabase import execute_query, fetch_row, fetch_all
from app.services.storage import storage_service
from app.services.hash import generate_file_hash
from app.services.cache import cache_service
from app.routes.deps import get_current_user
from worker.tasks import process_csv_task, update_state_task

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("/upload")
async def upload_dataset(
    name: str,
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user)
):
    content = await file.read()
    file_hash = generate_file_hash(content)
    
    # 2. Duplicate handling (Critical)
    existing = await fetch_row(
        "SELECT id, status FROM projects WHERE user_id = $1 AND file_hash = $2",
        current_user_id, file_hash
    )
    
    if existing:
        return {
            "project_id": existing["id"], 
            "status": existing["status"], 
            "message": "Duplicate found. Returning existing project."
        }

    project_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    storage_path = f"uploads/{current_user_id}/{project_id}.{file_extension}"
    
    # Save to storage
    temp_path = f"/tmp/{project_id}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)
    
    try:
        storage_service.upload_file(temp_path, storage_path)
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

    file_url = storage_service.get_public_url(storage_path)
    
    # Trigger task and store task_id
    task = process_csv_task.delay(project_id, storage_path, current_user_id)
    
    await execute_query(
        """
        INSERT INTO projects (id, user_id, name, file_name, file_hash, file_url, status, task_id)
        VALUES ($1, $2, $3, $4, $5, $6, 'processing', $7)
        """,
        project_id, current_user_id, name, file.filename, file_hash, file_url, task.id
    )
    
    return {"project_id": project_id, "status": "processing", "task_id": task.id}

@router.get("/{project_id}/status")
async def get_status(project_id: str, current_user_id: str = Depends(get_current_user)):
    project = await fetch_row(
        "SELECT status, task_id, latest_state_id FROM projects WHERE id = $1 AND user_id = $2",
        project_id, current_user_id
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # 9. Latest State Optimization
    latest_state = None
    if project["latest_state_id"]:
        # Try to get from cache first
        cache_key = f"state:{project_id}"
        latest_state = cache_service.get(cache_key)
        
        if not latest_state:
            latest_state = await fetch_row(
                "SELECT json_url, version FROM project_states WHERE id = $1",
                project["latest_state_id"]
            )
            if latest_state:
                cache_service.set(cache_key, dict(latest_state))
    
    return {
        "status": project["status"],
        "task_id": project["task_id"],
        "latest_version": latest_state["version"] if latest_state else None,
        "json_url": latest_state["json_url"] if latest_state else None
    }

@router.post("/predict")
async def predict(
    project_id: str,
    latitude: float,
    longitude: float,
    context: Optional[dict] = {},
    current_user_id: str = Depends(get_current_user)
):
    # Verify project ownership and status
    project = await fetch_row("SELECT id FROM projects WHERE id = $1 AND user_id = $2 AND status = 'completed'", project_id, current_user_id)
    if not project:
        raise HTTPException(status_code=400, detail="Project not ready or access denied.")

    # 5. Concurrency Fix: Queue updates via Celery
    task = update_state_task.delay(project_id, latitude, longitude, context)
    
    return {
        "message": "Prediction and state update queued.",
        "task_id": task.id
    }
