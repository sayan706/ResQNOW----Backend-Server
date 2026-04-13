from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import uuid
import json
import os
from app.services.db.supabase import execute_query, fetch_row, fetch_all
from app.services.storage import storage_service
from app.services.hash import generate_file_hash
from app.services.cache import cache_service
from app.routes.deps import get_current_user
from worker.tasks import process_csv_task, update_state_task
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("/upload")
async def upload_dataset(
    name: str = Form(...),
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

class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    hour: Optional[int] = datetime.now().hour
    day_of_week: Optional[str] = datetime.now().strftime("%A").lower()
    weather: Optional[str] = "clear"
    lighting_condition: Optional[str] = "daylight"
    visibility_level: Optional[str] = "good"
    road_surface_condition: Optional[str] = "dry"
    traffic_density: Optional[str] = "medium"
    road_type: Optional[str] = "urban road"
    speed_limit: Optional[int] = 60
    traffic_control_presence: Optional[str] = "none"
    number_of_lanes: Optional[int] = 2
    sharp_turn_or_blind_curve: Optional[bool] = False
    road_construction_present: Optional[bool] = False
    is_festival_day: Optional[bool] = False
    is_public_holiday: Optional[bool] = False
    is_hotspot: Optional[bool] = False
    severity_trend: Optional[str] = "medium"
    crowd_level: Optional[str] = "low"
    special_traffic_diversion: Optional[bool] = False

@router.post("/{project_id}/predict")
async def predict(
    project_id: str,
    req: PredictRequest,
    current_user_id: str = Depends(get_current_user)
):
    # 1. Verify project and get latest state
    project = await fetch_row(
        "SELECT latest_state_id, status FROM projects WHERE id = $1 AND user_id = $2",
        project_id, current_user_id
    )
    if not project or project["status"] != 'completed':
        raise HTTPException(status_code=400, detail="Project not ready or access denied.")

    # 2. Load latest state JSON
    state = await fetch_row("SELECT json_url FROM project_states WHERE id = $1", project["latest_state_id"])
    if not state:
        raise HTTPException(status_code=404, detail="Project state not found.")

    # In a real app, we'd download the JSON from Supabase Storage and parse it.
    # For now, we'll assume the state is accessible or we verify the closest deployment.
    # TO SIMULATE: We'll skip the heavy download and use a placeholder for nearest ambulance
    # unless we want to implement the full download & parse logic here.
    
    # 3. Calculate Risk
    data = req.dict()
    features = prediction_service.preprocess_features(data)
    prob, level = prediction_service.calculate_risk(features)
    ai_reasoning = await prediction_service.get_ai_reasoning(data, prob, level)

    # 4. Queue state update (async)
    update_state_task.delay(project_id, req.latitude, req.longitude, data)

    return {
        "project_id": project_id,
        "input": data,
        "risk_profile": {
            "probability": prob,
            "level": level,
            "ai_reasoning": ai_reasoning
        },
        "assigned_ambulance": {
            "status": "calculating",
            "message": "The nearest ambulance is being dispatched based on the synchronized project state."
        }
    }
