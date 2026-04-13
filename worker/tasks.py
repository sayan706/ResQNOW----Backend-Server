import os
import json
import uuid
import asyncio
from datetime import datetime
from worker.celery_app import celery_app
from app.services.storage import storage_service
from app.services.optimization import (
    process_ambulance_optimization, 
    reconstruct_state_from_dict, 
    add_real_time_point, 
    serialize_state_to_json
)
from app.services.db.supabase import execute_query, fetch_row, fetch_all
from app.services.cache import cache_service
from app.services.prediction_service import prediction_service

@celery_app.task(name="worker.tasks.process_csv_task", bind=True)
def process_csv_task(self, project_id: str, storage_path: str, user_id: str):
    """
    Initial background processing of CSV into ambulance placement zones.
    """
    try:
        # 1. Download file from storage
        local_csv = f"/tmp/{uuid.uuid4()}.csv"
        storage_service.download_file(storage_path, local_csv)
        
        # 2. Run optimization logic
        result_json = process_ambulance_optimization(local_csv)
        os.remove(local_csv)
        
        # 3. Handle Versioning (Initial is always 1)
        version = 1
        
        # 4. Storage Logic: {user_id}/{project_id}/v{version}.json
        json_filename = f"{user_id}/{project_id}/v{version}.json"
        storage_service.upload_json(json_filename, json.dumps(result_json))
        json_url = storage_service.get_public_url(json_filename)
        
        # 5. Database Update (Atomic)
        loop = asyncio.get_event_loop()
        state_id = loop.run_until_complete(finalize_initial_state(project_id, json_url, version))
        
        # Clear cache for this project
        cache_service.delete(f"state:{project_id}")
        
        return {"status": "success", "project_id": project_id, "state_id": state_id}
    except Exception as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(execute_query(
            "UPDATE projects SET status = 'failed' WHERE id = $1", project_id
        ))
        return {"status": "failed", "error": str(e)}

async def finalize_initial_state(project_id, json_url, version):
    # Insert project_state
    row = await fetch_row(
        "INSERT INTO project_states (project_id, json_url, state_type, version) VALUES ($1, $2, 'initial', $3) RETURNING id",
        project_id, json_url, version
    )
    state_id = row['id']
    # Update project with status and latest_state_id
    await execute_query(
        "UPDATE projects SET status = 'completed', latest_state_id = $1 WHERE id = $2",
        state_id, project_id
    )
    return str(state_id)

@celery_app.task(name="worker.tasks.update_state_task", bind=True)
def update_state_task(self, project_id: str, latitude: float, longitude: float, context: dict):
    """
    Update system state with new real-time prediction point.
    - Synchronizes with latest state
    - Performs coverage check
    - Dynamically updates JSON state and version
    """
    try:
        loop = asyncio.get_event_loop()
        
        # 1. Fetch Latest Metadata
        project = loop.run_until_complete(fetch_row(
            """
            SELECT p.user_id, s.json_url, s.version 
            FROM projects p
            JOIN project_states s ON p.latest_state_id = s.id
            WHERE p.id = $1
            """, 
            project_id
        ))
        user_id = project['user_id']
        old_json_url = project['json_url']
        
        # 2. Download and Parse Current State
        local_json = f"/tmp/{uuid.uuid4()}.json"
        
        # Surgical path extraction: everything after the bucket name in the URL
        bucket_prefix = f"/{storage_service.bucket_name}/"
        if bucket_prefix in old_json_url:
            relative_path = old_json_url.split(bucket_prefix, 1)[-1].split("?")[0]
        else:
            # Fallback for unexpected URL formats
            relative_path = old_json_url.split("/")[-1].split("?")[0]

        storage_service.download_file(relative_path, local_json)
        
        with open(local_json, 'r') as f:
            state_data = json.load(f)
        os.remove(local_json)
        
        all_results = reconstruct_state_from_dict(state_data)
        
        # 3. Predict Risk Level for the new point
        features = prediction_service.preprocess_features(context)
        _, level_str = prediction_service.calculate_risk(features)
        # Use Red, Orange, Green strings as requested by user
        risk_level = "Red" if level_str in ["Critical", "High"] else "Orange" if level_str == "Medium" else "Green"
        
        # Determine Period
        hour = context.get('hour', datetime.now().hour)
        period = 'Morning' if 6<=hour<12 else 'Afternoon' if 12<=hour<18 else 'Evening' if 18<=hour<22 else 'Night'

        # 4. Update the state (Coverage check + Add Zone/Ambulance)
        updated_state = add_real_time_point(all_results, period, latitude, longitude, risk_level)
        
        # 5. Atomic DB Bump
        new_state_db = loop.run_until_complete(fetch_row(
            """
            INSERT INTO project_states (project_id, json_url, state_type, version)
            SELECT $1, 'pending', 'updated', COALESCE(MAX(version), 0) + 1
            FROM project_states WHERE project_id = $1
            RETURNING id, version
            """,
            project_id
        ))
        state_id = new_state_db['id']
        version = new_state_db['version']
        
        # 6. Serialize and Upload
        new_json_data = serialize_state_to_json(updated_state)
        json_filename = f"{user_id}/{project_id}/v{version}.json"
        storage_service.upload_json(json_filename, new_json_data)
        json_url = storage_service.get_public_url(json_filename)
        
        # 7. Finalize Linked State
        loop.run_until_complete(execute_query(
            "UPDATE project_states SET json_url = $1 WHERE id = $2", json_url, state_id
        ))
        loop.run_until_complete(execute_query(
            "UPDATE projects SET latest_state_id = $1 WHERE id = $2", state_id, project_id
        ))
        
        # Invalidate cache
        cache_service.delete(f"state:{project_id}")
        
        return {"status": "updated", "version": version, "state_id": str(state_id), "risk": risk_level}
    except Exception as e:
        print(f"Update Task Failed: {str(e)}")
        return {"status": "failed", "error": str(e)}
