import os
import json
import uuid
import asyncio
from worker.celery_app import celery_app
from app.services.storage import storage_service
from app.services.optimization import process_ambulance_optimization
from app.services.db.supabase import execute_query, fetch_row, fetch_all
from app.services.cache import cache_service

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
        
        # 4. Storage Logic: /project-files/{user_id}/{project_id}/v{version}.json
        json_filename = f"project-files/{user_id}/{project_id}/v{version}.json"
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
    Update system state with new real-time prediction point (Concurrent-safe).
    """
    try:
        loop = asyncio.get_event_loop()
        
        # 3. Robust Versioning Logic: Atomic version increment
        # We fetch project info including user_id for storage path
        project = loop.run_until_complete(fetch_row("SELECT user_id FROM projects WHERE id = $1", project_id))
        user_id = project['user_id']
        
        # Atomic Insert with MAX(version) + 1
        new_state = loop.run_until_complete(fetch_row(
            """
            INSERT INTO project_states (project_id, json_url, state_type, version)
            SELECT $1, 'pending', 'updated', COALESCE(MAX(version), 0) + 1
            FROM project_states WHERE project_id = $1
            RETURNING id, version
            """,
            project_id
        ))
        
        state_id = new_state['id']
        version = new_state['version']
        
        # 4. JSON Storage Logic: /project-files/{user_id}/{project_id}/v{version}.json
        # Mocking logic update: in a real app, you'd load previous state, add points, solve/update
        # For now, we'll simulate the state file update
        simulated_data = {"version": version, "update": {"lat": latitude, "lng": longitude}}
        json_filename = f"project-files/{user_id}/{project_id}/v{version}.json"
        storage_service.upload_json(json_filename, json.dumps(simulated_data))
        json_url = storage_service.get_public_url(json_filename)
        
        # Finalize the URL and link as latest
        loop.run_until_complete(execute_query(
            "UPDATE project_states SET json_url = $1 WHERE id = $2", json_url, state_id
        ))
        loop.run_until_complete(execute_query(
            "UPDATE projects SET latest_state_id = $1 WHERE id = $2", state_id, project_id
        ))
        
        # Invalidate cache
        cache_service.delete(f"state:{project_id}")
        
        return {"status": "updated", "version": version, "state_id": str(state_id)}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
