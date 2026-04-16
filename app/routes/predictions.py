from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
import json

from app.routes.deps import get_current_user
from app.services.prediction_service import prediction_service
from app.services.db.supabase import execute_query

router = APIRouter(prefix="/prediction", tags=["prediction"])

class StandAlonePredictRequest(BaseModel):
    zone_name: Optional[str] = None
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

@router.post("/analyze")
async def analyze_zone(
    req: StandAlonePredictRequest,
    current_user_id: str = Depends(get_current_user)
):
    try:
        # Preprocess features into normalized weights
        data = req.dict()
        features = prediction_service.preprocess_features(data)
        
        # Calculate Risk Math
        prob, level = prediction_service.calculate_risk(features)
        
        # Get AI Interpretation and Rule-based recommendations
        ai_reasoning = await prediction_service.get_ai_reasoning(data, prob, level)
        key_factors = prediction_service.identify_key_risk_factors(features)
        recs = prediction_service.generate_recommendations(data, level)
        
        result_payload = {
            "risk_profile": {
                "probability": prob,
                "level": level,
                "ai_reasoning": ai_reasoning,
                "key_risk_factors": key_factors,
                "recommended_actions": recs
            }
        }
        
        # Save to PostgreSQL table
        await execute_query(
            """
            INSERT INTO predictions_history 
            (user_id, zone_name, latitude, longitude, input_data, risk_probability, risk_level, key_factors, recommendations, ai_reasoning)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            current_user_id,
            data.get("zone_name", "Unnamed Zone"),
            data["latitude"],
            data["longitude"],
            json.dumps(data),
            prob,
            level,
            json.dumps(key_factors),
            json.dumps(recs),
            ai_reasoning
        )
        
        return {
            "status": "success",
            "message": "Prediction created and saved to database successfully.",
            "data": result_payload
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
