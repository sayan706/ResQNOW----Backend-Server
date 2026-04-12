import os
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

WEIGHTS = {
    "time_of_day": 8, "weekend": 4, "season_month": 3, "lighting_condition": 5,
    "weather": 10, "visibility_level": 7, "rain_intensity": 6, "road_surface_condition": 8,
    "traffic_density": 9, "road_type": 7, "speed_factor": 6, "traffic_control_presence": 4,
    "number_of_lanes": 4, "sharp_turn_or_blind_curve": 4, "road_construction_present": 4,
    "historical_accident_count": 6, "severity_trend": 4, "hotspot": 5,
    "is_festival_day": 4, "is_public_holiday": 3, "crowd_level": 5,
    "special_traffic_diversion": 3, "night_event": 4,
    "hospital_distance": 2, "ambulance_response_time_min": 3, "emergency_support": 1,
}

MAX_THEORETICAL_SCORE = float(sum(WEIGHTS.values()))
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

class PredictionService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def _get_time_period(self, hour: int) -> str:
        if 6 <= hour < 12: return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        elif 18 <= hour < 22: return 'Evening'
        else: return 'Night'

    def preprocess_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        f: Dict[str, float] = {}
        
        # Time & Calendar
        tod_map = {"morning": 0.3, "afternoon": 0.4, "evening": 0.6, "night": 0.85, "late night": 1.0}
        f["time_of_day"] = tod_map.get(data.get("time_of_day", "morning"), 0.5)
        f["weekend"] = 1.0 if data.get("is_weekend") else 0.0
        f["season_month"] = 0.8 if data.get("month", 1) in {6, 7, 8, 9, 11, 12, 1} else 0.3
        
        # Environment
        weather_map = {"clear": 0.0, "cloudy": 0.2, "rain": 0.5, "heavy rain": 0.85, "fog": 0.9, "storm": 1.0}
        f["weather"] = weather_map.get(data.get("weather", "clear"), 0.3)
        vis_map = {"good": 0.0, "moderate": 0.5, "poor": 1.0}
        f["visibility_level"] = vis_map.get(data.get("visibility_level", "good"), 0.3)
        rain_map = {"none": 0.0, "light": 0.3, "medium": 0.6, "heavy": 1.0}
        f["rain_intensity"] = rain_map.get(data.get("rain_intensity", "none"), 0.0)
        surf_map = {"dry": 0.0, "wet": 0.4, "muddy": 0.6, "icy": 1.0, "damaged": 0.8}
        f["road_surface_condition"] = surf_map.get(data.get("road_surface_condition", "dry"), 0.2)
        
        # Traffic
        td_map = {"low": 0.1, "medium": 0.4, "high": 0.75, "very high": 1.0}
        f["traffic_density"] = td_map.get(data.get("traffic_density", "medium"), 0.4)
        rt_map = {"highway": 0.75, "urban road": 0.5, "rural road": 0.4, "intersection": 0.9, "market area": 0.8}
        f["road_type"] = rt_map.get(data.get("road_type", "urban road"), 0.5)
        
        # Risk Profile
        f["hotspot"] = 1.0 if data.get("is_hotspot") else 0.0
        st_map = {"low": 0.2, "medium": 0.5, "high": 1.0}
        f["severity_trend"] = st_map.get(data.get("severity_trend", "medium"), 0.5)
        
        # Emergency
        art = data.get("ambulance_response_time_min", 15)
        f["ambulance_response_time_min"] = min(art / 30.0, 1.0)

        return f

    def calculate_risk(self, features: Dict[str, float]) -> Tuple[float, str]:
        raw_score = sum(features.get(k, 0.0) * w for k, w in WEIGHTS.items())
        probability = (raw_score / MAX_THEORETICAL_SCORE) * 100.0
        
        level = "Low"
        if probability >= 75: level = "Critical"
        elif probability >= 50: level = "High"
        elif probability >= 25: level = "Medium"
        
        return round(probability, 2), level

    async def get_ai_reasoning(self, data: Dict[str, Any], prob: float, level: str) -> str:
        if not self.api_key:
            return "AI reasoning unavailable: GEMINI_API_KEY not configured."
        
        prompt = f"""
        Analyze the following accident risk:
        Location: ({data['latitude']}, {data['longitude']})
        Weather: {data.get('weather')}
        Traffic: {data.get('traffic_density')}
        Risk Level: {level} ({prob}%)
        
        Provide:
        1. Reasoning for this risk level.
        2. 3 concrete emergency recommendations.
        3. Pre-positioning advice for ambulances in this zone.
        """
        
        try:
            for model_name in GEMINI_MODELS:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content(prompt)
                    if res.text: return res.text
                except Exception as e:
                    print(f"Gemini error with {model_name}: {str(e)}")
                    continue
            return "AI failed to generate a response. Check server logs."
        except Exception as e:
            return f"Service Error: {str(e)}"

    def get_haversine_time(self, lat1, lon1, lat2, lon2):
        R = 6371
        la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = la2 - la1, lo2 - lo1
        a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
        dist = R * 2 * math.asin(math.sqrt(a))
        return dist * 1.5 / 30 * 60 # min

prediction_service = PredictionService()
