import os
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import errors
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
GEMINI_MODELS = [
    "gemini-3-flash-preview",  # Top preference (State-of-the-art Flash)
    "gemini-2.5-flash",        # Strong fallback
    "gemini-2.0-flash",        # Reliable backup
]

class PredictionService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def _get_time_period(self, hour: int) -> str:
        if 6 <= hour < 12: return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        elif 18 <= hour < 22: return 'Evening'
        else: return 'Night'

    def preprocess_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        f: Dict[str, float] = {}
        
        # Time & Calendar
        hour = data.get("hour", datetime.now().hour)
        tod_map = {"morning": 0.3, "afternoon": 0.4, "evening": 0.6, "night": 0.85, "late night": 1.0}
        time_of_day = 'morning' if 6<=hour<12 else 'afternoon' if 12<=hour<17 else 'evening' if 17<=hour<21 else 'night'
        f["time_of_day"] = tod_map.get(time_of_day, 0.5)
        
        day = data.get("day_of_week", "").lower()
        f["weekend"] = 1.0 if day in ("saturday", "sunday") else 0.0
        f["season_month"] = 0.8 if datetime.now().month in {6, 7, 8, 9, 11, 12, 1} else 0.3
        
        # Environment
        light_map = {"daylight": 0.0, "twilight": 0.4, "dark_streetlights": 0.6, "dark_no_streetlights": 1.0}
        f["lighting_condition"] = light_map.get(data.get("lighting_condition", "daylight"), 0.3)
        
        weather = data.get("weather", "clear").lower()
        weather_map = {"clear": 0.0, "cloudy": 0.2, "rain": 0.5, "heavy rain": 0.85, "fog": 0.9, "storm": 1.0}
        f["weather"] = weather_map.get(weather, 0.3)
        
        vis_map = {"good": 0.0, "moderate": 0.5, "poor": 1.0}
        f["visibility_level"] = vis_map.get(data.get("visibility_level", "good"), 0.3)
        
        f["rain_intensity"] = 0.8 if "heavy" in weather or "storm" in weather else 0.5 if "rain" in weather else 0.0
        
        surf_map = {"dry": 0.0, "wet": 0.4, "muddy": 0.6, "icy": 1.0, "damaged": 0.8}
        f["road_surface_condition"] = surf_map.get(data.get("road_surface_condition", "dry"), 0.2)
        
        # Traffic & Road
        td_map = {"low": 0.1, "medium": 0.4, "high": 0.75, "very high": 1.0}
        f["traffic_density"] = td_map.get(data.get("traffic_density", "medium"), 0.4)
        
        rt_map = {"highway": 0.75, "urban road": 0.5, "rural road": 0.4, "intersection": 0.9, "market area": 0.8}
        f["road_type"] = rt_map.get(data.get("road_type", "urban road"), 0.5)
        
        speed = data.get("speed_limit", 60)
        f["speed_factor"] = 1.0 if speed >= 120 else 0.7 if speed >= 80 else 0.45 if speed >= 60 else 0.25
        
        tc_map = {"signal": 0.0, "stop_sign": 0.3, "pedestrian_crossing": 0.4, "none": 1.0}
        f["traffic_control_presence"] = tc_map.get(data.get("traffic_control_presence", "none"), 1.0)
        
        lanes = data.get("number_of_lanes", 2)
        f["number_of_lanes"] = 0.8 if lanes == 1 else 0.4 if lanes == 2 else 0.2
        
        f["sharp_turn_or_blind_curve"] = 1.0 if data.get("sharp_turn_or_blind_curve") else 0.0
        f["road_construction_present"] = 1.0 if data.get("road_construction_present") else 0.0
        
        # Risk Profile & Special
        f["hotspot"] = 1.0 if data.get("is_hotspot") else 0.0
        st_map = {"low": 0.2, "medium": 0.5, "high": 1.0}
        f["severity_trend"] = st_map.get(data.get("severity_trend", "medium"), 0.5)
        
        f["is_festival_day"] = 1.0 if data.get("is_festival_day") else 0.0
        f["is_public_holiday"] = 1.0 if data.get("is_public_holiday") else 0.0
        
        cl_map = {"low": 0.1, "medium": 0.4, "high": 0.75, "very high": 1.0}
        f["crowd_level"] = cl_map.get(data.get("crowd_level", "low"), 0.3)
        f["special_traffic_diversion"] = 1.0 if data.get("special_traffic_diversion") else 0.0
        
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
        if not self.client:
            return "AI reasoning unavailable: GEMINI_API_KEY not configured."
        
        prompt = f"""
        You are a senior traffic-safety and emergency-response analyst AI.
        
        ═══  ZONE RISK ASSESSMENT DATA  ═══
        Location: ({data.get('latitude')}, {data.get('longitude')})
        Time: {data.get('hour')}:00, {data.get('day_of_week')}
        Weather: {data.get('weather')}
        Visibility: {data.get('visibility_level')}
        Road Surface: {data.get('road_surface_condition')}
        Traffic Density: {data.get('traffic_density')}
        Road Type: {data.get('road_type')}
        Speed Limit: {data.get('speed_limit')} km/h
        Risk Level: {level} ({prob}%)
        Special Context: {'Festival' if data.get('is_festival_day') else 'Normal'}, {'Hotspot' if data.get('is_hotspot') else ''}
        
        Provide:
        1. **AI Risk Reasoning**: Explain *why* this zone has the predicted risk level based on the data.
        2. **Key Risk Factors**: List the most significant factors driving the risk.
        3. **Emergency Actions**: Provide 4-6 concrete, actionable recommendations for emergency teams.
        4. **Deployment Insight**: Advise on ambulance pre-positioning for this specific scenario.
        
        Format with clear numbered headings. Be specific and actionable.
        """
        
        for model_name in GEMINI_MODELS:
            try:
                print(f"Trying Gemini model: {model_name}")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                
                # Extract text safely
                final_text = ""
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            final_text += part.text
                
                if final_text.strip():
                    return final_text.strip()
                
            except errors.APIError as e:
                # 404: Model not found/deprecated, 429: Quota exceeded
                if e.code in [404, 429]:
                    print(f"Skipping {model_name} due to error {e.code}: {e.message}")
                    continue
                print(f"Gemini API error with {model_name}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error with {model_name}: {str(e)}")
                continue
                
        return "AI failed to generate a response (Quota/API limits reached). Please refer to baseline recommendation."

    def get_haversine_time(self, lat1, lon1, lat2, lon2):
        R = 6371
        la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = la2 - la1, lo2 - lo1
        a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
        dist = R * 2 * math.asin(math.sqrt(a))
        return dist * 1.5 / 30 * 60 # min

prediction_service = PredictionService()
