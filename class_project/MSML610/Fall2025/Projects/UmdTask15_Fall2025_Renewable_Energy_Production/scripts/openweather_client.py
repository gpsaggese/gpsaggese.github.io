import os
import requests

def fetch_openweather_current(lat: float, lon: float, units: str = "metric") -> dict:
    """
    Fetch current weather from OpenWeatherMap using OPENWEATHER_API_KEY.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set in the environment")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": units}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def extract_features_from_openweather(payload: dict) -> dict:
    """
    Map OpenWeather fields to your model's weather features.
    """
    temp_c = float(payload["main"]["temp"])
    cloud_cover = float(payload.get("clouds", {}).get("all", 0.0))
    wind_speed = float(payload.get("wind", {}).get("speed", 0.0))

    # Not provided by the free current weather endpoint
    solar_radiation = 0.0

    return {
        "temp_c": temp_c,
        "cloud_cover": cloud_cover,
        "solar_radiation": solar_radiation,
        "wind_speed": wind_speed,
    }
