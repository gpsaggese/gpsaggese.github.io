'''
Units: we used SI (Celsius, mm, m/s)
'''

from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
from meteostat import Hourly, Stations, Point, units

def nearest_station(lat: float, lon: float, max_km: int = 25) -> Optional[str]:
    stations = Stations().nearby(lat, lon).inventory('hourly')
    df = stations.fetch(10)
    if df.empty:
        return None
    # Return the first station id (WMO / USAF / WBAN composite when available)
    return df.index[0]

def hourly_weather_for_point(lat: float, lon: float, start: datetime, end: datetime, use_imperial: bool=False) -> pd.DataFrame:
    loc = Point(lat, lon)
    data = Hourly(loc, start, end)
    if use_imperial:
        data = data.convert(units.imperial)
    df = data.fetch().reset_index().rename(columns={"time":"datetime"})
    # Keep typical predictors
    cols = ["datetime","temp","dwpt","rhum","prcp","snow","wdir","wspd","wpgt","pres","tsun"]
    return df[[c for c in cols if c in df.columns]]

def hourly_weather_for_station(station_id: str, start: datetime, end: datetime, use_imperial: bool=False) -> pd.DataFrame:
    data = Hourly(station_id, start, end)
    if use_imperial:
        data = data.convert(units.imperial)
    df = data.fetch().reset_index().rename(columns={"time":"datetime"})
    cols = ["datetime","temp","dwpt","rhum","prcp","snow","wdir","wspd","wpgt","pres","tsun"]
    return df[[c for c in cols if c in df.columns]]
