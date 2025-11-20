# Merge hourly Meteostat weather onto the ETL output.
# For each origin airport, find the nearest station (<= ~25km),
# fetch hourly weather across the date range, and join on the hour.
# Writes Parquet with microsecond timestamps so Spark can read it cleanly.


'''

No weather data provided in the project description, 
so had to do research outside of the project based on what was provided in the dataset in order to research another dataset that incorporated weather data/conditions based on the data and city wise information. 
This made it difficult because I had to communicate with one of the teacher assistants to clarify this, so this took some time to find a valid source for the weather data, which would be compatible and in-line with the other 3 datasets we had. 


'''

import os
from datetime import timedelta
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from weather_meteostat import nearest_station, hourly_weather_for_station


def load_parquet_dataset(path: str) -> pd.DataFrame:
    """read a partitioned parquet folder as one dataset"""
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table()  # all columns; fine for a single pass
    return table.to_pandas()


def save_parquet(df: pd.DataFrame, path: str):
    """write Parquet in a Spark-friendly way (microsecond timestamps, no index)"""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        coerce_timestamps="us",           # <- key: Spark expects us/ms, not ns
        allow_truncated_timestamps=True
    )


def run(
    processed_dir: str = "data/processed",
    in_name: str = "flights_enriched",
    out_path: str = "data/processed/flights_with_weather.parquet",
    station_cache_path: str = "data/processed/origin_station_cache.csv",
    max_km: int = 25,
    use_imperial: bool = False,
):
    # cut down on noisy pandas warnings from downstream libs
    warnings.filterwarnings("ignore", message="Support for nested sequences for 'parse_dates'")
    warnings.filterwarnings("ignore", message="'H' is deprecated")

    flights_path = os.path.join(processed_dir, in_name)
    flights = load_parquet_dataset(flights_path)

    # sanity: these columns need to exist (created by ETL)
    needed = ["ORIGIN_AIRPORT", "ORIGIN_LAT", "ORIGIN_LON", "dep_hour_rounded"]
    for c in needed:
        if c not in flights.columns:
            raise ValueError(f"Expected '{c}' in {flights_path}. Did ETL run?")

    # only rows with usable coords can be mapped to a station
    flights_ok = flights.dropna(subset=["ORIGIN_LAT", "ORIGIN_LON"]).copy()

    # build or load a tiny airport->station cache so re-runs are faster
    if os.path.exists(station_cache_path):
        airport_station = pd.read_csv(station_cache_path)
    else:
        airport_station = (
            flights_ok[["ORIGIN_AIRPORT", "ORIGIN_LAT", "ORIGIN_LON"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        airport_station["station_id"] = airport_station.apply(
            lambda r: nearest_station(float(r["ORIGIN_LAT"]), float(r["ORIGIN_LON"]), max_km=max_km),
            axis=1
        )
        airport_station.to_csv(station_cache_path, index=False)

    # join station ids back to flights
    flights_ok = flights_ok.merge(
        airport_station,
        on=["ORIGIN_AIRPORT", "ORIGIN_LAT", "ORIGIN_LON"],
        how="left"
    )

    # make sure both time columns we’ll join on are tz-naive + hourly
    dep_ts = pd.to_datetime(flights_ok["dep_hour_rounded"], errors="coerce")
    try:
        dep_ts = dep_ts.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    flights_ok["dep_hour_rounded"] = dep_ts.dt.floor("h")  # lowercase 'h' to avoid deprecation

    # time window: pad by a day on each side to be safe
    start = flights_ok["dep_hour_rounded"].min() - timedelta(days=1)
    end   = flights_ok["dep_hour_rounded"].max() + timedelta(days=1)

    # fetch weather once per station
    weather_frames = []
    stations = flights_ok["station_id"].dropna().unique()
    for sid in stations:
        w = hourly_weather_for_station(sid, start=start.to_pydatetime(), end=end.to_pydatetime(), use_imperial=use_imperial)
        if w.empty:
            continue
        w["station_id"] = sid
        # make this tz-naive + hourly too
        w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce")
        try:
            w["datetime"] = w["datetime"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        w["datetime"] = w["datetime"].dt.floor("h")
        weather_frames.append(w)

    if weather_frames:
        weather = pd.concat(weather_frames, ignore_index=True)
    else:
        weather = pd.DataFrame(columns=["station_id", "datetime"])

    # join on (station_id, hour)
    merged = flights_ok.merge(
        weather,
        left_on=["station_id", "dep_hour_rounded"],
        right_on=["station_id", "datetime"],
        how="left"
    )

    # we don't actually need the weather-side datetime after the join
    if "datetime" in merged.columns:
        merged = merged.drop(columns=["datetime"])

    # stick back flights that had missing coords (they just won't have weather)
    missing_mask = ~flights.index.isin(flights_ok.index)
    flights_missing = flights.loc[missing_mask]
    final = pd.concat([merged, flights_missing], ignore_index=True, sort=False)

    save_parquet(final, out_path)
    print(f"Wrote merged dataset: {out_path}, shape={final.shape}")


if __name__ == "__main__":
    run()
