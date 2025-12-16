import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def hhmm_to_hour(x):
    try:
        x = int(x)
        return x // 100
    except:
        return None

def main():
    print("[INFO] Loading flights.csv...")
    flights = pd.read_csv(RAW / "flights.csv", low_memory=False)

    print("[INFO] Converting date/time...")
    flights["DATE"] = pd.to_datetime(flights[["YEAR","MONTH","DAY"]])
    flights["DEP_HOUR"] = flights["SCHEDULED_DEPARTURE"].apply(hhmm_to_hour)
    flights["ARR_HOUR"] = flights["SCHEDULED_ARRIVAL"].apply(hhmm_to_hour)

    print("[INFO] Building hourly table...")

    # Count departures
    dep = (flights.groupby(["ORIGIN_AIRPORT", "DATE", "DEP_HOUR"])
           .size().reset_index(name="departures"))

    # Count arrivals
    arr = (flights.groupby(["DESTINATION_AIRPORT", "DATE", "ARR_HOUR"])
           .size().reset_index(name="arrivals"))

    # Rename columns to match
    dep.rename(columns={"ORIGIN_AIRPORT": "AIRPORT", "DEP_HOUR": "HOUR"}, inplace=True)
    arr.rename(columns={"DESTINATION_AIRPORT": "AIRPORT", "ARR_HOUR": "HOUR"}, inplace=True)

    print("[INFO] Merging dep + arr...")
    hourly = pd.merge(dep, arr, on=["AIRPORT", "DATE", "HOUR"], how="outer")

    hourly["departures"] = hourly["departures"].fillna(0).astype(int)
    hourly["arrivals"] = hourly["arrivals"].fillna(0).astype(int)
    hourly["total_flights"] = hourly["departures"] + hourly["arrivals"]

    # Simple labels
    hourly["congestion_level"] = pd.cut(
        hourly["total_flights"],
        bins=[-1, 20, 50, hourly["total_flights"].max()+1],
        labels=["Low", "Medium", "High"]
    )

    # Save as CSV (safe!)
    output_path = PROCESSED / "hourly_congestion.csv"
    hourly.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved: {output_path}")

if __name__ == "__main__":
    main()