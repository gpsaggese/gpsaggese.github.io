import os
from pyspark.sql import SparkSession, functions as F


def build_spark(app_name: str = "airline-delay-etl") -> SparkSession:
    # start a local Spark session. arrow speeds up pandas-style stuff if we ever collect small subsets
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def read_csvs(spark: SparkSession, raw_dir: str):
    # read the three main CSVs from the Kaggle dataset (flights, airlines, airports)
    flights = spark.read.option("header", True).csv(os.path.join(raw_dir, "flights.csv"))
    airlines = spark.read.option("header", True).csv(os.path.join(raw_dir, "airlines.csv"))
    airports = spark.read.option("header", True).csv(os.path.join(raw_dir, "airports.csv"))
    return flights, airlines, airports


def cast_and_clean(flights):
    """
    clean up types on the main numeric/boolean-ish columns and create some helper columns.
    - YEAR/MONTH/DAY → ints
    - DELAYS, CANCELLED, DIVERTED, AIR_TIME, DISTANCE → doubles
    - FL_DATE → actual date column
    - dep_hour_rounded → timestamp rounded to the scheduled departure HOUR, which we'll use later
      when we try to merge hourly weather (weather data is hourly, not per-minute)
    """
    flights = (
        flights
        .withColumn("YEAR", F.col("YEAR").cast("int"))
        .withColumn("MONTH", F.col("MONTH").cast("int"))
        .withColumn("DAY", F.col("DAY").cast("int"))
        .withColumn("DEPARTURE_DELAY", F.col("DEPARTURE_DELAY").cast("double"))
        .withColumn("ARRIVAL_DELAY", F.col("ARRIVAL_DELAY").cast("double"))
        .withColumn("CANCELLED", F.col("CANCELLED").cast("double"))
        .withColumn("DIVERTED", F.col("DIVERTED").cast("double"))
        .withColumn("AIR_TIME", F.col("AIR_TIME").cast("double"))
        .withColumn("DISTANCE", F.col("DISTANCE").cast("double"))
        .withColumn("FL_DATE", F.to_date(F.concat_ws("-", "YEAR", "MONTH", "DAY")))
    )

    # best-effort "hour bucket" for weather join:
    # SCHEDULED_DEPARTURE is usually HHMM as a string, e.g. "1735".
    # take the hour part (17), build "YYYY-MM-DD HH:00:00", cast that to timestamp.
    hour_int = F.when(
        F.length(F.col("SCHEDULED_DEPARTURE")) >= 3,
        (F.col("SCHEDULED_DEPARTURE").cast("int") / 100).cast("int")
    )
    hour_str = F.format_string("%02d", F.coalesce(hour_int, F.lit(0)))
    dep_ts = F.to_timestamp(
        F.concat_ws(
            " ",
            F.col("FL_DATE"),
            F.concat(hour_str, F.lit(":00:00"))
        )
    )
    flights = flights.withColumn("dep_hour_rounded", dep_ts)

    return flights


def make_label(flights, threshold_minutes: int = 15):
    """
    create the classification target "is_delayed":
    - if flight was cancelled or diverted, we don't force a label because there's no arrival time
      (these rows will basically be ignored in supervised training/eval)
    - else: 1 if ARRIVAL_DELAY >= threshold (default 15 min, like BTS on-time definition),
            0 otherwise
    """
    flights = flights.withColumn(
        "is_delayed",
        F.when(
            (F.col("CANCELLED") == 1) | (F.col("DIVERTED") == 1),
            F.lit(None).cast("int")
        ).otherwise(
            (F.col("ARRIVAL_DELAY") >= F.lit(threshold_minutes)).cast("int")
        )
    )
    return flights


def join_airlines_airports(flights, airlines, airports):
    """
    join airline names + airport metadata (city/state/lat/lon)
    we alias columns up front so we don't get duplicate column names from Spark
    """

    # airline info:
    # airlines.csv has IATA_CODE (like "AA") and AIRLINE (full name)
    airlines_sel = airlines.select(
        F.col("IATA_CODE").alias("AIRLINE"),
        F.col("AIRLINE").alias("AIRLINE_NAME")
    )
    flights = flights.join(airlines_sel, on="AIRLINE", how="left")

    # origin airport info
    airports_origin = airports.select(
        F.col("IATA_CODE").alias("ORIGIN_AIRPORT"),
        F.col("CITY").alias("ORIGIN_CITY"),
        F.col("STATE").alias("ORIGIN_STATE"),
        F.col("LATITUDE").alias("ORIGIN_LAT"),
        F.col("LONGITUDE").alias("ORIGIN_LON"),
    )
    flights = flights.join(airports_origin, on="ORIGIN_AIRPORT", how="left")

    # destination airport info
    airports_dest = airports.select(
        F.col("IATA_CODE").alias("DESTINATION_AIRPORT"),
        F.col("CITY").alias("DEST_CITY"),
        F.col("STATE").alias("DEST_STATE"),
        F.col("LATITUDE").alias("DEST_LAT"),
        F.col("LONGITUDE").alias("DEST_LON"),
    )
    flights = flights.join(airports_dest, on="DESTINATION_AIRPORT", how="left")

    # coords from airports.csv come in as strings; cast once here so downstream math behaves
    flights = (
        flights
        .withColumn("ORIGIN_LAT", F.col("ORIGIN_LAT").cast("double"))
        .withColumn("ORIGIN_LON", F.col("ORIGIN_LON").cast("double"))
        .withColumn("DEST_LAT",   F.col("DEST_LAT").cast("double"))
        .withColumn("DEST_LON",   F.col("DEST_LON").cast("double"))
    )

    return flights


def write_parquet(df, out_dir: str, name: str):
    import shutil, pathlib
    path = os.path.join(out_dir, name)
    p = pathlib.Path(path)
    if p.exists():
        shutil.rmtree(p)  # nuke any leftover/LFS-restored parts
    (df
        # .repartition(8)  # optional: stable number of output files
        .write.mode("overwrite").parquet(path))
    print(f"Wrote: {path}")


def run(raw_dir: str = "data/raw", processed_dir: str = "data/processed", threshold_minutes: int = 15):
    """
    main driver:
    - read raw CSVs
    - clean + cast + build FL_DATE and dep_hour_rounded
    - generate label
    - attach airline + airport metadata
    - write one parquet dataset we’ll reuse everywhere else
    """
    spark = build_spark()

    # quick sanity just to help catch path mistakes early instead of failing deep in Spark
    for fname in ["flights.csv", "airlines.csv", "airports.csv"]:
        full_path = os.path.join(raw_dir, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Expected {full_path} but it doesn't exist")

    flights, airlines, airports = read_csvs(spark, raw_dir)
    flights = cast_and_clean(flights)
    flights = make_label(flights, threshold_minutes)
    flights = join_airlines_airports(flights, airlines, airports)
    write_parquet(flights, processed_dir, "flights_enriched")

    spark.stop()


if __name__ == "__main__":
    run()