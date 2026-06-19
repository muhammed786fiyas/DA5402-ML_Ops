"""
DA5402 A8 - Spark pipeline for NYC Yellow Taxi cleansing + transformation.

Stages:
  1. Read parquet trip files from a directory
  2. Cleanse: drop nulls on key cols, dedupe, normalize timestamps
  3. Heavy join with the zone lookup table on PULocationID and DOLocationID
  4. Apply a Python UDF to compute average speed per trip-hour
  5. Write the result as a partitioned parquet dataset

Run from the master node:
  spark-submit \
      --master spark://<master-ip>:7077 \
      --deploy-mode client \
      --conf spark.sql.shuffle.partitions=64 \
      --conf spark.executor.memory=20g \
      --conf spark.executor.cores=8 \
      pipeline_spark.py

Or just `python pipeline_spark.py` if you've set MASTER_URL below to local[*]
for a single-machine sanity check.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, StructType, StructField, TimestampType, LongType


# Columns we care about. Keeping a fixed schema list so any extra cols added
# in newer parquet revisions don't break the pipeline.
TRIP_COLS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "total_amount",
]

TRIP_SCHEMA = StructType([
    StructField("tpep_pickup_datetime",  TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("trip_distance",         DoubleType(),    True),
    StructField("PULocationID",          LongType(),      True),
    StructField("DOLocationID",          LongType(),      True),
    StructField("fare_amount",           DoubleType(),    True),
    StructField("total_amount",          DoubleType(),    True),
])



def build_session(master_url: str, app_name: str = "da5402_a8_spark") -> SparkSession:
    """One place to control all the Spark knobs we care about for the bench."""
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        # Arrow makes pandas <-> Spark transfer 10x faster, big deal for UDFs
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        # Let AQE coalesce shuffle partitions adaptively
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Sensible default for our 2-node cluster
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "uncompressed")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )
    return builder.getOrCreate()


def speed_udf_python(distance, pickup_ts, dropoff_ts):
    """
    Pure-python UDF (the slow kind). We use this on purpose to demonstrate
    JVM <-> Python serialization overhead vs Ray's native execution.

    Returns mph. None for malformed/zero-duration trips.
    """
    if distance is None or pickup_ts is None or dropoff_ts is None:
        return None
    duration_sec = (dropoff_ts - pickup_ts).total_seconds()
    if duration_sec <= 0:
        return None
    hours = duration_sec / 3600.0
    speed = distance / hours
    # Cap absurd values - sensors get noisy, drivers don't actually do 200mph
    if speed < 0 or speed > 200:
        return None
    return float(speed)


def run(input_dir: str, zone_lookup: str, output_dir: str, master_url: str) -> dict:
    """Run the full pipeline and return a metrics dict."""
    metrics = {"framework": "spark", "started_at": datetime.utcnow().isoformat()}
    spark = build_session(master_url)
    # Quieter logs during benchmark
    spark.sparkContext.setLogLevel("WARN")

    t_start = time.perf_counter()

    # --- 1. Ingestion ---
    t0 = time.perf_counter()
    trips_raw = spark.read.schema(TRIP_SCHEMA).parquet(input_dir)
    zones = spark.read.option("header", "true").csv(zone_lookup)
    # Trigger a count so we time ingestion in isolation. Cache to avoid re-reading.
    trips_raw = trips_raw.cache()
    raw_count = trips_raw.count()
    metrics["ingest_sec"] = round(time.perf_counter() - t0, 3)
    metrics["raw_rows"] = raw_count
    print(f"[ingest] {raw_count:,} rows in {metrics['ingest_sec']}s")

    # --- 2. Cleansing ---
    t0 = time.perf_counter()
    cleaned = (
        trips_raw
        .dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                        "PULocationID", "DOLocationID", "trip_distance"])
        .dropDuplicates()
        # Drop nonsense rows: zero/negative distance, zero/negative duration
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
        # Truncate pickup to the hour - we'll group by this later
        .withColumn("pickup_hour", F.date_trunc("hour", F.col("tpep_pickup_datetime")))
    )
    cleaned = cleaned.cache()
    clean_count = cleaned.count()
    metrics["clean_sec"] = round(time.perf_counter() - t0, 3)
    metrics["clean_rows"] = clean_count
    print(f"[clean]  {clean_count:,} rows in {metrics['clean_sec']}s")

    # --- 3a. Heavy join on both PU and DO location IDs ---
    # zones table is small (~265 rows) but we join it twice to enrich both
    # pickup and dropoff locations. We force a broadcast hint to avoid an
    # expensive shuffle for such a small dim table.
    zones_pu = zones.select(
        F.col("LocationID").cast("int").alias("PULocationID"),
        F.col("Borough").alias("pu_borough"),
        F.col("Zone").alias("pu_zone"),
    )
    zones_do = zones.select(
        F.col("LocationID").cast("int").alias("DOLocationID"),
        F.col("Borough").alias("do_borough"),
        F.col("Zone").alias("do_zone"),
    )

    t0 = time.perf_counter()
    joined = (
        cleaned
        .join(F.broadcast(zones_pu), on="PULocationID", how="inner")
        .join(F.broadcast(zones_do), on="DOLocationID", how="inner")
    )
    # Force materialization of the join
    joined = joined.cache()
    join_count = joined.count()
    metrics["join_sec"] = round(time.perf_counter() - t0, 3)
    metrics["join_rows"] = join_count
    print(f"[join]   {join_count:,} rows in {metrics['join_sec']}s")

    # --- 3b. Python UDF for derived feature ---
    speed_spark_udf = F.udf(speed_udf_python, DoubleType())
    t0 = time.perf_counter()
    enriched = joined.withColumn(
        "avg_speed_mph",
        speed_spark_udf(
            F.col("trip_distance"),
            F.col("tpep_pickup_datetime"),
            F.col("tpep_dropoff_datetime"),
        ),
    ).filter(F.col("avg_speed_mph").isNotNull())
    # Aggregate per pickup zone + hour - this is the actual ML-ready feature table
    feature_df = (
        enriched
        .groupBy("pickup_hour", "pu_borough", "pu_zone")
        .agg(
            F.count("*").alias("trip_count"),
            F.avg("avg_speed_mph").alias("mean_speed"),
            F.avg("trip_distance").alias("mean_distance"),
            F.avg("total_amount").alias("mean_fare"),
        )
    )
    feature_df = feature_df.cache()
    feat_count = feature_df.count()
    metrics["udf_and_agg_sec"] = round(time.perf_counter() - t0, 3)
    metrics["feature_rows"] = feat_count
    print(f"[udf]    {feat_count:,} feature rows in {metrics['udf_and_agg_sec']}s")

    # --- 4. Export ---
    t0 = time.perf_counter()
    out_path = os.path.join(output_dir, "spark_features.parquet")
    (
        feature_df
        .repartition(8)
        .write
        .mode("overwrite")
        .parquet(out_path)
    )
    metrics["export_sec"] = round(time.perf_counter() - t0, 3)
    print(f"[export] wrote to {out_path} in {metrics['export_sec']}s")

    metrics["total_sec"] = round(time.perf_counter() - t_start, 3)
    metrics["finished_at"] = datetime.utcnow().isoformat()
    print(f"\n[total] {metrics['total_sec']}s")

    spark.stop()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.expanduser("/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas/data/trips"))
    ap.add_argument("--zones", default=os.path.expanduser("/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas/data/taxi_zone_lookup.csv"))
    ap.add_argument("--output", default=os.path.expanduser("/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas/output"))
    ap.add_argument("--master", default="spark://192.168.1.10:7077",
                    help="Use 'local[*]' for single-node debug")
    ap.add_argument("--metrics-out", default=os.path.expanduser("/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas/logs/spark_metrics.json"))
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    m = run(args.input, args.zones, args.output, args.master)
    with open(args.metrics_out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"\nmetrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
