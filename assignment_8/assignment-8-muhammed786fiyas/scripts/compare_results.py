"""
Post-benchmark analysis:
  1. Verify Spark and Ray produced the same feature rows (parity check)
  2. Build the timing comparison table
  3. Plot CPU/RAM usage from monitor CSVs
  4. Compute the UDF-overhead deep-dive number

Run:
  python compare_results.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


ROOT = Path(os.path.expanduser("/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas"))
LOGS = ROOT / "logs"
OUT = ROOT / "output"
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)


def load_metrics():
    s = json.loads((LOGS / "spark_metrics.json").read_text())
    r = json.loads((LOGS / "ray_metrics.json").read_text())
    return s, r


def load_features(path: Path) -> pd.DataFrame:
    """Read all parquet parts under a directory or a single file."""
    if path.is_dir():
        df = pq.read_table(str(path)).to_pandas()
    else:
        df = pd.read_parquet(path)
    return df


def parity_check():
    print("\n=== PARITY CHECK ===")
    spark_path = OUT / "spark_features.parquet"
    ray_path = OUT / "ray_features.parquet"

    if not spark_path.exists() or not ray_path.exists():
        print("  output dirs missing - run both pipelines first")
        return None

    sdf = load_features(spark_path)
    rdf = load_features(ray_path)

    # Normalize column names from Ray's auto-generated agg cols
    # Ray's groupby names them like mean(avg_speed_mph), count()
    rdf = rdf.rename(columns={
        "count()": "trip_count",
        "mean(avg_speed_mph)": "mean_speed",
        "mean(trip_distance)": "mean_distance",
        "mean(total_amount)": "mean_fare",
    })

    print(f"  spark rows: {len(sdf):,}")
    print(f"  ray   rows: {len(rdf):,}")

    key = ["pickup_hour", "pu_borough", "pu_zone"]
    # Make pickup_hour comparable - both should be timestamps already
    sdf["pickup_hour"] = pd.to_datetime(sdf["pickup_hour"])
    rdf["pickup_hour"] = pd.to_datetime(rdf["pickup_hour"])

    merged = sdf.merge(rdf, on=key, suffixes=("_spark", "_ray"), how="inner")
    print(f"  matched on {key}: {len(merged):,} groups")

    # Compare numeric aggregates - allow small float tolerance
    diffs = {}
    for col in ["trip_count", "mean_speed", "mean_distance", "mean_fare"]:
        sc = f"{col}_spark"
        rc = f"{col}_ray"
        if sc in merged.columns and rc in merged.columns:
            d = (merged[sc] - merged[rc]).abs()
            diffs[col] = {
                "max_abs_diff": float(d.max()),
                "mean_abs_diff": float(d.mean()),
            }
    print("  numeric agreement:")
    for k, v in diffs.items():
        print(f"    {k:14s} max|Δ|={v['max_abs_diff']:.6f}  mean|Δ|={v['mean_abs_diff']:.6f}")

    parity_pct = len(merged) / max(len(sdf), len(rdf)) * 100 if max(len(sdf), len(rdf)) else 0.0
    print(f"  group-level parity: {parity_pct:.2f}%")

    return {
        "spark_rows": len(sdf),
        "ray_rows": len(rdf),
        "matched_rows": len(merged),
        "parity_pct": round(parity_pct, 2),
        "numeric_diffs": diffs,
    }


def timing_table(s_metrics, r_metrics):
    rows = []
    stages = [
        ("Ingestion",       "ingest_sec"),
        ("Cleansing",       "clean_sec"),
        ("Heavy Join",      "join_sec"),
        ("UDF + Aggregate", "udf_and_agg_sec"),
        ("Export",          "export_sec"),
        ("TOTAL",           "total_sec"),
    ]
    for label, key in stages:
        sv = s_metrics.get(key, float("nan"))
        rv = r_metrics.get(key, float("nan"))
        speedup = (sv / rv) if (rv and not np.isnan(rv) and rv > 0) else float("nan")
        rows.append({
            "Stage": label,
            "Spark (s)": sv,
            "Ray (s)": rv,
            "Spark/Ray ratio": round(speedup, 2),
        })
    return pd.DataFrame(rows)


def plot_resources():
    if not HAVE_MPL:
        print("  matplotlib not available, skipping plots")
        return
    monitor_csvs = sorted(LOGS.glob("*_node*.csv"))
    if not monitor_csvs:
        print("  no monitor csv files found")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for csv_path in monitor_csvs:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["t"] = df["ts"] - df["ts"].iloc[0]
        label = csv_path.stem
        axes[0].plot(df["t"], df["cpu_pct"], label=label, linewidth=1)
        axes[1].plot(df["t"], df["mem_used_gb"], label=label, linewidth=1)
    axes[0].set_ylabel("CPU %")
    axes[0].set_title("CPU utilization")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Memory used (GB)")
    axes[1].set_xlabel("seconds since start")
    axes[1].set_title("Memory utilization")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = REPORT / "resource_usage.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  saved {out}")


def plot_timings(table_df: pd.DataFrame):
    if not HAVE_MPL:
        return
    plot_df = table_df[table_df["Stage"] != "TOTAL"]
    x = np.arange(len(plot_df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, plot_df["Spark (s)"], w, label="Spark")
    ax.bar(x + w/2, plot_df["Ray (s)"], w, label="Ray")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Stage"], rotation=20, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("Per-stage execution time: Spark vs Ray")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = REPORT / "timing_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  saved {out}")


def peak_resources():
    """Peak CPU% and RAM (GB) per node from monitor CSVs."""
    rows = []
    for csv_path in sorted(LOGS.glob("*_node*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rows.append({
            "log": csv_path.stem,
            "peak_cpu_pct": float(df["cpu_pct"].max()),
            "peak_mem_gb":  float(df["mem_used_gb"].max()),
            "mean_cpu_pct": round(float(df["cpu_pct"].mean()), 1),
        })
    return pd.DataFrame(rows)


def main():
    s_metrics, r_metrics = load_metrics()
    parity = parity_check()

    print("\n=== TIMING TABLE ===")
    tt = timing_table(s_metrics, r_metrics)
    print(tt.to_string(index=False))
    tt.to_csv(REPORT / "timing_table.csv", index=False)

    print("\n=== UDF DEEP-DIVE ===")
    s_udf = s_metrics.get("udf_and_agg_sec", float("nan"))
    r_udf = r_metrics.get("udf_and_agg_sec", float("nan"))
    print(f"  Spark UDF + agg : {s_udf}s")
    print(f"  Ray   UDF + agg : {r_udf}s")
    if r_udf and r_udf > 0 and not np.isnan(s_udf):
        print(f"  Speedup (Spark/Ray): {s_udf/r_udf:.2f}x")

    print("\n=== RESOURCE PEAKS ===")
    pr = peak_resources()
    if not pr.empty:
        print(pr.to_string(index=False))
        pr.to_csv(REPORT / "resource_peaks.csv", index=False)

    print("\n=== PLOTS ===")
    plot_timings(tt)
    plot_resources()

    summary = {
        "spark_total_sec": s_metrics.get("total_sec"),
        "ray_total_sec":   r_metrics.get("total_sec"),
        "winner": "Ray" if r_metrics.get("total_sec", 1e9) < s_metrics.get("total_sec", 1e9) else "Spark",
        "parity": parity,
    }
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSUMMARY -> {REPORT/'summary.json'}")
    print(f"WINNER  -> {summary['winner']}")


if __name__ == "__main__":
    main()
