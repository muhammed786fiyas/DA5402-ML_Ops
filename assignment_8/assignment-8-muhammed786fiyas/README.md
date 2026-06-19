# DA5402 Assignment 8 — Spark vs Ray Distributed Benchmark

A 2-node distributed-computing benchmark comparing **Apache Spark 4.1.1** and **Ray 2.55.1** on a real ETL pipeline over the **NYC Yellow Taxi 2023 Q1** dataset (~9.4 million rows).

**Final result:** Spark wins by 2.8× on this workload.

---

## Cluster Setup

| Node | Role | Hardware | IP |
|---|---|---|---|
| Master | Spark driver + Ray head | Lenovo IdeaPad Slim 5 AI, 32 GB RAM, 8 cores | 10.250.139.217 |
| Worker | Spark worker + Ray worker | HP ENVY 14, 15 GB RAM, 8 cores | 10.250.139.214 |

Both nodes connected over the same campus WiFi subnet, same software stack:

- **Python** 3.13.5
- **Java** 21.0.10
- **PySpark** 4.1.1
- **Ray** 2.55.1 with `ray[data]`
- **PyArrow** 19.0.1
- **pandas** 2.3.3
- **numpy** 2.4.4

---

## Repository Structure

```
├── assignment-8-muhammed786fiyas
  ├── README.md                          # This file
  ├── .gitignore                         # Excludes venvs, raw data
  ├── data/
  │   ├── taxi_zone_lookup.csv           # 12 KB — zone name lookup, broadcast joined
  │   └── trips/                         # 145 MB raw NYC Yellow Taxi 2023 Q1 parquets
  │                                      #   (gitignored — fetch via scripts/01_get_data.sh)
  ├── scripts/
  │   ├── 01_get_data.sh                 # Downloads NYC taxi parquet files
  │   ├── 02_setup_env.sh                # Creates venv & installs all dependencies
  │   ├── 03_start_spark.sh              # Spark master/worker daemon launcher
  │   ├── 04_start_ray.sh                # Ray head/worker daemon launcher
  │   ├── 05_run_benchmarks.sh           # Convenience wrapper for full benchmark
  │   ├── spark_clean.py              # Spark pipeline (5 stages + instrumentation)
  │   ├── ray_clean.py                # Ray Data pipeline (parity-equivalent)
  │   ├── compare_results.py             # Pipeline parity & timing comparison
  │   └── monitor.py                     # Per-second psutil resource sampler
  ├── logs/                              # Run-time outputs
  │   ├── spark_metrics.json             # Cluster Spark stage timings (official)
  │   ├── spark_run.log                  # Cluster Spark full pipeline log
  │   ├── spark_metrics_singlenode.json  # Single-node Spark baseline
  │   ├── ray_metrics.json               # Cluster Ray stage timings (official)
  │   ├── ray_run.log                    # Cluster Ray full pipeline log
  │   ├── ray_metrics_singlenode.json    # Single-node Ray baseline
  │   ├── spark_node_master.csv          # CPU/memory per second (master)
  │   └── ray_node_master.csv            # CPU/memory per second (master)
  ├── output/                            # Pipeline outputs
  │   ├── spark_features.parquet/        # 200,848 feature rows from Spark cluster
  │   └── ray_features.parquet           # 197,392 feature rows from Ray cluster (post-aggregated)
  ├── report/                            # Analysis artifacts
  │   ├── summary.json                   # Final winner & headline numbers
  │   ├── timing_table.csv               # Stage-by-stage timing comparison
  │   ├── resource_peaks.csv             # Peak CPU & memory per framework
  │   ├── timing_comparison.png          # Bar chart of stage timings
  │   ├── resource_usage.png             # CPU/memory time series
  │   ├── screenshots/                   # Cluster orchestration evidence
  │   │   ├── spark_2node_ui.png         # Spark UI showing 2-node cluster
  │   │   ├── spark_completed_apps.png   # Completed app history
  │   │   ├── ray_2node_status.png       # ray status output (2 nodes)
  │   │   ├── ray_cluster_resources.png  # Cluster resources during run
  │   │   └── comparison_output.png      # Final compare_results.py output
  │   └── DA5402_Assignment_8_Report.pdf # Written report (3-5 pages)
  └── screencast/
  └── video_link.md                 # Google Drive link to recorded walkthrough
```
---

## Pipeline

Both implementations process the same 5-stage ETL on identical input:

1. **Ingest** — Read parquet files, project to typed schema (`TRIP_SCHEMA`)
2. **Cleanse** — Drop null pickup/dropoff, validate trip distance > 0, fare > 0, etc.
3. **Heavy Join** — Broadcast-join with the 265-row taxi zone lookup
4. **UDF + Aggregate** — Python UDF computes `avg_speed_mph` per trip; group by `(pickup_hour, pu_borough, pu_zone)`, aggregate count/mean speed/mean distance/mean fare
5. **Export** — Write final feature parquet

The Spark pipeline uses Catalyst SQL with a Python UDF (worst-case JVM↔Python serialization) and explicit broadcast hint. The Ray pipeline uses Ray Data with native Python `MapBatches`, `HashAggregate`, and broadcasted lookup.

---

## How to Reproduce

### 1. Get the data

```bash
bash scripts/01_get_data.sh
```

Downloads NYC Yellow Taxi 2023 Jan/Feb/Mar parquet files to `data/trips/`.

### 2. Set up the environment

```bash
bash scripts/02_setup_env.sh
source venv/bin/activate
```

Creates a Python 3.13.5 virtual environment with all pinned dependencies.

> Requires Python 3.13.5 and Java 21 system-wide. On Ubuntu, install Python 3.13.5 via pyenv (`pyenv install 3.13.5`) and Java via `sudo apt install openjdk-21-jdk`.

### 3. Single-node validation

To verify the pipelines work locally before clustering:

```bash
python scripts/pipeline_spark.py --master "local[8]" --metrics-out logs/spark_metrics.json
python scripts/pipeline_ray.py   --address local             --metrics-out logs/ray_metrics.json
python scripts/compare_results.py
```

### 4. 2-node cluster benchmark

**On the master node** (10.250.139.217):

```bash
# Start Spark master
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  bash scripts/03_start_spark.sh master

# Start Ray head
ray start --head --node-ip-address=10.250.139.217 --port=6379 --dashboard-host=0.0.0.0
```

**On the worker node** (10.250.139.214):

```bash
# Start Spark worker (must match master's Java + Spark version)
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  ~/spark/sbin/start-worker.sh --cores 8 --memory 12g spark://10.250.139.217:7077

# Join Ray cluster
ray start --address='10.250.139.217:6379'
```

**Back on master**, run pipelines against the cluster:

```bash
python scripts/pipeline_spark.py \
    --master "spark://10.250.139.217:7077" \
    --metrics-out logs/spark_metrics.json

python scripts/pipeline_ray.py \
    --address "10.250.139.217:6379" \
    --metrics-out logs/ray_metrics.json

python scripts/compare_results.py
```

---

## Headline Results

| | Spark cluster | Ray cluster |
|---|---|---|
| Total runtime | **189.0 s** | **533.2 s** |
| Rows ingested | 9,384,487 | 9,384,487 |
| Rows after cleanse | 9,248,121 | 9,248,121 |
| Output feature rows | 200,848 | 197,392 |
| Pipeline parity | — | 98.28% |
| Peak CPU | 78.2% | 84.4% |
| Peak memory | 12.2 GB | 12.1 GB |

### Stage-by-stage

| Stage | Spark (s) | Ray (s) | Spark/Ray |
|---|---:|---:|---:|
| Ingestion | 25.5 | 5.3 | 4.80× |
| Cleansing | 31.2 | 117.3 | 0.27× |
| Heavy Join | 41.3 | 105.7 | 0.39× |
| UDF + Aggregate | 86.9 | 152.7 | 0.57× |
| Export | 4.0 | 152.2 | 0.03× |
| **Total** | **189.0** | **533.2** | **0.35×** |

Ray reads parquet faster (no JVM startup), but Spark's vectorized Catalyst engine and JVM-optimized broadcast join make every other stage substantially faster — even with a Python UDF in the hot path.

---

## Engineering Notes

The cluster setup surfaced several real-world distributed-systems challenges:

- **Spark Snappy native lib missing on worker** — fixed via `spark.sql.parquet.compression.codec=uncompressed`
- **Cross-month parquet schema drift in NYC data** — fixed via `spark.sql.parquet.enableVectorizedReader=false`
- **Timezone parity 0% → 98.43%** — fixed via `spark.sql.session.timeZone=UTC`
- **Ray strict version checking** — required exact Python 3.13.5 match across nodes (built via pyenv)
- **PyArrow C-extension serialization mismatch** — required matching pyarrow versions across nodes
- **Pandas 3.x ↔ 2.x datetime ABI break** — required matching pandas versions for shuffle stages

These would be eliminated in production by deploying both nodes from a single Docker image. See the report for full discussion.

---

## Cluster Walkthrough Video

A ~5-minute walkthrough showing both clusters operational across two physical laptops, version matching, pipeline outputs, and final comparison is available via the link in `screencast/video_link.txt`.

---

## Authors
Muhammed Fiyas  
DA25M018



