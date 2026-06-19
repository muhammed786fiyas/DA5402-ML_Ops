#!/bin/bash
# Runs Spark pipeline, then Ray pipeline, and finally the comparison.
# Resource monitor must be started SEPARATELY on each node before this script.
#
# Usage on the master/head node, after both clusters are up:
#   bash 05_run_benchmarks.sh

set -e

PROJ="/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas"
source "$PROJ/venv/bin/activate"

# Edit these to match your cluster
SPARK_MASTER="spark://10.250.139.217:7077"
RAY_ADDRESS="auto"

mkdir -p "$PROJ/logs" "$PROJ/output"

echo "=========================================="
echo " Running Spark pipeline"
echo "=========================================="
python "$PROJ/scripts/pipeline_spark.py" \
    --master "$SPARK_MASTER" \
    --metrics-out "$PROJ/logs/spark_metrics.json" \
    2>&1 | tee "$PROJ/logs/spark_run.log"

# Cool down so the OS isn't holding hot caches that bias Ray's read
echo ""
echo "[*] cooling caches for 10s..."
sync && echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
sleep 10

echo ""
echo "=========================================="
echo " Running Ray pipeline"
echo "=========================================="
python "$PROJ/scripts/pipeline_ray.py" \
    --address "$RAY_ADDRESS" \
    --metrics-out "$PROJ/logs/ray_metrics.json" \
    2>&1 | tee "$PROJ/logs/ray_run.log"

echo ""
echo "=========================================="
echo " Generating comparison report"
echo "=========================================="
python "$PROJ/scripts/compare_results.py"

echo ""
echo "Done. Artifacts in $PROJ/report/"
