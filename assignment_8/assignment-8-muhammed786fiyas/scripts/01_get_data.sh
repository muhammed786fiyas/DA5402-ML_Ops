#!/bin/bash
# Pulls NYC Yellow Taxi parquet files + the taxi zone lookup CSV
# Run this on BOTH nodes if you don't have a shared NFS mount.
# If you have NFS, only run on the node that exports the share.

set -e

DATA_DIR="${1:-/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas/data}"
mkdir -p "$DATA_DIR/trips"
cd "$DATA_DIR/trips"

# 3 months — Jan, Feb, Mar 2023. Each file ~400MB parquet.
# Full year works too if you have RAM/disk; just add more months below.
MONTHS=("2023-01" "2023-02" "2023-03")

for m in "${MONTHS[@]}"; do
    f="yellow_tripdata_${m}.parquet"
    if [ ! -f "$f" ]; then
        echo "downloading $f ..."
        wget -c --progress=bar:force:noscroll "https://d37ci6vzurychx.cloudfront.net/trip-data/$f"
    else
        echo "$f already exists, skipping"
    fi
done

# Zone lookup table — small CSV, used for the join
cd "$DATA_DIR"
if [ ! -f taxi_zone_lookup.csv ]; then
    wget -c --progress=bar:force:noscroll "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
fi

echo ""
echo "Done. Files in $DATA_DIR:"
du -h "$DATA_DIR"/* "$DATA_DIR"/trips/* 2>/dev/null
