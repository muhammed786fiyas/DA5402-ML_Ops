"""
Lightweight resource sampler. Run this on each node in a separate terminal
before kicking off a benchmark. It samples per-second CPU% and RSS, writes
to a CSV, and stops on Ctrl-C.

Example:
  python monitor.py --label spark_master --out logs/spark_master_node1.csv
"""

import argparse
import csv
import os
import signal
import time

import psutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run identifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stop = {"flag": False}

    def handle(_sig, _frm):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    print(f"[monitor:{args.label}] writing to {args.out} - Ctrl-C to stop")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "cpu_pct", "mem_used_gb", "mem_pct"])
        # Prime psutil's CPU counter
        psutil.cpu_percent(interval=None)
        while not stop["flag"]:
            time.sleep(args.interval)
            cpu = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            w.writerow([
                round(time.time(), 2),
                cpu,
                round(vm.used / 1024**3, 3),
                vm.percent,
            ])
            f.flush()
    print(f"\n[monitor:{args.label}] stopped.")


if __name__ == "__main__":
    main()
