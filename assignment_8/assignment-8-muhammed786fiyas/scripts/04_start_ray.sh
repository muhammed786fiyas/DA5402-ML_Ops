#!/bin/bash
# Ray cluster bring-up.
# Run HEAD on master, WORKER on second machine.
#
# Edit HEAD_HOST below to head node's LAN IP.

set -e

HEAD_HOST="10.250.139.217"   # <-- CHANGE THIS
HEAD_PORT=6379
DASHBOARD_PORT=8265

source "/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas/venv/bin/activate"

case "$1" in
    head)
        echo "[*] Starting Ray head node on $HEAD_HOST:$HEAD_PORT"
        # --node-ip-address pins ray to the LAN interface so the worker can reach it.
        # --dashboard-host=0.0.0.0 lets you open the dashboard from any machine.
        ray start --head \
            --node-ip-address="$HEAD_HOST" \
            --port="$HEAD_PORT" \
            --dashboard-host=0.0.0.0 \
            --dashboard-port="$DASHBOARD_PORT" \
            --num-cpus=8 \
            --object-store-memory=8000000000   # 8GB plasma store
        echo ""
        echo "[+] Dashboard: http://$HEAD_HOST:$DASHBOARD_PORT"
        echo "[+] To join from worker:  ray start --address=$HEAD_HOST:$HEAD_PORT"
        ;;
    worker)
        echo "[*] Joining Ray cluster at $HEAD_HOST:$HEAD_PORT"
        ray start \
            --address="$HEAD_HOST:$HEAD_PORT" \
            --num-cpus=8 \
            --object-store-memory=8000000000
        ;;
    status)
        ray status
        ;;
    stop)
        ray stop --force
        ;;
    *)
        echo "Usage: $0 {head|worker|status|stop}"
        exit 1
        ;;
esac
