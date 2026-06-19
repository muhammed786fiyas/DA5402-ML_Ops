#!/bin/bash
# Start Spark 4.1.1 standalone cluster using the official sbin scripts.
# Same format as friend's README.

set -e
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Master IP (master is on YOUR laptop). Friend's worker connects to this.
MASTER_HOST="10.250.139.217"
MASTER_PORT=7077
MASTER_WEBUI_PORT=8080
WORKER_WEBUI_PORT=8081

# Use the real Spark 4.1.1 distribution
export SPARK_HOME="$HOME/spark-4.1.1-bin-hadoop3"
export PATH="$SPARK_HOME/sbin:$SPARK_HOME/bin:$PATH"

# Activate venv so PYSPARK_PYTHON points at python3.13
source "/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas/venv/bin/activate"
export PYSPARK_PYTHON="$(which python3)"
export PYSPARK_DRIVER_PYTHON="$(which python3)"

case "$1" in
    master)
        echo "[*] Starting Spark master on $MASTER_HOST:$MASTER_PORT"
        SPARK_MASTER_HOST="$MASTER_HOST" \
        SPARK_MASTER_PORT="$MASTER_PORT" \
        SPARK_MASTER_WEBUI_PORT="$MASTER_WEBUI_PORT" \
            "$SPARK_HOME/sbin/start-master.sh"
        sleep 2
        echo "[+] UI:  http://$MASTER_HOST:$MASTER_WEBUI_PORT"
        echo "[+] URL: spark://$MASTER_HOST:$MASTER_PORT"
        ;;
    worker)
        echo "[*] Starting Spark worker, connecting to spark://$MASTER_HOST:$MASTER_PORT"
        "$SPARK_HOME/sbin/start-worker.sh" \
            --webui-port "$WORKER_WEBUI_PORT" \
            --cores 8 \
            --memory 12g \
            "spark://$MASTER_HOST:$MASTER_PORT"
        sleep 2
        echo "[+] Worker UI: http://localhost:$WORKER_WEBUI_PORT"
        ;;
    stop)
        "$SPARK_HOME/sbin/stop-worker.sh" 2>/dev/null || true
        "$SPARK_HOME/sbin/stop-master.sh" 2>/dev/null || true
        pkill -f "deploy.master.Master" 2>/dev/null || true
        pkill -f "deploy.worker.Worker" 2>/dev/null || true
        echo "[+] Stopped"
        ;;
    status)
        ps aux | grep -E "[d]eploy.master.Master|[d]eploy.worker.Worker" || echo "  not running"
        ;;
    *)
        echo "Usage: $0 {master|worker|stop|status}"
        exit 1
        ;;
esac
