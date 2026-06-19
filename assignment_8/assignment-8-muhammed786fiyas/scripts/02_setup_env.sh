#!/bin/bash
# One-shot environment setup. Run on BOTH nodes.
# Assumes Ubuntu/Debian-ish Linux. Adjust for your distro.

set -e

# --- Java for Spark ---
# Spark 3.5.x runs fine on Java 11 or 17. Picking 17 (current LTS).
if ! command -v java &> /dev/null; then
    echo "[*] Installing OpenJDK 17"
    sudo apt-get update -qq
    sudo apt-get install -y openjdk-17-jdk-headless
fi

# --- Python venv ---
PROJ_DIR="/home/muhammed786fiyas/Desktop/Projects/ml_ops/assignment_8/assignment-8-muhammed786fiyas"
VENV_DIR="$PROJ_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

# Pinned versions — Spark 3.5.x and Ray 2.9.x are mutually stable.
# pyarrow >=14 needed by both.
pip install \
    "pyspark==3.5.1" \
    "ray[data]==2.51.2" \
    "pyarrow>=14.0.0,<16.0.0" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0,<2.0.0" \
    "psutil" \
    "matplotlib" \
    "tabulate"

# JAVA_HOME — Spark needs this exported
JAVA_HOME_PATH="$(readlink -f $(which java) | sed 's:/bin/java::')"
echo ""
echo "[*] Add this to your ~/.bashrc on BOTH nodes:"
echo "    export JAVA_HOME=$JAVA_HOME_PATH"
echo "    export PATH=\$JAVA_HOME/bin:\$PATH"
echo ""
echo "[*] Verify versions:"
java -version 2>&1 | head -1
python3 -c "import pyspark; print('PySpark', pyspark.__version__)"
python3 -c "import ray; print('Ray', ray.__version__)"
