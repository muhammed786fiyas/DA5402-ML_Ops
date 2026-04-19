# DA5402: A1 – The "Manual" MLOps Challenge

- **Name:** Muhammed Fiyas
- **Roll No:** DA25M018

---

## Project Overview

This project implements a complete **manual MLOps lifecycle** without using automated MLOps tools (e.g., MLflow, DVC). It demonstrates manual data versioning, configuration management, model registry, API deployment, deployment logging, production monitoring, and drift detection with retraining triggers.

---

## Project Structure

```
assignment-1-muhammed786fiyas/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   ├── processed/
│   │   ├── v1_train.csv
│   │   ├── v2_train.csv
│   │   ├── v3_train.csv
│   │   ├── v4_train.csv
│   │   └── v5_train.csv
│   ├── production/
│   │   ├── v1_production.csv
│   │   ├── v2_production.csv
│   │   ├── v3_production.csv
│   │   ├── v4_production.csv
│   │   └── v5_production.csv
│   └── manifest.txt
│
├── models/
│   ├── model_v2.pkl
│   ├── model_v3.pkl
│   ├── model_v4.pkl
│   ├── model_v5.pkl
│   ├── v2_metadata.json
│   ├── v3_metadata.json
│   ├── v4_metadata.json
│   ├── v5_metadata.json
│   └── model_metadata.log
│
├── report_and_recording/
|   ├── DA5402_MLOps_Assignment_1_report.pdf
│   └── screen_recording
|
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── inference.py
│   ├── monitor.py
│   └── smoke_test.py
│
├── api_predictions_log.csv
├── deployment_log.csv
├── config.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Environment Setup

This project uses **Python 3.10**. Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Phase A – Data Preparation

```bash
python src/data_prep.py
```

Reads the raw dataset, drops leakage columns, performs one-hot encoding, splits chronologically, saves versioned CSVs, and appends an entry to `data/manifest.txt`. Dataset version is controlled via `config.yaml → data.current_version`.

---

### Phase B – Model Training & Registry

```bash
python src/train.py
```

Loads the versioned dataset, trains a RandomForest model, saves `model_<version>.pkl` and `<version>_metadata.json`, and appends an entry to `models/model_metadata.log`. Reproducibility is ensured through config-driven hyperparameters and a fixed `random_state`.

---

### Phase C – Deployment

```bash
uvicorn src.inference:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`. Loads the correct model version dynamically, logs deployment to `deployment_log.csv`, and serves the `/predict` endpoint.

**Smoke test:**
```bash
python src/smoke_test.py
```

---

### Phase D – Monitoring & Drift Detection

```bash
python src/monitor.py
```

Sends the production dataset through the API, reads predictions from `api_predictions_log.csv`, computes the production error rate, and triggers a drift alert if the threshold is exceeded. Configurable in `config.yaml`:

```yaml
monitoring:
  minimum_required_logs: 500
  drift_threshold: 0.05
```

---

## Retraining Workflow

To retrain without overwriting previous model versions:

**Step 1** – Increment the version in `config.yaml`:
```yaml
data:
  current_version: "v5"
```

**Step 2** – Generate a new versioned dataset:
```bash
python src/data_prep.py
```

**Step 3** – Train the new model:
```bash
python src/train.py
```

**Step 4** – Restart the API:
```bash
uvicorn src.inference:app --reload
```

> Incrementing the version is necessary to preserve existing model files. Without it, re-running training overwrites the current `model_<version>.pkl`.

---

## Reproducing the Full Pipeline

```bash
python src/data_prep.py
python src/train.py
uvicorn src.inference:app --reload
python src/monitor.py
```

---

## Notes

- No automated MLOps tools (MLflow, DVC, etc.) were used — all versioning is implemented manually.
- Monitoring is based on actual API prediction logs.
- Drift detection threshold is configurable via `config.yaml`.
