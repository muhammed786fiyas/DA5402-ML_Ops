# DA5402 Assignment 5 — Application Monitoring with Prometheus & Grafana

**Student:** Muhammed Fiyas  
**Roll no:** DA25M018    
**Course:** DA5402 MLOps  
**Assignment:** A5 — Observability with Prometheus & Grafana  

---

## Screencast Demo

A full demonstration of the monitoring stack including:
- Repo structure and logging
- Streamlit app (Single + Bulk mode)
- Prometheus metrics endpoint
- Prometheus targets and alert rules
- AlertManager email alert proof
- AlertManager silence
- Grafana dashboard with live metrics

📹 [Watch Screencast on Google Drive](https://drive.google.com/file/d/1IQcPd-JmnNwzN4nUb4O8sTNUjQ7W0E8t/view?usp=sharing)

---

## Overview

This project instruments a CPU-only AI image captioning application with a full production-grade monitoring stack. The application uses the BLIP model to generate captions for uploaded images, while Prometheus scrapes metrics, AlertManager sends email notifications, and Grafana visualizes everything in a real-time dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Browser                          │
│              http://localhost:8501                       │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   Streamlit App       │
         │   BLIP Image Caption  │
         │   Single + Bulk Mode  │
         │   :8501 (UI)          │
         │   :8001 (metrics)     │
         └───┬───────────────────┘
             │ scrapes :8001
    ┌────────▼────────┐
    │   Prometheus    │◄── also scrapes node_exporter :9100
    │   :9090         │
    └────┬────────────┘
         │ fires alerts
    ┌────▼────────────┐        ┌──────────────────┐
    │  AlertManager   │───────►│  webhook.py :5001│
    │  :9093          │        │  Mailtrap API    │
    └─────────────────┘        └──────────────────┘
         │ data source
    ┌────▼────────────┐
    │    Grafana      │
    │    :3000        │
    └─────────────────┘
```

---

## Repo Structure

```
assignment-5-muhammed786fiyas/
├── alertmanager/
│   ├── alertmanager.yml        # AlertManager routing config (webhook)
│   └── webhook.py              # Flask webhook — forwards alerts to Mailtrap API
├── app/
│   ├── logs/                   # Application log files (committed intentionally)
│   │   └── app_YYYY-MM-DD_HH-MM-SS.log
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py           # Centralized logger (file + console)
│   ├── app.py                  # Streamlit app with Prometheus instrumentation
│   └── requirements.txt        # Python dependencies
├── grafana/
│   └── dashboard.json          # Exported Grafana dashboard
├── prometheus/
│   ├── alert_rules.yml         # Alerting rules (AppDown, HighCPU, SlowInference etc.)
│   ├── prometheus.yml          # Scrape config (app + node_exporter + prometheus)
│   └── recording_rules.yml     # Pre-computed PromQL rules
├── report/                     # Assignment report (PDF)
├── .env.example                # Template for environment variables
├── .gitignore
├── docker-compose.yml          # Prometheus + Grafana + AlertManager
├── environment.yml             # Conda environment (mlops_a5, Python 3.10)
└── README.md
```

---

## Metrics Instrumented

### Counters
| Metric | Labels | Description |
|--------|--------|-------------|
| `images_processed_total` | `mode` | Total images captioned (single/bulk) |
| `app_requests_total` | `mode`, `status` | Total requests by success/error |
| `app_errors_total` | `error_type`, `mode` | Errors by type |
| `bulk_batches_total` | — | Total ZIP uploads processed |
| `client_requests_total` | `session_id`, `mode` | Per-session request tracking |

### Gauges
| Metric | Description |
|--------|-------------|
| `active_requests` | Currently in-flight requests |
| `model_loaded` | 1 if BLIP model in memory, 0 otherwise |
| `last_inference_time_seconds` | Most recent caption latency |
| `bulk_queue_size` | Images remaining in current bulk batch |
| `app_memory_usage_mb` | Python process RAM usage |

### Histograms
| Metric | Description |
|--------|-------------|
| `inference_latency_seconds` | Caption generation time distribution |
| `image_size_kb` | Uploaded image size distribution |
| `bulk_batch_size_images` | Images per bulk upload distribution |

### Summaries
| Metric | Description |
|--------|-------------|
| `image_processing_seconds` | Processing time `_sum` and `_count` |
| `caption_length_characters` | Caption length `_sum` and `_count` |

---

## Alert Rules

| Alert | Severity | Condition |
|-------|----------|-----------|
| `AppDown` | critical | App unreachable for >1 minute |
| `HighErrorRate` | critical | Error rate >0.1/s for >2 minutes |
| `SlowInferenceP95` | warning | P95 latency >15s for >1 minute |
| `NoImagesProcessed` | warning | No images processed in 10 minutes |
| `HighCPUUsage` | warning | CPU >80% for >2 minutes |
| `CriticalCPUUsage` | critical | CPU >95% for >1 minute |
| `HighMemoryUsage` | warning | RAM >85% for >2 minutes |
| `BulkUploadCPUSaturation` | warning | Bulk active + CPU >80% |

---

## Setup Instructions

### Prerequisites
- Conda (Miniconda or Anaconda)
- Docker + Docker Compose
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/DA5402-MLOps-JAN26/assignment-5-muhammed786fiyas.git
cd assignment-5-muhammed786fiyas
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate mlops_a5
```

### 3. Set Environment Variables
```bash
cp .env.example .env
nano .env  # add your Mailtrap API token
```

### 4. Install node_exporter
```bash
wget https://github.com/prometheus/node_exporter/releases/download/v1.8.1/node_exporter-1.8.1.linux-amd64.tar.gz
tar xvf node_exporter-1.8.1.linux-amd64.tar.gz
cd node_exporter-1.8.1.linux-amd64
./node_exporter &
```

### 5. Start Monitoring Stack
```bash
docker compose up -d
```

### 6. Run the Streamlit App
```bash
conda activate mlops_a5
cd app
streamlit run app.py
```

### 7. Run the Alert Webhook
```bash
conda activate mlops_a5
python alertmanager/webhook.py
```

---

## Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit App | http://localhost:8501 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |
| AlertManager | http://localhost:9093 | — |
| App Metrics | http://localhost:8001/metrics | — |
| node_exporter | http://localhost:9100/metrics | — |

---

## Grafana Dashboard

The dashboard (`grafana/dashboard.json`) contains 13 panels organized into 4 rows:

- **Row 1 — Health:** App Status, Total Images, Active Requests, Model Loaded, Memory, Last Inference Time
- **Row 2 — Throughput & Latency:** Single vs Bulk throughput, P50/P95/P99 latency percentiles
- **Row 3 — System Resources:** CPU % correlated with bulk uploads, RAM % and Disk %
- **Row 4 — Errors & Alerts:** Error rate by type, Request success vs error, Alert events table

### Import Dashboard
1. Go to `http://localhost:3000`
2. Dashboards → New → Import
3. Paste contents of `grafana/dashboard.json`
4. Select Prometheus as data source
5. Click Import

---

## Alerting Threshold Justification

Thresholds were chosen based on empirical observations:
- Single BLIP inference on CPU takes **0.9–2s** → P95 warning at **15s** (~10× baseline)
- Bulk upload of 10 images drives CPU to **60–80%** → alert at **80%**
- Error rate threshold of **0.1/s** = more than 1 error per 10 seconds
- Memory warning at **85%** — BLIP model uses ~1.4GB, leaving headroom for OS

---

## Deliverables

- [x] Streamlit app with Single + Bulk image captioning
- [x] Prometheus instrumentation (Counter, Gauge, Histogram, Summary)
- [x] node_exporter system metrics
- [x] Alert rules with multiple severity levels
- [x] AlertManager webhook → Mailtrap email alerts
- [x] AlertManager silence (maintenance window)
- [x] Grafana dashboard (13 panels, 7 commandments)
- [x] Application logging (utils/logger.py)
- [x] dashboard.json export
- [x] Screencast recording
- [ ] Report (see report/ folder)

---
