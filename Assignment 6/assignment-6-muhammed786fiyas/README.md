# DA5402 Assignment 6 — Orchestrated Web Scraper Pipeline

**Author:** Muhammed Fiyas  
**Roll No:** DA25M018  
**Course:** DA5402 MLOps 


---

## Overview

An automated **Web-to-DB pipeline** built with Apache Airflow 3.1.8 that watches for incoming CSV files containing URLs, scrapes HTML and image assets from those sites, stores metadata in a SQLite database, and sends real-time email alerts for pipeline events.

---

## Pipeline Architecture

```
                    ┌─────────────────┐
                    │   wait_for_csv  │  ← FileSensor watches data/input/*.csv
                    │   (FileSensor)  │     poke every 30s, timeout 2min
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ (skipped/timeout)           │ (success)
              ▼                             ▼
  ┌───────────────────────┐     ┌───────────────────────┐
  │   dry_pipeline_alert  │     │     init_database      │
  │   (@task, smtplib)    │     │     (SQLite setup)     │
  └───────────────────────┘     └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │       parse_csv        │
                                │  (reads URLs via XCom) │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │  scrape_url.expand()  │  ← Dynamic task mapping
                                │  pool=scraping_pool   │  ← Max 3 concurrent
                                │  retries=3, backoff   │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │    store_and_alert    │
                                │  (SQLite INSERT +     │
                                │   collect broken URLs)│
                                └──────────┬────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                  ▼
              ┌───────────────────────┐        ┌───────────────────────┐
              │  send_broken_link_    │        │  send_collection_     │
              │  alerts (@task)       │        │  stats (@task)        │
              │  Email: 404/timeouts  │        │  Email: DB summary    │
              └───────────────────────┘        └───────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestrator | Apache Airflow 3.1.8 |
| Executor | CeleryExecutor |
| Message Broker | Redis 7.2 |
| Metadata DB | PostgreSQL 16 |
| Scraping DB | SQLite |
| Web Scraping | requests + BeautifulSoup4 |
| Email Alerts | Gmail SMTP (smtplib) |
| Containerization | Docker Compose |

---

## Project Structure

```
assignment-6-muhammed786fiyas/
├── dags/
│   └── web_scraper_dag.py      # Main Airflow DAG
├── data/
│   └── input/                  # Drop CSV files here
│       └── .gitkeep
├── scraped_data/               # SQLite DB stored here
│   └── .gitkeep
├── report_and_screencast/
│   ├── report.pdf
│   └── screencast.mp4
├── logs/                       # Airflow task logs (auto-generated)
├── config/                     # Airflow config (auto-generated)
├── plugins/                    # Airflow plugins
│   └── .gitkeep
├── docker-compose.yaml         # Full Airflow stack
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (gitignored)
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Docker & Docker Compose
- Gmail account with App Password (2FA enabled)

### Step 1 — Clone and enter repo
```bash
git clone <repo-url>
cd assignment-6-muhammed786fiyas
```

### Step 2 — Create `.env` file
```bash
echo "AIRFLOW_UID=$(id -u)" > .env
echo "SMTP_PASSWORD=your_16_char_app_password" >> .env
```

> **Get Gmail App Password:** Google Account → Security → 2-Step Verification → App Passwords → Generate

### Step 3 — Start Airflow
```bash
docker compose up airflow-init   # initialize DB and admin user
docker compose up -d             # start all services
```

### Step 4 — Open Airflow UI
Navigate to `http://localhost:8080`  
Login: `airflow` / `airflow`

### Step 5 — Configure Airflow UI

**Create Pool** (Admin → Pools → Add):
| Field | Value |
|-------|-------|
| Pool Name | `scraping_pool` |
| Slots | `3` |
| Description | Pool for throttling web scraping tasks |

**Create SMTP Connection** (Admin → Connections → Add):
| Field | Value |
|-------|-------|
| Conn ID | `smtp_default` |
| Conn Type | Email |
| Host | `smtp.gmail.com` |
| Login | your Gmail address |
| Password | your App Password |
| Port | `587` |
| Extra Fields JSON | `{"ssl": false, "starttls": true}` |

**Create File Connection** (Admin → Connections → Add):
| Field | Value |
|-------|-------|
| Conn ID | `fs_default` |
| Conn Type | File (path) |
| Path | `/` |

---

## Running the Pipeline

### Trigger a Scraping Run
1. Drop a CSV file into `data/input/`:
```bash
cat > data/input/urls.csv << 'EOF'
url
https://books.toscrape.com
https://quotes.toscrape.com
https://httpstat.us/404
EOF
```
2. Go to Airflow UI → `web_scraper_pipeline` → **Trigger**

### CSV Format
```csv
url
https://example.com
https://another-site.com
```
Supported column names: `url`, `URL`, `urls`

---

## Pipeline Features

### A. Data Sensing & Triggering
- `FileSensor` monitors `data/input/` every 30 seconds
- Timeout: 2 minutes (testing) / 12 hours (production)
- **Dry Pipeline Alert** email sent if no CSV detected within timeout

### B. Concurrent Scraping with Worker Pool
- Dynamic task mapping — one `scrape_url` task per URL
- All scraping tasks throttled via `scraping_pool` (3 slots max)
- Prevents overwhelming target servers

### C. Data Persistence
- Scraped metadata stored in SQLite at `scraped_data/scraper.db`
- Duplicate URLs skipped via `INSERT OR IGNORE`
- Schema: `url, title, html_length, image_count, image_links, status_code, scraped_at`
- **Collection Stats** email sent when DB reaches threshold (default: 3 pages)

### D. Failure Handling & Alerting
- 404 / timeout / connection errors → **Broken Link Alert** email
- Retries: 3 attempts with exponential backoff (2min → 4min → 8min)
- Soft fail on FileSensor timeout → triggers Dry Pipeline alert

---

## Email Alerts

| Alert | Trigger | Content |
|-------|---------|---------|
| ⚠️ Dry Pipeline | No CSV within timeout | DAG name, timestamp |
| 🔴 Broken Link | 404 / timeout on URL | Table of failed URLs + errors |
| 📊 Collection Stats | DB threshold reached | Total pages, images, recent URLs |

---

## Test Results

### Test 1 — Successful 3-URL Scrape ✅
```
CSV: books.toscrape.com, quotes.toscrape.com, httpstat.us/404
Result: 2 pages scraped, 1 broken link detected
Emails: Broken link alert + Collection stats sent
```

### Test 2 — Broken Link Alert ✅
```
Triggered by: httpstat.us/404 in CSV
Email received: "[Airflow Alert] 1 Broken Link(s) Detected"
```

### Test 3 — Dry Pipeline Alert ✅
```
Triggered by: No CSV in data/input/ (sensor timeout)
Email received: "[Airflow Alert] Dry Pipeline — No CSV Detected"
DAG: wait_for_csv=Skipped, dry_pipeline_alert=Success
```

---


## AI Attribution

BeautifulSoup scraping logic and relative URL handling were assisted by AI.

**Prompt used:**
```
"Write a BeautifulSoup function to extract all image src URLs from an HTML page,
handling both relative and absolute URLs"
```

DAG structure, task dependencies (`>>`), Pool configurations, and retry logic
were authored by the student as per assignment requirements.

---

