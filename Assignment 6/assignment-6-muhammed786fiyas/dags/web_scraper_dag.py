"""
DA5402 Assignment 6 - Web Scraper Pipeline
Author: muhammed786fiyas

Automated Web-to-DB pipeline that watches for CSV files containing URLs,
scrapes HTML and image assets, stores metadata in SQLite, and sends
email alerts for dry pipelines, broken links, and batch collection stats.

AI Attribution:
    Tool: Claude (Anthropic)
    BeautifulSoup image URL extraction logic (relative/absolute URL handling)
    was developed with AI assistance.
    Prompt: "Extract all image src URLs from HTML using BeautifulSoup,
             handling both relative and absolute URLs"
    DAG structure, task dependencies (>>), and Pool configurations
    were authored by the student.
"""

import csv
import glob
import logging
import os
import sqlite3
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.decorators import task
from airflow.providers.standard.sensors.filesystem import FileSensor
from bs4 import BeautifulSoup

# ============================================================
# CONFIGURATION
# ============================================================

DATA_INPUT_DIR   = "/opt/airflow/data/input"
SCRAPED_DATA_DIR = "/opt/airflow/scraped_data"
DB_PATH          = "/opt/airflow/scraped_data/scraper.db"
ALERT_EMAIL      = "beacontech4@gmail.com"
BATCH_THRESHOLD  = 3        # pages needed to trigger stats email (10 in production)
SCRAPING_POOL    = "scraping_pool"

# ============================================================
# DEFAULT ARGS
# ============================================================

default_args = {
    "owner": "muhammed786fiyas",
    "depends_on_past": False,
    "email": [ALERT_EMAIL],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
}

# ============================================================
# HELPERS
# ============================================================

def init_db():
    """Create SQLite table if it does not already exist."""
    os.makedirs(SCRAPED_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scraped_pages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT UNIQUE,
            title       TEXT,
            html_length INTEGER,
            image_count INTEGER,
            image_links TEXT,
            status_code INTEGER,
            scraped_at  TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_db_count():
    """Return total number of scraped pages stored in the DB."""
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM scraped_pages").fetchone()[0]
    conn.close()
    return count


def find_latest_csv():
    """Return path of the most recently modified CSV in the input directory."""
    files = glob.glob(os.path.join(DATA_INPUT_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_INPUT_DIR}")
    return max(files, key=os.path.getmtime)


def send_email(subject: str, html: str):
    """Send an HTML email via Gmail SMTP using environment credentials."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("AIRFLOW__SMTP__SMTP_MAIL_FROM")
    msg["To"] = ALERT_EMAIL
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(
            os.environ.get("AIRFLOW__SMTP__SMTP_USER"),
            os.environ.get("AIRFLOW__SMTP__SMTP_PASSWORD"),
        )
        server.sendmail(msg["From"], [ALERT_EMAIL], msg.as_string())


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="web_scraper_pipeline",
    description="Web-to-DB scraping pipeline with email alerts",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "scraping", "assignment6"],
) as dag:

    # FileSensor pokes every 30s, times out after 2 min (120s).
    # soft_fail=True marks the task Skipped on timeout instead of Failed,
    # which triggers dry_pipeline_alert via trigger_rule="all_skipped".
    # Production timeout: 43200s (12 hours)
    wait_for_csv = FileSensor(
        task_id="wait_for_csv",
        filepath=os.path.join(DATA_INPUT_DIR, "*.csv"),
        poke_interval=30,
        timeout=120,
        mode="poke",
        soft_fail=True,
        fs_conn_id="fs_default",
    )

    @task(task_id="dry_pipeline_alert", trigger_rule="all_skipped")
    def dry_pipeline_alert():
        """Email alert sent when no CSV file arrives within the sensor timeout."""
        html = """
            <h3>⚠️ Dry Pipeline Alert</h3>
            <p>No new CSV file was detected in the input directory
            within the expected time window.</p>
            <p><b>DAG:</b> web_scraper_pipeline</p>
            <p>Please check if the target list upload is delayed.</p>
        """
        send_email("[Airflow Alert] Dry Pipeline — No CSV Detected", html)
        logging.info("Dry pipeline alert sent.")

    @task(task_id="init_database")
    def init_database():
        """Ensure the SQLite database and schema exist before scraping begins."""
        init_db()
        logging.info("Database ready at %s", DB_PATH)

    @task(task_id="parse_csv")
    def parse_csv():
        """Read URLs from the latest CSV and return them as a list via XCom."""
        csv_path = find_latest_csv()
        urls = []
        try:
            with open(csv_path, "r") as f:
                for row in csv.DictReader(f):
                    url = row.get("url") or row.get("URL") or row.get("urls")
                    if url and url.strip():
                        urls.append(url.strip())
        except Exception as e:
            logging.error("Failed to parse CSV: %s", e)
            raise

        if not urls:
            raise ValueError("CSV file contains no valid URLs")

        logging.info("Parsed %d URLs from %s", len(urls), csv_path)
        return urls

    # Pool throttles concurrent scraping to 3 tasks at a time.
    # scrape_url.expand() creates one task instance per URL dynamically.
    @task(
        task_id="scrape_url",
        pool=SCRAPING_POOL,
        retries=3,
        retry_delay=timedelta(minutes=2),
        retry_exponential_backoff=True,
        multiple_outputs=True,
    )
    def scrape_url(url: str):
        """Scrape a single URL and return its metadata."""
        result = {
            "url": url,
            "title": "",
            "html_length": 0,
            "image_count": 0,
            "image_links": "",
            "status_code": 0,
            "success": False,
            "error": "",
        }

        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; AirflowScraper/1.0)"}
            response = requests.get(url, timeout=10, headers=headers)
            result["status_code"] = response.status_code

            if response.status_code == 404:
                result["error"] = f"404 Not Found: {url}"
                return result

            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}: {url}"
                return result

            soup = BeautifulSoup(response.text, "html.parser")

            title_tag = soup.find("title")
            result["title"] = title_tag.get_text(strip=True) if title_tag else "No Title"

            # Resolve relative image src to absolute URLs (AI-assisted)
            images = []
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if src:
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        from urllib.parse import urlparse
                        base = urlparse(url)
                        src = f"{base.scheme}://{base.netloc}{src}"
                    images.append(src)

            result["html_length"] = len(response.text)
            result["image_count"] = len(images)
            result["image_links"] = ",".join(images[:20])
            result["success"] = True
            logging.info("Scraped %s — title: %s, images: %d",
                         url, result["title"], len(images))

        except requests.exceptions.Timeout:
            result["error"] = f"Timeout: {url}"
            logging.error("Timeout: %s", url)

        except requests.exceptions.ConnectionError:
            result["error"] = f"Connection Error: {url}"
            logging.error("Connection error: %s", url)

        except Exception as e:
            result["error"] = str(e)
            logging.error("Unexpected error scraping %s: %s", url, e)

        return result

    @task(task_id="store_and_alert")
    def store_and_alert(scraped_results: list):
        """
        Insert successful results into SQLite (duplicates skipped via INSERT OR IGNORE).
        Collect failed URLs to pass to the broken link alert task.
        """
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        inserted_count = 0
        broken_links = []

        for result in scraped_results:
            url = result.get("url", "")
            if not result.get("success"):
                broken_links.append({"url": url, "error": result.get("error", "")})
                logging.warning("Broken link: %s — %s", url, result.get("error"))
                continue

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO scraped_pages
                    (url, title, html_length, image_count, image_links, status_code, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    url,
                    result.get("title", ""),
                    result.get("html_length", 0),
                    result.get("image_count", 0),
                    result.get("image_links", ""),
                    result.get("status_code", 0),
                    datetime.now().isoformat(),
                ))
                if cursor.rowcount > 0:
                    inserted_count += 1
                    logging.info("Inserted: %s", url)
                else:
                    logging.info("Duplicate skipped: %s", url)
            except Exception as e:
                logging.error("DB insert error for %s: %s", url, e)

        conn.commit()
        conn.close()
        logging.info("Inserted %d new records", inserted_count)

        return {
            "inserted_count": inserted_count,
            "broken_links": broken_links,
            "total_db_count": get_db_count(),
        }

    @task(task_id="send_broken_link_alerts")
    def send_broken_link_alerts(store_result: dict):
        """Send a broken link alert email if any URLs failed during scraping."""
        broken_links = store_result.get("broken_links", [])
        if not broken_links:
            logging.info("No broken links — skipping alert.")
            return

        rows = "".join(
            f"<tr><td>{item['url']}</td><td>{item['error']}</td></tr>"
            for item in broken_links
        )
        html = f"""
            <h3>🔴 Broken Link Alert</h3>
            <p><b>{len(broken_links)}</b> broken link(s) detected.</p>
            <table border="1" cellpadding="5">
                <tr><th>URL</th><th>Error</th></tr>
                {rows}
            </table>
            <p><b>DAG:</b> web_scraper_pipeline</p>
        """
        send_email(f"[Airflow Alert] {len(broken_links)} Broken Link(s) Detected", html)
        logging.info("Broken link alert sent for %d URLs", len(broken_links))

    @task(task_id="send_collection_stats")
    def send_collection_stats(store_result: dict):
        """Send a DB summary email once the batch threshold is reached."""
        total_db_count = store_result.get("total_db_count", 0)
        inserted_count = store_result.get("inserted_count", 0)

        if total_db_count < BATCH_THRESHOLD:
            logging.info("Threshold not reached (%d/%d) — skipping stats email.",
                         total_db_count, BATCH_THRESHOLD)
            return

        conn = sqlite3.connect(DB_PATH)
        total = conn.execute(
            "SELECT COUNT(*) FROM scraped_pages").fetchone()[0]
        total_images = conn.execute(
            "SELECT SUM(image_count) FROM scraped_pages").fetchone()[0] or 0
        recent = conn.execute(
            "SELECT url, title, scraped_at FROM scraped_pages ORDER BY id DESC LIMIT 5"
        ).fetchall()
        conn.close()

        recent_rows = "".join(
            f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>"
            for r in recent
        )
        html = f"""
            <h3>📊 Collection Statistics Report</h3>
            <p>Batch threshold of <b>{BATCH_THRESHOLD}</b> pages reached!</p>
            <table border="1" cellpadding="5">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Pages Scraped</td><td>{total}</td></tr>
                <tr><td>Total Images Found</td><td>{total_images}</td></tr>
                <tr><td>Inserted This Run</td><td>{inserted_count}</td></tr>
            </table>
            <h4>5 Most Recently Scraped Pages:</h4>
            <table border="1" cellpadding="5">
                <tr><th>URL</th><th>Title</th><th>Scraped At</th></tr>
                {recent_rows}
            </table>
            <p><b>DAG:</b> web_scraper_pipeline</p>
        """
        send_email(f"[Airflow] Collection Stats - {total} pages in DB", html)
        logging.info("Collection stats email sent. Total pages: %d", total)

    # ============================================================
    # TASK DEPENDENCIES
    #
    #  wait_for_csv ──(skipped)──► dry_alert
    #       │
    #       ▼
    #    db_init ──► parse_csv ──► scrape_url[*] ──► store_result
    #                                                      │         │
    #                                               broken_alerts  stats_email
    # ============================================================

    dry_alert     = dry_pipeline_alert()
    db_init       = init_database()
    urls          = parse_csv()
    scraped       = scrape_url.expand(url=urls)
    store_result  = store_and_alert(scraped)
    broken_alerts = send_broken_link_alerts(store_result)
    stats_email   = send_collection_stats(store_result)

    wait_for_csv >> dry_alert
    wait_for_csv >> db_init >> urls >> scraped >> store_result
    store_result >> broken_alerts
    store_result >> stats_email