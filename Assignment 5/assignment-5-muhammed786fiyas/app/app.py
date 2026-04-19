"""
DA5402 Assignment 5 — AI Application with Prometheus Instrumentation
App: CPU-only image captioning using BLIP (blip-image-captioning-base)
Metrics: Exposed on :8001/metrics for Prometheus scraping
"""

import io
import os
import time
import uuid
import zipfile
import threading

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, REGISTRY
)
from utils.logger import get_logger

logger = get_logger("image_captioning")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROMETHEUS METRICS DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

def get_metric(cls, name, desc, labels=None, **kwargs):
    """Create a metric or retrieve it from the registry if already registered."""
    try:
        if labels:
            return cls(name, desc, labels, **kwargs)
        return cls(name, desc, **kwargs)
    except ValueError:
        for collector, names in REGISTRY._collector_to_names.items():
            if any(name in n for n in names):
                return collector
        raise

# ── Counters ──────────────────────────────────────────────────────────────────
images_processed_total  = get_metric(Counter, 'images_processed_total',  'Total images processed', ['mode'])
requests_total          = get_metric(Counter, 'app_requests_total',       'Total requests to the app', ['mode', 'status'])
app_errors_total        = get_metric(Counter, 'app_errors_total',         'Total errors encountered', ['error_type', 'mode'])
bulk_batches_total      = get_metric(Counter, 'bulk_batches_total',       'Total bulk ZIP uploads processed')
client_requests_total   = get_metric(Counter, 'client_requests_total',    'Requests tracked by session', ['session_id', 'mode'])

# ── Gauges ────────────────────────────────────────────────────────────────────
active_requests     = get_metric(Gauge, 'active_requests',             'Number of currently active requests')
model_loaded        = get_metric(Gauge, 'model_loaded',                '1 if BLIP model loaded else 0')
last_inference_time = get_metric(Gauge, 'last_inference_time_seconds', 'Latency of most recent inference', ['mode'])
bulk_queue_size     = get_metric(Gauge, 'bulk_queue_size',             'Images remaining in current bulk batch')
memory_usage_mb     = get_metric(Gauge, 'app_memory_usage_mb',         'Python process memory usage in MB')

# ── Histograms ────────────────────────────────────────────────────────────────
inference_latency    = get_metric(Histogram, 'inference_latency_seconds', 'Caption generation time per image', ['mode'],
                                  buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0])
image_size_kb        = get_metric(Histogram, 'image_size_kb',             'Size of uploaded images in KB', ['mode'],
                                  buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000])
bulk_batch_size_hist = get_metric(Histogram, 'bulk_batch_size_images',    'Number of images per bulk upload',
                                  buckets=[1, 2, 5, 10, 20, 50, 100])

# ── Summaries ─────────────────────────────────────────────────────────────────
processing_time_summary = get_metric(Summary, 'image_processing_seconds',  'Summary of image processing time', ['mode'])
caption_length_summary  = get_metric(Summary, 'caption_length_characters', 'Summary of generated caption lengths', ['mode'])


# ─────────────────────────────────────────────────────────────────────────────
# 2. METRICS HTTP SERVER
# ─────────────────────────────────────────────────────────────────────────────

def start_metrics_server():
    try:
        start_http_server(8001)
        logger.info("Prometheus metrics server started on :8001/metrics")
    except OSError:
        logger.warning("Metrics server already running on :8001")

if "metrics_server_started" not in st.session_state:
    threading.Thread(target=start_metrics_server, daemon=True).start()
    st.session_state["metrics_server_started"] = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading BLIP model... (first run takes ~1 min)")
def load_model():
    logger.info("Loading BLIP model from HuggingFace...")
    model_loaded.set(0)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    model_loaded.set(1)
    logger.info("BLIP model loaded successfully")
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# 4. MEMORY TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def update_memory_gauge():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    kb = int(line.split()[1])
                    memory_usage_mb.set(kb / 1024)
                    break
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 5. CAPTION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_caption(image, processor, model, mode):
    logger.info(f"Generating caption | mode={mode}")
    active_requests.inc()
    update_memory_gauge()
    start = time.time()
    try:
        inputs = processor(image, return_tensors="pt")
        with processing_time_summary.labels(mode=mode).time():
            output = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        latency = time.time() - start

        inference_latency.labels(mode=mode).observe(latency)
        last_inference_time.labels(mode=mode).set(latency)
        caption_length_summary.labels(mode=mode).observe(len(caption))
        images_processed_total.labels(mode=mode).inc()
        requests_total.labels(mode=mode, status="success").inc()

        logger.info(f"Caption generated | mode={mode} | latency={latency:.2f}s | caption='{caption}'")
        return caption, latency

    except Exception as e:
        logger.error(f"Caption generation failed | mode={mode} | error={str(e)}")
        app_errors_total.labels(error_type="model_error", mode=mode).inc()
        requests_total.labels(mode=mode, status="error").inc()
        raise e
    finally:
        active_requests.dec()


# ─────────────────────────────────────────────────────────────────────────────
# 6. STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Image Captioning — DA5402 A5", page_icon="🖼️", layout="wide")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    session_id = st.session_state.session_id

    st.title("🖼️ CPU Image Captioning")
    st.caption(f"Session ID: `{session_id}` | Metrics: [localhost:8001/metrics](http://localhost:8001/metrics)")
    st.markdown("---")

    try:
        processor, model = load_model()
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        st.error(f"Failed to load model: {e}")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        mode = st.radio("Select Mode", ["Single Image", "Bulk (ZIP)"], index=0)

    # ── SINGLE MODE ───────────────────────────────────────────────────────────
    if mode == "Single Image":
        with col2:
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_file:
            file_bytes = uploaded_file.read()
            size_kb = len(file_bytes) / 1024
            image_size_kb.labels(mode="single").observe(size_kb)
            client_requests_total.labels(session_id=session_id, mode="single").inc()
            logger.info(f"Single image upload | size={size_kb:.1f}KB | session={session_id}")
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            with c2:
                with st.spinner("Generating caption..."):
                    try:
                        caption, latency = generate_caption(image, processor, model, mode="single")
                        st.success("Caption generated")
                        st.markdown(f"### Caption\n> *{caption}*")
                        st.metric("Inference Time", f"{latency:.2f}s")
                        st.metric("Image Size", f"{size_kb:.1f} KB")
                    except Exception as e:
                        logger.error(f"Single mode error | session={session_id} | error={e}")
                        st.error(f"Error: {e}")

    # ── BULK MODE ─────────────────────────────────────────────────────────────
    else:
        with col2:
            uploaded_zip = st.file_uploader("Upload a ZIP file of images", type=["zip"])

        if uploaded_zip:
            client_requests_total.labels(session_id=session_id, mode="bulk").inc()
            try:
                with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zf:
                    image_files = [
                        f for f in zf.namelist()
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                        and not f.startswith("__MACOSX")
                    ]
                    if not image_files:
                        logger.warning(f"No valid images in ZIP | session={session_id}")
                        st.warning("No valid images found in ZIP.")
                        app_errors_total.labels(error_type="zip_error", mode="bulk").inc()
                        return

                    total = len(image_files)
                    bulk_batch_size_hist.observe(total)
                    bulk_batches_total.inc()
                    logger.info(f"Bulk upload started | total={total} images | session={session_id}")
                    st.info(f"Found **{total}** images. Processing...")

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    for i, filename in enumerate(image_files):
                        bulk_queue_size.set(total - i)
                        status_text.text(f"Processing {i+1}/{total}: {filename}")
                        try:
                            file_data = zf.read(filename)
                            size_kb = len(file_data) / 1024
                            image_size_kb.labels(mode="bulk").observe(size_kb)
                            image = Image.open(io.BytesIO(file_data)).convert("RGB")
                            caption, latency = generate_caption(image, processor, model, mode="bulk")
                            results.append({"File": filename, "Caption": caption,
                                            "Latency (s)": f"{latency:.2f}", "Size (KB)": f"{size_kb:.1f}"})
                        except Exception as e:
                            logger.error(f"Bulk file error | file={filename} | error={e}")
                            app_errors_total.labels(error_type="file_error", mode="bulk").inc()
                            results.append({"File": filename, "Caption": f"ERROR: {e}",
                                            "Latency (s)": "—", "Size (KB)": "—"})
                        progress_bar.progress((i + 1) / total)

                    bulk_queue_size.set(0)
                    success_count = sum(1 for r in results if not r["Caption"].startswith("ERROR"))
                    logger.info(f"Bulk upload complete | success={success_count}/{total} | session={session_id}")
                    status_text.text("All images processed")
                    st.markdown("### Results")
                    st.dataframe(results, use_container_width=True)
                    ca, cb, cc = st.columns(3)
                    ca.metric("Total Images", total)
                    cb.metric("Successful", success_count)
                    cc.metric("Errors", total - success_count)

            except zipfile.BadZipFile:
                logger.error(f"Invalid ZIP file | session={session_id}")
                st.error("Invalid ZIP file.")
                app_errors_total.labels(error_type="zip_error", mode="bulk").inc()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Metrics Snapshot")
        st.markdown(f"**Session ID:** `{session_id}`")
        st.markdown("**Metrics endpoint:** `:8001/metrics`")
        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("- [Prometheus](http://localhost:9090)")
        st.markdown("- [Grafana](http://localhost:3000)")
        st.markdown("- [AlertManager](http://localhost:9093)")
        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown("- **Model:** BLIP-base")
        st.markdown("- **Device:** CPU")
        st.markdown("- **Task:** Image Captioning")


if __name__ == "__main__":
    main()