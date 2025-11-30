# app.py
"""
AI-Powered Road Safety - Streamlit Web App (Corrected)
Single-file demo that supports:
 - Image detection (simulated or YOLO if ultralytics is available)
 - Short video detection (frame sampling)
 - Hotspot demo (CSV or sample points)
Deployment: push to GitHub and deploy at https://share.streamlit.io

Requirements (put these into requirements.txt):
streamlit
opencv-python-headless
Pillow
numpy
matplotlib
pandas
# Optional for real YOLO:
# ultralytics
# torch
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import time
import os
import tempfile
import pandas as pd

st.set_page_config(page_title="AI Road Safety Demo", layout="wide")

# Header
st.title("AI-Powered Road Safety & Accident Prevention — Demo")
st.markdown(
    "This demo shows image/video violation detection and a simple hotspot visualization. "
    "For real YOLO detection install the `ultralytics` package and enable it in the sidebar."
)

# Sidebar controls
st.sidebar.header("Configuration")
use_real = st.sidebar.checkbox("Use real YOLO detection (ultralytics)", value=False)
confidence = st.sidebar.slider("Detection confidence (if using real model)", 0.1, 0.95, 0.4)
mode = st.sidebar.radio("Mode", ["Image", "Video", "Hotspot Demo"])

# Try to import YOLO if requested
yolo_available = False
model = None
if use_real:
    try:
        from ultralytics import YOLO  # type: ignore
        # try to load a small model; if weights need downloading this may take time
        model = YOLO("yolov8n.pt")
        yolo_available = True
    except Exception as e:
        st.sidebar.warning("ultralytics not available or failed to load model. Falling back to simulated mode.")
        yolo_available = False
        model = None

# Utility: draw boxes on PIL image
def draw_boxes_pil(img_pil, boxes, labels=None, scores=None):
    img = np.array(img_pil.convert("RGB"))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = ""
        if labels and i < len(labels):
            txt = f"{labels[i]}"
        if scores and i < len(scores):
            txt += f" {scores[i]:.2f}"
        if txt:
            cv2.putText(img, txt, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(img)

# Simulated detector: returns random boxes
def simulated_detection(image, n=3):
    w, h = image.size
    boxes = []
    labels = []
    scores = []
    rng = np.random.default_rng(int(time.time() % 1e6))
    for i in range(n):
        x1 = int(rng.integers(0, int(w * 0.6)))
        y1 = int(rng.integers(0, int(h * 0.6)))
        x2 = int(min(w - 1, x1 + int(rng.integers(max(1,int(w * 0.05)), max(2,int(w * 0.3))))))
        y2 = int(min(h - 1, y1 + int(rng.integers(max(1,int(h * 0.05)), max(2,int(h * 0.3))))))
        boxes.append((x1, y1, x2, y2))
        labels.append(rng.choice(["car", "bike", "truck", "pedestrian"]))
        scores.append(float(rng.random() * 0.4 + 0.5))
    return boxes, labels, scores

# Wrapper for real YOLO inference (if available)
def yolo_detect_on_pil(pil_img, conf_threshold=0.4):
    """
    Returns boxes, labels, scores
    boxes: list of (x1,y1,x2,y2)
    """
    if model is None:
        return [], [], []
    try:
        results = model.predict(source=np.array(pil_img), imgsz=640, conf=conf_threshold, verbose=False)
        if len(results) == 0:
            return [], [], []
        res = results[0]
        boxes = []
        labels = []
        scores = []
        # res.boxes.data: each row [x1,y1,x2,y2,score,class]
        for det in getattr(res, "boxes").data.tolist():
            x1, y1, x2, y2, score, class_id = det
            boxes.append((x1, y1, x2, y2))
            label = model.names[int(class_id)] if hasattr(model, "names") else str(int(class_id))
            labels.append(label)
            scores.append(float(score))
        return boxes, labels, scores
    except Exception as e:
        st.warning("YOLO inference failed; falling back to simulated detection for this frame.")
        return simulated_detection(pil_img)

# IMAGE MODE
if mode == "Image":
    st.header("Image Violation Detection")
    uploaded = st.file_uploader("Upload an image (road / traffic)", type=["png", "jpg", "jpeg"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)
            if use_real and yolo_available and model is not None:
                st.write("Running YOLO detection...")
                boxes, labels, scores = yolo_detect_on_pil(image, conf_threshold=confidence)
                out_img = draw_boxes_pil(image, boxes, labels, scores)
                st.image(out_img, caption="Detections (YOLO)", use_column_width=True)
            else:
                st.write("Simulated detection (real model not enabled)")
                boxes, labels, scores = simulated_detection(image)
                out_img = draw_boxes_pil(image, boxes, labels, scores)
                st.image(out_img, caption="Simulated Detections", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to process image: {e}")

# VIDEO MODE
elif mode == "Video":
    st.header("Video Violation Detection (Short clips)")
    uploaded = st.file_uploader("Upload video (mp4 / mov / avi) - keep short (<=30s)", type=["mp4", "mov", "avi"])
    if uploaded:
        # Save to a temp file (safer for cv2.VideoCapture)
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.flush()
            tfile_name = tfile.name
            tfile.close()

            cap = cv2.VideoCapture(tfile_name)
            if not cap.isOpened():
                st.error("Failed to open uploaded video. Try another file.")
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 10
                # Sample roughly 1 frame per second (or at least 1 frame)
                sample_every = max(1, int(round(fps)))
                st.write(f"Video frames: {frame_count}, FPS: {fps:.1f}, sampling every {sample_every} frames")
                progress_bar = st.progress(0.0)
                frames_processed = 0
                detected_clips = []
                max_samples_show = 9

                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # sample frames
                    if i % sample_every == 0:
                        try:
                            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        except Exception:
                            pil = Image.fromarray(frame)

                        if use_real and yolo_available and model is not None:
                            boxes, labels, scores = yolo_detect_on_pil(pil, conf_threshold=confidence)
                        else:
                            boxes, labels, scores = simulated_detection(pil, n=3)

                        out_img = draw_boxes_pil(pil, boxes, labels, scores)
                        buf = io.BytesIO()
                        out_img.save(buf, format="JPEG")
                        detected_clips.append(buf.getvalue())

                    frames_processed += 1
                    # update progress
                    if frame_count > 0:
                        progress_bar.progress(min(1.0, frames_processed / frame_count))

                cap.release()
                try:
                    os.remove(tfile_name)
                except Exception:
                    pass

                st.success(f"Processed {frames_processed} frames. Showing sampled detections below.")
                cols = st.columns(3)
                for idx, img_bytes in enumerate(detected_clips[:max_samples_show]):
                    cols[idx % 3].image(img_bytes, use_column_width=True)

        except Exception as e:
            st.error(f"Error processing uploaded video: {e}")

# HOTSPOT DEMO
elif mode == "Hotspot Demo":
    st.header("Hotspot Prediction Demo")
    st.write("Upload a CSV of historical incidents or use the sample dataset to see a simple hotspot map.")
    st.write("CSV format suggestion: lat,lon,count (or upload any CSV with 'lat' and 'lon' columns).")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None
    else:
        rng = np.random.default_rng(123)
        lats = 17.35 + rng.random(50) * 0.1
        lons = 78.4 + rng.random(50) * 0.1
        counts = rng.integers(1, 10, size=50)
        df = pd.DataFrame({"lat": lats, "lon": lons, "count": counts})

    if df is not None:
        # Try to normalize column names
        if "lat" in df.columns and "lon" in df.columns:
            map_df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
        elif "latitude" in df.columns and "longitude" in df.columns:
            map_df = df
        else:
            # Attempt best-effort: look for columns that look like lat/lon
            candidates = [c for c in df.columns if "lat" in c.lower() or "lon" in c.lower()]
            if len(candidates) >= 2:
                map_df = df.rename(columns={candidates[0]: "latitude", candidates[1]: "longitude"})
            else:
                st.error("CSV must have 'lat' and 'lon' (or 'latitude' and 'longitude') columns.")
                map_df = None

        if map_df is not None:
            # Streamlit map expects columns named 'lat'/'lon' or 'latitude'/'longitude'
            st.map(map_df[["latitude", "longitude"]].assign(latitude=map_df["latitude"], longitude=map_df["longitude"]))
            st.write("Hotspot list (top 10 by 'count' if present):")
            if "count" in df.columns:
                st.dataframe(df.sort_values("count", ascending=False).head(10))
            else:
                st.dataframe(df.head(10))
            st.write("Note: This is a demo heatmap. For production use spatial aggregation + advanced models (GNN/LSTM).")

# Footer / Notes
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Deployment Steps**:\n"
    "1. Create GitHub repo and add `app.py` and `requirements.txt`.\n"
    "2. Push to GitHub.\n"
    "3. Deploy via share.streamlit.io selecting your repo and `app.py`.\n\n"
    "**Optional:** Use GitHub Actions to run tests or containerize the app for other hosts."
)
st.sidebar.markdown("**Credits**: Demo prototype for YUVA AI Road Safety project. Replace simulated detection with real models and add secure data handling for production.")
