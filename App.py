# app.py
"""
AI-Powered Road Safety - Streamlit Web App (Final)
- Works even if OpenCV (cv2) cannot be imported on the host.
- If cv2 is missing, Video mode is disabled with a helpful message.
- Image detection and Hotspot demo still work without cv2.
- Optional: enable YOLO by installing ultralytics and torch.

Deployment:
1. Add this file to your GitHub repo as app.py
2. Add requirements.txt and runtime.txt from below
3. Deploy on https://share.streamlit.io selecting your repo and app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import os
import tempfile
import pandas as pd

st.set_page_config(page_title="AI Road Safety Demo", layout="wide")

# Try importing cv2 but allow app to continue if it's missing or incompatible
cv2 = None
try:
    import cv2 as _cv2  # type: ignore
    cv2 = _cv2
except Exception as e:
    # Use st.warning only after the page has started — store a flag instead
    cv2 = None
    cv2_import_error = str(e)

# Try to import ultralytics YOLO (optional)
model = None
yolo_available = False
try:
    # Import only if ultralytics installed; don't throw if missing
    from ultralytics import YOLO  # type: ignore
    # If YOLO weights aren't present, model initialization may download weights (avoid on cloud if you don't want downloads)
    try:
        model = YOLO("yolov8n.pt")
        yolo_available = True
    except Exception:
        model = None
        yolo_available = False
except Exception:
    model = None
    yolo_available = False

# Header
st.title("AI-Powered Road Safety & Accident Prevention — Demo")
st.markdown(
    "This demo shows image/video violation detection and a simple hotspot visualization. "
    "Video mode requires OpenCV (cv2) to be importable on the server. "
    "YOLO detection is optional and requires `ultralytics` + compatible `torch`."
)

# Sidebar controls
st.sidebar.header("Configuration")
use_real = st.sidebar.checkbox("Use real YOLO detection (ultralytics)", value=False)
confidence = st.sidebar.slider("Detection confidence (if using real model)", 0.1, 0.95, 0.4)
mode = st.sidebar.radio("Mode", ["Image", "Video", "Hotspot Demo"])

if cv2 is None:
    st.sidebar.error(
        "OpenCV (cv2) not available in this environment. Video mode will be disabled. "
        "To enable, add 'opencv-python-headless' to requirements.txt and set a compatible Python runtime (see README)."
    )

# Utility: draw boxes on PIL image
def draw_boxes_pil(img_pil, boxes, labels=None, scores=None):
    img = np.array(img_pil.convert("RGB"))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2_color = (0, 255, 0) if cv2 is not None else (0, 255, 0)
        # Use OpenCV if available, otherwise use PIL drawing fallback via numpy + cv2
        try:
            # cv2.rectangle works whether or not cv2 was the import origin; if cv2 missing, this block won't run
            if cv2 is not None:
                cv2.rectangle(img, (x1, y1), (x2, y2), cv2_color, 2)
                txt = ""
                if labels and i < len(labels):
                    txt = f"{labels[i]}"
                if scores and i < len(scores):
                    txt += f" {scores[i]:.2f}"
                if txt:
                    cv2.putText(img, txt, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Fallback: draw rectangles using numpy operations (simple)
                rr1 = max(0, y1)
                rr2 = min(img.shape[0] - 1, y2)
                cc1 = max(0, x1)
                cc2 = min(img.shape[1] - 1, x2)
                img[rr1:rr1+2, cc1:cc2] = [0, 255, 0]
                img[rr2-1:rr2+1, cc1:cc2] = [0, 255, 0]
                img[rr1:rr2, cc1:cc1+2] = [0, 255, 0]
                img[rr1:rr2, cc2-1:cc2+1] = [0, 255, 0]
        except Exception:
            pass
    return Image.fromarray(img)

# Simulated detector: returns random boxes
def simulated_detection(image, n=3):
    w, h = image.size
    boxes = []
    labels = []
    scores = []
    rng = np.random.default_rng(int(time.time() % 1e6))
    for i in range(n):
        x1 = int(rng.integers(0, max(1, int(w * 0.6))))
        y1 = int(rng.integers(0, max(1, int(h * 0.6))))
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
    if model is None or not yolo_available:
        return [], [], []
    try:
        results = model.predict(source=np.array(pil_img), imgsz=640, conf=conf_threshold, verbose=False)
        if len(results) == 0:
            return [], [], []
        res = results[0]
        boxes = []
        labels = []
        scores = []
        for det in getattr(res, "boxes").data.tolist():
            x1, y1, x2, y2, score, class_id = det
            boxes.append((x1, y1, x2, y2))
            label = model.names[int(class_id)] if hasattr(model, "names") else str(int(class_id))
            labels.append(label)
            scores.append(float(score))
        return boxes, labels, scores
    except Exception:
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
    if cv2 is None:
        st.header("Video Violation Detection (Disabled)")
        st.error(
            "Video mode is disabled because OpenCV (cv2) could not be imported in this environment. "
            "To enable video processing, add 'opencv-python-headless' to requirements.txt and use a compatible Python runtime (e.g., python-3.11)."
        )
    else:
        st.header("Video Violation Detection (Short clips)")
        uploaded = st.file_uploader("Upload video (mp4 / mov / avi) - keep short (<=30s)", type=["mp4", "mov", "avi"])
        if uploaded:
            try:
                # Save to temp file for OpenCV
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
        # Normalize lat/lon column names
        if "lat" in df.columns and "lon" in df.columns:
            map_df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
        elif "latitude" in df.columns and "longitude" in df.columns:
            map_df = df
        else:
            candidates = [c for c in df.columns if "lat" in c.lower() or "lon" in c.lower()]
            if len(candidates) >= 2:
                map_df = df.rename(columns={candidates[0]: "latitude", candidates[1]: "longitude"})
            else:
                st.error("CSV must have 'lat' and 'lon' (or 'latitude' and 'longitude') columns.")
                map_df = None

        if map_df is not None:
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
    "1. Create GitHub repo and add `app.py`, `requirements.txt`, and `runtime.txt`.\n"
    "2. Push to GitHub.\n"
    "3. Deploy via share.streamlit.io selecting your repo and `app.py`.\n\n"
    "**Optional:** Use GitHub Actions to run tests or containerize the app for other hosts."
)
st.sidebar.markdown("**Credits**: Demo prototype for YUVA AI Road Safety project. Replace simulated detection with real models and add secure data handling for production.")

