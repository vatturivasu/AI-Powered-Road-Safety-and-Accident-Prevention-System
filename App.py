# App code begins here
for i in range(frame_count):
ret, frame = cap.read()
if not ret:
break
if i % sample_every == 0:
pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
if use_real and model is not None:
results = model.predict(source=np.array(pil), imgsz=640, conf=confidence)
res = results[0]
boxes = []
labels = []
scores = []
for det in res.boxes.data.tolist():
x1,y1,x2,y2,score,class_id = det
boxes.append((x1,y1,x2,y2))
labels.append(model.names[int(class_id)] if hasattr(model,'names') else str(int(class_id)))
scores.append(score)
else:
boxes, labels, scores = simulated_detection(pil, n=3)
out_img = draw_boxes_pil(pil, boxes, labels, scores)
buf = io.BytesIO()
out_img.save(buf, format='JPEG')
detected_clips.append(buf.getvalue())
frames_processed += 1
progress.progress(min(1.0, frames_processed/frame_count))
cap.release()
st.success(f"Processed {frames_processed} frames. Showing sampled detections below.")
cols = st.columns(3)
for idx, img_bytes in enumerate(detected_clips[:9]):
cols[idx%3].image(img_bytes, use_column_width=True)




# Hotspot Demo
elif mode == 'Hotspot Demo':
st.header("Hotspot Prediction Demo")
st.write("Upload a CSV of historical incidents or use the sample dataset to see a simple hotspot heatmap.")
st.write("CSV format: lat,lon,count (or use sample).")
uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
import pandas as pd
if uploaded:
df = pd.read_csv(uploaded)
else:
# sample data (random points in a bounding box near a city)
rng = np.random.default_rng(123)
lats = 17.35 + rng.random(50)*0.1
lons = 78.4 + rng.random(50)*0.1
counts = rng.integers(1,10,size=50)
df = pd.DataFrame({'lat':lats, 'lon':lons, 'count':counts})
st.map(df.rename(columns={'lat':'latitude','lon':'longitude'})[['lat','lon']].assign(latitude=df['lat'], longitude=df['lon']))
st.write("Hotspot list (top 10 by count):")
st.dataframe(df.sort_values('count', ascending=False).head(10))
st.write("Note: This is a demo heatmap visualization. For production, use proper spatial aggregation and GNN/LSTM models for prediction.")




# Footer / Notes
st.sidebar.markdown("---")
st.sidebar.markdown("**Deployment Steps**:\n1. Create GitHub repo and add `app.py` and `requirements.txt`.\n2. Push to GitHub.\n3. Deploy via share.streamlit.io selecting your repo and `app.py`.\n\n**Optional:** Use GitHub Actions to automate tests or containerize app with Docker for other hosts.")


st.sidebar.markdown("**Credits**: This demo app is a simplified prototype for the YUVA AI Road Safety project. Replace simulated detection with real models and add secure data handling for production.")




# End of file
