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
                x1, y1, x2, y2, score, class_id = det
                boxes.append((x1, y1, x2, y2))
                labels.append(model.names[int(class_id)] if hasattr(model, 'names') else str(int(class_id)))
                scores.append(score)
        else:
            boxes, labels, scores = simulated_detection(pil, n=3)

        out_img = draw_boxes_pil(pil, boxes, labels, scores)
        buf = io.BytesIO()
        out_img.save(buf, format='JPEG')
        detected_clips.append(buf.getvalue())

    frames_processed += 1
    progress.progress(min(1.0, frames_processed / frame_count))

