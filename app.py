import streamlit as st
import cv2
import tempfile
import os
import time
from detector import detect_vehicles, detect_from_image, detect_from_video, estimate_speed, save_video

st.set_page_config(page_title="Vehicle Detection", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Detection System")

# Sidebar
st.sidebar.header("Settings")
source = st.sidebar.radio("Source", ["Image", "Video", "Webcam"])
line_position = st.sidebar.slider("Line Position", 0.1, 0.9, 0.5)

# ─── IMAGE ───────────────────────────────────────────
if source == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        annotated, counts = detect_from_image(uploaded.read())
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Detected Vehicles", use_column_width=True)

        st.subheader("Vehicle Count")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚗 Car", counts['car'])
        col2.metric("🏍️ Motorcycle", counts['motorcycle'])
        col3.metric("🚌 Bus", counts['bus'])
        col4.metric("🚛 Truck", counts['truck'])

# ─── VIDEO ───────────────────────────────────────────
elif source == "Video":
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Processing... wait a moment"):
            frames, all_counts, avg_speed = detect_from_video(
                tmp_path,
                line_position
            )

            # Detected video save 
            output_path = tmp_path.replace('.mp4', '_detected.mp4')
            save_video(frames, output_path, fps=10)

        if frames:
            st.success(f"✅ {len(frames)} frames detected ")

            # Avg speed
            st.metric("⚡ Est. Avg Speed", f"~{avg_speed} km/h")

            # Final cumulative count
            st.subheader("Total Vehicle Count")
            final = all_counts[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🚗 Car", final['car'])
            col2.metric("🏍️ Motorcycle", final['motorcycle'])
            col3.metric("🚌 Bus", final['bus'])
            col4.metric("🚛 Truck", final['truck'])

            # Smooth video playback
            st.subheader("Detected Video")
            st.video(output_path)

            # Cleanup
            os.remove(tmp_path)
            os.remove(output_path)

# ─── WEBCAM ──────────────────────────────────────────
elif source == "Webcam":
    st.info("Webcam live feed — click Start")

    col1, col2 = st.columns(2)
    run = col1.button("▶ Start")
    stop = col2.button("⏹ Stop")
    frame_window = st.image([])
    count_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        prev_boxes = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break

            annotated, counts, crossed, results = detect_vehicles(frame, line_position)

            for i, box in enumerate(results[0].boxes):
                box_coords = box.xyxy[0].tolist()
                speed = estimate_speed(prev_boxes.get(i), box_coords, fps)
                prev_boxes[i] = box_coords

                if speed > 0:
                    x1, y1 = int(box_coords[0]), int(box_coords[1])
                    cv2.putText(
                        annotated, f'{speed} km/h',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2
                    )

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_window.image(annotated_rgb, use_column_width=True)

            with count_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🚗 Car", counts['car'])
                c2.metric("🏍️ Motorcycle", counts['motorcycle'])
                c3.metric("🚌 Bus", counts['bus'])
                c4.metric("🚛 Truck", counts['truck'])

        cap.release()