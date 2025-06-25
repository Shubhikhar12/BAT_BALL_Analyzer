import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import altair as alt
from moviepy import VideoFileClip

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Helper: Angle calculation
def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# Helper: Shot classification
def classify_shot(angle):
    if angle < 20:
        return "Scoop / Reverse Sweep / Tight Defense", "Low"
    elif angle < 70:
        return "Straight Drive / On Drive / Pull", "Medium"
    elif angle < 120:
        return "Cover Drive / Lofted Drive", "High"
    else:
        return "Uppercut / Cut / Hook", "Very High"

# Helper: Extract frames from video
def extract_frames(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    os.makedirs("frames", exist_ok=True)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join("frames", f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        frame_count += 1
    cap.release()
    return frame_paths

# Streamlit UI
st.set_page_config(page_title="Bat Swing Analyzer", layout="wide")
st.title("🏏 Bat Swing Angle Analyzer")

st.markdown("Upload a cricket **video** or **image** to analyze elbow swing angle and predict shot type.")

# Image Upload
st.subheader("📷 Upload Single Frame (JPG/PNG)")
image_file = st.file_uploader("Upload Frame", type=["jpg", "jpeg", "png"])
if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    results = model(image)
    keypoints = results[0].keypoints
    if keypoints:
        kpts = keypoints.xy[0].cpu().numpy()
        shoulder, elbow, wrist = kpts[6], kpts[8], kpts[10]
        angle = get_angle(shoulder, elbow, wrist)
        shot, category = classify_shot(angle)
        st.image(image_file, caption="Uploaded Frame", use_column_width=True)
        st.markdown("### 🤖 Pose Analysis Result")
        st.markdown(f"""
        - **Right Shoulder Coordinates**: {shoulder.tolist()}
        - **Right Elbow Coordinates**: {elbow.tolist()}
        - **Right Wrist Coordinates**: {wrist.tolist()}
        - ✅ **Elbow Angle**: `{angle:.1f}°`
        """)
        st.success(f"🏏 Predicted Shot Based on Angle {angle:.1f}°: **{shot}**")
    else:
        st.error("❌ No keypoints detected.")

# Video Upload
st.subheader("🎞️ Batch Video Analysis")
video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    st.video(video_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.info("🔍 Extracting frames... please wait.")
    frames = extract_frames(video_path, frame_interval=10)
    analysis_results = []

    for frame in frames:
        img = cv2.imread(frame)
        results = model(img)
        keypoints = results[0].keypoints
        if keypoints and len(keypoints.xy[0]) >= 11:
            kpts = keypoints.xy[0].cpu().numpy()
            shoulder, elbow, wrist = kpts[6], kpts[8], kpts[10]
            angle = get_angle(shoulder, elbow, wrist)
            shot, category = classify_shot(angle)
            analysis_results.append({
                "frame": os.path.basename(frame),
                "angle": round(angle, 2),
                "shot_type": shot,
                "category": category
            })

    if analysis_results:
        df = pd.DataFrame(analysis_results)
        st.success(f"✅ {len(df)} frames analyzed.")
        st.dataframe(df)

        st.subheader("📊 Shot Frequency")
        chart = alt.Chart(df).mark_bar().encode(
            x='shot_type:N',
            y='count():Q',
            color='category:N'
        ).properties(title="Shot Types Across Video")
        st.altair_chart(chart, use_container_width=True)

        st.subheader("🧠 Player Summary")
        st.markdown(f"""
        - **Average Angle**: `{df['angle'].mean():.2f}°`
        - **Good Shot % (angle > 60°)**: `{(df['angle'] > 60).mean()*100:.1f}%`
        - **Mistimed Shot % (angle < 30°)**: `{(df['angle'] < 30).mean()*100:.1f}%`
        """)
    else:
        st.warning("⚠️ No valid pose data found in frames.")
