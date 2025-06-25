from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Get absolute path to frame
base_dir = os.path.dirname(os.path.abspath(__file__))  # path to 'src'
frame_path = os.path.join(base_dir, '..', 'frames', 'frame_264.jpg')
frame_path = os.path.abspath(frame_path)

# Check if the image exists
if not os.path.exists(frame_path):
    print(f"❌ File not found: {frame_path}")
    sys.exit(1)

# Load image
image_bgr = cv2.imread(frame_path)

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Run pose detection
results = model(image_bgr)

# Extract keypoints
keypoints = results[0].keypoints
if keypoints is None:
    print("❌ No keypoints detected.")
    sys.exit(1)

# Extract coordinates of RIGHT shoulder, elbow, wrist
kpts = keypoints.xy[0].cpu().numpy()
shoulder = kpts[6]
elbow = kpts[8]
wrist = kpts[10]

# Draw keypoints and connections
image_annotated = image_bgr.copy()
for pt in [shoulder, elbow, wrist]:
    cv2.circle(image_annotated, tuple(int(x) for x in pt), 6, (0, 255, 0), -1)  # Green

cv2.line(image_annotated, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (255, 0, 0), 2)  # Blue
cv2.line(image_annotated, tuple(elbow.astype(int)), tuple(wrist.astype(int)), (0, 0, 255), 2)      # Red

# Function to calculate angle
def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Calculate and display elbow angle
angle = get_angle(shoulder, elbow, wrist)
cv2.putText(image_annotated, f"Angle: {int(angle)} deg", tuple(elbow.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text

# Convert BGR to RGB for Matplotlib display
image_rgb = cv2.cvtColor(image_annotated, cv2.COLOR_BGR2RGB)

# Show the image
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Bat Swing Angle")
plt.axis("off")
plt.show()

print(f"🟩 Bat swing angle (elbow): {round(angle, 2)} degrees")
