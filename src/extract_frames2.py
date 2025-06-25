import cv2

# ✅ Full path to your video file
video_path = "E:/Data Analyst/BAT_BALL Analyzer/data/rohit-sharma-shot_xiIEWgbS.mp4"

# ✅ Open the video
cap = cv2.VideoCapture(video_path)
frame_number = 0

# ✅ Loop through the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # ✅ Show frame number on screen
    display = frame.copy()
    cv2.putText(display, f"Frame: {frame_number}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ✅ Display frame
    cv2.imshow("Video Frame", display)

    # ✅ Wait for keypress
    key = cv2.waitKey(30)

    # ✅ Press 's' to save the current frame
    if key == ord('s'):
        filename = f"E:/Data Analyst/BAT_BALL Analyzer/frames/frame_{frame_number}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")

    # ✅ Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
