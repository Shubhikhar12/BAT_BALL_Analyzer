import cv2

# ✅ Corrected Path (relative to where you run the Python file)
image_path = "frames/frame_306.jpg"

# ✅ Step 1: Load image
img = cv2.imread(image_path)

# ✅ Step 2: Validate image load
if img is None:
    print("❌ Error: Could not load image. Check the file path.")
    exit()

# ✅ Step 3: Copy for drawing
clone = img.copy()

# ✅ Step 4: Handle mouse click event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"✅ Impact Point: X={x}, Y={y}")
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.imshow("Click to Mark Impact Point", img)

# ✅ Step 5: Show image and wait for click
cv2.imshow("Click to Mark Impact Point", img)
cv2.setMouseCallback("Click to Mark Impact Point", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
