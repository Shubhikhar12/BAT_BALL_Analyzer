import cv2
import os
import csv

# ✅ Step 1: Set image path (change this per frame)
image_path = "frames/frame_938.jpg"

# ✅ Step 2: CSV output path
csv_path = "impact_points.csv"

# ✅ Step 3: Load image
img = cv2.imread(image_path)
if img is None:
    print("❌ Error: Could not load image. Check the path.")
    exit()

# ✅ Step 4: Clone the image for drawing
clone = img.copy()

# ✅ Step 5: Define all impact types
impact_types = [
    "Sweet Spot", 
    "Toe", 
    "Edge", 
    "Upper Edge", 
    "Bottom Edge", 
    "Handle", 
    "Missed"
]
selected_type = [impact_types[0]]  # default

# ✅ Step 6: Function to choose impact type
def choose_classification():
    print("\n🎯 Choose the Impact Type:")
    for i, val in enumerate(impact_types):
        print(f"{i+1}. {val}")
    try:
        choice = int(input("Enter your choice (1 to {}): ".format(len(impact_types))))
        if 1 <= choice <= len(impact_types):
            selected_type[0] = impact_types[choice - 1]
        else:
            print("⚠️ Invalid choice. Using default.")
    except:
        print("⚠️ Invalid input. Using default.")

# ✅ Step 7: Handle mouse clicks
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"✅ Marked: X={x}, Y={y}, Type={selected_type[0]}")

        # Draw red dot
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.imshow("Click to Mark Impact", img)

        # Save to CSV
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([os.path.basename(image_path), x, y, selected_type[0]])
        print("💾 Saved to impact_points.csv")

# ✅ Step 8: Start
choose_classification()
cv2.imshow("Click to Mark Impact", img)
cv2.setMouseCallback("Click to Mark Impact", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
