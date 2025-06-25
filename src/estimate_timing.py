import pandas as pd

# ✅ Step 1: Simulated dataset (replace this with your actual CSV later)
# You can also load from 'impact_points.csv' if you store approach frames too
data = {
    'Approach_Frame': ['frame_303.jpg', 'frame_304.jpg', 'frame_300.jpg', 'frame_302.jpg'],
    'Image': ['frame_306.jpg', 'frame_307.jpg', 'frame_264.jpg', 'frame_938.jpg']  # Impact frames
}
df = pd.DataFrame(data)

# ✅ Step 2: Define timing estimation function
def estimate_timing(approach_frame: str, impact_frame: str) -> str:
    # Extract numbers like 303 from "frame_303.jpg"
    a = int(approach_frame.split('_')[1].split('.')[0])
    b = int(impact_frame.split('_')[1].split('.')[0])
    diff = b - a
    
    if diff <= 1:
        return "Early"
    elif 2 <= diff <= 3:
        return "Ideal"
    else:
        return "Late"

# ✅ Step 3: Apply the function to estimate timing
df['Timing'] = df.apply(lambda row: estimate_timing(row['Approach_Frame'], row['Image']), axis=1)

# ✅ Step 4: Print the result
print("\n🏏 Batting Timing Estimation:")
print(df)
