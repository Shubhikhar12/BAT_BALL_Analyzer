import pandas as pd
import matplotlib.pyplot as plt

# ✅ Step 1: Load CSV
df = pd.read_csv("impact_point.csv")

# ✅ Step 2: Setup plot
plt.figure(figsize=(8, 10))
colors = {
    "Sweet Spot": "green",
    "Toe": "blue",
    "Edge": "red",
    "Upper Edge": "orange",
    "Bottom Edge": "purple",
    "Handle": "brown",
    "Missed": "gray"
}

# ✅ Step 3: Scatter by impact type
for impact_type in df['Type'].unique():
    subset = df[df['Type'] == impact_type]
    plt.scatter(subset['X'], subset['Y'], 
                label=impact_type,
                color=colors.get(impact_type, 'black'),
                s=60, edgecolors='k')

# ✅ Step 4: Flip Y-axis (images have top-left origin)
plt.gca().invert_yaxis()

# ✅ Step 5: Labels and legend
plt.title("Bat-Ball Impact Zones")
plt.xlabel("X Coordinate (horizontal on bat)")
plt.ylabel("Y Coordinate (vertical on bat)")
plt.legend()
plt.grid(True)

# ✅ Step 6: Show plot
plt.tight_layout()
plt.show()
