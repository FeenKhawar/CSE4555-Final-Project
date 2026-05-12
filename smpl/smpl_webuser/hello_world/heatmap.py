import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../RobustnessResults.csv'))

df = pd.read_csv(csv_path)

heatmap_data = df.pivot(
    index="Std of Gaussian Noise for Pose",
    columns="Std of Gaussian Noise for Shape",
    values="Mean Vertex Displacement"
)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    cbar_kws={"label": "Mean Vertex Displacement"},
    linewidths=0.5,
    linecolor="white",
)
plt.gca().invert_yaxis()
plt.xlabel("Std of Gaussian Noise for Shape")
plt.ylabel("Std of Gaussian Noise for Pose")
plt.title("Robustness Analysis: Mean Vertex Displacement")
plt.tight_layout()
plt.savefig("robustness_heatmap.png", dpi=300)
plt.show()