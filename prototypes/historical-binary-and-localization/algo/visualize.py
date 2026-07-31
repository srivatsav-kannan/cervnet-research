import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load computed results from Excel
excel_path = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/results.xlsx'
df = pd.read_excel(excel_path)

# List of metrics to visualize
metrics = [
    "row2", "row3s", "row3x", "row4s", "row4x",
    "C2-7Cobb", "C2-6Cobb", "SVA", "CCI", "CCL",
    "VBA3", "VBA4", "VBA5", "VBA6", "VBA7"
]

# Number of images per row/column (customizable)
num_cols = 4  # Set number of columns
num_rows = int(np.ceil(len(metrics) / num_cols))  # Auto-calculate rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))  # Create subplot grid
fig.suptitle("C-Spine Metrics Visualization", fontsize=16)

# Iterate over metrics and plot
for i, metric in enumerate(metrics):
    row, col = divmod(i, num_cols)  # Get row/column position

    # Extract metric values from dataset
    metric_values = df[metric].values

    # Normalize values for visualization (optional)
    metric_values = (metric_values - np.min(metric_values)) / (np.max(metric_values) - np.min(metric_values) + 1e-8)

    # Reshape to a heatmap-like visualization (adjust shape if needed)
    img_size = int(np.sqrt(len(metric_values)))  # Approximate square shape
    metric_image = metric_values[:img_size ** 2].reshape((img_size, img_size))

    # Plot as a heatmap
    axes[row, col].imshow(metric_image, cmap='viridis', interpolation='nearest')
    axes[row, col].set_title(metric, fontsize=10)
    axes[row, col].axis("off")  # Hide axes

# Hide unused subplots if any
for i in range(len(metrics), num_rows * num_cols):
    row, col = divmod(i, num_cols)
    axes[row, col].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
plt.show()
