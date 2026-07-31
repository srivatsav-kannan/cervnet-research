import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths
cs_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalized/CS"
healthy_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalized/Healthy"

def compute_stats(image_folder):
    means, stds = [], []
    all_pixels = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale

            if img is None:
                continue  # Skip unreadable files

            means.append(np.mean(img))
            stds.append(np.std(img))
            all_pixels.extend(img.flatten())  # Flatten image to get pixel values

    return means, stds, np.array(all_pixels)

# Compute statistics
healthy_means, healthy_stds, healthy_pixels = compute_stats(healthy_path)
cs_means, cs_stds, cs_pixels = compute_stats(cs_path)

# Print average mean and std for both datasets
print(f"Healthy - Mean: {np.mean(healthy_means):.2f}, Std: {np.mean(healthy_stds):.2f}")
print(f"CS - Mean: {np.mean(cs_means):.2f}, Std: {np.mean(cs_stds):.2f}")

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(healthy_pixels, bins=50, alpha=0.5, color='blue', label='Healthy')
plt.hist(cs_pixels, bins=50, alpha=0.5, color='red', label='CS')
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Pixel Intensity Distribution")
plt.show()
