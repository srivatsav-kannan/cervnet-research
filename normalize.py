import os
import cv2
import numpy as np
from skimage import exposure
from tqdm import tqdm

# Define paths
dataset1_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAug/CS"
dataset2_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAug/Healthy"
output1_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalized/CS"
output2_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalized/Healthy"

os.makedirs(output1_path, exist_ok=True)
os.makedirs(output2_path, exist_ok=True)


# Function to load images
def load_images(dataset_path):
    images = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images


# Load all images from both datasets
images1 = load_images(dataset1_path)
images2 = load_images(dataset2_path)
all_images = images1 + images2

# Compute the global reference histogram
all_pixels = np.concatenate([img.flatten() for img in all_images])
global_histogram, bin_edges = np.histogram(all_pixels, bins=256, range=(0, 255), density=True)


def match_histogram(image, reference_histogram, bin_edges):
    image_histogram, _ = np.histogram(image.flatten(), bins=bin_edges, density=True)

    # Compute the cumulative distribution functions (CDFs)
    cdf_image = np.cumsum(image_histogram)
    cdf_ref = np.cumsum(reference_histogram)

    # Normalize CDFs
    cdf_image = cdf_image / cdf_image[-1]
    cdf_ref = cdf_ref / cdf_ref[-1]

    # Use linear interpolation to match the CDF of the reference
    matched_values = np.interp(cdf_image, cdf_ref, bin_edges[:-1])  # Fix: Remove last bin edge

    # Apply mapping
    matched_image = np.interp(image.flatten(), bin_edges[:-1], matched_values).reshape(image.shape)

    return matched_image


# Process datasets
def process_dataset(input_path, output_path, reference_histogram, bin_edges):
    for filename in tqdm(os.listdir(input_path)):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            normalized_img = match_histogram(img, reference_histogram, bin_edges)
            cv2.imwrite(os.path.join(output_path, filename), normalized_img, [cv2.IMWRITE_JPEG_QUALITY, 100])


# Apply histogram matching to both datasets
process_dataset(dataset1_path, output1_path, global_histogram, bin_edges)
process_dataset(dataset2_path, output2_path, global_histogram, bin_edges)

print("Histogram matching completed successfully!")
