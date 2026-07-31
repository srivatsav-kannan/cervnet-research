import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

# Define paths
dataset1_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalized/CS"
dataset2_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalized/Healthy"
output1_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalizedCLAHE/CS"
output2_path = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAugNormalizedCLAHE/Healthy"

os.makedirs(output1_path, exist_ok=True)
os.makedirs(output2_path, exist_ok=True)

# Function to apply CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print(type(image))
    print(image.shape)
    print(image.dtype)
    return clahe.apply(image)

# Process datasets
def process_dataset(input_path, output_path):
    for filename in tqdm(os.listdir(input_path)):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            clahe_img = apply_clahe(img)
            cv2.imwrite(os.path.join(output_path, filename), clahe_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

# Apply CLAHE to both datasets
process_dataset(dataset1_path, output1_path)
process_dataset(dataset2_path, output2_path)

print("CLAHE processing completed successfully!")
