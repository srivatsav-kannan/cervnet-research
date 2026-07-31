import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Directories
cropped_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/All'
enhanced_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Enhanced'

# CLAHE Parameters
clip_limit = 3.0  # Controls contrast enhancement (higher = more contrast)
tile_grid_size = (8, 8)  # Defines the size of local regions

# Apply CLAHE to a single image
def enhance_image_clahe(image_array):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_img = clahe.apply(image_array)  # Apply CLAHE
    return enhanced_img

# Process all images in a directory
def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        # Maintain directory structure
        relative_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                # Load image in grayscale
                input_filepath = os.path.join(root, file)
                image = Image.open(input_filepath).convert('L')
                image_array = np.array(image)

                # Apply CLAHE enhancement
                enhanced_array = enhance_image_clahe(image_array)

                # Save enhanced image
                plt.imsave(os.path.join(output_path, file), enhanced_array, cmap='gray')

# Execute enhancement
process_images(cropped_dir, enhanced_dir)
print("CLAHE-based image enhancement completed. Enhanced images saved in:", enhanced_dir)
