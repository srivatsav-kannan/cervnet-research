import os
import sys

import cv2
import numpy as np
from PIL import Image
import scipy.special as sp
import matplotlib.pyplot as plt

# Parameters
cropped_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-boundingboxes-cropped/'
enhanced_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-enhanced/'

k = 2
rho = 0.4

# k-symbol gamma function
def k_gamma(value, k):
    return sp.gamma(value / k)

# Enhance a single image using the proposed algorithm
def enhance_image(image_array):
    # plt.imshow(image_array)
    # plt.show()
    # Normalize the image to [0, 1]
    normalized_img = image_array / 255.0

    # plt.imshow(normalized_img)
    # plt.show()
    # Compute pixel probabilities (prevent divide-by-zero)
    total_intensity = np.sum(normalized_img)
    if total_intensity == 0:
        total_intensity = 1  # Avoid division by zero
    pixel_probabilities = normalized_img / total_intensity

    # Apply the enhancement equation
    try:
        enhancement_factor = pixel_probabilities**(1 - (rho / k))
        gamma_norm = k_gamma(2 - (rho / k), k)
        enhanced_img = normalized_img * enhancement_factor / gamma_norm
        # plt.imshow(enhanced_img)
        # plt.show()
    except Exception as e:
        print("Error during enhancement calculation:", e)
        return image_array  # Return original image on failure

    # Rescale back to [0, 255] and clip
    # enhanced_img = enhanced_img * 255.0
    print(enhanced_img)
    print(type(enhanced_img))
    # plt.imshow(enhanced_img)
    # plt.show()

    # plt.imsave('img.png', enhanced_img)
    # sys.exit(0)
    # enhanced_img = enhanced_img.astype(np.uint8)

    return enhanced_img

# Enhance all images in the directory
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
                # Load the image
                input_filepath = os.path.join(root, file)
                image = Image.open(input_filepath).convert('L')  # Convert to grayscale

                # Apply enhancement
                image_array = np.array(image)
                enhanced_array = enhance_image(image_array)

                plt.imsave(os.path.join(output_path, file), enhanced_array)

# Execute enhancement
process_images(cropped_dir, enhanced_dir)
print("Image enhancement completed and saved in:", enhanced_dir)
