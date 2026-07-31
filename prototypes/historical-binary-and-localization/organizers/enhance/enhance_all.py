import os
import cv2
import numpy as np
from skimage import exposure, filters, restoration
from PIL import Image

# Directories
# input_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-boundingboxes-cropped/'
# output_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-enhanced2/'

input_dir = '/Users/srivatsavkannan/Datasets/images'
output_dir = '/Users/srivatsavkannan/Datasets/images2'
# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Histogram Equalization
def histogram_equalization(image):
    return exposure.equalize_hist(image) * 255


# Contrast Stretching
def contrast_stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    return exposure.rescale_intensity(image, in_range=(p2, p98))


# Sharpening
def sharpening(image):
    return filters.unsharp_mask(image, radius=1, amount=1) * 255


# Edge Enhancement

def edge_enhancement(image):
    edges = filters.sobel(image)
    enhanced = image + edges * 255
    return np.clip(enhanced, 0, 255)
# Noise Reduction
def noise_reduction(image):
    return restoration.denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15)


# Process and enhance each image
def enhance_image(image_path, output_path):
    # Load the image as grayscale
    image = Image.open(image_path).convert('L')
    image_array = np.array(image, dtype=np.float32)

    # Normalize to [0, 1] for processing
    image_array /= 255.0

    # Apply enhancement techniques
    # image_array = histogram_equalization(image_array)

    image_array = contrast_stretching(image_array)
    # image_array = sharpening(image_array)
    # image_array = edge_enhancement(image_array)
    image_array = noise_reduction(image_array)
    #
    # # Convert back to uint8
    enhanced_image = np.clip(image_array, 0, 1) * 255
    enhanced_image = enhanced_image.astype(np.uint8)

    # Save the enhanced image
    enhanced_pil = Image.fromarray(enhanced_image)
    enhanced_pil.save(output_path)


# Process all images in the input directory
def process_images(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                input_filepath = os.path.join(root, file)
                output_filepath = os.path.join(output_path, file)

                try:
                    enhance_image(input_filepath, output_filepath)
                    print(f"Enhanced: {input_filepath} -> {output_filepath}")
                except Exception as e:
                    print(f"Failed to enhance {input_filepath}: {e}")


# Run the enhancement
process_images(input_dir, output_dir)
print("Enhancement process completed.")
