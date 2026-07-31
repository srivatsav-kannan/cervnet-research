import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def normalize_array(arr):
    """Apply Min-Max scaling to normalize array to [0, 1]."""
    arr_min, arr_max = np.min(arr), np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def generate_andrew_curve(image_array, size=(32, 32)):
    """Generate the Andrew Curve from the flattened image array."""
    flattened = image_array.flatten()
    num_pixels = len(flattened)
    t_values = np.linspace(0, 10, num=np.prod(size))  # Ensure correct number of elements
    function_values = []

    for t in t_values:
        f_t = 0
        for j in range(num_pixels):
            factor = (np.sin(j * np.pi * t) / np.sqrt(2.0 ** j)) if t != 0 else 1
            f_t += flattened[j] * factor
        function_values.append(f_t)

    return np.array(function_values)

def generate_spectrogram(andrew_curve, size):
    """Generate and normalize the spectrogram image from the Andrew Curve."""

    spectrogram = np.reshape(andrew_curve, size)
    normalized_spectrogram = normalize_array(spectrogram)
    return (normalized_spectrogram * 255).astype(np.uint8)

def process_images(input_dir, output_dir, size=(32, 32), step_size=0.1, t_range=10):
    """Process images to generate spectrograms."""
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
                input_filepath = os.path.join(root, file)
                image = Image.open(input_filepath).convert('L')  # Convert to grayscale

                # Resize and standardize image
                resized_image = image.resize(size)
                resized_array = np.array(resized_image, dtype=np.float32)
                standardized_array = (resized_array - np.mean(resized_array)) / np.std(resized_array)

                # Generate Andrew Curve and Spectrogram
                andrew_curve = generate_andrew_curve(standardized_array, size=size)
                spectrogram = generate_spectrogram2(andrew_curve)

                # Display spectrogram
                # plt.imshow(spectrogram, cmap='gray')
                # plt.title(f"Spectrogram of {file}")
                # plt.show()

                # Save the spectrogram image
                spectrogram_image = Image.fromarray(spectrogram)
                spectrogram_image.save(os.path.join(output_path, file))

# Directories
input_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized/val'
output_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized-en2/val'
os.makedirs(output_dir, exist_ok=True)
# Process images
process_images(input_dir, output_dir)
