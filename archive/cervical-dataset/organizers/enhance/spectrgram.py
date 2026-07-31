import os
import numpy as np
import csv
from PIL import Image
from matplotlib import pyplot as plt, cm

def normalize_array(arr):
    """Normalize array to the range [0, 1]."""
    arr_min, arr_max = np.min(arr), np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def generate_spectrogram_image(image_path, output_csv_path, output_image_path, size=(32, 32), step_size=0.1, t_range=10):
    """
    Generate a spectrogram image from the given image path based on Andrew Curve-like processing.

    Args:
        image_path (str): Path to the input image.
        output_csv_path (str): Path to save the function values as a CSV file.
        output_image_path (str): Path to save the final spectrogram image.
        size (tuple): Desired size of the output spectrogram.
        step_size (float): Step size for generating t_values.
        t_range (float): Range for t_values.
    """
    # Step 1: Load and preprocess the image (image := GrayScaleMatrix(image_path))
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    resized_image = image.resize(size)  # resized_image := Resize(image, (size, size))
    image_array = np.array(resized_image, dtype=np.float32)

    # Step 2: Standardize the image (nr := (resized_image - mean(resized_image)); standardize_image := nr / std_dev)
    standardized_image = (image_array - np.mean(image_array)) / np.std(image_array)
    flatten_image = standardized_image.flatten()  # flatten_image := Flatten(standardized_image)
    num_pixels = len(flatten_image)  # num_pixels := len(flatten_image)

    # Step 3: Generate t_values (t_values := [0, step_size, ..., range])
    t_values = []
    t = 0
    while t <= t_range:
        t_values.append(t)
        t += step_size

    # Step 4: Compute function values for each t (for i := 0 to t_size)
    function_values = []
    for t in t_values:
        f_t = 0
        for j in range(num_pixels):  # for j := 0 to num_pixels
            if t == 0:
                f_t += flatten_image[j]  # f_t := f_t + flatten_image[j]
            else:
                factor = np.sin(j * np.pi * t) / np.sqrt(2.0 ** j)  # factor := sin(j π t) / sqrt(2^j)
                value = flatten_image[j] * factor  # value := flatten_image[j] * factor
                f_t += value  # f_t := f_t + value
        function_values.append(f_t)  # append f_t to function_values

    # Step 5: Save function values to CSV (store function_values to csv)
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(function_values)

    # Step 6: Normalize and reshape to spectrogram (spectrogram := Reshape(function_values, size))
    normalized_function_values = normalize_array(function_values)  # Normalize(function_values)
    spectrogram = np.reshape(normalized_function_values, size)  # Reshape(function_values, size)

    # Step 7: Convert to uint8 and apply colormap (spectrogram_unit8 := unit8(normalized_spectrogram))
    spectrogram_uint8 = (spectrogram * 255).astype(np.uint8)
    color_spectrogram = cm.jet(spectrogram_uint8)  # color_spectrogram := ColorMap(spectrogram_unit8)
    color_spectrogram = (color_spectrogram[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

    # Step 8: Save the spectrogram as an image (store color_spectrogram as image in output_path)
    spectrogram_image = Image.fromarray(color_spectrogram)
    spectrogram_image.save(output_image_path)

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
                output_csv_path = os.path.join(output_path, file.replace('.png', '.csv').replace('.jpg', '.csv'))
                output_image_path = os.path.join(output_path, file)

                # Generate spectrogram
                generate_spectrogram_image(input_filepath, output_csv_path, output_image_path, size, step_size, t_range)

# Directories
input_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized/train'
output_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized-en/train'

# Process images
process_images(input_dir, output_dir)
