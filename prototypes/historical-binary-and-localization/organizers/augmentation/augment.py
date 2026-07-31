# import os
# import platform
# import random
# import sys
#
# import tensorflow as tf
# from scipy.ndimage import gaussian_filter
# from tensorflow import keras
# from keras.src.legacy.preprocessing.image import ImageDataGenerator
#
# import tqdm as tqdm
#
# def apply_gaussian_blur(image):
#     stddev = random.uniform(0.5, 1.5)  # Random standard deviation
#     image_np = image
#     blurred_image = gaussian_filter(image_np, sigma=stddev)  # Apply Gaussian filter
#     return tf.convert_to_tensor(blurred_image, dtype=tf.float32)  # Convert back to tensor
#
# # Function to perform image augmentatio
# def augment_images(input_folder, output_folder, greater):
#     greater -= 1
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Get list of image files in the input folder
#     image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
#     # datagen = ImageDataGenerator(
#     #     # rotation_range=10,
#     #     horizontal_flip=True,
#     #     # width_shift_range=0.2,
#     #     # height_shift_range=0.2,
#     #     rescale=1. / 255,
#     #     brightness_range=[0.8, 1.2],
#     #     fill_mode='nearest')
#     # datagen = ImageDataGenerator(
#     #     rotation_range=15,  # Rotation range between -15 and 15 degrees
#     #     preprocessing_function=lambda x: apply_gaussian_blur(x),  # Apply Gaussian Blur
#     #     # Apply Gaussian Blur
#     #     rescale=1. / 255,
#     # )
#     datagen = ImageDataGenerator(
#         rescale=1. / 255,
#         brightness_range=[0.8, 1.2],  # Adjust brightness within a controlled range
#         contrast_stretching=True,  # Custom contrast normalization
#         channel_shift_range=0.2,  # Randomly shift color channels
#         hue_shift_range=0.1,  # Small hue variations
#         saturation_range=[0.8, 1.2],  # Change color saturation slightly
#         gaussian_noise=0.02,  # Introduce slight noise
#         blur_variation=[0.0, 1.5],  # Randomized slight blurring
#         sharpening_intensity=[0.8, 1.2]  # Controlled sharpening for texture variability
#     )
#
#     # Loop through each image file
#     num = 0
#     for filename in image_files:
#         if filename == '.Ds_Store':
#             continue
#         img = tf.io.read_file(os.path.join(input_folder, filename))
#         try:
#             img = tf.image.decode_image(img, channels=3)  # Ensure it's a 3-channel image
#         except:
#             continue
#         # Expand dimensions to match expected input shape for ImageDataGenerator
#         img = tf.expand_dims(img, 0)
#
#         # Generate augmented images using ImageDataGenerator
#         augmented_images = datagen.flow(img, batch_size=1)
#
#         # Save augmented images to output folder
#         for i, batch in enumerate(augmented_images):
#             augmented_img = tf.image.convert_image_dtype(batch[0], tf.uint8)
#             tf.io.write_file(os.path.join(output_folder, f"{filename[:-4]}_{i}.png"), tf.io.encode_jpeg(augmented_img))
#             print(os.path.join(output_folder, f"{filename[:-4]}_{i}.png"))
#             if i >= greater:  # Generate multiple augmented images
#                 break  # Limiting to greater augmented images per input image
#
#  # Note: Unaugmented datasets have not been shared
#
# input_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org/Healthy"
# output_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Aug2/Healthy"
#
# os.makedirs(output_folder, exist_ok=True)
# augment_images(input_folder, output_folder, 26)
# # Greater is no. of extra u get.
# # 26, 22

import os
import random
import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to apply Gaussian blur
def apply_gaussian_blur(image):
    stddev = random.uniform(2.0, 4.0)  # Random standard deviation
    image_np = image
    blurred_image = gaussian_filter(image_np, sigma=[stddev, stddev, 0])  # Apply Gaussian filter (no effect on channels)
    return tf.convert_to_tensor(blurred_image, dtype=tf.float32)  # Convert back to tensor

# Function for contrast stretching
def contrast_stretching(image):
    """Applies contrast stretching using percentile-based normalization."""
    image_np = image.numpy().astype(np.float32)
    p2, p98 = np.percentile(image_np, (2, 98))
    image_np = np.clip((image_np - p2) * 255.0 / (p98 - p2 + 1e-5), 0, 255).astype(np.uint8)
    return tf.convert_to_tensor(image_np, dtype=tf.float32)

# Custom preprocessing function
def custom_preprocessing(image):
    image = apply_gaussian_blur(image)  # Apply Gaussian blur
    image = contrast_stretching(image)  # Apply contrast stretching
    return image

# Function to perform image augmentation
def augment_images(input_folder, output_folder, greater):
    greater -= 1
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # ImageDataGenerator with valid parameters
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        brightness_range=[1.1, 1.5],  # Adjust brightness
        channel_shift_range=0.2,  # Random color shifts
        preprocessing_function=custom_preprocessing  # Apply custom transformations
    )

    # Process each image
    for filename in image_files:
        if filename.lower() == '.ds_store':  # Ignore system files
            continue
        img_path = os.path.join(input_folder, filename)

        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)  # Ensure 3-channel image
        except:
            continue

        img = tf.expand_dims(img, 0)  # Match batch shape

        # Generate augmented images
        augmented_images = datagen.flow(img, batch_size=1)

        # Save augmented images
        for i, batch in enumerate(augmented_images):
            augmented_img = tf.image.convert_image_dtype(batch[0], tf.uint8)
            save_path = os.path.join(output_folder, f"{filename[:-4]}_{i}.png")
            tf.io.write_file(save_path, tf.io.encode_jpeg(augmented_img))
            print(f"Saved: {save_path}")

            if i >= greater:  # Stop after generating required augmentations
                break

# Define input and output paths
input_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org/Healthy"
output_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug2/Healthy"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Run augmentation
augment_images(input_folder, output_folder, 22)  # `greater` is the number of extra images per input
