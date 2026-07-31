import os
import platform
import random
import sys

import tensorflow as tf
from scipy.ndimage import gaussian_filter
from tensorflow import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def apply_gaussian_blur(image):
    stddev = random.uniform(0.5, 1.5)  # Random standard deviation
    image_np = image
    blurred_image = gaussian_filter(image_np, sigma=stddev)  # Apply Gaussian filter
    return tf.convert_to_tensor(blurred_image, dtype=tf.float32)  # Convert back to tensor

# Function to perform image augmentatio
def augment_images(input_folder, output_folder, greater):
    greater -= 1
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    datagen = ImageDataGenerator(
        # rotation_range=10,
        # horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.1,
        rescale=1. / 255,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest')
    # datagen = ImageDataGenerator(
    #     rotation_range=15,  # Rotation range between -15 and 15 degrees
    #     preprocessing_function=lambda x: apply_gaussian_blur(x),  # Apply Gaussian Blur
    #     # Apply Gaussian Blur
    #     rescale=1. / 255,
    # )


    # Loop through each image file
    num = 0
    for filename in image_files:
        if filename == '.Ds_Store':
            continue
        img = tf.io.read_file(os.path.join(input_folder, filename))
        try:
            img = tf.image.decode_image(img, channels=3)  # Ensure it's a 3-channel image
        except:
            continue
        # Expand dimensions to match expected input shape for ImageDataGenerator
        img = tf.expand_dims(img, 0)

        # Generate augmented images using ImageDataGenerator
        augmented_images = datagen.flow(img, batch_size=1)

        # Save augmented images to output folder
        for i, batch in enumerate(augmented_images):
            augmented_img = tf.image.convert_image_dtype(batch[0], tf.uint8)
            tf.io.write_file(os.path.join(output_folder, f"{filename[:-4]}aa.jpg"), tf.io.encode_jpeg(augmented_img))
            print(os.path.join(output_folder, f"{filename[:-4]}aa.jpg"))
            if i >= greater:  # Generate multiple augmented images
                break  # Limiting to greater augmented images per input image

 # Note: Unaugmented datasets have not been shared

input_folder = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCropped/CS"
output_folder = "/Users/srivatsavkannan/Datasets/CervicalNew10/TrainROSCroppedAug/CS"

os.makedirs(output_folder, exist_ok=True)
augment_images(input_folder, output_folder, 1)
# Greater is no. of extra u get.
# 26, 22


