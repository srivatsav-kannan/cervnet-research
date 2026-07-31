import os
import platform
import sys

import tensorflow as tf
from tensorflow import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator

import tqdm as tqdm


# Function to perform image augmentatio
def augment_images(input_folder, output_folder, greater):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.3],  # Random brightness between 0.8 and 1.2
        fill_mode='nearest')
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
            tf.io.write_file(os.path.join(output_folder, f"{filename[:-4]}_{i}.jpg"), tf.io.encode_jpeg(augmented_img))
            print(os.path.join(output_folder, f"{filename[:-4]}_{i}.jpg"))
            if i >= greater:  # Generate multiple augmented images
                break  # Limiting to greater augmented images per input image

 # Note: Unaugmented datasets have not been shared

input_folder = "/Users/srivatsavkannan/Datasets/C-Spine Xray/XRay_Atlas_Curve/Sigmoid2"
output_folder = "/Users/srivatsavkannan/Datasets/C-Spine Xray/XRay_Atlas_aug/Sigmoid2"

augment_images(input_folder, output_folder, 2)
