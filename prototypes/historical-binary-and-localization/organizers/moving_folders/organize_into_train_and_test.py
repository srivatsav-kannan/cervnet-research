import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import shutil

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = ["CS", "Healthy"]

train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

# Define the directories for the new folders
output_base_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized"
train_output_dir = os.path.join(output_base_dir, "train")
val_output_dir = os.path.join(output_base_dir, "val")

# Ensure output directories exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Create subdirectories for each class in training and validation folders
for class_name in class_names:
    os.makedirs(os.path.join(train_output_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_output_dir, class_name), exist_ok=True)


# Function to copy images to their respective folders
def organize_images(dataset, output_dir):
    for images, labels in dataset:
        for i in range(images.shape[0]):
            # Get the image and its corresponding class name
            img = images[i].numpy().astype("uint8")
            label = labels[i].numpy()
            class_name = class_names[label]

            # Save the image to the appropriate class folder
            class_folder = os.path.join(output_dir, class_name)
            img_name = f"{len(os.listdir(class_folder)) + 1}.png"
            img_path = os.path.join(class_folder, img_name)

            # Save the image
            tf.keras.preprocessing.image.save_img(img_path, img)


# Organize training and validation datasets
organize_images(train_ds, train_output_dir)
organize_images(val_ds, val_output_dir)

print("Images organized into train and validation folders successfully.")

sys.exit(0)