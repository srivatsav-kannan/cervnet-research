import os
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

# Directories
train_dir = "/Users/srivatsavkannan/Datasets/CervicalNew2Recreate/TrainROS"
val_dir = "/Users/srivatsavkannan/Datasets/CervicalNew2Recreate/ValROS"
train_cropped_dir = "/Users/srivatsavkannan/Datasets/CervicalNew2Recreate/TrainROSCropped"
val_cropped_dir = "/Users/srivatsavkannan/Datasets/CervicalNew2Recreate/ValROSCropped"

os.makedirs(train_cropped_dir, exist_ok=True)
os.makedirs(val_cropped_dir, exist_ok=True)

# Load model
model = tf.keras.models.load_model('ssd_bounding.keras')

# Image size
IMAGE_SIZE = (224, 224)


def process_and_save_images(source_dir, dest_dir):
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        dest_class_path = os.path.join(dest_dir, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize(IMAGE_SIZE)

                img_array = np.array(image) / 1.0
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Predict bounding box
                predicted_bbox, _ = model.predict(img_array)
                predicted_bbox = predicted_bbox[0]

                # Extract coordinates
                min_x = int(max(0, predicted_bbox[0]))
                min_y = int(max(0, predicted_bbox[1]))
                max_x = int(min(224, predicted_bbox[2]))
                max_y = int(min(224, predicted_bbox[3]))

                # Crop and resize
                cropped_image = image.crop((min_x, min_y, max_x, max_y)).resize(IMAGE_SIZE)

                # Save image with original filename
                cropped_image.save(os.path.join(dest_class_path, filename))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


# Process training and validation datasets
process_and_save_images(train_dir, train_cropped_dir)
process_and_save_images(val_dir, val_cropped_dir)
