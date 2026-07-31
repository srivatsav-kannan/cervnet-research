import os
import json
import shutil

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Directories
img_dir = '/Users/srivatsavkannan/Datasets/CompressedCervicalDataset/All'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
train_output_dir = '/Users/srivatsavkannan/Datasets/CompressedCervicalDataset/Train/'
val_output_dir = '/Users/srivatsavkannan/Datasets/CompressedCervicalDataset/Val/'

# Ensure output directories exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Image dimensions
WIDTH = 224
HEIGHT = 224


def load_data(img_dir, json_dir):
    images = []
    labels = []
    filenames = []  # To store original filenames

    for filename in os.listdir(img_dir):
        if filename.endswith('.png'):
            # Load image
            img_path = os.path.join(img_dir, filename)
            json_path = os.path.join(json_dir, filename.replace('.png', '.json'))

            if os.path.exists(json_path):
                # Open the image
                img = Image.open(img_path).convert("RGB")
                image = cv2.imread(img_path)
                image = cv2.resize(image, (WIDTH, HEIGHT))

                original_width, original_height = img.size

                # Resize the image
                img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
                images.append(image)

                with open(json_path, 'r') as f:
                    data = json.load(f)

                # List to store points for bounding box
                points = []

                for shape in data['shapes']:
                    point = shape['points'][0]  # Get the point (x, y)

                    # Adjust the point coordinates according to the resized image
                    x = (point[0] / original_width) * WIDTH
                    y = (point[1] / original_height) * HEIGHT
                    adjusted_point = (x, y)

                    # Add the adjusted point to the list for bounding box calculation
                    points.append(adjusted_point)

                if points:
                    min_x = min(p[0] for p in points)
                    max_x = max(p[0] for p in points)
                    min_y = min(p[1] for p in points)
                    max_y = max(p[1] for p in points)
                    labels.append([min_x, min_y, max_x, max_y])

                # Save the original filename
                filenames.append(filename)

    return np.array(images), np.array(labels), filenames


# Load data
X, y, filenames = load_data(img_dir, json_dir)
print("Data Loading Completed")

# Train-test split
X_train, X_val, y_train, y_val, filenames_train, filenames_val = train_test_split(
    X, y, filenames, test_size=0.2, random_state=42
)


# Save the train and validation images with original filenames
def save_images(X, filenames, output_dir):
    for i, img_array in enumerate(X):

        # Save the image with the original filename
        shutil.copy(os.path.join(img_dir, filenames[i]), os.path.join(output_dir, filenames[i]))
        # img.save(os.path.join(output_dir, filenames[i]))


# Save train and validation images
save_images(X_train, filenames_train, train_output_dir)
save_images(X_val, filenames_val, val_output_dir)

print(f"Images successfully saved to Train: {train_output_dir} and Val: {val_output_dir}")
