import os
import sys

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.preprocessing import MinMaxScaler

# Constants
train_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized/train"
val_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized/val"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
class_names = ["CS", "Healthy"]

# Load datasets
train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Feature extraction function
def extract_features(image, label):
    """Extract features from an image as per pseudo-code."""
    print(image)
    print(label)
    sys.exit(0)
    image = Image.fromarray(image.numpy()).convert('L')
    resized_image = image.resize((32, 32))  # Resize to (32, 32)
    image_array = np.array(resized_image, dtype=np.float32)

    rank = np.linalg.matrix_rank(image_array)  # Rank of the matrix
    determinant = np.linalg.det(image_array) if image_array.shape[0] == image_array.shape[1] else 0  # Determinant
    eigenvalues = np.linalg.eigvals(image_array) if image_array.shape[0] == image_array.shape[1] else [0]

    flattened_image = image_array.flatten()
    features = np.concatenate([flattened_image, [rank], [determinant], eigenvalues.real])
    return features, label

# Process dataset to extract features
def process_dataset(dataset):
    """Process dataset to extract features and labels."""
    features = []
    labels = []

    for images, batch_labels in dataset:
        for img, lbl in zip(images, batch_labels):
            x, y = extract_features(img, lbl.numpy())
            features.append(x)
            labels.append(y)

    return np.array(features), np.array(labels)

# Extract features from train and validation datasets
x_train, y_train = process_dataset(train_ds)
x_val, y_val = process_dataset(val_ds)

# Normalize features
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(x_train, y_train)

# Evaluate classifier
y_pred = classifier.predict(x_val)
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))
