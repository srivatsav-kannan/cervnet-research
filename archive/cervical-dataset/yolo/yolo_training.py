import sys

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image, ImageDraw
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_dir = "checkpointsk/"
radius = 5

RESIZE_WIDTH, RESIZE_HEIGHT = 224, 224  # Desired dimensions
BATCH_SIZE = 16

# Bounding box expansion margins
EXPAND_TOP = 75 / 2
EXPAND_BOTTOM = 37.5 / 2
EXPAND_LEFT = 20 / 2
EXPAND_RIGHT = 100 / 2

# Create a black image
# Define data loading functions
def load_data(img_dir, json_dir):
    other_img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
    images, bounding_boxes, class_labels = [], [], []
    WIDTH, HEIGHT = 224, 224
    excel_path = '/Users/srivatsavkannan/Datasets/C-Spine Xray/datasets.xlsx'

    # Load the Excel file
    df = pd.read_excel(excel_path)

    for filename in os.listdir(img_dir):
        if filename.endswith('.png'):
            # Extract the row number from the filename
            filename = filename[:7] + '.png'
            row = filename[:4]
            row = int(row)
            print(row)
            print(df.iat[row, 0])
            # Extract the disease classification from the Excel file
            disease_classification = df.iat[row, 3]  # Subtracting 1 to align with 0-indexing
            disease_classification -= 1  # Subtract 1 from the value
            class_labels.append(disease_classification)

            # Load and process the image
            img_path = os.path.join(img_dir, filename)
            json_path = os.path.join(json_dir, filename.replace('.png', '.json'))

            if os.path.exists(json_path):
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                img_size = Image.open(os.path.join(other_img_dir, filename))
                original_width, original_height = img_size.size

                images.append(tf.keras.preprocessing.image.img_to_array(img))

                # Load and process JSON data
                with open(json_path, 'r') as f:
                    data = json.load(f)

                points = []
                for shape in data['shapes']:
                    point = shape['points'][0]  # Get the point (x, y)
                    x = (point[0] / original_width) * WIDTH
                    y = (point[1] / original_height) * HEIGHT
                    points.append((x, y))

                # Calculate bounding box coordinates
                if points:
                    min_x = max(0, min(p[0] for p in points) - EXPAND_LEFT)
                    max_x = min(WIDTH, max(p[0] for p in points) + EXPAND_RIGHT)
                    min_y = max(0, min(p[1] for p in points) - EXPAND_TOP)
                    max_y = min(HEIGHT, max(p[1] for p in points) + EXPAND_BOTTOM)
                    bounding_boxes.append([min_x, min_y, max_x, max_y])
    print(np.array(images).shape)
    print(np.array(bounding_boxes).shape)
    print(np.array(class_labels).shape)
    return np.array(images), np.array(bounding_boxes), np.array(class_labels)


# Load and preprocess datasets
img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Train2'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'

X_train, bounding_boxes_train, train_class_labels = load_data(img_dir, json_dir)

img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Val2'

X_val, bounding_boxes_val, val_class_labels = load_data(img_dir, json_dir)

# Print shapes of training data
print("X_train shape:", X_train.shape)
print("bounding_boxes_train shape:", bounding_boxes_train.shape)
print("train_class_labels shape:", train_class_labels.shape)

# Print shapes of validation data
print("X_val shape:", X_val.shape)
print("bounding_boxes_val shape:", bounding_boxes_val.shape)
print("val_class_labels shape:", val_class_labels.shape)

print("Data Types:")
print("X_train data type:", X_train.dtype)
print("bounding_boxes_train data type:", bounding_boxes_train.dtype)
print("train_class_labels data type:", train_class_labels.dtype)
print("X_val data type:", X_val.dtype)
print("bounding_boxes_val data type:", bounding_boxes_val.dtype)
print("val_class_labels data type:", val_class_labels.dtype)

bounding_boxes_train = bounding_boxes_train.astype(np.float32)
bounding_boxes_val = bounding_boxes_val.astype(np.float32)

train_class_labels = train_class_labels.astype(np.int32)
val_class_labels = val_class_labels.astype(np.int32)

print("Data Types After Fixing:")
print("X_train data type:", X_train.dtype)
print("bounding_boxes_train data type:", bounding_boxes_train.dtype)
print("train_class_labels data type:", train_class_labels.dtype)
print("X_val data type:", X_val.dtype)
print("bounding_boxes_val data type:", bounding_boxes_val.dtype)
print("val_class_labels data type:", val_class_labels.dtype)

# Combine images, bounding boxes, and classification labels into unified datasets
with tf.device('/cpu:0'):
    train_combined = tf.data.Dataset.from_tensor_slices((
        X_train, {"bounding_box": bounding_boxes_train, "classification": train_class_labels}
    )).batch(BATCH_SIZE)

    val_combined = tf.data.Dataset.from_tensor_slices((
        X_val, {"bounding_box": bounding_boxes_val, "classification": val_class_labels}
    )).batch(BATCH_SIZE)


# Define YOLO-like multi-task model
def build_yolo_multitask(input_shape=(224, 224, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # Feature extraction backbone (like YOLO's Darknet)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Detection head for bounding box prediction
    bounding_box_head = layers.Flatten()(x)
    bounding_box_head = layers.Dense(128, activation='relu')(bounding_box_head)
    bounding_box_head = layers.Dense(4, activation='linear', name='bounding_box')(bounding_box_head)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    classification_head = layers.Dense(1, activation='sigmoid', name='classification')(x)

    model = models.Model(inputs=inputs, outputs=[bounding_box_head, classification_head])
    return model


# Compile model
model = build_yolo_multitask()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss={
        'bounding_box': 'mse',  # Mean squared error for bounding box regression
        'classification': 'binary_crossentropy'  # Binary cross-entropy for classification
    },
    metrics={
        'bounding_box': ['mae'],  # Mean absolute error for bounding box
        'classification': ['accuracy']  # Classification accuracy
    }
)

model.summary()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras"),  # Save model as 'model_epoch_XX.h5'
    save_freq='epoch',  # Save every epoch
    save_weights_only=False,  # Save the entire model (not just weights)
    verbose=1  # Print a message when saving
)

# Training the model
history = model.fit(
    train_combined,
    validation_data=val_combined,
    epochs=20,
    callbacks=[checkpoint_callback],
)

model.save('yolo.keras')

