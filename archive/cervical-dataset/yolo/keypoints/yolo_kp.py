import os
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageDraw
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image

# Constants
checkpoint_dir = "checkpoints/"
RESIZE_WIDTH, RESIZE_HEIGHT = 224, 224
BATCH_SIZE = 16
NUM_KEYPOINTS = 23  # Number of keypoints (23 points with x, y)

points_dict = {
    0: "C2 centroid",
    1: "C2 bottom left",
    2: "C2 bottom right",
    3: "C3 top left",
    4: "C3 top right",
    5: "C3 bottom left",
    6: "C3 bottom right",
    7: "C4 top left",
    8: "C4 top right",
    9: "C4 bottom left",
    10: "C4 bottom right",
    11: "C5 top left",
    12: "C5 top right",
    13: "C5 bottom left",
    14: "C5 bottom right",
    15: "C6 top left",
    16: "C6 top right",
    17: "C6 bottom left",
    18: "C6 bottom right",
    19: "C7 top left",
    20: "C7 top right",
    21: "C7 bottom left",
    22: "C7 bottom right",
}


# Data Loading Function
def load_keypoint_data(img_dir, json_dir, resize_width=224, resize_height=224):
    """
    Load image data and corresponding key points from JSON files.

    Args:
        img_dir (str): Directory containing images.
        json_dir (str): Directory containing JSON annotation files.
        resize_width (int): Target width for resizing images.
        resize_height (int): Target height for resizing images.

    Returns:
        np.array: Images as numpy arrays.
        np.array: Key points as numpy arrays of shape (num_images, NUM_KEYPOINTS * 2).
    """
    images = []
    key_points = []
    other_img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
    for filename in os.listdir(img_dir):
        filename = filename[:7]+'.png'
        if filename.endswith('.png'):
            img_path = os.path.join(img_dir, filename)
            json_path = os.path.join(json_dir, filename.replace('.png', '.json'))

            # Load and resize the image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((resize_width, resize_height))
            draw = ImageDraw.Draw(img)
            images.append(tf.keras.preprocessing.image.img_to_array(img))

            # Load and process JSON key points
            if os.path.exists(json_path):
                print(json_path, 'exists')
                with open(json_path, 'r') as f:
                    data = json.load(f)

                points = []
                sorted_shapes = sorted(data['shapes'], key=lambda x: list(points_dict.values()).index(x['label']))
                print(sorted_shapes)

                for shape in sorted_shapes:
                    point = shape['points'][0]
                    img_size = Image.open(os.path.join(other_img_dir, filename))
                    original_width, original_height = img_size.size
                    points.extend([
                        (point[0] / original_width) * resize_width,  # Normalize x-coordinate
                        (point[1] / original_height) * resize_height  # Normalize y-coordinate
                    ])
                    a = (point[0]/original_width) * resize_width
                    b = (point[1]/original_height) * resize_height
                    draw.ellipse((a-2,b-2,a+2,b+2), fill=255)
                # Ensure exactly NUM_KEYPOINTS points (pad with zeros if needed)
                # plt.imshow(img)
                # plt.show()
                while len(points) < NUM_KEYPOINTS * 2:
                    print("WRONG")
                    points.extend([0.0, 0.0])
                key_points.append(points)
            else:
                print(json_path, 'does not exist')

    print(np.array(images).shape)
    print(np.array(key_points).shape)

    return np.array(images), np.array(key_points)

# Load and preprocess datasets
img_dir_train = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Train2'
json_dir_train = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
img_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Val2'
json_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'


X_train, y_train = load_keypoint_data(img_dir_train, json_dir_train)
X_val, y_val = load_keypoint_data(img_dir_val, json_dir_val)

# Print dataset shapes
print("X_train shape:", X_train.shape)  # (num_images, 224, 224, 3)
print("y_train shape:", y_train.shape)  # (num_images, NUM_KEYPOINTS * 2)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# Combine images and key points into datasets
with tf.device('/cpu:0'):
    train_combined = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_combined = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

# Define SSD-like Model for Key Point Detection
def build_yolo_model(input_shape=(224, 224, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # Feature extraction backbone (simplified YOLO-like CNN)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # YOLO detection head
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)


    kp_output = layers.Dense(46, name="bbox_output")(x)

    # Build the model
    model = models.Model(inputs=inputs, outputs=[kp_output])
    return model

# Compile SSD model
model = build_yolo_model()
model.compile(
    optimizer='adam',
    loss='mse',  # Mean squared error for key point regression
    metrics=['mae']  # Mean absolute error
)
model.summary()

# Define early stopping to stop training when val_loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=21,  # Number of epochs to wait before stopping if no improvement
    restore_best_weights=True,  # Restore best weights at the end
    verbose=1
)

# Train the model with callbacks
history = model.fit(
    train_combined,
    validation_data=val_combined,
    epochs=4,
    callbacks=[early_stop]  # Add callbacks
)

# Save the final model
model.save('yolo.keras')
