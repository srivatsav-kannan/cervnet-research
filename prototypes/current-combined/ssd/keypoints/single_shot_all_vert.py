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
checkpoint_dir = "checkpoints4/"
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


@tf.keras.utils.register_keras_serializable()
class MeanAveragePrecision(tf.keras.metrics.Metric):
    def __init__(self, name="mAP", threshold=0.05, image_size=(224, 224), **kwargs):
        super(MeanAveragePrecision, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.image_size = image_size
        self.map_score = self.add_weight(name="map", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)  # Track number of updates

    def update_state(self, y_true, y_pred, sample_weight=None):
        def calculate_map_np(gt_keypoints, pred_keypoints, threshold, image_size):
            num_images, num_keypoints = gt_keypoints.shape[0], gt_keypoints.shape[1] // 2
            tp = np.zeros(num_keypoints)
            fp = np.zeros(num_keypoints)
            fn = np.zeros(num_keypoints)
            width, height = image_size

            for i in range(num_images):
                for j in range(num_keypoints):
                    gt_x, gt_y = gt_keypoints[i][2 * j], gt_keypoints[i][2 * j + 1]
                    pred_x, pred_y = pred_keypoints[i][2 * j], pred_keypoints[i][2 * j + 1]

                    # Calculate normalized distance
                    distance = np.sqrt(((gt_x - pred_x) / width) ** 2 + ((gt_y - pred_y) / height) ** 2)

                    if distance <= threshold:
                        tp[j] += 1  # True positive
                    else:
                        fp[j] += 1  # False positive

                # Count missing keypoints (false negatives)
                for j in range(num_keypoints):
                    if np.sum(pred_keypoints[i][2 * j:2 * j + 2]) == 0:
                        fn[j] += 1

            # Compute precision and recall
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            # Compute average precision (AP) for each keypoint
            ap = (precision + recall) / 2

            # Compute Mean Average Precision (mAP)
            return np.mean(ap).astype(np.float32)

        # Compute mAP using numpy_function
        map_value = tf.numpy_function(
            func=calculate_map_np,
            inp=[y_true, y_pred, self.threshold, self.image_size],
            Tout=tf.float32
        )

        # Ensure map_value has a defined shape
        map_value.set_shape([])

        # Update metric state
        self.map_score.assign_add(map_value)
        self.count.assign_add(1)  # Increment update count

    def result(self):
        return self.map_score / (self.count + 1e-8)  # Return the mean mAP

    def reset_state(self):
        self.map_score.assign(0)
        self.count.assign(0)


# Data Loading Function
def load_keypoint_data(img_dir, img_dir2, json_dir, json_dir2, resize_width=224, resize_height=224):
    images = []
    key_points = []
    for filename in os.listdir(img_dir):
        if filename.endswith('.png'):
            other_img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
            filename = filename[:7] + '.png'
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
                    a = (point[0] / original_width) * resize_width
                    b = (point[1] / original_height) * resize_height
                    draw.ellipse((a - 2, b - 2, a + 2, b + 2), fill=255)
                # Ensure exactly NUM_KEYPOINTS points (pad with zeros if needed)
                # plt.imshow(img)
                # plt.show()
                while len(points) < NUM_KEYPOINTS * 2:
                    print("WRONG")
                    points.extend([0.0, 0.0])
                key_points.append(points)
            else:
                print(json_path, 'does not exist')

    for filename in os.listdir(img_dir2):
        if filename.endswith('.jpg'):
            other_img_dir = img_dir2
            img_path = os.path.join(img_dir2, filename)
            json_path = os.path.join(json_dir2, filename.replace('.jpg', '.json'))

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
                    a = (point[0] / original_width) * resize_width
                    b = (point[1] / original_height) * resize_height
                    draw.ellipse((a - 2, b - 2, a + 2, b + 2), fill=255)
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
img_dir_train = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Train3'
json_dir_train = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
img_dir2_train = '/Users/srivatsavkannan/Datasets/CervicalNew2/Train/CS'
json_dir2_train = '/Users/srivatsavkannan/Datasets/CervicalNew2/json'

img_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Val3'
json_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
img_dir2_val = '/Users/srivatsavkannan/Datasets/CervicalNew2/Val/CS'
json_dir2_val = '/Users/srivatsavkannan/Datasets/CervicalNew2/json'

X_train, y_train = load_keypoint_data(img_dir=img_dir_train, img_dir2=img_dir2_train, json_dir=json_dir_train, json_dir2=json_dir2_train)
X_val, y_val = load_keypoint_data(img_dir=img_dir_val, img_dir2=img_dir2_val, json_dir=json_dir_val, json_dir2=json_dir2_val)

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
def build_keypoint_ssd_model(input_shape=(224, 224, 3), num_keypoints=23):
    output_size = num_keypoints * 2  # Each key point has (x, y) coordinates

    inputs = layers.Input(shape=input_shape)

    # Feature extraction backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Multi-scale feature maps for SSD
    feature_map_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    feature_map_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))(feature_map_1)
    feature_map_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))(feature_map_2)

    # Combine feature maps (flatten and concatenate)
    flat_1 = layers.Flatten()(feature_map_1)
    flat_2 = layers.Flatten()(feature_map_2)
    flat_3 = layers.Flatten()(feature_map_3)
    combined_features = layers.Concatenate()([flat_1, flat_2, flat_3])

    # Key point regression head
    keypoint_output = layers.Dense(output_size, name="keypoint_output")(combined_features)

    # Build the model
    model = models.Model(inputs=inputs, outputs=keypoint_output)
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for key point regression
        metrics=['mae', MeanAveragePrecision()]  # Mean absolute error
    )
    return model


# Compile SSD model
model = build_keypoint_ssd_model()
model.summary()




early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mAP',   # Track validation mAP
    patience=100,
    mode='max',          # Because we want to maximize mAP
    restore_best_weights=True  # Restore the best model when stopping
)

# Train the model
history = model.fit(
    train_combined,
    validation_data=val_combined,
    epochs=40,
    callbacks=[early_stopping],
)

with open("history_ssd5.json", "w") as f:
    json.dump(history.history, f)

# Save the final model
model.save('ssd_keypoint_best.keras')
