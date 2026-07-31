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
import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable()
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, name="MeanIoU", **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name="iou_sum", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        def calculate_iou_np(gt_bboxes, pred_bboxes):
            """Calculate IoU for multiple bounding boxes."""
            num_images, num_bboxes = gt_bboxes.shape[0], gt_bboxes.shape[1] // 4
            ious = []

            def calculate_iou(bbox1, bbox2):
                x_min_inter = max(bbox1[0], bbox2[0])
                y_min_inter = max(bbox1[1], bbox2[1])
                x_max_inter = min(bbox1[2], bbox2[2])
                y_max_inter = min(bbox1[3], bbox2[3])

                inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
                bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                union_area = bbox1_area + bbox2_area - inter_area

                return inter_area / union_area if union_area > 0 else 0

            for i in range(num_images):
                image_ious = []
                for j in range(num_bboxes):
                    gt_bbox = gt_bboxes[i, j * 4: j * 4 + 4]
                    pred_bbox = pred_bboxes[i, j * 4: j * 4 + 4]
                    image_ious.append(calculate_iou(gt_bbox, pred_bbox))

                ious.append(np.mean(image_ious) if image_ious else 0)

            return np.mean(ious).astype(np.float32)

        # Compute IoU using numpy_function
        iou_value = tf.numpy_function(
            func=calculate_iou_np,
            inp=[y_true, y_pred],
            Tout=tf.float32
        )

        iou_value.set_shape([])
        self.iou_sum.assign_add(iou_value)
        self.count.assign_add(1)

    def result(self):
        return self.iou_sum / (self.count + 1e-8)

    def reset_state(self):
        self.iou_sum.assign(0)
        self.count.assign(0)

import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable()
class MeanAveragePrecisionIoU(tf.keras.metrics.Metric):
    def __init__(self, name="mAP_IoU", iou_threshold=0.75, **kwargs):
        super(MeanAveragePrecisionIoU, self).__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        self.map_score = self.add_weight(name="map", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        def calculate_map_iou_np(gt_bboxes, pred_bboxes, iou_threshold):
            num_images, num_bboxes = gt_bboxes.shape[0], gt_bboxes.shape[1] // 4
            tp = np.zeros(num_bboxes)
            fp = np.zeros(num_bboxes)
            fn = np.zeros(num_bboxes)

            def calculate_iou(bbox1, bbox2):
                """Calculate the Intersection over Union (IoU) of two bounding boxes."""
                x_min_inter = max(bbox1[0], bbox2[0])
                y_min_inter = max(bbox1[1], bbox2[1])
                x_max_inter = min(bbox1[2], bbox2[2])
                y_max_inter = min(bbox1[3], bbox2[3])

                inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
                bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                union_area = bbox1_area + bbox2_area - inter_area

                return inter_area / union_area if union_area > 0 else 0

            for i in range(num_images):
                for j in range(num_bboxes):
                    gt_bbox = gt_bboxes[i, j * 4: j * 4 + 4]
                    pred_bbox = pred_bboxes[i, j * 4: j * 4 + 4]

                    iou = calculate_iou(gt_bbox, pred_bbox)

                    if iou >= iou_threshold:
                        tp[j] += 1
                    else:
                        fp[j] += 1

                # Count false negatives
                for j in range(num_bboxes):
                    if np.sum(pred_bboxes[i, j * 4:j * 4 + 4]) == 0:
                        fn[j] += 1

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            ap = (precision + recall) / 2  # Average Precision
            return np.mean(ap).astype(np.float32)

        # Compute IoU-based mAP
        map_value = tf.numpy_function(
            func=calculate_map_iou_np,
            inp=[y_true, y_pred, self.iou_threshold],
            Tout=tf.float32
        )

        map_value.set_shape([])
        self.map_score.assign_add(map_value)
        self.count.assign_add(1)

    def result(self):
        return self.map_score / (self.count + 1e-8)

    def reset_state(self):
        self.map_score.assign(0)
        self.count.assign(0)


# Create a black image
# Define data loading functions
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def load_bounding_box_data(img_dir, img_dir2, json_dir, json_dir2, resize_width=224, resize_height=224):
    images = []
    bounding_boxes = []

    # Expansion margins for bounding boxes
    EXPAND_LEFT, EXPAND_RIGHT, EXPAND_TOP, EXPAND_BOTTOM = 5, 5, 5, 5

    # Function to process a single directory
    def process_directory(img_dir, json_dir):
        for filename in os.listdir(img_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                if filename.endswith('.png'):
                    filename = filename[:7]+'.png'
                # Construct paths
                img_path = os.path.join(img_dir, filename)
                json_path = os.path.join(json_dir, filename.replace('.png', '.json').replace('.jpg', '.json'))

                # Load and resize the image
                img = Image.open(img_path).convert("RGB")
                img = img.resize((resize_width, resize_height))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)

                # Load and process JSON key points
                if os.path.exists(json_path):
                    print(json_path, 'exists')
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Collect all keypoints
                    points = []
                    for shape in data['shapes']:
                        point = shape['points'][0]  # Assuming each shape has one point
                        img_size = Image.open(img_path)
                        original_width, original_height = img_size.size
                        x = (point[0] / original_width) * resize_width  # Normalize x-coordinate
                        y = (point[1] / original_height) * resize_height  # Normalize y-coordinate
                        points.append((x, y))

                    # Calculate bounding box coordinates
                    if points:
                        min_x = max(0, min(p[0] for p in points) - EXPAND_LEFT)
                        max_x = min(resize_width, max(p[0] for p in points) + EXPAND_RIGHT)
                        min_y = max(0, min(p[1] for p in points) - EXPAND_TOP)
                        max_y = min(resize_height, max(p[1] for p in points) + EXPAND_BOTTOM)
                        bounding_boxes.append([min_x, min_y, max_x, max_y])

                        # Optional: Visualize bounding box and keypoints
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=2)
                        for (x, y) in points:
                            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="blue")
                        # Uncomment to display the image with bounding box and keypoints
                        # if filename.endswith('.jpg'):
                        #     plt.imshow(img)
                        #     plt.show()
                else:
                    print(json_path, 'does not exist')

    # Process both directories
    process_directory(img_dir, json_dir)
    process_directory(img_dir2, json_dir2)

    print(np.array(images).shape)
    print(np.array(bounding_boxes).shape)
    print(np.array(bounding_boxes).dtype)

    return np.array(images), np.array(bounding_boxes)

img_dir_train = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Train2'
json_dir_train = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
img_dir2_train = '/Users/srivatsavkannan/Datasets/CervicalNew2/Train/CS'
json_dir2_train = '/Users/srivatsavkannan/Datasets/CervicalNew2/json'

img_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Val2'
json_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
img_dir2_val = '/Users/srivatsavkannan/Datasets/CervicalNew2/Val/CS'
json_dir2_val = '/Users/srivatsavkannan/Datasets/CervicalNew2/json'

X_train, bounding_boxes_train = load_bounding_box_data(img_dir=img_dir_train, img_dir2=img_dir2_train, json_dir=json_dir_train, json_dir2=json_dir2_train)
X_val, bounding_boxes_val = load_bounding_box_data(img_dir=img_dir_val, img_dir2=img_dir2_val, json_dir=json_dir_val, json_dir2=json_dir2_val)




# Combine images, bounding boxes, and classification labels into unified datasets
with tf.device('/cpu:0'):
    train_combined = tf.data.Dataset.from_tensor_slices((
        X_train, bounding_boxes_train
    )).batch(BATCH_SIZE)

    val_combined = tf.data.Dataset.from_tensor_slices((
        X_val, bounding_boxes_val
    )).batch(BATCH_SIZE)


# Define SSD-like model


def build_ssd_model(input_shape=(224, 224, 3), num_classes=2):
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

    # Bounding box regression head (4 outputs: x, y, width, height)
    bbox_output = layers.Dense(4, name="bbox_output")(combined_features)
    # Build the model
    model = models.Model(inputs=inputs, outputs=[bbox_output])
    return model


# Compile SSD model
model = build_ssd_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss={
        'bbox_output': 'mse',  # Mean squared error for bounding box regression
    },
    metrics={
        'bbox_output': ['mae', MeanAveragePrecisionIoU(), MeanIoU()],  # Mean absolute error for bounding box
    }
)

model.summary()

early_stopping = EarlyStopping(
    monitor='val_MeanIoU',  # Track validation loss for bounding box
    patience=100,  # Stop after 10 epochs of no improvement
    restore_best_weights=True,  # Restore model to best checkpoint
    verbose=1  # Print early stopping message
)

# Training the model
history = model.fit(
    train_combined,
    validation_data=val_combined,
    epochs=40,
    callbacks=[early_stopping],
)

with open("history_ssd.json", "w") as f:
    json.dump(history.history, f)

model.save('ssd_final.keras')
