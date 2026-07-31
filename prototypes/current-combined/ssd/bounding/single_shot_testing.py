import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
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
EXPAND_TOP = 10
EXPAND_BOTTOM = 10
EXPAND_LEFT = 10
EXPAND_RIGHT = 10

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
# Create a black image
# Define data loading functions
# model = tf.keras.models.load_model('ssd_keypoint_middle.keras',  custom_objects={'mAP': MeanAveragePrecision()})
model = tf.keras.models.load_model('ssd_keypoint_middle.keras')
model.summary()

def load_bounding_box_data(img_dir, json_dir, img_dir2, json_dir2, resize_width=224, resize_height=224):
    images = []
    bounding_boxes = []



    # Function to process a single directory
    def process_directory(img_dir, json_dir):
        if img_dir == '':
            return
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
                print(img_array.shape)
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
                        # print(min_x, min_y, max_x, max_y)
                        #
                        # # Optional: Visualize bounding box and keypoints
                        # draw = ImageDraw.Draw(img)
                        #
                        # draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=2)
                        # keypoints = model.predict(np.array([img_array]))
                        # keypoints = np.resize(keypoints, (23, 2))
                        #
                        # # Extract bounding box from keypoints
                        # min_x = max(0, min(p[0] for p in keypoints) - EXPAND_LEFT)
                        # max_x = min(224, max(p[0] for p in keypoints) + EXPAND_RIGHT)
                        # min_y = max(0, min(p[1] for p in keypoints) - EXPAND_TOP)
                        # max_y = min(224, max(p[1] for p in keypoints) + EXPAND_BOTTOM)
                        #
                        # draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=2)
                        #
                        # for (x, y) in points:
                        #     draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="green")
                        #
                        # for (x, y) in keypoints:
                        #     draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")

                        # Uncomment to display the image with bounding box and keypoints
                        # if filename.endswith('.jpg'):
                        #     plt.imshow(img)
                        #     plt.show()
                else:
                    print(json_path, 'does not exist')

    # Process both directories
    process_directory(img_dir, json_dir)
    # process_directory(img_dir2, json_dir2)

    print(np.array(images).shape)
    print(np.array(bounding_boxes).shape)
    print(np.array(bounding_boxes).dtype)

    return np.array(images), np.array(bounding_boxes)


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1: [x_min, y_min, x_max, y_max] for the first bounding box
        bbox2: [x_min, y_min, x_max, y_max] for the second bounding box
    Returns:
        IoU: Intersection over Union value
    """
    # Calculate the intersection coordinates
    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    x_max_inter = min(bbox1[2], bbox2[2])
    y_max_inter = min(bbox1[3], bbox2[3])

    # Calculate intersection area
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate areas of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    # Compute IoU
    iou = inter_area / union_area
    return iou

img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Val2'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON'
img_dir2_val = '/Users/srivatsavkannan/Datasets/CervicalNew10/Val/CS'
json_dir2_val = '/Users/srivatsavkannan/Datasets/CervicalNew2/json'
X_val, bounding_boxes_val = load_bounding_box_data(img_dir, json_dir, img_dir2_val, json_dir2_val)

# Load and preprocess datasets
# img_dir = '/Users/srivatsavkannan/Datasets/CervicalNew5/Train'
# json_dir = '/Users/srivatsavkannan/Datasets/CervicalNew2/json'
# X_train, bounding_boxes_train = load_bounding_box_data(img_dir, json_dir, '', '')





# Combine images, bounding boxes, and classification labels into unified datasets
with tf.device('/cpu:0'):
    # train_combined = tf.data.Dataset.from_tensor_slices((
    #     X_train, bounding_boxes_train
    # )).batch(BATCH_SIZE)

    val_combined = tf.data.Dataset.from_tensor_slices((
        X_val, bounding_boxes_val
    )).batch(BATCH_SIZE)



# Compile model

# Function to visualize saliency map
true_labels = []
predicted_labels = []

iou_list = []
gts = []
predicteds = []
# Loop through validation dataset


def calculate_map(gt_keypoints, pred_keypoints, threshold=0.05, image_size=(224, 224)):
    """
    Calculate Mean Average Precision (mAP) for keypoints.

    Args:
        gt_keypoints (np.array): Ground truth keypoints of shape (num_images, NUM_KEYPOINTS * 2).
        pred_keypoints (np.array): Predicted keypoints of shape (num_images, NUM_KEYPOINTS * 2).
        threshold (float): Distance threshold for keypoint matching (normalized).
        image_size (tuple): Width and height of the images.

    Returns:
        float: Mean Average Precision (mAP).
    """
    num_images, num_keypoints = gt_keypoints.shape[0], gt_keypoints.shape[1] // 2
    tp = np.zeros(num_keypoints)
    fp = np.zeros(num_keypoints)
    fn = np.zeros(num_keypoints)

    width, height = image_size

    for i in range(num_images):
        for j in range(num_keypoints):
            # Ground truth and predicted keypoints for the current keypoint
            gt_x, gt_y = gt_keypoints[i][2 * j], gt_keypoints[i][2 * j + 1]
            pred_x, pred_y = pred_keypoints[i][2 * j], pred_keypoints[i][2 * j + 1]

            # Calculate normalized distance
            distance = np.sqrt(((gt_x - pred_x) / width) ** 2 + ((gt_y - pred_y) / height) ** 2)

            # Match based on threshold
            if distance <= threshold:
                tp[j] += 1  # True positive
            else:
                fp[j] += 1  # False positive

        # Count missing keypoints (false negatives)
        for j in range(num_keypoints):
            if np.sum(pred_keypoints[i][2 * j:2 * j + 2]) == 0:
                fn[j] += 1

    # Calculate precision for each keypoint
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)  # Avoid division by zero

    # Calculate average precision (AP) for each keypoint
    ap = (precision + recall) / 2  # F1 score for simplicity

    # Mean Average Precision (mAP)
    map_score = np.mean(ap)

    return map_score

for batch in val_combined:
    images, labels = batch
    actual_bboxes = labels.numpy()

    predicted_bboxes = []
    predicted_labels_batch = []

    for i in range(images.shape[0]):
        image = images[i].numpy() / 1.0  # Normalize
        img_array = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict keypoints
        keypoints = model.predict(img_array)
        keypoints = np.resize(keypoints, (23, 2))

        # Extract bounding box from keypoints
        min_x = max(0, min(p[0] for p in keypoints) - EXPAND_LEFT)
        max_x = min(224, max(p[0] for p in keypoints) + EXPAND_RIGHT)
        min_y = max(0, min(p[1] for p in keypoints) - EXPAND_TOP)
        max_y = min(224, max(p[1] for p in keypoints) + EXPAND_BOTTOM)
        predicted_bbox = [min_x, min_y, max_x, max_y]
        predicted_bboxes.append(predicted_bbox)


        # Calculate IoU
        iou = calculate_iou(actual_bboxes[i], predicted_bbox)
        iou_list.append(iou)
        print("IoU:", iou)

        if (iou > 0.8):
            # Visualize keypoints and bounding boxes
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image.astype(np.uint8))
            # ax.scatter([p[0] for p in keypoints], [p[1] for p in keypoints], c='red', marker='o',
            #            label='Predicted Keypoints')
            rect_pred = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=3, edgecolor='red',
                                      facecolor='none', label='Predicted BBox')
            rect_gt = plt.Rectangle((actual_bboxes[i][0], actual_bboxes[i][1]),
                                    actual_bboxes[i][2] - actual_bboxes[i][0],
                                    actual_bboxes[i][3] - actual_bboxes[i][1], linewidth=3, edgecolor='green',
                                    facecolor='none', label='Ground Truth BBox')
            ax.add_patch(rect_pred)
            ax.add_patch(rect_gt)
            ax.legend()
            # plt.show()

    # Append results for evaluation
    gts.extend(actual_bboxes)
    predicteds.extend(predicted_bboxes)

# Compute mean IoU and mAP
average_iou = np.mean(iou_list)
print(f"Average IoU: {average_iou:.4f}")

score = calculate_map(np.array(gts), np.array(predicteds), threshold=0.05, image_size=(224, 224))
print("Mean Average Precision (mAP):", score)


average_iou = np.mean(iou_list)
print(f"Average IoU: {average_iou:.4f}")

gts = np.array(gts)
predicteds = np.array(predicteds)
print("GT shape: ", gts.shape)
print("Predicted Shape: ", predicteds.shape)
score = calculate_map(gts, predicteds, threshold=0.05, image_size=(224, 224))
print("Mean Average Precision (mAP): ", score)
