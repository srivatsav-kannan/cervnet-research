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
                img = img.resize((RESIZE_WIDTH, RESIZE_HEIGHT))
                img2 = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(img2)
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
                    draw.rectangle([min_x, min_y, max_x, max_y], outline="yellow", width=3)
                    # plt.imshow(img2)
                    # plt.show()
                    bounding_boxes.append([min_x, min_y, max_x, max_y])

    return np.array(images), np.array(bounding_boxes), np.array(class_labels)

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

# Load and preprocess datasets
img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Train2'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'

# X_train, bounding_boxes_train, train_class_labels = load_data(img_dir, json_dir)

img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Val2'
X_val, bounding_boxes_val, val_class_labels = load_data(img_dir, json_dir)

# # Print shapes of training data
# print("X_train shape:", X_train.shape)
# print("bounding_boxes_train shape:", bounding_boxes_train.shape)
# print("train_class_labels shape:", train_class_labels.shape)
#
# # Print shapes of validation data
# print("X_val shape:", X_val.shape)
# print("bounding_boxes_val shape:", bounding_boxes_val.shape)
# print("val_class_labels shape:", val_class_labels.shape)
#
# print("Data Types:")
# print("X_train data type:", X_train.dtype)
# print("bounding_boxes_train data type:", bounding_boxes_train.dtype)
# print("train_class_labels data type:", train_class_labels.dtype)
# print("X_val data type:", X_val.dtype)
# print("bounding_boxes_val data type:", bounding_boxes_val.dtype)
# print("val_class_labels data type:", val_class_labels.dtype)
#
# bounding_boxes_train = bounding_boxes_train.astype(np.float32)
# bounding_boxes_val = bounding_boxes_val.astype(np.float32)
#
# train_class_labels = train_class_labels.astype(np.int32)
# val_class_labels = val_class_labels.astype(np.int32)
#
# print("Data Types After Fixing:")
# print("X_train data type:", X_train.dtype)
# print("bounding_boxes_train data type:", bounding_boxes_train.dtype)
# print("train_class_labels data type:", train_class_labels.dtype)
# print("X_val data type:", X_val.dtype)
# print("bounding_boxes_val data type:", bounding_boxes_val.dtype)
# print("val_class_labels data type:", val_class_labels.dtype)

# Combine images, bounding boxes, and classification labels into unified datasets
with tf.device('/cpu:0'):
    # train_combined = tf.data.Dataset.from_tensor_slices((
    #     X_train, {"bbox_output": bounding_boxes_train, "class_output": train_class_labels}
    # )).batch(BATCH_SIZE)

    val_combined = tf.data.Dataset.from_tensor_slices((
        X_val, {"bbox_output": bounding_boxes_val, "class_output": val_class_labels}
    )).batch(BATCH_SIZE)

def compute_saliency_map(model, image, class_index=0):
    """
    Computes a saliency map for the classification head of the model.

    Args:
        model: The multi-task YOLO-like model.
        image: Input image as a numpy array of shape (H, W, C).
        class_index: The class index to compute saliency for (default 0 for binary classification).

    Returns:
        saliency_map: Normalized saliency map as a numpy array of shape (H, W).
    """
    image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        _, classification_output = model(image_tensor)  # Ignore bounding box head
        loss = classification_output[:, class_index]  # Focus on the specified class output

    # Compute the gradient of the loss with respect to the input image
    gradients = tape.gradient(loss, image_tensor)[0]

    # Take the absolute value of the gradients to get the saliency map
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1).numpy()

    # Normalize the saliency map to [0, 1] for visualization
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)

    return saliency


# Compile model
model = tf.keras.models.load_model('ssd.keras')
model.summary()

# Function to visualize saliency map
true_labels = []
predicted_labels = []

iou_list = []
gts = []
predicteds = []
# Loop through validation dataset
for batch in val_combined:
    images, labels = batch
    actual_bboxes = labels["bbox_output"].numpy()
    actual_labels = labels["class_output"].numpy()

    # Predict bounding boxes and classifications
    predicted_bboxes, predicted_labels_batch = model.predict(images)
    print(predicted_labels_batch, predicted_labels_batch.shape)
    predicted_labels_batch = (predicted_labels_batch > 0.5).astype(int).flatten()

    # Append to arrays
    true_labels.extend(actual_labels)
    predicted_labels.extend(predicted_labels_batch)

    for i in range(images.shape[0]):
        image = images[i].numpy() / 255.0  # Normalize for visualization
        actual_bbox = actual_bboxes[i]
        actual_label = actual_labels[i]
        predicted_bbox = predicted_bboxes[i]
        predicted_label = predicted_labels_batch[i]
        gts.append(actual_bbox)
        predicteds.append(predicted_bbox)
        iou = calculate_iou(actual_bbox, predicted_bbox)
        print(iou)
        iou_list.append(iou)

        # Visualize results
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Draw actual and predicted bounding boxes
        img_with_boxes = Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_with_boxes)

        # Actual bounding box
        draw.rectangle(
            [actual_bbox[0], actual_bbox[1], actual_bbox[2], actual_bbox[3]], outline="green", width=3
        )

        # Predicted bounding box
        print("Predicted Bounding Box: ", predicted_bbox)
        print("Actual Bounding Box: ", actual_bbox)
        draw.rectangle(
            [predicted_bbox[0], predicted_bbox[1], predicted_bbox[2], predicted_bbox[3]], outline="red", width=3
        )
        grad_cam_overlay = compute_saliency_map(model, image)
        ax.imshow(img_with_boxes)
        ax.set_title(f"Actual: {actual_label}, Predicted: {predicted_label}")
        ax.axis("off")

        red_patch = mpatches.Patch(color='red', label='Predicted')
        green_patch = mpatches.Patch(color='green', label='Ground Truth')
        ax.legend(handles=[red_patch, green_patch], loc='upper right')

        plt.show()

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

average_iou = np.mean(iou_list)
print(f"Average IoU: {average_iou:.4f}")

gts = np.array(gts)
predicteds = np.array(predicteds)
print("GT shape: ", gts.shape)
print("Predicted Shape: ", predicteds.shape)
score = calculate_map(gts, predicteds, threshold=0.05, image_size=(224, 224))
print("Mean Average Precision (mAP): ", score)

# Evaluate the model
model.evaluate(val_combined)

# Convert to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Generate classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Cervical Spondylosis", "Healthy"], digits=4))

# Generate confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print(conf_matrix)

# IoU = 0.8711
# mAp = 0.9167
