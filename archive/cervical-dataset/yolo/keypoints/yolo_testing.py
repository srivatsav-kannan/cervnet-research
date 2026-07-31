import os
import json
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Constants
RESIZE_WIDTH, RESIZE_HEIGHT = 224, 224
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

import numpy as np

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
    print(gt_keypoints)
    print(pred_keypoints)
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



# Visualization Function
def visualize_keypoints(images, gt_keypoints, pred_keypoints=None):
    """
    Visualize ground truth and optionally predicted keypoints on images.

    Args:
        images (np.array): Array of images as numpy arrays.
        gt_keypoints (np.array): Ground truth keypoints of shape (num_images, NUM_KEYPOINTS * 2).
        pred_keypoints (np.array): Predicted keypoints of shape (num_images, NUM_KEYPOINTS * 2). Default is None.
    """

    print("Keypoints Visualization Started")

    for i in range(len(images)):
        img = (images[i] / 255.0).astype(np.float32)  # Normalize for visualization
        img = Image.fromarray((img * 255).astype(np.uint8))

        draw = ImageDraw.Draw(img)

        # Draw ground truth keypoints
        for j in range(0, len(gt_keypoints[i]), 2):
            x, y = gt_keypoints[i][j], gt_keypoints[i][j + 1]
            print(x,y)
            draw.ellipse((x - 1, y - 3, x + 1, y + 1), fill="green", outline="green")

        print("Pred Now")
        # Draw predicted keypoints if available
        if pred_keypoints is not None:
            for j in range(0, len(pred_keypoints[i]), 2):
                x, y = pred_keypoints[i][j], pred_keypoints[i][j + 1]
                print(x,y)
                draw.ellipse((x - 1, y - 3, x + 1, y + 1), fill="red", outline="red")

        map_score = calculate_map(gt_keypoints, pred_keypoints, threshold=0.05)
        print(map_score)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Predicted Keypoints")

        # Add legend for colors
        plt.scatter([], [], color='red', label='Predicted')
        plt.scatter([], [], color='green', label='Ground Truth')
        plt.legend(loc='upper right')

        plt.show()
        #
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Predicted Keypoints")
        plt.show()

# Load the trained model
model = tf.keras.models.load_model('yolo.keras')

# Load validation data
img_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized11/Val2'
json_dir_val = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
X_val, y_val = load_keypoint_data(img_dir_val, json_dir_val)

# Predict keypoints

# sys.exit(0)
with tf.device('/cpu:0'):
    pred_keypoints = model.predict(X_val)


#
# Visualize results

visualize_keypoints(X_val, y_val, pred_keypoints)
