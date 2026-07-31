import os
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

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

# Directories
train_dir = "/Users/srivatsavkannan/Datasets/CervicalNew9/TrainROS"
val_dir = "/Users/srivatsavkannan/Datasets/CervicalNew9/ValROS"
train_cropped_dir = "/Users/srivatsavkannan/Datasets/CervicalNew9/TrainROSCropped2"
val_cropped_dir = "/Users/srivatsavkannan/Datasets/CervicalNew9/ValROSCropped2"

os.makedirs(train_cropped_dir, exist_ok=True)
os.makedirs(val_cropped_dir, exist_ok=True)

# Load model
model = tf.keras.models.load_model('ssd_keypoint_middle.keras',  custom_objects={'mAP': MeanAveragePrecision()})

# Image size
IMAGE_SIZE = (224, 224)
# EXPAND_LEFT, EXPAND_RIGHT, EXPAND_TOP, EXPAND_BOTTOM = 5,5,5,5
EXPAND_TOP = 20
EXPAND_BOTTOM = 20
EXPAND_LEFT = 20
EXPAND_RIGHT = 20
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
                keypoints = model.predict(img_array)
                keypoints = np.resize(keypoints, (23, 2))
                # plt.imshow(img_array[0].astype(np.uint8))
                # plt.scatter([p[0] for p in keypoints], [p[1] for p in keypoints], c='red', marker='o')
                # plt.title("Predicted Keypoints")
                # plt.show()


                min_x = max(0, min(p[0] for p in keypoints) - EXPAND_LEFT)
                max_x = min(224, max(p[0] for p in keypoints) + EXPAND_RIGHT)
                min_y = max(0, min(p[1] for p in keypoints) - EXPAND_TOP)
                max_y = min(224, max(p[1] for p in keypoints) + EXPAND_BOTTOM)

                predicted_bbox = [min_x, min_y, max_x, max_y]

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
