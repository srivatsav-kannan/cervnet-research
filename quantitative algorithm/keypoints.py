import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Constants
RESIZE_WIDTH, RESIZE_HEIGHT = 224, 224


# Load and preprocess images
def load_images(img_dir, resize_width=224, resize_height=224):
    images = []
    filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    for filename in filenames:
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((resize_width, resize_height))
        images.append(tf.keras.preprocessing.image.img_to_array(img))

    return np.array(images), filenames


# Visualization Function
def visualize_keypoints(images, pred_keypoints):
    print("Keypoints Visualization Started")

    for i in range(len(images)):
        img = (images[i] / 255.0).astype(np.float32)  # Normalize for visualization
        img = Image.fromarray((img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # Draw predicted keypoints
        for j in range(0, len(pred_keypoints[i]), 2):
            x, y = pred_keypoints[i][j], pred_keypoints[i][j + 1]
            draw.ellipse((x - 1, y - 3, x + 1, y + 1), fill="red", outline="red")

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Predicted Keypoints")

        # Add legend for colors
        plt.scatter([], [], color='red', label='Predicted')
        plt.legend(loc='upper right')

        plt.show()


# Load the trained model
model = tf.keras.models.load_model('ssd_keypoint_best.keras')

# Load images
img_dir_val = '/Users/srivatsavkannan/Datasets/CervicalNew/Train/CS'
X_val, filenames = load_images(img_dir_val)
print(X_val.shape)
# Predict keypoints
with tf.device('/cpu:0'):
    pred_keypoints = model.predict(X_val)

# Visualize results
visualize_keypoints(X_val, pred_keypoints)
