import os

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from skimage.feature import hog

def extract_hog_features(image):
    """Extract HOG features from a single image."""
    gray_img = rgb2gray(image)
    features = hog(
        gray_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features

def get_saliency_map(model, image, tabular_data, hog_data):
    """
    Generate a saliency map for a given image input.

    Args:
    - model: Trained Keras model
    - image: (1, H, W, C) preprocessed image input
    - tabular_data: (1, T) preprocessed tabular input
    - hog_data: (1, HOG) preprocessed HOG input

    Returns:
    - Heatmap overlaid on the original image
    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert NumPy to Tensor
    tabular_data = tf.convert_to_tensor(tabular_data, dtype=tf.float32)
    hog_data = tf.convert_to_tensor(hog_data, dtype=tf.float32)

    # Use GradientTape to compute gradients of the output w.r.t input image
    with tf.GradientTape() as tape:
        # Ensure the image input is watched
        tape.watch(image)

        # Get the model output
        predictions = model([tabular_data, image, hog_data])
        loss = predictions[0]  # Focus on the single output

    # Compute gradients of output w.r.t image input
    grads = tape.gradient(loss, image)

    # Get absolute values and normalize gradients
    grads = tf.reduce_max(tf.abs(grads), axis=-1)[0]  # Max over RGB channels

    # Normalize to [0,1]
    grads = (grads - tf.reduce_min(grads)) / (tf.reduce_max(grads) - tf.reduce_min(grads) + 1e-10)

    # Convert to NumPy
    grads = grads.numpy()

    # Resize saliency map to match image size
    saliency_map = cv2.resize(grads, (image.shape[2], image.shape[1]))

    # Convert grayscale saliency map to color heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on image
    image = np.squeeze(image)  # Remove batch dimension
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize image to [0,1]
    image = np.uint8(255 * image)  # Convert to uint8

    overlaid_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlaid_img

file_path = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/results3.xlsx'
data = pd.read_excel(file_path, header=0).dropna()

print("started")
model = tf.keras.models.load_model('saint_77_trans_gen_aug_before_crop4.keras')
print("model loaded")
for image_name in os.listdir('/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug_Cropped2/CS'):
    dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug_Cropped2/CS/' + image_name
    image = load_img(dir, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
#
#     # Extract HOG features for this image
    hog_features = extract_hog_features(image[0])  # Remove batch dimension
    hog_features = np.expand_dims(hog_features, axis=0)  # Match batch format
#
#     # Find corresponding quantitative data
    seq_number = int(image_name[:7])  # Extract ID from filename
    row = data[data['pic_id'] == seq_number]
#
    if row.empty:
        print(f"No matching row found for {image_name}")
        continue
#
    quant_data = row.drop(columns=['pic_id']).values
    # quant_data = np.expand_dims(quant_data, axis=0)  # Match batch format
    print(quant_data.shape)
#
#     # Generate and display saliency map
    saliency_image = get_saliency_map(model, image, quant_data, hog_features)
    plt.imshow(saliency_image)
    plt.axis('off')
    plt.show()
