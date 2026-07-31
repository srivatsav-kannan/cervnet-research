import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from skimage.feature import hog
from skimage.color import rgb2gray


# Load and process images
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


def load_images(image_dir, data):
    labels, images, gt_stats = [], [], []
    dirs = [image_dir + '/CS', image_dir + '/Healthy']
    for idx, image_dir in enumerate(dirs):
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith('.png'):
                seq_number = int(filename[:7])
                row = data[data['pic_id'] == seq_number]
                if row.empty:
                    continue
                label_row = row.drop(columns=['pic_id']).iloc[0]
                gt_stats.append(label_row.values)
                img_path = os.path.join(image_dir, filename)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (224, 224))
                labels.append(idx)
                images.append(image)
    return np.array(gt_stats), np.array(labels), np.array(images) / 255.0


# Load dataset
file_path = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/results3.xlsx'
data = pd.read_excel(file_path, header=0).dropna()

train_image_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Aug_Cropped2'
val_image_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug_Cropped2'
X_train, y_train, images_train = load_images(train_image_dir, data)
X_val, y_val, images_val = load_images(val_image_dir, data)

hog_train = np.array([extract_hog_features(img) for img in images_train])
hog_val = np.array([extract_hog_features(img) for img in images_val])

# Load model
model = tf.keras.models.load_model('saint_77_trans_gen_aug_before_crop4.keras')

import matplotlib.pyplot as plt
import numpy as np

# Placeholder lists for different cases
tp_images, fp_images, tn_images, fn_images = [], [], [], []

# Loop through validation set and classify images
for i in range(len(X_val)):
    X_val_batch = np.expand_dims(X_val[i], axis=0)
    images_val_batch = np.expand_dims(images_val[i], axis=0)
    hog_val_batch = np.expand_dims(hog_val[i], axis=0)

    # Predict using the model
    pred_prob = model.predict([X_val_batch, images_val_batch, hog_val_batch])[0][0]
    pred_label = 1 if pred_prob > 0.5 else 0
    actual_label = y_val[i]

    # Compute CAM
    # heatmap = compute_cam(model, [X_val_batch, images_val_batch, hog_val_batch], pred_label)
    # heatmap_image = overlay_attention_map(heatmap, images_val[i])

    # Classify image
    if pred_label == 0 and actual_label == 0:
        tp_images.append(images_val[i])  # True Positive
    elif pred_label == 0 and actual_label == 1:
        fp_images.append(images_val[i])  # False Positive
    elif pred_label == 1 and actual_label == 1:
        tn_images.append(images_val[i])  # True Negative
    elif pred_label == 1 and actual_label == 0:
        fn_images.append(images_val[i])  # False Negative

    # Stop if we have at least one of each case
    if len(tp_images) > 0 and len(fp_images) > 0 and len(tn_images) > 0 and len(fn_images) > 0:
        break

# Select one image from each category
tp = tp_images[0] if tp_images else (np.zeros((224, 224, 3)))
fp= fp_images[0] if fp_images else (np.zeros((224, 224, 3)))
tn = tn_images[0] if tn_images else (np.zeros((224, 224, 3)))
fn = fn_images[0] if fn_images else (np.zeros((224, 224, 3)))

# Plot the 2x2 grid
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# TP
axes[0].imshow(tp)
axes[0].set_title("True Positive")
axes[0].axis("off")

# FP
axes[1].imshow(fp)
axes[1].set_title("False Positive")
axes[1].axis("off")

# TN
axes[2].imshow(tn)
axes[2].set_title("True Negative")
axes[2].axis("off")

# FN
# axes[3].imshow(fn)
# axes[3].set_title("False Negative")
# axes[3].axis("off")

plt.tight_layout()
plt.show()