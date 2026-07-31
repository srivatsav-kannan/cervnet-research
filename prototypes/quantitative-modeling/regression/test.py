import cv2
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
regression_model = tf.keras.models.load_model('image_to_stats_model.keras')
classification_model = tf.keras.models.load_model('saint_solo.keras')
def load_images(image_dir, data):
    labels = []
    images = []
    gt_stats = []
    dirs = [image_dir + '/CS', image_dir + '/Healthy']
    for idx, image_dir in enumerate(dirs):
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith('.png'):
                print(filename)
                # Decode filename to extract sequence number, gender, and age
                # print(filename)
                seq_number = int(filename[:4])  # First 4 digits: sequence number

                # Find the corresponding row in the dataset
                row = data[data['Number'] == seq_number]

                if row.empty:
                    print(f"No matching row found for {filename}")
                    continue

                # Drop the Disease Classification column to use all other columns as label
                label_row = row.drop(columns=['Disease classification: 1. Cervical spondylosis; 2. Healthy']).iloc[0]

                gt_stats.append(label_row.values)

                img_path = os.path.join(image_dir, filename)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (224, 224))

                labels.append(idx)
                images.append(image)

    return np.array(images)/255.0, np.array(labels), np.array(gt_stats)  # Normalize images

# File paths to the dataset and image directories
file_path = '/Users/srivatsavkannan/Datasets/C-Spine Xray/datasets.xlsx'
data = pd.read_excel(file_path, header=1).dropna()

train_image_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Aug'
val_image_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug'


# Load training and validation images and labels
X_train, y_train, gt_stats_train = load_images(train_image_dir, data)
predicted_stats_train = regression_model.predict(X_train)
print("Train Done..")

X_val, y_val, gt_stats_val = load_images(val_image_dir, data)
predicted_stats_val = regression_model.predict(X_val)

print(X_train.shape)
print(X_val.shape)
print('\n')
print(y_train.shape)
print(y_val.shape)
print('\n')
print(gt_stats_train.shape)
print(gt_stats_val.shape)
print('\n')
print(predicted_stats_train.shape)
print(predicted_stats_val.shape)

predicted_labels_from_stats = classification_model.predict(predicted_stats_val).flatten()
predicted_labels_from_stats = (predicted_labels_from_stats > 0.5).astype(int)

# 2. Using ground truth stats
predicted_labels_from_gt_stats = classification_model.predict(gt_stats_val).flatten()
predicted_labels_from_gt_stats = (predicted_labels_from_gt_stats > 0.5).astype(int)

# Calculate accuracy for both predictions
accuracy_from_stats = accuracy_score(y_val, predicted_labels_from_stats)
accuracy_from_gt_stats = accuracy_score(y_val, predicted_labels_from_gt_stats)

# Print results
print(f"Accuracy using predicted stats: {accuracy_from_stats}")
print(f"Accuracy using ground truth stats: {accuracy_from_gt_stats}")

print("Classification Report for label from predicted stats\n")
print((classification_report(y_val, predicted_labels_from_stats, digits=4)))

print("Classification Report for label from GT stats\n")
print((classification_report(y_val, predicted_labels_from_gt_stats, digits=4)))
