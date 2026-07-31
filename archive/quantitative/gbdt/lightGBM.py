import cv2
import os
import sys
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np



old = []
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

    return np.array(gt_stats), np.array(labels), np.array(images)/255.0  # Normalize images

# File paths to the dataset and image directories
file_path = '/Users/srivatsavkannan/Datasets/C-Spine Xray/datasets.xlsx'
data = pd.read_excel(file_path, header=1).dropna()

train_image_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Aug'
val_image_dir = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug'

# Load training and validation images and labels
X_train, y_train, images_train = load_images(train_image_dir, data)
X_val, y_val, images_val = load_images(val_image_dir, data)

print(X_train.shape)
print(X_val.shape)
print(images_train.shape)

print(y_train.shape)
print(y_val.shape)
print(images_val.shape)

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Build the model
input_dim = X_train.shape[1]
image_input_dim = (224,224,3)
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42)
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")