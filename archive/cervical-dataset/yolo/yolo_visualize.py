import sys

import matplotlib.pyplot as plt
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

train_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized"
val_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized"
train_cropped_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized12/Train3"
val_cropped_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized12/Val3"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = ["normal"]

train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

model = tf.keras.models.load_model('./checkpointsk/v1/model_epoch_17.keras')

for batch_idx, batch in enumerate(train_ds):
    images, labels = batch
    predicted_bboxes, predicted_labels_batch = model.predict(images)
    predicted_labels_batch = (predicted_labels_batch > 0.5).astype(int).flatten()
    print(labels)
    for i in range(images.shape[0]):
        # image = images[i].numpy() / 255.0  # Normalize for visualization
        image = images[i].numpy().astype(np.uint8)
        print(image)
        actual_label = labels[i]
        predicted_bbox = predicted_bboxes[i]
        predicted_label = predicted_labels_batch[i]
        img_with_boxes = Image.fromarray(image)
        draw = ImageDraw.Draw(img_with_boxes)

        print(predicted_bbox)
        min_x = int(max(0, predicted_bbox[0]))
        min_y = int(max(0, predicted_bbox[1]))
        max_x = int(min(224, predicted_bbox[2]))
        max_y = int(min(224, predicted_bbox[3]))

        # Crop the image using the predicted bounding box
        print("Predicted Bounding Box: ", predicted_bbox)

        draw.rectangle(
            [predicted_bbox[0], predicted_bbox[1], predicted_bbox[2], predicted_bbox[3]], outline="red", width=3
        )

        plt.imshow(img_with_boxes)
        plt.title(predicted_label)
        plt.show()

