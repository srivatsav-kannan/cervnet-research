import sys

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from PIL import Image, ImageDraw
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping




checkpoint_dir = "checkpoints_unet/"
radius = 5

RESIZE_WIDTH, RESIZE_HEIGHT = 224, 224  # Desired dimensions
BATCH_SIZE = 16

# Create a black image
# Define data loading functions
def load_data(img_dir, json_dir):
    other_img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
    images, labels, class_labels = [], [], []
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
                img_size = Image.open(os.path.join(other_img_dir, filename))
                original_width, original_height = img_size.size
                # img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
                #
                # heatmap = Image.new("RGB", (224, 224), color=(0, 0, 0))
                # draw = ImageDraw.Draw(heatmap)

                images.append(tf.keras.preprocessing.image.img_to_array(img))

                # Load and process JSON data
                with open(json_path, 'r') as f:
                    data = json.load(f)

                points = []
                for shape in data['shapes']:
                    x = (shape['points'][0][0] / original_width) * WIDTH
                    y = (shape['points'][0][1] / original_height) * HEIGHT
                    # draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255,255,255))
                    print([x,y])
                    points.extend([x, y])

                if len(points) == 46:  # Ensure consistency
                    labels.append(points)

    return np.array(images), np.array(labels), np.array(class_labels)



# Convert keypoint labels to heatmaps
def convert_to_heatmaps(labels, img_size=(224, 224)):
    heatmaps = []
    for label in labels:
        heatmap = np.zeros(img_size, dtype=np.float32)
        for i in range(0, len(label), 2):
            x, y = int(label[i]), int(label[i + 1])
            print(x,y)
            if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
                heatmap[y,x] = 1.0
                # plt.imshow(heatmap)
                # plt.show()
        # plt.imshow(heatmap)
        # plt.show()
        heatmap = tf.keras.layers.GaussianNoise(stddev=1.0)(tf.convert_to_tensor(heatmap[np.newaxis, ..., np.newaxis]))
        heatmap = heatmap[0, ..., 0].numpy()
        heatmap /= np.sum(heatmap)  # Normalize
        # plt.imshow(heatmap)
        # plt.show()
        heatmaps.append(heatmap)

    return np.array(heatmaps)


# Load and preprocess datasets
img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG-aug-train/'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'

X_train, keypoint_labels_train, train_class_labels = load_data(img_dir, json_dir)
y_train_heatmaps = convert_to_heatmaps(keypoint_labels_train, img_size=(224, 224))

img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG-aug-test/'

X_val, keypoint_labels_val, val_class_labels = load_data(img_dir, json_dir)
y_val_heatmaps = convert_to_heatmaps(keypoint_labels_val, img_size=(224, 224))


# Print shapes of training data
print("X_train shape:", X_train.shape)
print("keypoint_labels_train shape:", keypoint_labels_train.shape)
print("train_class_labels shape:", train_class_labels.shape)
print("y_train_heatmaps shape:", y_train_heatmaps.shape)

# Print shapes of validation data
print("X_val shape:", X_val.shape)
print("keypoint_labels_val shape:", keypoint_labels_val.shape)
print("val_class_labels shape:", val_class_labels.shape)
print("y_val_heatmaps shape:", y_val_heatmaps.shape)

print("Data Types:")
print("X_train data type:", X_train.dtype)
print("keypoint_labels_train data type:", keypoint_labels_train.dtype)
print("train_class_labels data type:", train_class_labels.dtype)
print("y_train_heatmaps data type:", y_train_heatmaps.dtype)
print("X_val data type:", X_val.dtype)
print("keypoint_labels_val data type:", keypoint_labels_val.dtype)
print("val_class_labels data type:", val_class_labels.dtype)
print("y_val_heatmaps data type:", y_val_heatmaps.dtype)

keypoint_labels_train = keypoint_labels_train.astype(np.float32)
keypoint_labels_val = keypoint_labels_val.astype(np.float32)

train_class_labels = train_class_labels.astype(np.int32)
val_class_labels = val_class_labels.astype(np.int32)

print("Data Types After Fixing:")
print("X_train data type:", X_train.dtype)
print("keypoint_labels_train data type:", keypoint_labels_train.dtype)
print("train_class_labels data type:", train_class_labels.dtype)
print("y_train_heatmaps data type:", y_train_heatmaps.dtype)
print("X_val data type:", X_val.dtype)
print("keypoint_labels_val data type:", keypoint_labels_val.dtype)
print("val_class_labels data type:", val_class_labels.dtype)
print("y_val_heatmaps data type:", y_val_heatmaps.dtype)

# # Split into train and validation sets
# X_train, X_val, y_train_heatmaps, y_val_heatmaps = train_test_split(X, heatmaps, test_size=0.2, random_state=42)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# # Extract classification labels from dataset
# def extract_labels(dataset):
#     labels = []
#     cnt = 0
#     for _, label_batch in dataset:
#         labels.extend(label_batch.numpy())
#         cnt += 1
#         print(label_batch.numpy().shape)
#     print(cnt)
#     return np.array(labels)
#
#
# train_class_labels = extract_labels(train_class_ds)
# val_class_labels = extract_labels(val_class_ds)

print(np.array(X_train).shape)
print(np.array(y_train_heatmaps).shape)
print(np.array(train_class_labels).shape)
# Combine images, heatmaps, and classification labels into unified datasets
with tf.device('/cpu:0'):
    train_combined = tf.data.Dataset.from_tensor_slices((
        X_train, {"saliency": y_train_heatmaps, "classification": train_class_labels}
    )).batch(BATCH_SIZE)

    val_combined = tf.data.Dataset.from_tensor_slices((
        X_val, {"saliency": y_val_heatmaps, "classification": val_class_labels}
    )).batch(BATCH_SIZE)


# Define MT-UNet model
def build_mt_unet(input_shape=(224, 224, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)

    # Output heads
    saliency_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='saliency')(c9)

    global_pool = layers.GlobalAveragePooling2D()(c5)
    dense1 = layers.Dense(128, activation='relu')(global_pool)
    dense1 = layers.Dropout(0.25)(dense1)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification')(dense1)

    model = models.Model(inputs=inputs, outputs=[saliency_output, classification_output])
    return model

# Compile model
# Set up ModelCheckpoint to save the best model based on validation accuracy
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "model_epoch2_{epoch:02d}.keras"),  # Save model as 'model_epoch_XX.h5'
    save_freq='epoch',  # Save every epoch
    save_weights_only=False,  # Save the entire model (not just weights)
    verbose=1  # Print a message when saving
)

# Set up EarlyStopping with a patience of 100 epochs
early_stopping = EarlyStopping(
    monitor='val_classification_accuracy', # Metric to monitor
    mode='max',                            # We want to maximize the metric
    patience=100,                          # Number of epochs with no improvement before stopping
    verbose=1,                             # Show output when stopping early
    restore_best_weights=True             # Restore the best weights when stopping early
)

# model = build_mt_unet()
model = tf.keras.models.load_model(os.path.join(checkpoint_dir, "model_epoch_12.keras"))
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss={
        'saliency': 'kld',  # Kullback-Leibler Divergence for regression
        'classification': 'binary_crossentropy'  # Sparse categorical cross-entropy for classification
    },
    metrics={
        'saliency': ['accuracy'],  # Accuracy on saliency map (optional metric)
        'classification': ['accuracy']  # Classification accuracy
    }
)

# Model summary
model.summary()

# Training the model
history = model.fit(
    train_combined,
    validation_data=val_combined,
    epochs=20,
    callbacks=[checkpoint_callback],
)


model.save('unet.keras')
