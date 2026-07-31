import os
import sys

import skimage
import tensorflow as tf
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG19, EfficientNetB7
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = ["CS", "Healthy"]

# Directories
train_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Cropped_SMOTE/"
val_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Cropped_SMOTE/"

# Load datasets (images only)
train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# HOG Feature Extraction
def extract_hog_features(image):
    """Extract HOG features from a single image."""
    # image = skimage.transform.resize(image.numpy(), (64, 128), anti_aliasing=True)
    gray_img = rgb2gray(image.numpy())
    features = hog(
        gray_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    print('Shape: ', features.shape)

    return features


# Function to compute HOG features and return a dataset with both images and HOG features
def create_hog_dataset(dataset):
    """Compute HOG features for the dataset and return a combined dataset."""
    hog_features = []
    image_data = []
    labels = []

    # Loop through each batch in the dataset
    for images, batch_labels in dataset:
        for img, label in zip(images.numpy(), batch_labels.numpy()):
            img = tf.convert_to_tensor(img)  # Convert back to TensorFlow tensor if necessary
            features = extract_hog_features(img)
            hog_features.append(features)
            image_data.append(img)
            labels.append(label)

    hog_features = np.array(hog_features)
    image_data = np.array(image_data)
    labels = np.array(labels)
    print(image_data.shape)
    print(hog_features.shape)
    print(labels.shape)
    # Return dataset as a tuple of image, HOG features, and labels
    return image_data, hog_features, labels


# Create HOG dataset for training and validation
train_images, train_hog_features, train_labels = create_hog_dataset(train_ds)
val_images, val_hog_features, val_labels = create_hog_dataset(val_ds)

# Function to create the model
def create_model(input_shape_cnn, input_shape_hog, num_classes):
    """Create the model with CNN and HOG feature extraction branches."""

    # Define CNN branch (using VGG19 pre-trained weights)
    image_input = layers.Input(shape=input_shape_cnn, name="image_input")
    eff = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape_cnn)
    eff.trainable = False

    cnn_input = eff(image_input, training=False)
    cnn_flatten = layers.GlobalAveragePooling2D()(cnn_input)
    cnn_dense = layers.Dense(256, activation="relu", name="CNN_Dense1")(cnn_flatten)
    cnn_output = layers.Dense(128, activation="relu", name="CNN_Dense2")(cnn_dense)

    # Define HOG input branch with the correct shape (26244 features)
    hog_input = layers.Input(shape=input_shape_hog, name="HOG_Input")
    hog_dense = layers.Dense(256, activation="relu", name="HOG_Dense1")(hog_input)
    hog_output = layers.Dense(128, activation="relu", name="HOG_Dense2")(hog_dense)

    # Combine HOG and CNN branches
    combined = layers.Concatenate(name="Concatenated_Features")([cnn_output, hog_output])
    combined_dense = layers.Dense(128, activation="relu", name="Combined_Dense")(combined)
    final_output = layers.Dense(num_classes, activation="softmax", name="Output_Layer")(combined_dense)

    # Define and compile the model
    model = models.Model(inputs=[image_input, hog_input], outputs=final_output, name="HOG_CNN_Model")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

# Define input shapes
input_shape_cnn = (224, 224, 3)  # Image shape for CNN
input_shape_hog = (train_hog_features.shape[1],)  # HOG features shape (26244,)

# Create the model
model = create_model(input_shape_cnn, input_shape_hog, NUM_CLASSES)

# Model summary
model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, mode='max',
                                              restore_best_weights=True)
# Train the model

with tf.device('/cpu:0'):
   train_images = tf.convert_to_tensor(train_images, np.float32)
   train_hog_features = tf.convert_to_tensor(train_hog_features, np.float32)
   train_labels = tf.convert_to_tensor(train_labels, np.float32)
   val_images = tf.convert_to_tensor(val_images, np.float32)
   val_hog_features = tf.convert_to_tensor(val_hog_features, np.float32)
   val_labels = tf.convert_to_tensor(val_labels, np.float32)

history = model.fit(
    [train_images, train_hog_features], train_labels,
    epochs=EPOCHS,
    validation_data=([val_images, val_hog_features], val_labels),
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)
# Evaluate the model
model.save("HOG_CNN_Model_smote_eff.h5")

model = tf.keras.models.load_model("HOG_CNN_Model_smote_eff.h5")
print(val_images[0].shape)

eval_results = model.evaluate(x=[val_images, val_hog_features], y=val_labels)
print(f"Validation Loss: {eval_results[0]}, Validation Accuracy: {eval_results[1]}")

#Plot training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), train_loss, 'b', label='Training Loss')
plt.plot(range(EPOCHS), val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), train_acc, 'b', label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
