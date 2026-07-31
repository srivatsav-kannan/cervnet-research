import os
import sys

import tensorflow as tf
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG19, EfficientNetB7
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = ["CS", "Healthy"]

# Directories
train_dir = "/Users/srivatsavkannan/Datasets/EnhancedCervicalDataset/Train_Org_Cropped"
val_dir = "/Users/srivatsavkannan/Datasets/EnhancedCervicalDataset/Val_Org_Cropped"

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
    gray_img = rgb2gray(image.numpy())
    features = hog(
        gray_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
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

# Define input shapes
input_shape_cnn = (224, 224, 3)  # Image shape for CNN
input_shape_hog = (train_hog_features.shape[1],)  # HOG features shape (26244)

model = tf.keras.models.load_model("HOG_CNN_Model_smote_eff_clahe.h5")

with tf.device('/cpu:0'):
   train_images = tf.convert_to_tensor(train_images, np.float32)
   train_hog_features = tf.convert_to_tensor(train_hog_features, np.float32)
   train_labels = tf.convert_to_tensor(train_labels, np.float32)
   val_images = tf.convert_to_tensor(val_images, np.float32)
   val_hog_features = tf.convert_to_tensor(val_hog_features, np.float32)
   val_labels = tf.convert_to_tensor(val_labels, np.float32)

# eval_results = model.evaluate(x=[val_images, val_hog_features], y=val_labels)
# print(f"Validation Loss: {eval_results[0]}, Validation Accuracy: {eval_results[1]}")

# Generate predictions from the model
y_true = []
y_pred = []

prediction = model.predict([val_images, val_hog_features])
print(prediction.shape)
prediction = np.argmax(prediction, axis=-1)
print(prediction.shape)

# Convert lists to numpy arrays
y_true = val_labels
y_pred = prediction

correct_preds = []
incorrect_preds = []
tp = []
fp = []
tn = []
fn = []

# Calculate accuracy, precision, recall, and F1-score
class_names = ["CS", "Healthy"]

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Generate classification report with 4 significant figures
report = classification_report(y_true, y_pred, digits=4)

# Output metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:")
print(report)

# Create and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate sensitivity and specificity
# Sensitivity (Recall) for each class
sensitivity = recall_score(y_true, y_pred, average=None)
# Specificity for each class
specificity = []
for i in range(len(class_names)):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp))

print("Sensitivity (Recall) for each class:", sensitivity)
print("Specificity for each class:", specificity)

# AUC and ROC Curve (one-vs-all for multi-class)
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3, 4])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'black']
for i, color in zip(range(len(class_names)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
