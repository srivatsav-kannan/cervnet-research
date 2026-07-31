import os
import sys

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.resnet50 import preprocess_input

# Specify class names if known
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Split dataset into training and testing sets

# train_dir = "/Users/srivatsavkannan/Datasets/Experiment/Train_Org_ROS_RUS"
# val_dir = "/Users/srivatsavkannan/Datasets/Experiment/Val_Org_ROS_RUS"

train_dir = "/Users/srivatsavkannan/Datasets/CervicalNew/TrainCropped"
val_dir = "/Users/srivatsavkannan/Datasets/CervicalNew/ValCropped"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = ["CS", "Healthy"]

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

# Configure dataset for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.load_model('model.keras')
model.summary()

# Generate predictions from the model
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images).flatten()  # Get predictions as a 1D array
    binary_predictions = (predictions > 0.5).astype(int)  # Convert to binary values (threshold at 0.5)
    print("True Labels:", labels.numpy())
    print("Predicted Probabilities:", predictions)
    print("Binary Predictions:", binary_predictions)
    y_true.extend(labels.numpy())
    y_pred.extend(binary_predictions)


# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

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