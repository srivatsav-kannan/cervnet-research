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

# Define constants


def extract_features(dataset, model):
    feature_extractor = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    all_features = []
    all_labels = []

    for images, labels in dataset:
        # Preprocess images if required (ResNet50 uses preprocess_input)
        # print(images.shape)
        # print(images[0])
        preprocessed_images = preprocess_input(images)
        # print(preprocessed_images[0])
        # plt.imshow(preprocessed_images[0])
        # plt.show()
        features = feature_extractor.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# Specify class names if known
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# Split dataset into training and testing sets
train_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized12/train"
val_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized12/val"

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

# Define the directories for the new folders
# output_base_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized"
# train_output_dir = os.path.join(output_base_dir, "train")
# val_output_dir = os.path.join(output_base_dir, "val")
#
# # Ensure output directories exist
# os.makedirs(train_output_dir, exist_ok=True)
# os.makedirs(val_output_dir, exist_ok=True)
#
# # Create subdirectories for each class in training and validation folders
# for class_name in class_names:
#     os.makedirs(os.path.join(train_output_dir, class_name), exist_ok=True)
#     os.makedirs(os.path.join(val_output_dir, class_name), exist_ok=True)
#
#
# # Function to copy images to their respective folders
# def organize_images(dataset, output_dir):
#     for images, labels in dataset:
#         for i in range(images.shape[0]):
#             # Get the image and its corresponding class name
#             img = images[i].numpy().astype("uint8")
#             label = labels[i].numpy()
#             class_name = class_names[label]
#
#             # Save the image to the appropriate class folder
#             class_folder = os.path.join(output_dir, class_name)
#             img_name = f"{len(os.listdir(class_folder)) + 1}.png"
#             img_path = os.path.join(class_folder, img_name)
#
#             # Save the image
#             tf.keras.preprocessing.image.save_img(img_path, img)
#
#
# # Organize training and validation datasets
# organize_images(train_ds, train_output_dir)
# organize_images(val_ds, val_output_dir)
#
# print("Images organized into train and validation folders successfully.")
#
# sys.exit(0)

# Configure dataset for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.load_model('documenting/cs20/cs20.h5')
model.summary()


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib

svm_classifier = joblib.load('documenting/cs20/svm_model20.pkl')
val_features, val_labels = extract_features(val_ds, model)
# train_features, train_labels = extract_features(train_ds, model)
print(val_features.shape)
val_predictions = svm_classifier.predict(val_features)

y_true = val_labels
y_pred = val_predictions
# for images, labels in val_ds:
#     y_true.extend(labels.numpy())
#     y_pred.extend(np.argmax(model.predict(images), axis=1))

# # Convert lists to numpy arrays
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

