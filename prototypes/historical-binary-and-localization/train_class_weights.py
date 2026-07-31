import os
import sys
from collections import Counter

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.efficientnet import preprocess_input

# Define constants
checkpoint_dir = "documenting/cs27/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Class names
class_names = ["CS", "Healthy"]

# Directories
train_dir = "/Users/srivatsavkannan/Datasets/Experiment/Train_Org_ROS_RUS/"
val_dir = "/Users/srivatsavkannan/Datasets/Experiment/Val_Org_ROS_RUS/"

# Image properties
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Load dataset
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

# Compute class weights
train_labels = np.concatenate([labels.numpy() for _, labels in train_ds])
class_counts = Counter(train_labels)
total_samples = sum(class_counts.values())

class_weights = {
    label: total_samples / (NUM_CLASSES * count)
    for label, count in class_counts.items()
}
print(type(class_weights))
print(f"Class Weights: {class_weights}")
class_weights = {0: 0.25, 1:1.75}
print(f"Class Weights: {class_weights}")
# sys.exit(0)
# Configure dataset for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Model Architecture
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB7(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)
base_model.trainable = False  # Freeze the base model

inputs = tf.keras.Input(shape=input_shape)

# Pass inputs through the base model
x = base_model(inputs, training=False)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Compile model with class weights
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=100, mode='max', restore_best_weights=True
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras"),
    save_freq='epoch',
    save_weights_only=False,
    verbose=1
)

# Train model
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,  # Apply class weights
    callbacks=[early_stop]
)

# Save final model
model.save('model.keras')
# Training and Validation Plots
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, 'b', label='Training Loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

model.save(os.path.join(checkpoint_dir, "transl_eff_class_weights.keras"))

# Evaluation
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = (model.predict(images).flatten() > 0.5)
    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Classification Report
report = classification_report(y_true, y_pred, digits=4)

# Print Metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Sensitivity (Recall) & Specificity
sensitivity = recall_score(y_true, y_pred, average=None)
specificity = []
for i in range(len(class_names)):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp))

print("Sensitivity (Recall) per class:", sensitivity)
print("Specificity per class:", specificity)

# ROC Curve
y_true_bin = label_binarize(y_true, classes=[0, 1])
y_pred_bin = label_binarize(y_pred, classes=[0, 1])

fpr, tpr, _ = roc_curve(y_true_bin[:, 0], y_pred_bin[:, 0])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# SVM Feature Extraction
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def extract_features(dataset, model):
    feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    all_features = []
    all_labels = []

    for images, labels in dataset:
        features = feature_extractor.predict(preprocess_input(images))
        all_features.append(features)
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = extract_features(train_ds, model)
val_features, val_labels = extract_features(val_ds, model)

svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(train_features, train_labels)

train_accuracy = accuracy_score(train_labels, svm_classifier.predict(train_features))
val_accuracy = accuracy_score(val_labels, svm_classifier.predict(val_features))

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Save SVM Model
joblib.dump(svm_classifier, os.path.join(checkpoint_dir, 'svm_model27.pkl'))
