import os
import sys
from collections import Counter

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

checkpoint_dir = "checkpoints_train/"

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_labels = [], []

        # Iterate through the validation dataset
        for images, labels in self.validation_data:
            val_images.append(images.numpy())
            val_labels.append(labels.numpy())

        val_images = np.concatenate(val_images)
        val_labels = np.concatenate(val_labels)

        # Get predictions
        predictions = self.model.predict(val_images)
        predicted_labels = np.argmax(predictions, axis=1)

        # Compute F1 score for the minority class (class 1)
        f1 = f1_score(val_labels, predicted_labels, pos_label=1)
        print(f"\nEpoch {epoch + 1}: F1 Score (Class 1): {f1:.4f}")

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
# train_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized12/Train4"
# val_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized12/Val4"
#
# train_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Cropped_Aug"
# val_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Cropped_Aug"


train_dir = "/Users/srivatsavkannan/Datasets/CXRA/train"
val_dir = "/Users/srivatsavkannan/Datasets/CXRA/val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 2
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
# class_names = ["CS", "Healthy"]
class_names = ["PNEUMONIA", "NORMAL"]

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

# for images,labels in train_ds:
#     print(images.shape)
#     print(images.dtype)
#     print(images[0])
#     plt.imshow(images[0])
#     plt.show()
#     sys.exit(0)


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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, mode='max',
                                              restore_best_weights=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras"),  # Save model as 'model_epoch_XX.h5'
    save_freq='epoch',  # Save every epoch
    save_weights_only=False,  # Save the entire model (not just weights)
    verbose=1  # Print a message when saving
)

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stop])

model.save('model2.keras')

y_true = []
y_pred = []

for images, labels in val_ds:
    a = (model.predict(images).flatten() > 0.5)
    print(labels.numpy())
    print(a)
    y_true.extend(labels.numpy())
    y_pred.extend(a)

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
#
#
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#
# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()




model = tf.keras.models.load_model('documenting/cs25/ssd_cropped_transl_eff.keras')


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_features, train_labels = extract_features(train_ds, model)
val_features, val_labels = extract_features(val_ds, model)

# Train an SVM on the extracted features
svm_classifier = SVC(kernel='linear', C=1)
print(train_features.shape)
svm_classifier.fit(train_features, train_labels)

# Make predictions on the validation data
val_predictions = svm_classifier.predict(val_features)
train_predictions = svm_classifier.predict(train_features)
# Evaluate the SVM classifier
accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation Accuracy: {accuracy:.4f}')

accuracy = accuracy_score(train_labels, train_predictions)
print(f'Training Accuracy: {accuracy:.4f}')
import joblib
#
# # # Save the model
joblib.dump(svm_classifier, 'documenting/cs25/svm_model25.pkl')



