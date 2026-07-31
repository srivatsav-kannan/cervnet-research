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


# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE

def compute_grad_cam(model, image, layer_name='conv5_block3_3_conv'):
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(pooled_grads):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()  # Normalize

    return cam, predicted_class.numpy()

# Specify class names if known
class_names = ["Lordotic", "Straight", "Sigmoid1", "Sigmoid2", "Kyphotic"]

# Split dataset into training and testing sets
train_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/XRay_Atlas_Curve_aug/"
val_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/XRay_Atlas_Curve_aug/"

train_ds = image_dataset_from_directory(
    train_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

val_ds = image_dataset_from_directory(
    val_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

# Configure dataset for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
#
# #Get Pretrained ResNet50 model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

base_model.trainable = False


# Add custom head with global average pooling and dense layer
x = base_model.output

# First pair of Conv2D and Dropout layers
# x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
# x = tf.keras.layers.Dropout(rate=0.5)(x)
#
# # Second pair of Conv2D and Dropout layers
# x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
# x = tf.keras.layers.Dropout(rate=0.5)(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)


# Compile the model with adam optimizer and sparse_categorical_crossentropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Define model callback for early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, mode='max', restore_best_weights=True)

# Train the model for 20 epochs
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stop])
#
model.save('cervicalCurve3.h5')
model = tf.keras.models.load_model('cervicalCurve3.h5')

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

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#
plt.show()

# Evaluate the model

# Iterate through the test dataset and generate true and predicted predictions

y_true = []
y_pred = []
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(model.predict(images), axis=1))

# # Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate accuracy, precision, recall, and F1-score
class_names = ["Lordotic", "Straight", "Sigmoid1", "Sigmoid2", "Kyphotic"]

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



num = 0.0
ovnum = 0.0

for batch in val_ds:
    images, labels = batch  # Unpack the tuple
    predictions = model.predict(images)

    for i in range(len(images)):
        image = images[i].numpy().astype(np.int32)
        label = labels[i]
        prediction = predictions[i]

        # Decode one-hot encoded labels
        true_label = label
        predicted_label = np.argmax(prediction)
        print(label, predicted_label)
        image_input = tf.expand_dims(images[i], axis=0)
        cam, _ = compute_grad_cam(model, image_input)

        cam_resized = tf.image.resize(cam[..., tf.newaxis], IMAGE_SIZE,
                                      method=tf.image.ResizeMethod.BILINEAR).numpy()
        cam_resized = cam_resized.squeeze()

        fig, axs = plt.subplots(1, 3, figsize=(12, 6))

        # Display the original image
        axs[0].imshow(image)
        axs[0].set_title(f"Ground Truth: {true_label}, Prediction: {predicted_label}")
        axs[0].axis('off')

        # Display the attention map over the image

        # cam_resized = tf.image.resize(cam, (image.shape[0], image.shape[1])).numpy()
        axs[1].imshow(cam_resized, cmap='jet', alpha=0.5)
        axs[1].set_title("Attention Map")
        axs[1].axis('off')

        threshold = 0.5  # You can adjust this threshold
        mask = cam_resized > threshold
        high_attention_image = np.zeros_like(image)
        high_attention_image[mask] = image[mask]

        axs[2].imshow(high_attention_image)
        axs[2].set_title("High Attention Areas")
        axs[2].axis('off')

        plt.show()

        if label == predicted_label:
            num += 1.0
        ovnum += 1.0
# Classification Report:
#               precision    recall  f1-score   support
#
#            0     0.7613    0.8662    0.8104       523
#            1     0.6390    0.8103    0.7145       522
#            2     0.3488    0.1546    0.2143        97
#            3     0.4615    0.1846    0.2637       195
#            4     0.6273    0.4570    0.5287       151
#
#     accuracy                         0.6694      1488
#    macro avg     0.5676    0.4945    0.5063      1488
# weighted avg     0.6386    0.6694    0.6377      1488

# When augmented
# Accuracy: 0.749902761571373
# Precision: 0.7488701237107818
# Recall: 0.749902761571373
# F1 Score: 0.7470095839069767
# Classification Report:
#               precision    recall  f1-score   support
#
#            0     0.8526    0.8891    0.8705       514
#            1     0.8831    0.8781    0.8806       525
#            2     0.6693    0.5195    0.5850       487
#            3     0.6484    0.7184    0.6816       593
#            4     0.6925    0.7323    0.7118       452
#
#     accuracy                         0.7499      2571
#    macro avg     0.7492    0.7475    0.7459      2571
# weighted avg     0.7489    0.7499    0.7470      2571