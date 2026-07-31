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
train_dir = "/Users/srivatsavkannan/Datasets/CXRA/train"
val_dir = "/Users/srivatsavkannan/Datasets/CXRA/val"


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2
AUTOTUNE = tf.data.experimental.AUTOTUNE
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

# model = tf.keras.models.load_model('model2.keras')
# print(model.summary())
model = tf.keras.models.load_model('documenting/cs23/cs23.keras')
print(model.summary())

correct_images = []
incorrect_images = []
correct_labels = []
incorrect_labels = []
predictions = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds_class = np.argmax(preds, axis=1)

    for i in range(len(images)):
        if preds_class[i] == labels[i].numpy():
            correct_images.append(images[i].numpy())
            correct_labels.append(labels[i].numpy())
        else:
            incorrect_images.append(images[i].numpy())
            incorrect_labels.append(labels[i].numpy())
        predictions.append(preds_class[i])


# Define a set number of images to repeat cyclically
cs_correct_images = [img for img, lbl in zip(correct_images, correct_labels) if lbl == 0]
healthy_correct_images = [img for img, lbl in zip(correct_images, correct_labels) if lbl == 1]
cs_incorrect_images = [img for img, lbl in zip(incorrect_images, incorrect_labels) if lbl == 0]
healthy_incorrect_images = [img for img, lbl in zip(incorrect_images, incorrect_labels) if lbl == 1]

# Helper to get a cyclic list
def get_cyclic_images(image_list, count):
    return [image_list[i % len(image_list)] for i in range(count)] if image_list else [None] * count

# Set the number of images for each type (repeat cyclically if fewer available)
num_images_per_type = 1  # Number of images per type (can be adjusted)
cs_correct = get_cyclic_images(cs_correct_images, num_images_per_type)
healthy_correct = get_cyclic_images(healthy_correct_images, num_images_per_type)
cs_incorrect = get_cyclic_images(cs_incorrect_images, num_images_per_type)
healthy_incorrect = get_cyclic_images(healthy_incorrect_images, num_images_per_type)

# Combine all information into a single cyclic list
image_types = [
    (cs_correct[i], 0, f"CS (Correctly Predicted) {i+1}", 0) for i in range(num_images_per_type)
] + [
    (healthy_correct[i], 1, f"Healthy (Correctly Predicted) {i+1}", 1) for i in range(num_images_per_type)
] + [
    (cs_incorrect[i], 0, f"CS (Incorrectly Predicted) {i+1}", 1) for i in range(num_images_per_type)
] + [
    (healthy_incorrect[i], 1, f"Healthy (Incorrectly Predicted) {i+1}", 0) for i in range(num_images_per_type)
]

# Compute Vanilla Saliency Map
def compute_vanilla_saliency(model, image, label_index):
    image = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = predictions[0, label_index]

    gradient = tape.gradient(loss, image)[0]
    saliency = tf.reduce_max(tf.abs(gradient), axis=-1)
    return saliency.numpy()

# Compute Grad-CAM heatmap
def compute_grad_cam(model, image, label_index):
    # Get the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        print("Grad-CAM is not possible as there are no convolutional layers in the model.")
        return None

    grad_cam_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    image = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_cam_model(image)
        loss = predictions[0, label_index]

    gradients = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(gradients, axis=(0, 1))
    grad_cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
    grad_cam = tf.nn.relu(grad_cam).numpy()
    return grad_cam

# Compute CAM
def compute_cam(model, image, label_index):
    # Check for Global Average Pooling (GAP) layer
    gap_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
            break
    if gap_layer is None:
        print("CAM is not possible as there is no GAP layer in the model.")
        return None

    # Extract weights and features
    feature_extractor = tf.keras.models.Model(
        inputs=model.input,
        outputs=gap_layer.input  # Get features before GAP
    )
    features = feature_extractor.predict(image[np.newaxis, ...])[0]
    label_index = 0
    weights = model.layers[-1].weights[0][:, label_index]  # Class weights from final dense layer

    cam = tf.reduce_sum(weights * features, axis=-1).numpy()
    cam = tf.nn.relu(cam)
    return cam

# Overlay attention map on image
def overlay_attention_map(attention_map, image, alpha=0.4, colormap="viridis"):
    if attention_map is None:
        return None  # Skip if attention map could not be computed

    attention_map_resized = tf.image.resize(
        attention_map[..., tf.newaxis],
        (image.shape[0], image.shape[1])
    ).numpy()
    attention_map_resized = attention_map_resized.squeeze()
    attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / (
        np.max(attention_map_resized) - np.min(attention_map_resized) + 1e-8
    )

    colormap = plt.cm.get_cmap(colormap)
    heatmap_colored = colormap(attention_map_resized)[..., :3]

    image_normalized = image.astype("float32") / 255.0
    superimposed_image = (1 - alpha) * image_normalized + alpha * heatmap_colored
    return np.clip(superimposed_image, 0, 1)

# Iterate through image types and plot attention maps
# Iterate cyclically over image types
fig, axes = plt.subplots(len(image_types), 2, figsize=(20, len(image_types) * 4))

for idx, (image, true_label, description, predicted_label) in enumerate(image_types):
    print(f"\n{description}:")
    print(f"True Label: {class_names[true_label]}, Predicted Label: {class_names[predicted_label]}")

    if image is None:
        print(f"Skipping {description} because no image is available.")
        continue

    # Compute attention maps
    # vanilla_saliency = compute_vanilla_saliency(model, image, true_label)
    grad_cam = compute_grad_cam(model, image, true_label)
    cam = compute_cam(model, image, true_label)

    # Overlay maps
    # vanilla_overlay = overlay_attention_map(vanilla_saliency, image)
    grad_cam_overlay = overlay_attention_map(grad_cam, image)
    cam_overlay = overlay_attention_map(cam, image)

    # Plot original image
    axes[idx, 0].imshow(image.astype("uint8"))
    axes[idx, 0].set_title(f"True: {class_names[true_label]}, Pred: {class_names[predicted_label]}")
    axes[idx, 0].axis("off")

    # # Plot Vanilla Saliency Map
    # if vanilla_overlay is not None:
    #     axes[idx, 1].imshow(vanilla_overlay)
    #     axes[idx, 1].set_title("Vanilla Saliency Map Overlay")
    # else:
    #     axes[idx, 1].text(0.5, 0.5, "Not Applicable", ha="center", va="center")
    # axes[idx, 1].axis("off")

    # Plot CAM
    if cam_overlay is not None:
        axes[idx, 1].imshow(cam_overlay)
        axes[idx, 1].set_title("CAM Overlay")
    else:
        axes[idx, 1].text(0.5, 0.5, "Not Applicable", ha="center", va="center")
    axes[idx, 1].axis("off")

plt.tight_layout()
plt.show()
