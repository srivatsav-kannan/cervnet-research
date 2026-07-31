import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Directories
img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
output_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-boundingboxes/'

# Parameters
IMG_SIZE = (224, 224)  # Resizing all images to 224x224 for ResNet50
WIDTH = IMG_SIZE[0]
HEIGHT = IMG_SIZE[1]
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001


# Function to load data and labels
def load_data(img_dir, json_dir):
    images = []
    labels = []

    for filename in os.listdir(img_dir):
        if filename.endswith('.png'):
            # Load image
            img_path = os.path.join(img_dir, filename)
            json_path = os.path.join(json_dir, filename.replace('.png', '.json'))

            if os.path.exists(json_path):
                # Open the image
                img = Image.open(img_path).convert("RGB")
                original_width, original_height = img.size

                # Resize the image
                img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
                images.append(img_to_array(img))
                with open(json_path, 'r') as f:
                    data = json.load(f)

                    # List to store points for bounding box
                points = []

                for shape in data['shapes']:
                    point = shape['points'][0]  # Get the point (x, y)

                    # Adjust the point coordinates according to the resized image
                    x = (point[0] / original_width) * WIDTH
                    y = (point[1] / original_height) * HEIGHT
                    adjusted_point = (x, y)

                    # Add the adjusted point to the list for bounding box calculation
                    points.append(adjusted_point)

                if points:
                    min_x = min(p[0] for p in points)
                    max_x = max(p[0] for p in points)
                    min_y = min(p[1] for p in points)
                    max_y = max(p[1] for p in points)
                    labels.append([min_x, min_y, max_x, max_y])

    return np.array(images), np.array(labels)


# Load data
X, y = load_data(img_dir, json_dir)
print("Data Loading Completed")
# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train Test Split Completed")


# Build the base model (ResNet50 + additional layers)
def build_base_model():
    # Load ResNet50 with ImageNet weights, exclude top layers
    base_resnet = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Add custom convolutional layers and global average pooling
    x = base_resnet.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    base_model = Model(inputs=base_resnet.input, outputs=x, name="resnet50_base")
    return base_model


# Build the complete regression model
def build_regression_model():
    base_model = build_base_model()

    # Add the regression head
    x = base_model.output
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    bbox_output = Dense(4, activation='linear', name="bbox_output")(x)  # 4 outputs for bounding box coordinates

    model = Model(inputs=base_model.input, outputs=bbox_output, name="bounding_box_regressor")
    return model


# Build and compile the model
model = build_regression_model()
print("Model Built")
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])

checkpoint_dir = "../unet/checkpoints_unet/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Define the ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras"),  # Save model as 'model_epoch_XX.h5'
    save_freq='epoch',  # Save every epoch
    save_weights_only=False,  # Save the entire model (not just weights)
    verbose=1  # Print a message when saving
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=100, mode='min',
                                              restore_best_weights=True)
model.summary()
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stop]
)

# Save the trained model
model.save("bounding_box_regressor2.keras")
model = tf.keras.models.load_model("bounding_box_regressor2.keras")

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss (MSE)')
plt.plot(epochs, val_loss, 'r', label='Validation loss (MSE)')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#
# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_mae, 'b', label='Training MAE')
plt.plot(epochs, val_mae, 'r', label='Validation MAE')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

def compute_saliency_map_regression(model, img_array):
    """
    Compute a saliency map for a regression model.

    Args:
        model: Trained regression model.
        img_array: Preprocessed image array of shape (H, W, 3), normalized to [0, 1].

    Returns:
        saliency_map: Saliency map (H, W).
    """
    # Convert the normalized image to a tensor and add batch dimension
    image = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)

    # Use GradientTape to record operations for gradient computation
    with tf.GradientTape() as tape:
        tape.watch(image)  # Watch the input image
        outputs = model(image)  # Get the model's output (regression predictions)

    # Compute gradients of the output w.r.t. the input image
    gradients = tape.gradient(outputs, image)[0]  # Gradients w.r.t the input image

    # Take the maximum absolute gradient across channels (R, G, B) for each pixel
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1).numpy()

    # Normalize the saliency map for better visualization
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

    return saliency_map



def plot_predictions_with_gradcam(img_dir, output_dir, model, last_conv_layer_name='conv5_block3_out', num_samples=5):
    # Randomly select a few images from the directory
    sample_files = np.random.choice(os.listdir(img_dir), num_samples, replace=False)

    for i, filename in enumerate(sample_files):
        # Load image
        img_path = os.path.join(img_dir, filename)
        json_path = os.path.join(output_dir, filename.replace('.png', '.json'))

        if os.path.exists(json_path):
            img = Image.open(img_path).convert("RGB")
            original_width, original_height = img.size

            # Resize the image
            img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            img_array = img_to_array(img) / 255.0  # Normalize image

            # Draw bounding box and ground truth
            draw = ImageDraw.Draw(img)
            with open(json_path, 'r') as f:
                data = json.load(f)

            points = []
            for shape in data['shapes']:
                point = shape['points'][0]  # Get the point (x, y)

                # Adjust the point coordinates according to the resized image
                x = (point[0] / original_width) * WIDTH
                y = (point[1] / original_height) * HEIGHT
                adjusted_point = (x, y)
                points.append(adjusted_point)

            if points:
                min_x = min(p[0] for p in points)
                max_x = max(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_y = max(p[1] for p in points)
                draw.rectangle([min_x, min_y, max_x, max_y], outline="yellow", width=3)

            # Predict bounding box
            predicted_bbox = model.predict(np.expand_dims(img_array, axis=0))[0]
            draw.rectangle(predicted_bbox, outline='red', width=3)

            # Compute Grad-CAM
            saliency_map = compute_saliency_map_regression(model, img_array)

            # Plot the results
            plt.figure(figsize=(12, 6))

            # Plot original image with predictions
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Sample {i + 1}: Predictions")
            plt.axis("off")

            # Plot Grad-CAM overlay
            plt.subplot(1, 2, 2)
            plt.imshow(img_array, alpha=0.7)  # Original image
            plt.imshow(saliency_map, cmap='hot', alpha=0.5)  # Saliency map overlay
            plt.title("Saliency Map Overlay")
            plt.axis("off")

            plt.tight_layout()
            plt.show()


# Example usage
plot_predictions_with_gradcam(img_dir, json_dir, model, num_samples=100)
