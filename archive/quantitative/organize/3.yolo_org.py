import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf

# Directories
train_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Aug2"
val_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug2"
train_cropped_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Train_Org_Aug_Cropped2"
val_cropped_dir = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug_Cropped2"

os.makedirs(train_cropped_dir, exist_ok=True)
os.makedirs(val_cropped_dir, exist_ok=True)

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
class_names = ["CS", "Healthy"]

# Load datasets
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

# Model
model = tf.keras.models.load_model('ssd.keras')

# Helper function to extract filenames
def get_filenames(dataset, original_dir):
    filenames = []
    for path in dataset.file_paths:
        filenames.append(os.path.basename(path))
    return filenames

# Get filenames
train_filenames = get_filenames(train_ds, train_dir)
val_filenames = get_filenames(val_ds, val_dir)

# Process training dataset
for batch_idx, batch in enumerate(train_ds):
    images, labels = batch
    predicted_bboxes, predicted_labels_batch = model.predict(images)

    for i in range(images.shape[0]):
        image = images[i].numpy().astype(np.uint8)
        actual_label = labels[i].numpy()
        predicted_bbox = predicted_bboxes[i]

        # Bounding box
        min_x = int(max(0, predicted_bbox[0]))
        min_y = int(max(0, predicted_bbox[1]))
        max_x = int(min(224, predicted_bbox[2]))
        max_y = int(min(224, predicted_bbox[3]))

        # Crop the image
        cropped_image = Image.fromarray(image).crop((min_x, min_y, max_x, max_y))
        cropped_image = cropped_image.resize((224,224))
        # Subdirectory based on label (0 = CS, 1 = Healthy)
        sub_dir = "CS" if actual_label == 0 else "Healthy"
        class_dir = os.path.join(train_cropped_dir, sub_dir)
        os.makedirs(class_dir, exist_ok=True)

        # Save the cropped image with original filename
        original_filename = train_filenames[batch_idx * BATCH_SIZE + i]
        cropped_image.save(os.path.join(class_dir, original_filename))

# Process validation dataset
for batch_idx, batch in enumerate(val_ds):
    images, labels = batch
    predicted_bboxes, predicted_labels_batch = model.predict(images)

    for i in range(images.shape[0]):
        image = images[i].numpy().astype(np.uint8)
        actual_label = labels[i].numpy()
        predicted_bbox = predicted_bboxes[i]

        # Bounding box
        min_x = int(max(0, predicted_bbox[0]))
        min_y = int(max(0, predicted_bbox[1]))
        max_x = int(min(224, predicted_bbox[2]))
        max_y = int(min(224, predicted_bbox[3]))

        # Crop the image
        cropped_image = Image.fromarray(image).crop((min_x, min_y, max_x, max_y))

        # Subdirectory based on label (0 = CS, 1 = Healthy)
        sub_dir = "CS" if actual_label == 0 else "Healthy"
        class_dir = os.path.join(val_cropped_dir, sub_dir)
        os.makedirs(class_dir, exist_ok=True)

        # Save the cropped image with original filename
        original_filename = val_filenames[batch_idx * BATCH_SIZE + i]
        cropped_image.save(os.path.join(class_dir, original_filename))
