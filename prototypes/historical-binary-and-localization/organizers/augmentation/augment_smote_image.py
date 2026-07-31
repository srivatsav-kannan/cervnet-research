import os
import numpy as np
from PIL import Image
from imblearn.over_sampling import SMOTE
from keras.preprocessing.image import array_to_img, img_to_array

def load_images_and_labels(cs_input_folder, healthy_input_folder, target_size=(128, 128)):
    """Load images and corresponding labels from CS and Healthy folders."""
    X = []
    y = []
    filenames = []

    # Load CS images
    for filename in os.listdir(cs_input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(cs_input_folder, filename)
            img = Image.open(img_path).convert('RGB').resize(target_size)
            X.append(img_to_array(img).flatten())
            y.append(0)  # Label for CS
            filenames.append((filename, "CS"))

    # Load Healthy images
    for filename in os.listdir(healthy_input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(healthy_input_folder, filename)
            img = Image.open(img_path).convert('RGB').resize(target_size)
            X.append(img_to_array(img).flatten())
            y.append(1)  # Label for Healthy
            filenames.append((filename, "Healthy"))

    return np.array(X), np.array(y), filenames

def save_images_after_smote(X_resampled, y_resampled, filenames, output_dir, target_size=(128, 128)):
    """Save resampled images to the appropriate directories."""
    for i, (image, label) in enumerate(zip(X_resampled, y_resampled)):
        label_name = "CS" if label == 0 else "Healthy"
        original_filename, original_label = filenames[i] if i < len(filenames) else (f"generated_{i}", label_name)

        # Determine save directory and filename
        save_dir = os.path.join(output_dir, label_name)
        os.makedirs(save_dir, exist_ok=True)

        save_filename = f"{i}.png"
        # if i >= len(filenames):  # For synthetic images, modify filename
        #     save_filename = f"{original_filename[:-4]}_0.png"
        # else:
        #     save_filename = original_filename

        # Reshape and save the image
        img = array_to_img(image.reshape((*target_size, 3)))
        print(image.shape)
        img.save(os.path.join(save_dir, save_filename))

def apply_smote_and_save(cs_input_folder, healthy_input_folder, output_dir, target_size=(128, 128)):
    """Apply SMOTE to balance classes and save images to output directory."""
    # Load images and labels
    X, y, filenames = load_images_and_labels(cs_input_folder, healthy_input_folder, target_size)
    print(X.shape)
    print(y.shape)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(X_resampled.shape)
    print(y_resampled.shape)

    # Save resampled images
    save_images_after_smote(X_resampled, y_resampled, filenames, output_dir, target_size)

# Directories
cs_input_folder = "/Users/srivatsavkannan/Datasets/EnhancedCervicalDataset/Train_Org_Cropped/CS"
healthy_input_folder = "/Users/srivatsavkannan/Datasets/EnhancedCervicalDataset/Train_Org_Cropped/Healthy"
output_dir = "/Users/srivatsavkannan/Datasets/EnhancedCervicalDataset/Train_Org_Cropped_SMOTE"

# Apply SMOTE and save images
apply_smote_and_save(cs_input_folder, healthy_input_folder, output_dir, target_size=(224, 224))
