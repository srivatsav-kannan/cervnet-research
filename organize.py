import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Directories
img_dir = '/Users/srivatsavkannan/Datasets/CervicalNew3/All'
train_output_dir = '/Users/srivatsavkannan/Datasets/CervicalNew8/Train/Healthy'
val_output_dir = '/Users/srivatsavkannan/Datasets/CervicalNew8/Val/Healthy'

# Ensure output directories exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Collect filenames of images
filenames = [f for f in os.listdir(img_dir) if f.endswith('.png')]


# Train-test split
filenames_train, filenames_val = train_test_split(filenames, test_size=0.3, random_state=42)

# Function to save images
def save_images(filenames, input_dir, output_dir):
    for filename in filenames:
        if filename.endswith('.png'):
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

# Save train and validation images
save_images(filenames_train, img_dir, train_output_dir)
save_images(filenames_val, img_dir, val_output_dir)

print(f"Images successfully saved to Train: {train_output_dir} and Val: {val_output_dir}")
