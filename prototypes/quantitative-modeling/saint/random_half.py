import os
import random
import shutil


def copy_random_half(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    # Set seed for reproducibility
    random.seed(42)

    # Get all images
    all_images = [f for f in os.listdir(src_folder) if f.endswith(".png")]
    print(len(all_images))
    # Separate images based on ending
    c_images = [img for img in all_images if img.endswith("c.png")]
    print(len(c_images))
    other_images = [img for img in all_images if not img.endswith("c.png")]

    # Ensure equal selection
    half_c = len(c_images) // 2
    half_other = len(other_images) // 2

    selected_c = random.sample(c_images, half_c)
    selected_other = random.sample(other_images, half_other)

    selected_images = selected_c + selected_other

    # Copy selected images to destination folder
    for img in selected_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(dest_folder, img))


# Example usage
src_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_comp+normal/CS"
dest_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_comp+normal_half/CS"
copy_random_half(src_folder, dest_folder)
