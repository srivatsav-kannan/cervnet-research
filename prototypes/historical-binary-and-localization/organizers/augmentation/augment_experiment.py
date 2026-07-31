import os
import random
import shutil


def balance_classes(cs_input_folder, healthy_input_folder, cs_output_folder, healthy_output_folder):
    """Balance classes using Random Oversampling (ROS) and Random Undersampling (RUS)."""

    # Ensure output folders exist
    os.makedirs(cs_output_folder, exist_ok=True)
    os.makedirs(healthy_output_folder, exist_ok=True)

    # Get list of image files in input folders
    cs_images = [f for f in os.listdir(cs_input_folder) if os.path.isfile(os.path.join(cs_input_folder, f))]
    healthy_images = [f for f in os.listdir(healthy_input_folder) if os.path.isfile(os.path.join(healthy_input_folder, f))]

    # Perform Random Oversampling (ROS) for Healthy images (each image is turned into 3 copies)
    healthy_augmented = []
    for filename in healthy_images:
        src_path = os.path.join(healthy_input_folder, filename)
        for i in range(3):  # Create 3 copies
            new_filename = f"{filename[:-4]}_aug{i}.png"
            dest_path = os.path.join(healthy_output_folder, new_filename)
            shutil.copyfile(src_path, dest_path)
            healthy_augmented.append(new_filename)

    total_healthy_augmented = len(healthy_augmented)

    # Perform Random Undersampling (RUS) for CS images (CS is reduced to twice the number of augmented Healthy images)
    cs_images_to_keep = random.sample(cs_images, 2 * total_healthy_augmented)

    # Copy selected CS images to output folder
    for filename in cs_images_to_keep:
        src_path = os.path.join(cs_input_folder, filename)
        dest_path = os.path.join(cs_output_folder, filename)
        shutil.copyfile(src_path, dest_path)

    print(f"Total Healthy Augmented: {total_healthy_augmented} (each image turned into 3)")
    print(f"Total CS after Undersampling: {len(cs_images_to_keep)} (2x Healthy)")
    print(f"Balanced CS images saved to {cs_output_folder}")
    print(f"Balanced Healthy images saved to {healthy_output_folder}")


# Directories
cs_input_folder = "/Users/srivatsavkannan/Datasets/Experiment/Train_Org_Cropped/CS"
healthy_input_folder = "/Users/srivatsavkannan/Datasets/Experiment/Train_Org_Cropped/Healthy"
cs_output_folder = "/Users/srivatsavkannan/Datasets/Experiment/Train_Org_Cropped_ROS_RUS/CS"
healthy_output_folder = "/Users/srivatsavkannan/Datasets/Experiment/Train_Org_Cropped_ROS_RUS/Healthy"

# Balance classes
balance_classes(cs_input_folder, healthy_input_folder, cs_output_folder, healthy_output_folder)
