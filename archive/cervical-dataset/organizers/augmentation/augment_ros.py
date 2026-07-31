import os
import random
import shutil

# Function to balance classes by oversampling Healthy images
def balance_classes(cs_input_folder, healthy_input_folder, cs_output_folder, healthy_output_folder):
    # Ensure output folders exist
    os.makedirs(cs_output_folder, exist_ok=True)
    os.makedirs(healthy_output_folder, exist_ok=True)

    # Get list of image files in input folders
    cs_images = [f for f in os.listdir(cs_input_folder) if os.path.isfile(os.path.join(cs_input_folder, f))]
    healthy_images = [f for f in os.listdir(healthy_input_folder) if os.path.isfile(os.path.join(healthy_input_folder, f))]

    total_cs = len(cs_images)
    total_healthy = len(healthy_images)
    # total_cs -= 1
    print(f"Total CS: {total_cs}, Total Healthy: {total_healthy}")

    # Copy all CS images to the output folder
    for filename in cs_images:
        src_path = os.path.join(cs_input_folder, filename)
        dest_path = os.path.join(cs_output_folder, filename)
        os.makedirs(cs_output_folder, exist_ok=True)
        shutil.copyfile(src_path, dest_path)

    # Oversample Healthy images to match the count of CS images
    if total_healthy < total_cs:
        healthy_images_to_duplicate = random.choices(healthy_images, k=total_cs - total_healthy)
        healthy_images_balanced = healthy_images + healthy_images_to_duplicate
    else:
        healthy_images_balanced = healthy_images

    # Copy balanced Healthy images to the output folder
    for i, filename in enumerate(healthy_images_balanced):
        if filename == '.DS_Store':  # Skip system files
            continue
        src_path = os.path.join(healthy_input_folder, filename)
        if i >= len(healthy_images):  # For duplicates, rename to avoid overwriting
            dest_path = os.path.join(healthy_output_folder, f"{filename[:-4]}_dup{i}.png")
        else:
            dest_path = os.path.join(healthy_output_folder, filename)
        os.makedirs(healthy_output_folder, exist_ok=True)
        shutil.copyfile(src_path, dest_path)

    print(f"CS images saved to {cs_output_folder}")
    print(f"Balanced Healthy images saved to {healthy_output_folder}")


# Directories
# cs_input_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Cropped/CS"
# healthy_input_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Cropped/Healthy"
# cs_output_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Cropped_ROS/CS"
# healthy_output_folder = "/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Cropped_ROS/Healthy"

cs_input_folder = "/Users/srivatsavkannan/Datasets/CXR/test/PNEUMONIA"
healthy_input_folder = "/Users/srivatsavkannan/Datasets/CXR/test/NORMAL"
cs_output_folder = "/Users/srivatsavkannan/Datasets/CXRA/val/PNEUMONIA"
healthy_output_folder = "/Users/srivatsavkannan/Datasets/CXRA/val/NORMAL"

os.makedirs(cs_output_folder, exist_ok=True)
os.makedirs(healthy_output_folder, exist_ok=True)
# Balance classes using ROS for Healthy
balance_classes(cs_input_folder, healthy_input_folder, cs_output_folder, healthy_output_folder)
