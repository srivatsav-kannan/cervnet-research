import os
import random
import shutil


# Function to balance classes by duplicating or removing images
def balance_classes(cs_input_folder, healthy_input_folder, cs_output_folder, healthy_output_folder):
    # Ensure output folders exist
    os.makedirs(cs_output_folder, exist_ok=True)
    os.makedirs(healthy_output_folder, exist_ok=True)

    # Get list of image files in input folders
    cs_images = [f for f in os.listdir(cs_input_folder) if os.path.isfile(os.path.join(cs_input_folder, f))]
    healthy_images = [f for f in os.listdir(healthy_input_folder) if
                      os.path.isfile(os.path.join(healthy_input_folder, f))]

    # Calculate midpoint
    total_cs = len(cs_images)
    total_healthy = len(healthy_images)
    midpoint = (total_cs + total_healthy) // 2

    print(f"Total CS: {total_cs}, Total Healthy: {total_healthy}, Midpoint: {midpoint}")

    # Balance CS images by randomly removing excess
    if total_cs > midpoint:
        cs_images_to_keep = random.sample(cs_images, midpoint)
    else:
        cs_images_to_keep = cs_images

    # Balance Healthy images by duplicating
    if total_healthy < midpoint:
        healthy_images_to_duplicate = random.choices(healthy_images, k=midpoint - total_healthy)
        healthy_images_balanced = healthy_images + healthy_images_to_duplicate
    else:
        healthy_images_balanced = healthy_images

    # Copy selected CS images to output folder
    for filename in cs_images_to_keep:
        src_path = os.path.join(cs_input_folder, filename)
        dest_path = os.path.join(cs_output_folder, filename)
        shutil.copyfile(src_path, dest_path)

    # Copy balanced Healthy images to output folder
    for i, filename in enumerate(healthy_images_balanced):
        if filename == '.DS_Store':
            continue
        print(filename)
        print(f"{filename[:-4]}_dup{i}.png")
        src_path = os.path.join(healthy_input_folder, filename)
        if i >= len(healthy_images):  # For duplicates, rename to avoid overwriting
            dest_path = os.path.join(healthy_output_folder, f"{filename[:-4]}_dup{i}.png")
        else:
            dest_path = os.path.join(healthy_output_folder, filename)
        shutil.copyfile(src_path, dest_path)

    print(f"Balanced CS images saved to {cs_output_folder}")
    print(f"Balanced Healthy images saved to {healthy_output_folder}")


# Directories
cs_input_folder = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized2/Val/CS"
healthy_input_folder = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized2/Val/Healthy"
cs_output_folder = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized2/Val/CS_aug"
healthy_output_folder = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Final_Organized2/Val/Healthy_aug"

# Balance classes
balance_classes(cs_input_folder, healthy_input_folder, cs_output_folder, healthy_output_folder)
