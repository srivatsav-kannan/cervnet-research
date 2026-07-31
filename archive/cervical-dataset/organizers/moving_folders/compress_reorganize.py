import os
from PIL import Image

def whatsapp_compress(input_path, output_path, resize_to=1080, quality=75):
    """
    Simulates WhatsApp image compression and saves the output.
    """
    img = Image.open(input_path)
    img = img.convert("RGB")  # Ensure compatibility with JPEG format

    # Resize if needed
    original_width, original_height = img.size
    if max(original_width, original_height) > resize_to:
        if original_width > original_height:
            new_width = resize_to
            new_height = int(resize_to * original_height / original_width)
        else:
            new_height = resize_to
            new_width = int(resize_to * original_width / original_height)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Save compressed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    img.save(output_path, "JPEG", quality=quality)

# Define paths
base_input_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized_aug2"
base_output_dir = "/Users/srivatsavkannan/Datasets/C-Spine Xray/Organized_aug2_compressed"

# List of subdirectories to process
subdirs = [
    "train/CS",
    "train/Healthy",
    "val/CS",
    "val/Healthy"
]

# Iterate over all images in the input directories
for subdir in subdirs:
    input_dir = os.path.join(base_input_dir, subdir)
    output_dir = os.path.join(base_output_dir, subdir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Only process files with valid image extensions
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
            whatsapp_compress(input_path, output_path)

print("All images have been compressed and saved in the new directory structure.")
