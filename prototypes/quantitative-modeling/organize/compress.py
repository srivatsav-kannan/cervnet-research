import os

import cv2
import numpy as np
from PIL import Image

def extreme_whatsapp_compression(input_image_path, output_image_path, quality=30, rounds=3, blur_intensity=3):
    """
    Simulates extreme WhatsApp-style compression by aggressively reducing quality, applying multiple saves,
    and adding blur artifacts.

    Args:
        input_image_path (str): Path to the original image.
        output_image_path (str): Path to save the compressed image.
        quality (int): JPEG compression quality (lower means more compression, default 30).
        rounds (int): Number of times the image is re-saved to simulate repeated compression.
        blur_intensity (int): Kernel size for Gaussian blur to simulate aggressive artifacting.
    """
    # Load image
    image = cv2.imread(input_image_path)

    # Convert from OpenCV BGR to RGB for PIL processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    image_pil = Image.fromarray(image)

    for i in range(rounds):
        # Convert back to OpenCV format for processing
        image_cv = np.array(image_pil)

        # Apply blur to simulate aggressive compression artifacts
        if blur_intensity > 0:
            image_cv = cv2.GaussianBlur(image_cv, (blur_intensity, blur_intensity), 0)

        # Convert back to PIL
        image_pil = Image.fromarray(image_cv)

        # Save with extreme compression
        temp_path = output_image_path.replace(".jpg", f"_round{i}.jpg")
        image_pil.save(temp_path, "JPEG", quality=quality, optimize=True)

        # Reload the image to simulate repeated compression
        image_pil = Image.open(temp_path)

    # Save the final compressed image
    image_pil.save(output_image_path, "JPEG", quality=quality, optimize=True)

    print(f"Compressed image saved at {output_image_path}")

# Example usage
input_path = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug_Cropped2/CS'
output_path = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org_Aug_Cropped_comp/CS'
for input_image in os.listdir(input_path):
    extreme_whatsapp_compression(input_path+'/'+input_image, output_path+'/'+input_image, quality=20, rounds=5, blur_intensity=5)
