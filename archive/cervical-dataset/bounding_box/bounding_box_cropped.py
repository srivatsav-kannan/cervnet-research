import os
import json
from PIL import Image, ImageDraw
from bounding_box import label_colors

# Directories
img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
output_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-boundingboxes/'
cropped_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-boundingboxes-cropped-512/'

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

# Color mapping for labels

# Resize dimensions
RESIZE_WIDTH = 1024
RESIZE_HEIGHT = 1024
CROP_SIZE = 512

# Bounding box expansion margins
EXPAND_TOP = 75*2
EXPAND_BOTTOM = 37.5*2
EXPAND_LEFT = 20*2
EXPAND_RIGHT = 100*2

# Process each image
for filename in os.listdir(img_dir):
    if filename.endswith('.png'):
        # Image path and corresponding JSON file path
        img_path = os.path.join(img_dir, filename)
        json_path = os.path.join(json_dir, filename.replace('.png', '.json'))

        if os.path.exists(json_path):
            # Open the image
            img = Image.open(img_path).convert("RGB")
            original_width, original_height = img.size

            # Resize the image
            img = img.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.Resampling.LANCZOS)
            img2 = Image.open(img_path).convert("RGB").resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.Resampling.LANCZOS)

            draw = ImageDraw.Draw(img)

            # Open the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)

            # List to store points for bounding box
            points = []

            # Draw points and labels
            for shape in data['shapes']:
                label = shape['label']
                color = label_colors.get(label, (255, 255, 255))  # Default to white if not in mapping
                point = shape['points'][0]  # Get the point (x, y)

                # Adjust the point coordinates according to the resized image
                x = (point[0] / original_width) * RESIZE_WIDTH
                y = (point[1] / original_height) * RESIZE_HEIGHT
                adjusted_point = (x, y)

                # Add the adjusted point to the list for bounding box calculation
                points.append(adjusted_point)

                # Draw the point
                radius = 5
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

            # Draw bounding box if there are points
            if points:
                min_x = min(p[0] for p in points)
                max_x = max(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_y = max(p[1] for p in points)

                # Expand the bounding box
                min_x = max(0, min_x - EXPAND_LEFT)
                max_x = min(RESIZE_WIDTH, max_x + EXPAND_RIGHT)
                min_y = max(0, min_y - EXPAND_TOP)
                max_y = min(RESIZE_HEIGHT, max_y + EXPAND_BOTTOM)

                # Draw the bounding box
                draw.rectangle([min_x, min_y, max_x, max_y], outline="yellow", width=3)

                # Crop the expanded bounding box region
                cropped_img = img2.crop((min_x, min_y, max_x, max_y))

                # Resize the cropped image to 224x224
                cropped_img = cropped_img.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)

                # Save the cropped image
                cropped_path = os.path.join(cropped_dir, filename)
                print(os.path.join(cropped_dir, filename))
                cropped_img.save(cropped_path)

            # Save the modified image
            # output_path = os.path.join(output_dir, filename)
            # img.save(output_path)

print("Processing complete.")
# print("Resized images with bounding boxes saved in:", output_dir)
print("Cropped and resized images saved in:", cropped_dir)
