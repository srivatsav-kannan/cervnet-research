# import os
# import json
# from PIL import Image, ImageDraw
#
# # Directories
# img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
# json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
# output_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-boundingboxes/'
#
# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)
#
# # Color mapping for labels
# label_colors = {
#     "C2 bottom left": (139, 0, 0),  # Dark Red
#     "C2 bottom right": (0, 128, 0),  # Green
#     "C2 centroid": (128, 0, 128),  # Purple
#     "C3 top left": (128, 0, 128),  # Purple
#     "C3 top right": (0, 128, 128),  # Cyan
#     "C3 bottom left": (144, 238, 144),  # Light Green
#     "C3 bottom right": (152, 251, 152),  # Light Green
#     "C4 top left": (128, 128, 128),  # Gray
#     "C4 top right": (139, 0, 0),  # Dark Red
#     "C4 bottom left": (255, 0, 0),  # Red
#     "C4 bottom right": (0, 128, 0),  # Green
#     "C5 top left": (255, 165, 0),  # Orange
#     "C5 top right": (128, 0, 128),  # Purple
#     "C5 bottom left": (255, 20, 147),  # Pink
#     "C5 bottom right": (0, 128, 128),  # Cyan
#     "C6 top left": (210, 180, 140),  # Tan
#     "C6 top right": (0, 128, 0),  # Green
#     "C6 bottom left": (139, 69, 19),  # Brown
#     "C6 bottom right": (0, 255, 0),  # Lime Green
#     "C7 top left": (144, 238, 144),  # Light Green
#     "C7 top right": (0, 0, 255),  # Blue
#     "C7 bottom left": (85, 107, 47),  # Olive
#     "C7 bottom right": (0, 0, 128),  # Navy Blue
# }
#
# # Process each image
# for filename in os.listdir(img_dir):
#     if filename.endswith('.png'):
#         # Image path and corresponding JSON file path
#         img_path = os.path.join(img_dir, filename)
#         json_path = os.path.join(json_dir, filename.replace('.png', '.json'))
#
#         if os.path.exists(json_path):
#             # Open the image
#             img = Image.open(img_path).convert("RGB")
#             draw = ImageDraw.Draw(img)
#
#             # Open the JSON file
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
#
#             # List to store points for bounding box
#             points = []
#
#             # Draw points and labels
#             for shape in data['shapes']:
#                 label = shape['label']
#                 color = label_colors.get(label, (255, 255, 255))  # Default to white if not in mapping
#                 point = shape['points'][0]  # Get the point (x, y)
#
#                 # Add the point to the list for bounding box calculation
#                 points.append(tuple(point))
#
#                 # Draw the point
#                 radius = 5
#                 x, y = point
#                 draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
#
#             # Draw bounding box if there are points
#             if points:
#                 min_x = min(p[0] for p in points)
#                 max_x = max(p[0] for p in points)
#                 min_y = min(p[1] for p in points)
#                 max_y = max(p[1] for p in points)
#
#                 # Draw the bounding box
#                 draw.rectangle([min_x, min_y, max_x, max_y], outline="yellow", width=3)
#
#             # Save the modified image
#             output_path = os.path.join(output_dir, filename)
#             img.save(output_path)
#
# print("Processing complete. Images with bounding boxes saved in:", output_dir)
import os
import json
from PIL import Image, ImageDraw

# Directories
img_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-PNG/'
json_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-JSON/'
output_dir = '/Users/srivatsavkannan/Datasets/C-Spine Xray/X-ray Atlas/datasets-boundingboxes/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Color mapping for labels
label_colors = {
    "C2 bottom left": (139, 0, 0),  # Dark Red
    "C2 bottom right": (0, 128, 0),  # Green
    "C2 centroid": (128, 0, 128),  # Purple
    "C3 top left": (128, 0, 128),  # Purple
    "C3 top right": (0, 128, 128),  # Cyan
    "C3 bottom left": (144, 238, 144),  # Light Green
    "C3 bottom right": (152, 251, 152),  # Light Green
    "C4 top left": (128, 128, 128),  # Gray
    "C4 top right": (139, 0, 0),  # Dark Red
    "C4 bottom left": (255, 0, 0),  # Red
    "C4 bottom right": (0, 128, 0),  # Green
    "C5 top left": (255, 165, 0),  # Orange
    "C5 top right": (128, 0, 128),  # Purple
    "C5 bottom left": (255, 20, 147),  # Pink
    "C5 bottom right": (0, 128, 128),  # Cyan
    "C6 top left": (210, 180, 140),  # Tan
    "C6 top right": (0, 128, 0),  # Green
    "C6 bottom left": (139, 69, 19),  # Brown
    "C6 bottom right": (0, 255, 0),  # Lime Green
    "C7 top left": (144, 238, 144),  # Light Green
    "C7 top right": (0, 0, 255),  # Blue
    "C7 bottom left": (85, 107, 47),  # Olive
    "C7 bottom right": (0, 0, 128),  # Navy Blue
}

# Resize dimensions
RESIZE_WIDTH = 512
RESIZE_HEIGHT = 512

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

                # Draw the bounding box
                draw.rectangle([min_x, min_y, max_x, max_y], outline="yellow", width=3)

            # Save the modified image
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)

print("Processing complete. Resized images with bounding boxes saved in:", output_dir)
