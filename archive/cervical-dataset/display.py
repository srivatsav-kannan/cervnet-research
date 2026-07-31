import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the folder containing the JSON files
json_folder_path = '/Users/srivatsavkannan/Datasets/X-ray Atlas/datasets-JSON'
image_folder_path = '/Users/srivatsavkannan/Datasets/X-ray Atlas/datasets-PNG'
# Define a color map for different labels
color_map = {
    "C2 bottom left": "red",
    "C2 bottom right": "blue",
    "C2 centroid": "pink",
    "C3 top left": "purple",
    "C3 top right": "brown",
    "C3 bottom left": "cyan",
    "C3 bottom right": "magenta",
    "C4 top left": "yellow",
    "C4 top right": "lime",
    "C4 bottom left": "olive",
    "C4 bottom right": "navy",
    "C5 top left": "teal",
    "C5 top right": "maroon",
    "C5 bottom left": "grey",
    "C5 bottom right": "black",
    "C6 top left": "lightcoral",
    "C6 top right": "darkorange",
    "C6 bottom left": "khaki",
    "C6 bottom right": "lightseagreen",
    "C7 top left": "mediumvioletred",
    "C7 top right": "mediumslateblue",
    "C7 bottom left": "green",
    "C7 bottom right": "orange"
}


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def display_image_with_annotations(image_path, annotations):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    print(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a plot
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)

    # Plot each annotation
    for shape in annotations['shapes']:
        label = shape['label']
        points = shape['points']
        color = color_map.get(label, "white")  # Use white if label is not in color_map

        for point in points:
            plt.scatter(point[0], point[1], color=color, label=label)

    # Create a legend with unique labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"Annotations for {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()


# Process each JSON file in the folder
print(json_folder_path)
for json_file in os.listdir(json_folder_path):
    if json_file.endswith('.json'):
        json_file_path = os.path.join(json_folder_path, json_file)
        data = read_json_file(json_file_path)

        # Construct the image path from the JSON file name
        image_file_name = os.path.splitext(json_file)[0] + '.png'
        image_path = os.path.join(image_folder_path, image_file_name)

        # Display the image with annotations
        display_image_with_annotations(image_path, data)
