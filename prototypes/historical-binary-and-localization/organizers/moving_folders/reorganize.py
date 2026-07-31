import os
import shutil
import pandas as pd
import cv2

# Define paths
path1 = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val'
path2 = '/Users/srivatsavkannan/Datasets/FinalCervicalDataset/Val_Org'


# Define class names
class_names = ["CS", "Healthy"]

for class_name in class_names:
    os.makedirs(os.path.join(path2, class_name), exist_ok=True)
# Read the Excel file
excel_path = '/Users/srivatsavkannan/Datasets/C-Spine Xray/datasets.xlsx'
df = pd.read_excel(excel_path)

# Create directories if they do not exist
# for class_name in class_names:
#     class_dir = os.path.join(path2, class_name)
#     if not os.path.exists(class_dir):
#         os.makedirs(class_dir)

# Process each row in the specified range

for i in range(1, 5003):  # Pandas index is 0-based
    try:
        # Get the values from the DataFrame
        a_value = str(df.iat[i, 0]).zfill(4)
        b_value = str(df.iat[i, 1])
        c_value = str(df.iat[i,  2]).zfill(2)
        e_value = int(df.iat[i, 3])
        if e_value is not None:
            e_value -= 1
            # Construct the image filename
            image_filename = f"{a_value}{b_value}{c_value}.png"
            image_path = os.path.join(path1, image_filename)

            # Check if the image exists
            if not os.path.isfile(image_path):
                print(f"Image not found: {image_path}")
                continue

            # Read the image
            image = cv2.imread(image_path)

            # Get the class name based on the value in column E
            class_name = class_names[e_value]
            class_dir = os.path.join(path2, class_name)

            # Save the image to the corresponding class directory
            output_path = os.path.join(class_dir, image_filename)
            os.makedirs(class_dir, exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"Saved row {i}")
    except Exception as e:
        print(f"Error processing row {i}: {e}")

print("Processing completed.")
