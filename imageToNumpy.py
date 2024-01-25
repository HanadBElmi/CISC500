import os
from PIL import Image, ImageOps
import numpy as np

# Set the directory path
directory_path = "/Users/hanadelmi/Documents/Research/2023/Fall 2023 - Thesis/Open-source datasets/Dataset_BUSI_with_GT/normal"

# Set the file name for the numpy array
output_file_name = "image_dataset.npy"

# Set a common size for all images
common_size = (256, 256)  # You can change this to whatever size you want

# Create a list to store the image data
image_data_list = []

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
        # Construct the full path to the image file
        image_path = os.path.join(directory_path, filename)

        # Open the image using PIL (Pillow)
        image = Image.open(image_path)

        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image to the common size using Resampling.LANCZOS (which is the same as ANTIALIAS)
        image = image.resize(common_size, Image.Resampling.LANCZOS)

        # Convert the resized image to a NumPy array
        image_data = np.array(image)

        # Append the image data to the list
        image_data_list.append(image_data)

# Convert the list of NumPy arrays into a 4D NumPy array
image_dataset = np.stack(image_data_list)

# Save the 4D NumPy array to the same directory as the images
np.save(os.path.join(directory_path, output_file_name), image_dataset)

# Print some information about the saved dataset
print(f"Number of images: {image_dataset.shape[0]}")
print(f"Each image shape: {image_dataset.shape[1:]}")
print(f"Dataset saved to: {os.path.join(directory_path, output_file_name)}")
