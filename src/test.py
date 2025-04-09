# import cv2
# import numpy as np
# import os

# # Load the image
# image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/grid_slices_3/row_0.png"  # Replace with your image path
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Step 1: Remove empty margins
# _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
# coords = cv2.findNonZero(binary)
# x, y, w, h = cv2.boundingRect(coords)
# cropped = image[y:y+h, x:x+w]

# # Step 2: Detect vertical projection profile
# gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray_crop, 50, 150)
# vertical_projection = np.sum(edges, axis=0)

# # Step 3: Find local minimums (gaps between items)
# threshold = 0.2 * np.max(vertical_projection)
# gaps = np.where(vertical_projection < threshold)[0]

# # Group close gaps
# min_gap = 30
# split_points = [gaps[0]]
# for i in range(1, len(gaps)):
#     if gaps[i] - split_points[-1] > min_gap:
#         split_points.append(gaps[i])

# # Final slice edges
# slice_edges = [0] + split_points + [cropped.shape[1]]

# # Create output directory
# output_dir = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/grid_slices_3/slices"
# os.makedirs(output_dir, exist_ok=True)

# # Slice and save
# for i in range(len(slice_edges) - 1):
#     x1, x2 = slice_edges[i], slice_edges[i + 1]
#     item_crop = cropped[:, x1:x2]
#     if item_crop.shape[1] > 30:
#         cv2.imwrite(f"{output_dir}/item_{i + 1}.png", item_crop)

# print(f"Sliced items into '{output_dir}' folder.")


import cv2
import os

# Load image
image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/grid_slices_1/row_1.png"  # Replace with your image path
image = cv2.imread(image_path)

# Get image dimensions
height, width, _ = image.shape
num_items = 5  # You can auto-detect this later too
item_width = width // num_items

# Create output folder
output_dir = "equal_slices"
os.makedirs(output_dir, exist_ok=True)

# Slice into equal parts
for i in range(num_items):
    x1 = i * item_width
    x2 = (i + 1) * item_width if i < num_items - 1 else width
    item = image[:, x1:x2]
    cv2.imwrite(f"{output_dir}/item_{i + 1}.png", item)
