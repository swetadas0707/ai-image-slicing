import cv2
import numpy as np
import os

def vertical_projection_slicing(image_path, output_folder="column_slices"):
    os.makedirs(output_folder, exist_ok=True)

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(os.path.join(output_folder, "thresh_debug.png"), thresh)

    # Calculate vertical projection profile
    projection = np.sum(thresh, axis=0)
    min_val = np.min(projection)
    threshold = min_val + (np.max(projection) - min_val) * 0.2

    # Detect low-value gaps in projection (column gaps)
    split_indices = []
    in_gap = False
    for i, val in enumerate(projection):
        if val < threshold and not in_gap:
            start = i
            in_gap = True
        elif val >= threshold and in_gap:
            end = i
            split_indices.append((start, end))
            in_gap = False

    # Derive slicing boundaries (midpoints of gaps)
    boundaries = [0] + [int((a + b) / 2) for a, b in split_indices] + [image.shape[1]]

    # Draw projection overlay
    overlay = image.copy()
    for x in boundaries:
        cv2.line(overlay, (x, 0), (x, image.shape[0]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "projection_overlay.png"), overlay)

    # Slice the image using boundaries
    col_paths = []
    for i in range(len(boundaries) - 1):
        x1, x2 = boundaries[i], boundaries[i + 1]
        if x2 - x1 < 30:  # filter small columns
            continue
        col = image[:, x1:x2]
        path = os.path.join(output_folder, f"col_{i}.png")
        cv2.imwrite(path, col)
        col_paths.append(path)

    print(f"✅ Saved {len(col_paths)} vertical slices using projection.")
    print(f"🖼️  Overlay image with boundaries saved as 'projection_overlay.png'.")


if __name__=="__main__":
    vertical_projection_slicing(
        image_path="/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/prj_vertical_slicing/row_1.png",
        output_folder="/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/prj_vertical_slicing/row_1")