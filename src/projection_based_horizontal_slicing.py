import cv2
import numpy as np
import os

def projection_slicing(image_path, output_folder="prj_img"):
    os.makedirs(output_folder, exist_ok=True)

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    # Compute horizontal projection
    projection = np.sum(binary, axis=1)

    # Smooth the projection to reduce noise (rolling average)
    smoothed = np.convolve(projection, np.ones(15)/15, mode='same')

    # Dynamic threshold: higher = stricter
    threshold = np.max(smoothed) * 0.3

    # Detect start and end points of content rows
    rows = []
    in_row = False
    start = 0
    for i, value in enumerate(smoothed):
        if value > threshold and not in_row:
            start = i
            in_row = True
        elif value <= threshold and in_row:
            end = i
            if end - start > 50:
                rows.append((start, end))
            in_row = False

    # Apply padding to each row and clip within image bounds
    row_paths = []
    overlay = image.copy()
    pad_top = 10
    pad_bottom = 20

    for idx, (y1, y2) in enumerate(rows):
        y1p = max(0, y1 - pad_top)
        y2p = min(image.shape[0], y2 + pad_bottom)

        row_img = image[y1p:y2p, :]
        row_path = os.path.join(output_folder, f"row_{idx}.png")
        cv2.imwrite(row_path, row_img)
        row_paths.append(row_path)

        # Draw rectangle on overlay
        cv2.rectangle(overlay, (0, y1p), (image.shape[1], y2p), (0, 255, 0), 2)

    # Save overlay
    cv2.imwrite(os.path.join(output_folder, "projection_overlay.png"), overlay)

    print(f"✅ Sliced {len(rows)} rows with smoothing, thresholding, and padding.")


if __name__ == "__main__":
    projection_slicing("/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/listing.png", output_folder="prj_slices_3")
