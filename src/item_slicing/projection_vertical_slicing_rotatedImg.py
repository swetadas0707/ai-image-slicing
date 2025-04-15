import cv2
import numpy as np
import os

def vertical_projection_slicing_on_rotated(image_path, output_folder="column_slices_rotated"):
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Load and rotate image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Step 2: Thresholding
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(os.path.join(output_folder, "rotated_thresh_debug.png"), thresh)

    # Step 3: Vertical projection profile (on rotated = horizontal in original)
    projection = np.sum(thresh, axis=0)
    min_val = np.min(projection)
    threshold = min_val + (np.max(projection) - min_val) * 0.2

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

    boundaries = [0] + [int((a + b) / 2) for a, b in split_indices] + [rotated.shape[1]]

    # Step 4: Debug overlay image (on rotated)
    overlay = rotated.copy()
    for x in boundaries:
        cv2.line(overlay, (x, 0), (x, rotated.shape[0]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "rotated_projection_overlay.png"), overlay)

    # Step 5: Slice and rotate back
    col_paths = []
    for i in range(len(boundaries) - 1):
        x1, x2 = boundaries[i], boundaries[i + 1]
        if x2 - x1 < 30:
            continue
        col = rotated[:, x1:x2]
        restored = cv2.rotate(col, cv2.ROTATE_90_COUNTERCLOCKWISE)
        path = os.path.join(output_folder, f"col_{i}.png")
        cv2.imwrite(path, restored)
        col_paths.append(path)

    print(f"✅ Saved {len(col_paths)} vertical slices from rotated image.")
    print(f"🖼️ Debug overlay saved as 'rotated_projection_overlay.png'.")



if __name__=="__main__":
    # Example usage
    vertical_projection_slicing_on_rotated(
        image_path="/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/prj_rotated_vertical_slicing/row_1.png",
        output_folder="/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/prj_rotated_vertical_slicing/row_1"
    )


