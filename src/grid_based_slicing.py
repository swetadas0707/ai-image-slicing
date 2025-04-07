import cv2
import numpy as np
import os

def merge_lines(lines, threshold=10):
    lines.sort()
    merged = [lines[0]]
    for current in lines[1:]:
        last = merged[-1]
        if abs(current[0] - last[0]) < threshold:
            avg = (current[0] + last[0]) // 2
            merged[-1] = (avg, avg)
        else:
            merged.append(current)
    return [l[0] for l in merged]


def slice_rows_by_visual_grid(image_path, output_folder="row_slices_grid"):
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Adaptive threshold to enhance contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Step 2: Hough line transform to find horizontal lines
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=image.shape[1] // 2,
        maxLineGap=10
    )

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # It's a horizontal line
                horizontal_lines.append((min(y1, y2), max(y1, y2)))

    # Step 3: Merge nearby lines
    merged_lines = merge_lines(horizontal_lines, threshold=15)

    # Ensure full image bounds are included
    if 0 not in merged_lines:
        merged_lines.insert(0, 0)
    if image.shape[0] not in merged_lines:
        merged_lines.append(image.shape[0])

    merged_lines = sorted(set(merged_lines))

    # Step 4: Create debug overlay
    overlay = image.copy()
    for y in merged_lines:
        cv2.line(overlay, (0, y), (image.shape[1], y), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_folder, "grid_overlay.png"), overlay)

    # Step 5: Crop between line pairs
    row_paths = []
    for i in range(len(merged_lines) - 1):
        y1, y2 = merged_lines[i], merged_lines[i+1]
        if y2 - y1 < 50:
            continue  # skip tiny rows
        row_crop = image[y1:y2, :]
        row_path = os.path.join(output_folder, f"row_{i}.png")
        cv2.imwrite(row_path, row_crop)
        row_paths.append(row_path)

    print(f"✅ Detected and saved {len(row_paths)} rows based on visual grid.")



if __name__ == "__main__":
    slice_rows_by_visual_grid("/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/listing.png", output_folder="grid_slices_3")
