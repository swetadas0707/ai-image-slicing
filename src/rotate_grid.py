from PIL import Image
import cv2
import numpy as np
import os

# def rotate_image(image_path):
#     # Load the image
#     image = Image.open(image_path)  # Replace with your image path

#     # Rotate the image by 
#     # If the image has an alpha channel, split it out before inverting
#     rotated_image = image.rotate(-90, expand=True)

#     # Save the result
#     rotated_image.save("rotated_output.jpg")  # Save as JPG or any desired format

def merge_lines(lines, threshold=10):
    if not lines:
        return []
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

def vertical_slicing_via_rotation(image_path, output_folder="vertical_slices"):
    os.makedirs(output_folder, exist_ok=True)

    # Load and rotate image 90 degrees clockwise
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # Thresholding to highlight horizontal (former vertical) lines
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(os.path.join(output_folder, "rotated_thresh_debug.png"), thresh)

    # Detect horizontal lines (which are vertical lines in the original image)
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=rotated.shape[1] // 3,
        maxLineGap=10
    )

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:
                horizontal_lines.append((min(y1, y2), max(y1, y2)))

    merged_lines = merge_lines(horizontal_lines, threshold=15)

    # Ensure boundaries are included
    if 0 not in merged_lines:
        merged_lines.insert(0, 0)
    if rotated.shape[0] not in merged_lines:
        merged_lines.append(rotated.shape[0])

    merged_lines = sorted(set(merged_lines))

    # Overlay for debug
    overlay = rotated.copy()
    for y in merged_lines:
        cv2.line(overlay, (0, y), (rotated.shape[1], y), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "rotated_grid_overlay.png"), overlay)

    # Crop and rotate slices back to original orientation
    col_paths = []
    for i in range(len(merged_lines) - 1):
        y1, y2 = merged_lines[i], merged_lines[i+1]
        if y2 - y1 < 50:
            continue
        row_crop = rotated[y1:y2, :]
        restored = cv2.rotate(row_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        path = os.path.join(output_folder, f"col_{i}.png")
        cv2.imwrite(path, restored)
        col_paths.append(path)

    print(f"✅ Detected and saved {len(col_paths)} vertical slices (columns).")



if __name__=="__main__":

    vertical_slicing_via_rotation("/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/book_items/row_0.png")  # Replace with your image path