import cv2
import os
import numpy as np

def merge_lines(lines, threshold=10, is_horizontal=True):
    if not lines:
        return []
    
    lines.sort()
    merged = [lines[0]]

    for current in lines[1:]:
        last = merged[-1]
        if abs(current[0] - last[0]) < threshold:
            # Merge them by averaging
            new_line = (
                (current[0] + last[0]) // 2,
                (current[1] + last[1]) // 2
            )
            merged[-1] = new_line
        else:
            merged.append(current)

    return [l[0] for l in merged]  # Return just the coordinate


def grid_lines(img_path, output_dir="gen_pics", padding=5):
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found")
        return
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding (to highlight grid lines)
    thresh = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Hough line transform (to detect grid lines)
    min_line_length = min(img.shape[0], img.shape[1]) // 2
    lines = cv2.HoughLinesP(
        thresh,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=10
    )

    horizontal = []
    vertical = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # horizontal line
                horizontal.append((min(y1, y2), max(y1, y2)))
            else:  # vertical line
                vertical.append((min(x1, x2), max(x1, x2)))


    # Merge similar lines
    horizontal = merge_lines(horizontal, threshold=10, is_horizontal=True)
    vertical = merge_lines(vertical, threshold=10, is_horizontal=False)

    # # Warn if grid looks too small/large
    # if len(horizontal) <= 2 or len(vertical) <= 2:
    #     print("Warning: Detected very few grid lines. Check thresholds or image structure.")
    # if len(horizontal) > 50 or len(vertical) > 50:
    #     print("Warning: Detected many grid lines. May need more filtering.")


    # Visual grid overlay for debugging
    overlay = img.copy()
    for y in horizontal:
        cv2.line(overlay, (0, int(y)), (img.shape[1], int(y)), (0, 255, 0), 1)
    for x in vertical:
        cv2.line(overlay, (int(x), 0), (int(x), img.shape[0]), (255, 0, 0), 1)

    cv2.imwrite(os.path.join(output_dir, "grid_overlay.png"), overlay)
    print(len(horizontal))
    print(len(vertical))

    count = 0
    for i in range(len(horizontal) - 1):
        for j in range(len(vertical) - 1):
            y1, y2 = horizontal[i], horizontal[i + 1]
            x1, x2 = vertical[j], vertical[j + 1]

            # Apply padding
            y1 = max(0, y1 - padding)
            y2 = min(img.shape[0], y2 + padding)
            x1 = max(0, x1 - padding)
            x2 = min(img.shape[1], x2 + padding)

            cell = img[y1:y2, x1:x2]

            # Filter tiny cells
            # if cell.shape[0] < 30 or cell.shape[1] < 30:
            #     continue

            cv2.imwrite(os.path.join(output_dir, f"cell_{i}_{j}.jpg"), cell)
            count += 1


if __name__ == "__main__":
    
    slice("/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/amazon-sneakers.png")
