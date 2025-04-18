import cv2
import numpy as np
import os

def extract_grid_items(image_path, output_folder="grid_items"):
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Edge detection

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    item_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 50:  # Ignore small contours (could be noise)
            continue
        item_boxes.append((x, y, w, h))

    # Sort boxes top to bottom, then left to right
    item_boxes = sorted(item_boxes, key=lambda b: (b[1] // 100, b[0]))  # Sorting by row and column

    # Print bounding box info for debugging
    for i, (x, y, w, h) in enumerate(item_boxes):
        print(f"Item {i}: x={x}, y={y}, w={w}, h={h}")


    # Calculate the average area of bounding boxes
    areas = [w * h for _, _, w, h in item_boxes]
    avg_area = np.mean(areas)
    std_area = np.std(areas)

    # Set a threshold for outlier areas based on the standard deviation from the average
    area_threshold = avg_area + 2 * std_area  # 2 standard deviations away from the mean

    # Filter out bounding boxes whose area is above the threshold
    filtered_boxes = []
    for (x, y, w, h) in item_boxes:
        area = w * h
        if area < area_threshold:  # Exclude outliers based on area
            filtered_boxes.append((x, y, w, h))

    # Crop and save each item
    for i, (x, y, w, h) in enumerate(filtered_boxes):
        crop = image[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(output_folder, f"item_{i}.png"), crop)

    # Optional: Save debug overlay
    debug = image.copy()
    for (x, y, w, h) in filtered_boxes:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "debug_overlay.png"), debug)

    print(f"✅ Extracted {len(filtered_boxes)} individual items from grid.")




if __name__ == "__main__":
    # image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/bts_black_frame.jpeg"  # Replace with your image path
    # extract_grid_items(image_path, output_folder="/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/bts_black_frame_1")

    # image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/medium.jpeg"
    # extract_grid_items(
    #     image_path=image_path,
    #     output_folder="/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/medium"
    # )

    image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/screenshot - 2025-04-17T181700.688.jpeg"
    extract_grid_items(
        image_path=image_path,
        output_folder="/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/test1"
    )