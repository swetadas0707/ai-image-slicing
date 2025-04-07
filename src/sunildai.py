import cv2
import numpy as np
import os

def enhanced_split_image_by_grid(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect lines using probabilistic Hough Transform
    min_line_length = min(img.shape[0], img.shape[1]) // 2
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=100, 
                           minLineLength=min_line_length, maxLineGap=10)
    
    # Separate horizontal and vertical lines
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
    def merge_lines(lines, threshold=10, is_horizontal=True):
        if not lines:
            return []
        
        lines.sort()
        merged = [lines[0]]
        
        for current in lines[1:]:
            last = merged[-1]
            if is_horizontal:
                # Check if current line is close to the last merged line
                if abs(current[0] - last[0]) < threshold:
                    # Merge them by averaging
                    new_line = (
                        (current[0] + last[0]) // 2,
                        (current[1] + last[1]) // 2
                    )
                    merged[-1] = new_line
                else:
                    merged.append(current)
            else:
                if abs(current[0] - last[0]) < threshold:
                    new_line = (
                        (current[0] + last[0]) // 2,
                        (current[1] + last[1]) // 2
                    )
                    merged[-1] = new_line
                else:
                    merged.append(current)
        
        return [l[0] for l in merged]  # Return just the coordinate
    
    horizontal = merge_lines(horizontal, threshold=20, is_horizontal=True)
    vertical = merge_lines(vertical, threshold=20, is_horizontal=False)
    
    if 0 not in horizontal:
        horizontal.insert(0, 0)
    if img.shape[0] not in horizontal:
        horizontal.append(img.shape[0])
    if 0 not in vertical:
        vertical.insert(0, 0)
    if img.shape[1] not in vertical:
        vertical.append(img.shape[1])
    
    horizontal = sorted(list(set(horizontal)))
    vertical = sorted(list(set(vertical)))
    
    # Split the image into grid cells
    for i in range(len(horizontal)-1):
        for j in range(len(vertical)-1):
            # Define the coordinates of the cell
            y1, y2 = horizontal[i], horizontal[i+1]
            x1, x2 = vertical[j], vertical[j+1]
            
            # Extract the cell
            cell = img[y1:y2, x1:x2]
            
            # Save the cell if it's not empty
            if cell.size > 0:
                cv2.imwrite(f"{output_folder}/cell_{i}_{j}.jpg", cell)
    
    print(f"Image split into {len(horizontal)-1}x{len(vertical)-1} pieces")

enhanced_split_image_by_grid("/home/sunilkarki/Pictures/amazon-sneakers.png", "puzzle_pieces")