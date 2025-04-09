import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load image
image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/listing_gray.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adaptive threshold for better contrast on light background
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)

# Morph to isolate horizontal row-like areas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 25))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find row contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter + sort top to bottom
rows = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if h > 60 and w > 200:  # Looser bounds
        rows.append((x, y, w, h))

rows = sorted(rows, key=lambda b: b[1])

# Save rows and show on image
output_dir = "note_items"
os.makedirs(output_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(rows):
    row_crop = image[y:y+h, x:x+w]
    cv2.imwrite(f"{output_dir}/row_{i}.png", row_crop)

# Visualization
annotated = image.copy()
for (x, y, w, h) in rows:
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)

plt.figure(figsize=(12, 10))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title("Detected Rows in Book Listing Page")
plt.axis("off")
plt.show()

print(f"✅ Detected {len(rows)} rows. Crops saved to: {output_dir}")
