import cv2
import matplotlib.pyplot as plt

# Load color image
image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/listing.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save grayscale image
gray_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/pics/listing_gray.png"
cv2.imwrite(gray_path, gray)

# Show the result
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Conversion")
plt.axis('off')
plt.show()

print(f"✅ Grayscale image saved to: {gray_path}")
