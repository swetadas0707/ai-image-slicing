# **<R&D Project>**

## **AI-based Image Segmentation**

This project is aimed to perform screenshot slicing for a listing page (e.g., a page showing multiple product cards, articles, or listings) and extract each individual item as an image. Here, we are going DOM-less that is, we want to slice images without using the DOM, just raw screenshots. For it to happen, we'd need to use image 
processing (OpenCV or AI-based models) to detect repeating patterns or boxes. 

### 🧪 **Listing Screenshot Slicing via OpenCV**
1. Load screenshot
2. Convert to grayscale
3. Apply edge detection / thresholding
4. Find contours / bounding boxes
5. Filter bounding boxes (e.g., size, alignment)
6. Crop and save individual item images


Detailed Analysis and Findings: https://www.notion.so/R-D-AI-Image-Slicing-1ce456391e80809490a3fe69837540bb?pvs=4