import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import os

def convert_pdf_to_image(pdf_path):
    """Convert a PDF page to an image with high resolution."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # Adjust if multiple pages need to be processed
    mat = fitz.Matrix(300 / 72, 300 / 72)  # Increasing resolution for clearer images
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save("temp_page_image.png")
    return "temp_page_image.png"

def preprocess_image(image_path):
    """Apply advanced preprocessing to enhance the image for contour detection."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur followed by adaptive thresholding
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Use morphological operations to expose the structure of tables and cells
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated, gray

def find_and_draw_contours(processed_img, original_img):
    """Detect contours from processed image and draw them on the original image."""
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Use perimeter and area filters to refine contour detection
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 100:  # Filter out small contours
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:  # Looking for quadrilateral contours which could be tables or cells
                x, y, w, h = cv2.boundingRect(approx)
                if 100 < w < 1000 and 20 < h < 600:  # Further size filtering to ensure suitability
                    cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw in green

    cv2.imshow("Detected Tables and Cells", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_ocr_process():
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print("No file selected.")
        return

    image_path = convert_pdf_to_image(pdf_path)
    processed_img, gray = preprocess_image(image_path)
    find_and_draw_contours(processed_img, gray)
    os.remove(image_path)

if __name__ == "__main__":
    run_ocr_process()
