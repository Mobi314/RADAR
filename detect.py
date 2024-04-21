import cv2
import numpy as np
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import os

def convert_pdf_to_image(pdf_path):
    """Convert a PDF page to an image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # assuming single-page documents
    mat = fitz.Matrix(300 / 72, 300 / 72)  # High resolution
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save("temp_page_image.png")
    return "temp_page_image.png"

def preprocess_image(image_path):
    """Enhance image for contour detection."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use bilateral filter for edge-preserving smoothing.
    smooth = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edged = cv2.Canny(smooth, 30, 200)
    # Dilate to close gaps between object edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    return dilated, gray

def find_and_draw_contours(processed_img, original_img):
    """Detect and draw contours based on processed image."""
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:  # Assuming rectangular tables and cells
            x, y, w, h = cv2.boundingRect(cnt)
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
