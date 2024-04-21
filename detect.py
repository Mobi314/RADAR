import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import datetime
import os

def convert_pdf_to_image(pdf_path):
    """Convert PDF file to a high-resolution image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # assuming single-page documents
    mat = fitz.Matrix(300 / 72, 300 / 72)  # Render at 300 DPI
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save("temp_page_image.png")
    return "temp_page_image.png"

def preprocess_image(image_path):
    """Load image and convert to grayscale if necessary, then enhance for better OCR."""
    image = cv2.imread(image_path)
    if image.ndim == 3:  # Check if the image is colored
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated, gray

def find_contours(processed_image):
    """Find contours on the processed image."""
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def display_and_filter_contours(original_image, contours):
    """Display contours and filter based on size and aspect ratio."""
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 1000 and 20 < h < 600:  # Example size filters
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            valid_contours.append(contour)
    cv2.imshow("Detected Tables and Cells", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return valid_contours

def extract_text_from_cells(gray_image, contours):
    """Extract cells as images, enhance, and perform OCR."""
    data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cell_image = gray_image[y:y+h, x:x+w]
        enhanced_image = Image.fromarray(cell_image).resize((w*3, h*3), Image.BICUBIC)
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(2.0)
        text = pytesseract.image_to_string(enhanced_image, lang='eng')
        data.append(text.strip())
    return data

def save_to_excel(data):
    """Save data to an Excel file with timestamp."""
    df = pd.DataFrame([data], columns=['Extracted Text'])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_excel(f"extracted_data_{timestamp}.xlsx", index=False)
    print("Data exported to Excel with timestamp.")

def run_ocr_process():
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print("No file selected.")
        return

    image_path = convert_pdf_to_image(pdf_path)
    processed_image, gray_image = preprocess_image(image_path)
    contours = find_contours(processed_image)
    valid_contours = display_and_filter_contours(gray_image, contours)
    data = extract_text_from_cells(gray_image, valid_contours)
    save_to_excel(data)
    os.remove(image_path)  # Clean up the page image

if __name__ == "__main__":
    run_ocr_process()
