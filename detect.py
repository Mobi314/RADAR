import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import datetime
import os

def enhance_lines(image):
    """Enhance lines in the image to help with cell detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # Create a rectangular kernel for extracting lines
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def convert_pdf_to_image(pdf_path):
    """Convert a PDF file to an image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # assuming single-page documents
    mat = fitz.Matrix(300/72, 300/72)  # Render at 300 DPI
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save("temp_page_image.png")
    return "temp_page_image.png"

def extract_cells_from_image(image_path):
    """Extract individual cells from the table."""
    image = cv2.imread(image_path)
    processed = enhance_lines(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter out irrelevant contours
            cell_image = image[y:y+h, x:x+w]
            cv2.imwrite(f"temp_cell_{x}_{y}.png", cell_image)
            cells.append((f"temp_cell_{x}_{y}.png", x, y, w, h))
    return cells

def ocr_and_cleanup_cells(cells):
    """Perform OCR on each cell and clean up images."""
    results = []
    for cell_path, x, y, w, h in cells:
        cell_image = Image.open(cell_path)
        enhanced = ImageEnhance.Contrast(cell_image).enhance(2.0)
        text = pytesseract.image_to_string(enhanced, lang='eng')
        results.append((x, y, text))
        os.remove(cell_path)  # Remove the temporary image file
    return results

def save_to_excel(data):
    """Save extracted data to an Excel file."""
    df = pd.DataFrame(data, columns=['X', 'Y', 'Text'])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_excel(f"extracted_data_{timestamp}.xlsx", index=False)
    print("Data exported to Excel.")

def run_ocr_process():
    """Main function to run the OCR process."""
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print("No file selected.")
        return

    image_path = convert_pdf_to_image(pdf_path)
    cells = extract_cells_from_image(image_path)
    data = ocr_and_cleanup_cells(cells)
    save_to_excel(data)
    os.remove(image_path)  # Clean up the page image

if __name__ == "__main__":
    run_ocr_process()
