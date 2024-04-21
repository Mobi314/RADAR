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
    return dilated, gray  # Return both processed and original grayscale image for further use

def display_image_with_contours(image, contours):
    """Display the image with detected contours."""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter out irrelevant contours
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Detected Tables and Cells", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_pdf_to_image(pdf_path):
    """Convert a PDF file to an image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # assuming single-page documents
    mat = fitz.Matrix(300/72, 300/72)  # Render at 300 DPI
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save("temp_page_image.png")
    return "temp_page_image.png"

def extract_and_process_cells(image_path):
    """Extract cells from the image and process them for OCR."""
    image = cv2.imread(image_path)
    processed, gray = enhance_lines(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    data = []
    headers = []
    first_row = True

    display_image_with_contours(gray, contours)  # Display image with detected cells outlined

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter out irrelevant contours
            cell_image = gray[y:y+h, x:x+w]
            cell_image = Image.fromarray(cell_image)
            cell_image = cell_image.resize((w*3, h*3), Image.BICUBIC)  # Enlarge image for better OCR
            enhancer = ImageEnhance.Contrast(cell_image)
            enhanced_image = enhancer.enhance(2.0)
            text = pytesseract.image_to_string(enhanced_image, lang='eng')

            if first_row:
                headers.append(text.strip())
            else:
                data.append(text.strip())

            # Display each cell image briefly for verification
            cv2.imshow("Cell Image", np.array(enhanced_image))
            cv2.waitKey(500)  # Display each cell image for 500 ms

    cv2.destroyAllWindows()
    return headers if first_row else data

def save_to_excel(headers, data):
    """Save extracted data to an Excel file with appropriate headers."""
    df = pd.DataFrame([data], columns=headers)
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
    headers, data = extract_and_process_cells(image_path)
    save_to_excel(headers, data)
    os.remove(image_path)  # Clean up the page image

if __name__ == "__main__":
    run_ocr_process()
