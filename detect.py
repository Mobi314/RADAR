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
    if len(image.shape) == 3:  # Check if the image is colored
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Image is already grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated, gray  # Returning both processed and original grayscale image for contour extraction

def display_image_with_contours(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter contours that likely represent cells (modify the criteria as needed)
        if w > 50 and h > 20 and w < 1000:  # Adjust width and height constraints to better fit table cells
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Detected Tables and Cells", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_and_process_cells(image_path, gray):
    """Extract cells from the image and process them for OCR."""
    processed, gray = enhance_lines(gray)  # Make sure gray is correctly passed
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    data = []

    display_image_with_contours(gray, contours)  # Ensure gray image is used here

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:  # Filter out irrelevant contours
            cell_image = gray[y:y+h, x:x+w]
            cell_image = Image.fromarray(cell_image)
            cell_image = cell_image.resize((w*3, h*3), Image.BICUBIC)  # Enlarge image for better OCR
            enhancer = ImageEnhance.Contrast(cell_image)
            enhanced_image = enhancer.enhance(2.0)
            text = pytesseract.image_to_string(enhanced_image, lang='eng')
            data.append(text.strip())

    return data

def run_ocr_process():
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print("No file selected.")
        return

    image_path = convert_pdf_to_image(pdf_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    data = extract_and_process_cells(image_path, gray)

    if not data or not any(data):  # Check if data is empty or all values are empty
        print("No data extracted from OCR.")
        return

    # Assuming the first row contains headers
    headers = data[0] if data else []
    if not headers:  # Check if headers are empty
        print("Header information is missing.")
        return

    # Create DataFrame and save to Excel
    try:
        df = pd.DataFrame(data[1:], columns=headers)  # Exclude the first row which is assumed to be headers
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_excel(f"extracted_data_{timestamp}.xlsx", index=False)
        print("Data exported to Excel.")
    except Exception as e:
        print(f"Error creating or saving DataFrame: {e}")

    os.remove(image_path)  # Clean up the page image

def save_to_excel(headers, data):
    df = pd.DataFrame(data, columns=headers)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_excel(f"extracted_data_{timestamp}.xlsx", index=False)
    print("Data exported to Excel with timestamp.")

if __name__ == "__main__":
    run_ocr_process()
