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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated, gray  # Returning the dilated image and the original gray image for contour extraction

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
    processed, _ = enhance_lines(gray)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    data = []

    display_image_with_contours(gray, contours)  # Display with overlays

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
    headers = data[0] if data else []  # Assuming the first row contains headers
    save_to_excel(headers, data[1:])  # Save all but the first row as data
    os.remove(image_path)  # Clean up the page image

def save_to_excel(headers, data):
    df = pd.DataFrame(data, columns=headers)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_excel(f"extracted_data_{timestamp}.xlsx", index=False)
    print("Data exported to Excel with timestamp.")

if __name__ == "__main__":
    run_ocr_process()
