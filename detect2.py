import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from datetime import datetime
import os
import io
import string
import re
import tensorflow as tf

def enhance_image_for_ocr(cell_image):
    """ Enhance the image for OCR by applying grayscale, CLAHE, and thresholding. """
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def convert_pdf_to_image(pdf_path):
    """ Convert the first page of a PDF to an image. """
    if not os.path.exists(pdf_path):
        print("PDF file does not exist.")
        return None
    
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        print("No pages found in document.")
        doc.close()
        return None

    page = doc.load_page(0)  # load the first page
    zoom = 2.5
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes()))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img

def perform_ocr_on_cell(cell_image):
    """ Perform OCR on the processed cell image using Tesseract. """
    processed_image = enhance_image_for_ocr(cell_image)
    pil_image = Image.fromarray(processed_image)
    text = pytesseract.image_to_string(pil_image, config='--psm 6')
    return format_continuous_text(text)

def extract_table_data(image, detected_cells):
    """ Extract data from detected cells and organize by rows and columns. """
    table_data = []
    for (x, y, w, h) in detected_cells:
        cell_image = image[y:y+h, x:x+w]
        cell_text = perform_ocr_on_cell(cell_image)
        table_data.append(cell_text)
    return table_data

def save_to_excel(table_data, base_filename="output"):
    """ Save the table data to an Excel file. """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_filename}_{current_time}.xlsx"
    df = pd.DataFrame(table_data)
    df.to_excel(filename, index=False, header=None)
    print(f"Data exported to Excel file {filename}")

def select_pdf_and_convert():
    """ Select a PDF file and convert it to an image, then perform OCR and save to Excel. """
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        return
    image = convert_pdf_to_image(pdf_path)
    detected_cells, _ = process_image_for_table_detection(image)
    if detected_cells:
        table_data = extract_table_data(image, detected_cells)
        save_to_excel(table_data)
    else:
        print("No tables detected.")
    cv2.destroyAllWindows()

def format_continuous_text(text):
    """ Clean and format the OCR output. """
    text = text.replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\bIl\b', '11', text)
    text = re.sub(r'\bO\b', '0', text)
    text = re.sub(r'(?<!\d)0(?!\d)', 'O', text)
    text = ''.join(char for char in text if char in string.printable or char == '\n')
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text

def process_image_for_table_detection(image):
    """ Process the image to detect cells which might contain tables or text. """
    processed_img = enhance_lines(image)
    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    detected_cells = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours):
            if hierarchy[i][3] != -1:  # Has a parent contour
                bounding_box = get_valid_bounding_box(contour)
                if bounding_box:
                    detected_cells.append(bounding_box)
    return detected_cells, image

def get_valid_bounding_box(contour):
    """ Validate and get bounding box for a contour. """
    x, y, w, h = cv2.boundingRect(contour)
    if w > 0 and h > 0 and (w/h < 15 and h/w < 15):
        return (x, y, w, h)
    return None

def enhance_lines(image):
    """ Enhance lines in the image to help with cell detection. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(image.shape[1] / 25)), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(image.shape[0] / 25))))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.morphologyEx(combined_lines, cv2.MORPH_CLOSE, kernel, iterations=3)
    return processed_img

if __name__ == "__main__":
    pdf_path = select_pdf_and_convert()
