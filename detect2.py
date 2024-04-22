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
    # Convert to grayscale using TensorFlow
    gray = tf.image.rgb_to_grayscale(cell_image)
    gray = gray.numpy().squeeze()

    # Dynamic CLAHE using OpenCV
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)

    # Use Otsu's method to apply binary threshold
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def convert_pdf_to_image(pdf_path):
    if not os.path.exists(pdf_path):
        print("PDF file does not exist.")
        return None
    
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        print("No pages found in document.")
        doc.close()
        return None

    try:
        page = doc.load_page(0)  # Load the first page
        zoom = 2.5
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes()))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
        doc.close()
        return img
    except IndexError:
        print("Page index out of range.")
        doc.close()
        return None

def perform_ocr_on_cell(cell_image):
    processed_image = enhance_image_for_ocr(cell_image)

    # OCR processing with Tesseract
    pil_image = Image.fromarray(processed_image)  # Convert to PIL Image for Tesseract
    text = pytesseract.image_to_string(pil_image, config='--psm 6')
    formatted_text = format_continuous_text(text)

    print(f"Extracted Text: {formatted_text}")
    return formatted_text

def format_continuous_text(text):
    # Normalize newlines, replace carriage returns with newline characters
    text = text.replace('\r', '\n')
    # Compact multiple newline characters into a single newline
    text = re.sub(r'\n+', '\n', text)
    # Correct common OCR misreads
    text = re.sub(r'\bIl\b', '11', text)  # Correct 'Il' to '11'
    text = re.sub(r'\bO\b', '0', text)    # Correct 'O' to '0' when isolated
    text = re.sub(r'(?<!\d)0(?!\d)', 'O', text)  # Correct '0' to 'O' when not surrounded by other digits
    # Remove non-printable characters except newlines
    text = ''.join(char for char in text if char in string.printable or char == '\n')
    # Trim whitespace around the text and between lines
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text

# Additional functions like `extract_table_data`, `save_to_excel`, `select_pdf_and_convert`, etc., remain the same.

if __name__ == "__main__":
    pdf_path = select_pdf_and_convert()
