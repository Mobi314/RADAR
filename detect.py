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
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # assuming single-page documents
    mat = fitz.Matrix(300 / 72, 300 / 72)  # Render at 300 DPI for better quality
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save("temp_page_image.png")
    return "temp_page_image.png"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh, gray

def find_and_draw_contours(processed_img, original_img):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter to avoid too small or too large contours
        if 50 < w < 1000 and 20 < h < 600:
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
            valid_contours.append((x, y, w, h))
    cv2.imshow("Detected Tables and Cells", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return valid_contours

def ocr_cells(gray, contours):
    data = []
    for x, y, w, h in contours:
        cell_image = gray[y:y+h, x:x+w]
        resized_image = Image.fromarray(cell_image).resize((w*3, h*3), Image.BICUBIC)
        enhancer = ImageEnhance.Contrast(resized_image)
        enhanced_image = enhancer.enhance(4.0)
        text = pytesseract.image_to_string(enhanced_image, config='--psm 6')
        data.append(text.strip())
    return data

def save_to_excel(data):
    if not data:  # Ensure there is data to write
        print("No data available to write to Excel.")
        return
    # Create DataFrame and handle possibility of data not fitting predefined columns
    try:
        df = pd.DataFrame([data], columns=['Extracted Text'])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_excel(f"extracted_data_{timestamp}.xlsx", index=False)
        print("Data exported to Excel with timestamp.")
    except Exception as e:
        print(f"Error creating or saving DataFrame: {e}")

def run_ocr_process():
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        print("No file selected.")
        return

    image_path = convert_pdf_to_image(pdf_path)
    processed_img, gray = preprocess_image(image_path)
    contours = find_and_draw_contours(processed_img, gray)
    data = ocr_cells(gray, contours)
    save_to_excel(data)
    os.remove(image_path)

if __name__ == "__main__":
    run_ocr_process()
