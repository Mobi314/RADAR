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
    page = doc.load_page(0)
    mat = fitz.Matrix(300 / 72, 300 / 72)  # Increase resolution for better OCR
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save("temp_page_image.png")
    return "temp_page_image.png"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)  # Dilate to connect components
    return dilated, gray

def find_and_draw_contours(processed_img, original_img):
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 50 < w < 1000 and 20 < h < 600:  # Filter to approximate expected cell size
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Draw in vibrant green
            valid_contours.append((x, y, w, h))
    cv2.imshow("Detected Table and Cells", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return valid_contours

def ocr_cells(gray, contours):
    data = []
    for x, y, w, h in contours:
        cell_image = gray[y:y+h, x:x+w]
        resized_image = Image.fromarray(cell_image).resize((w*3, h*3), Image.BICUBIC)
        enhancer = ImageEnhance.Contrast(resized_image)
        enhanced_image = enhancer.enhance(4.0)  # Significantly enhance contrast
        text = pytesseract.image_to_string(enhanced_image, config='--psm 6')  # Assume a single uniform block of text
        data.append(text.strip())
    return data

def save_to_excel(data, headers):
    df = pd.DataFrame([data], columns=headers)
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
    processed_img, gray = preprocess_image(image_path)
    contours = find_and_draw_contours(processed_img, gray)
    data = ocr_cells(gray, contours)
    headers = data[0] if data else ["Data"]  # Assume the first row contains headers or default to 'Data'
    save_to_excel(data[1:], headers)  # Exclude headers from data rows
    os.remove(image_path)

if __name__ == "__main__":
    run_ocr_process()
