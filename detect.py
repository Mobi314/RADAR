import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from datetime import datetime
import os
import tempfile

def enhance_image_for_ocr(image):
    """Enhance the image for better OCR recognition."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def enhance_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    horizontal_kernel_length = max(20, int(image.shape[1] / 30))
    vertical_kernel_length = max(20, int(image.shape[0] / 30))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.morphologyEx(combined_lines, cv2.MORPH_CLOSE, kernel, iterations=3)
    return processed_img

def convert_pdf_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    if not doc.page_count:  # Check if the document contains any pages
        print("No pages found in document.")
        doc.close()
        return None

    try:
        page = doc.load_page(0)  # Attempt to load the first page
    except IndexError:
        print("Page index out of range.")
        doc.close()
        return None

    zoom = 2.5
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))  # Convert to an array directly

    doc.close()
    return img

def safe_open_image(path):
    """Context manager for safely opening and closing images."""
    img = Image.open(path)
    try:
        yield img
    finally:
        img.close()

def process_image_for_table_detection(image):
    processed_img = enhance_lines(image)
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_cells = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out too small contours
            x, y, w, h = cv2.boundingRect(contour)
            detected_cells.append((x, y, w, h))  # Ensure always four elements

    return detected_cells, image  # Return both the cells and the image for further processing

def classify_cells(detected_cells):
    rows = {}
    for cell in detected_cells:
        if len(cell) == 4:
            x, y, w, h = cell  # This unpacks four elements
        else:
            print(f"Detected cell tuple has incorrect format: {cell}")
            continue  # Skip this iteration to avoid the error

        row_found = False
        for key in rows.keys():
            if abs(key - y) < 10:  # 10 is a threshold for grouping cells in the same row
                rows[key].append((x, y, w, h))
                row_found = True
                break
        if not row_found:
            rows[y] = [(x, y, w, h)]

    sorted_rows = []
    for y in sorted(rows.keys()):
        sorted_cells = sorted(rows[y], key=lambda cell: cell[0])
        sorted_rows.append(sorted_cells)
    return sorted_rows

    # Sort rows and cells within rows
    sorted_rows = []
    for y in sorted(rows.keys()):
        sorted_cells = sorted(rows[y], key=lambda cell: cell[0])  # Sort cells in a row by x-coordinate
        sorted_rows.append(sorted_cells)
    return sorted_rows

def format_continuous_text(text):
    return ' '.join(text.split())

def extract_table_data(image, detected_cells):
    table_data = []
    for x, y, w, h in detected_cells:
        cell_image = image[y:y+h, x:x+w]
        enhanced_image = enhance_image_for_ocr(cell_image)
        config = '--oem 1 --psm 6'
        cell_text = pytesseract.image_to_string(enhanced_image, config=config)
        cell_text = ' '.join(cell_text.split())  # Normalize the spacing
        table_data.append(cell_text)
    return table_data

def save_to_excel(table_data, base_filename="output"):
    """Save the extracted table data to an Excel file."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_filename}_{current_time}.xlsx"
    df = pd.DataFrame([table_data])  # Adjust as needed for row/column format
    df.to_excel(filename, index=False)
    print(f"Data exported to Excel file {filename}")

def select_pdf_and_convert():
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        return
    image_path = convert_pdf_to_image(pdf_path)
    detected_cells = process_image_for_table_detection(image_path)
    sorted_rows = classify_cells(detected_cells)
    if sorted_rows:
        table_data = extract_table_data(image_path, sorted_rows)
        save_to_excel(table_data)
    else:
        print("No tables detected.")
    cv2.destroyAllWindows()  # Close all OpenCV windows

def remove_image_file(image_path):
    """Remove an image file with more robust handling."""
    try:
        os.remove(image_path)
        print(f"File {image_path} successfully deleted.")
    except PermissionError as e:
        print(f"Error deleting file {image_path}: {str(e)}")

if __name__ == "__main__":
    pdf_path = select_pdf_and_convert()
    image_path = convert_pdf_to_image(pdf_path)
    detected_cells, image = process_image_for_table_detection(image_path)
    sorted_rows = classify_cells(detected_cells)
    table_data = extract_table_data(image, sorted_rows)
    if sorted_rows:
        table_data = extract_table_data(image_path, sorted_rows)
        save_to_excel(table_data)
    remove_image_file(image_path)
