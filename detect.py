import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
import pandas as pd

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
    page = doc.load_page(0)  # Only the first page
    zoom = 2.5
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    image_path = "page_0.png"
    pix.save(image_path)
    return image_path

def process_image_for_table_detection(image_path):
    image = cv2.imread(image_path)
    processed_img = enhance_lines(image)
    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    detected_cells = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            x, y, w, h = cv2.boundingRect(contour)
            detected_cells.append((x, y, w, h))
    return detected_cells

def classify_cells(detected_cells):
    # Group cells by rows based on y-coordinate proximity
    rows = {}
    for cell in detected_cells:
        x, y, w, h = cell
        row_found = False
        for key in rows.keys():
            if abs(key - y) < 10:  # 10 is a threshold for grouping cells in the same row
                rows[key].append((x, y, w, h))
                row_found = True
                break
        if not row_found:
            rows[y] = [(x, y, w, h)]

    # Sort rows and cells within rows
    sorted_rows = []
    for y in sorted(rows.keys()):
        sorted_cells = sorted(rows[y], key=lambda cell: cell[0])  # Sort cells in a row by x-coordinate
        sorted_rows.append(sorted_cells)
    return sorted_rows

def format_continuous_text(text):
    return ' '.join(text.split())

def extract_table_data(image_path, sorted_rows):
    image = Image.open(image_path)
    table_data = []
    for row in sorted_rows:
        row_data = []
        for x, y, w, h in row:
            cell_image = image.crop((x, y, x + w, y + h))
            config = '--oem 1 --psm 6'
            cell_text = pytesseract.image_to_string(cell_image, config=config)
            cell_text = format_continuous_text(cell_text)
            row_data.append(cell_text)
        table_data.append(row_data)
    return table_data

def save_to_excel(table_data, output_file="output.xlsx"):
    df = pd.DataFrame(table_data)
    if len(df.columns) > 0:  # Optionally set the first row as header if it's indeed headers
        df.columns = df.iloc[0]
        df = df[1:]
    df.to_excel(output_file, index=False)
    print(f"Data exported to Excel file {output_file}")

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

if __name__ == "__main__":
    select_pdf_and_convert()
