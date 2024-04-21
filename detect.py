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
import io

def enhance_image_for_ocr(image):
    """Enhance the image for better OCR recognition."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def enhance_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a combination of Gaussian Blur and adaptive threshold
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Modify kernel size based on expected cell size, these values might need tuning
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine the horizontal and vertical lines
    combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    # Additional morphological closing can help close gaps in lines
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.morphologyEx(combined_lines, cv2.MORPH_CLOSE, kernel, iterations=3)
    return processed_img

def perform_ocr_on_cell(cell_image):
    """Perform OCR on the provided cell image using PyTesseract."""
    enhanced_image = enhance_image_for_ocr(cell_image)  # Assuming this returns a binary image ready for OCR
    text = pytesseract.image_to_string(enhanced_image, config='--psm 6')
    return format_continuous_text(text)  # Cleans up the text

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
        page = doc.load_page(0)  # load the first page
    except IndexError:
        print("Page index out of range.")
        doc.close()
        return None

    zoom = 2.5
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.array(Image.open(io.BytesIO(pix.tobytes())))
    doc.close()

    if img.size == 0:
        print("Image loading failed, empty image array.")
        return None
    return img

def safe_open_image(path):
    """Context manager for safely opening and closing images."""
    img = Image.open(path)
    try:
        yield img
    finally:
        img.close()

def get_valid_bounding_box(contour):
    """Calculate and validate the bounding box of a contour."""
    x, y, w, h = cv2.boundingRect(contour)
    # You can add any additional validation or transformation here if needed
    if w > 0 and h > 0:  # Simple validation to ensure width and height are positive
        return (x, y, w, h)
    return None  # Return None if the bounding box is not valid

def safe_tuple_access(tpl, default=(0, 0, 0, 0)):
    """Safely access tuple elements, returning a default if not possible."""
    try:
        x, y, w, h = tpl
        return (x, y, w, h)
    except ValueError:
        print(f"Warning: Received malformed tuple {tpl}, using default {default}.")
        return default

def append_cell_if_valid(contour, detected_cells):
    bounding_box = get_valid_bounding_box(contour)
    assert len(bounding_box) == 4, "Bounding box must have exactly four elements"
    detected_cells.append(bounding_box)

def validate_and_append_cell(contour, detected_cells):
    if cv2.contourArea(contour) > 100:
        bounding_box = get_valid_bounding_box(contour)
        if bounding_box:
            detected_cells.append(bounding_box)
            print(f"Appending valid bounding box: {bounding_box}")
        else:
            print(f"Invalid or incomplete bounding box derived from contour.")

def process_image_for_table_detection(image):
    processed_img = enhance_lines(image)
    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_cells = []

    # Optionally, visualize the processed image for debugging
    cv2.imshow("Processed Image for Table Detection", processed_img)
    cv2.waitKey(0)

    for contour in contours:
        # Consider adding more sophisticated checks based on the contour properties
        if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 5000:  # Adjust area range as needed
            x, y, w, h = cv2.boundingRect(contour)
            if 0.5 < w/h < 2:  # Optional: Check for aspect ratio to filter out non-cell-like contours
                detected_cells.append((x, y, w, h))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detected Cells", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_cells, image

def classify_cells(detected_cells):
    rows = {}
    for cell in detected_cells:
        x, y, w, h = safe_tuple_access(cell)  # Safely unpacks tuple
        if w == 0 or h == 0:
            print(f"Skipping malformed cell: {cell}")
            continue  # Skip this cell if dimensions are zero

        row_found = False
        for key in rows.keys():
            if abs(key - y) < 10:  # Grouping cells in the same row
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

def format_continuous_text(text):
    return ' '.join(text.split())

def extract_table_data(image, detected_cells):
    table_data = []
    for cell in detected_cells:
        x, y, w, h = safe_tuple_access(cell)
        if w > 0 and h > 0:
            cell_image = image[y:y+h, x:x+w]
            cell_text = perform_ocr_on_cell(cell_image)
            table_data.append(cell_text)
        else:
            print(f"Skipped OCR on invalid cell with dimensions: {w}x{h}")
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
    if image_path:
        try:
            os.remove(image_path)
            print(f"File {image_path} successfully deleted.")
        except Exception as e:
            print(f"Error deleting file {image_path}: {e}")
    else:
        print("No valid image path provided for deletion.")

if __name__ == "__main__":
    pdf_path = select_pdf_and_convert()
