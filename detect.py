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

def enhance_image_for_ocr(cell_image):
    # Convert to grayscale
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    # Applying CLAHE to enhance contrast more adaptively
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Apply a median filter to reduce noise while preserving edges
    median_filtered = cv2.medianBlur(contrast, 5)

    # Thresholding to create a binary image, invert it as most texts are dark
    _, binary = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to close small holes and gaps in text
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closing

def enhance_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Consider using dynamic kernel sizes based on the content of the image
    horizontal_kernel_length = max(20, int(image.shape[1] / 25))  # Slightly smaller to capture wider columns
    vertical_kernel_length = max(20, int(image.shape[0] / 25))

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))

    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.morphologyEx(combined_lines, cv2.MORPH_CLOSE, kernel, iterations=3)
    return processed_img

def perform_ocr_on_cell(cell_image, numeric=False):
    # Enhance the image specifically tailored for OCR
    enhanced_image = enhance_image_for_ocr(cell_image)

    # Using different PSM modes based on whether we expect numeric or mixed content
    config = r'--oem 3 --psm 6'  # Assume a single uniform block of text
    if numeric:
        config = r'--oem 3 --psm 8 outputbase digits'  # Optimized for numeric extraction

    # Perform OCR using Pytesseract with the specified configuration
    text = pytesseract.image_to_string(enhanced_image, config=config)
    return format_continuous_text(text)

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
    x, y, w, h = cv2.boundingRect(contour)
    if w > 0 and h > 0 and (w/h < 15 and h/w < 15):  # Ensure reasonable dimensions
        return (x, y, w, h)
    return None  # This will prevent malformed tuples

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

def crop_image_with_padding(cell_image, padding=10):
    # Check if the image is smaller than the padding size
    if cell_image.shape[0] <= 2*padding or cell_image.shape[1] <= 2*padding:
        return cell_image  # Return the original if too small to pad
    return cell_image[padding:-padding, padding:-padding]

def process_image_for_table_detection(image):
    if image is None or image.size == 0:
        print("Empty or None image passed to process_image_for_table_detection.")
        return [], None

    # Apply line enhancement
    processed_img = enhance_lines(image)
    if processed_img is None:
        print("Failed to enhance image lines.")
        return [], None

    # Retrieve contours with hierarchy information
    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    detected_cells = []

    # Check for valid hierarchy to process contours
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Get the actual hierarchy array
        for i, contour in enumerate(contours):
            # Filter to include only child contours (assuming cells are children of table contours)
            if hierarchy[i][3] != -1:  # Has a parent contour
                bounding_box = get_valid_bounding_box(contour)
                if bounding_box:
                    detected_cells.append(bounding_box)
                    # Optionally draw each bounding box on the image for visualization
                    x, y, w, h = bounding_box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    print(f"Invalid bounding box for contour with area {cv2.contourArea(contour)}")

    # Display the processed image with detected cells for verification
    if detected_cells:
        print(f"Detected {len(detected_cells)} cells.")
        cv2.imshow("Detected Cells", image)
        cv2.waitKey(0)  # Wait for a key press to close the display window
        cv2.destroyAllWindows()
    else:
        print("No tables detected.")

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

def extract_and_process_cell(image, bounding_box, numeric=False):
    x, y, w, h = bounding_box
    # Extract the cell using the bounding box and apply padding
    cell_image = image[y:y+h, x:x+w]
    cropped_image = crop_image_with_padding(cell_image)
    
    # Perform OCR on the cropped image
    cell_text = perform_ocr_on_cell(cropped_image, numeric=numeric)
    return cell_text

def format_continuous_text(text):
    return ' '.join(text.split())

def extract_table_data(image, detected_cells):
    table_data = []
    padding = 5  # Padding to avoid reading borders

    for cell in detected_cells:
        x, y, w, h = cell
        # Apply padding, ensuring we do not exceed image boundaries
        x_padded, y_padded = max(0, x-padding), max(0, y-padding)
        w_padded, h_padded = min(image.shape[1] - x_padded, w + 2*padding), min(image.shape[0] - y_padded, h + 2*padding)
        cell_image = image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]

        # Perform OCR on each cell
        cell_text = perform_ocr_on_cell(cell_image)
        table_data.append(cell_text)

    return table_data

def save_to_excel(table_data, base_filename="output"):
    """Save the table data to an Excel file, ensuring no unintended header row."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_filename}_{current_time}.xlsx"
    df = pd.DataFrame(table_data)  # Confirming that no header row is formed from indices
    df.to_excel(filename, index=False, header=None)  # Ensure no header is used
    print(f"Data exported to Excel file {filename}")

def select_pdf_and_convert():
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
