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
import string
import re

def enhance_image_for_ocr(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # Slightly higher clipLimit
    contrast = clahe.apply(gray)
    
    # Applying a fixed global threshold might sometimes yield more consistent results
    _, binary = cv2.threshold(contrast, 120, 255, cv2.THRESH_BINARY_INV)  # Adjust threshold value based on sample images
    
    return binary

def is_text_thin(image):
    # Estimate text thickness based on the distribution of pixel values
    whites = np.sum(image > 128)
    blacks = np.sum(image <= 128)
    return whites / float(whites + blacks) > 0.5

def threshold_otsu(image):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    total = image.size
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = np.sum([i * hist[i] for i in range(256)])
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * ((mB - mF) ** 2)
        if between > maximum:
            maximum = between
            level = i
    return level

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

def perform_ocr_on_cell(cell_image):
    processed_image = enhance_image_for_ocr(cell_image)
    text = pytesseract.image_to_string(processed_image, config='--oem 3 --psm 6')

    # Show the processed image and close after 2 seconds
    cv2.imshow('Processed Cell Image', processed_image)
    cv2.waitKey(1000)  # Wait for 2000 milliseconds (2 seconds)
    cv2.destroyAllWindows()

    print(f"OCR Output: {text}")
    return text

def estimate_text_density(image):
    non_white_pixels = np.sum(image < 255)
    total_pixels = image.size
    return non_white_pixels / total_pixels

"""
def detect_content_type(image):
    # A simple approach based on the ratio of white to black pixels
    whites = np.sum(image == 255)
    blacks = np.sum(image == 0)
    if blacks / float(whites + blacks) > 0.5:  # More dense text regions might indicate numeric content
        return 'numeric'
    return 'alphanumeric'
"""
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

def extract_table_data(image, detected_cells):
    """Extract data from detected cells and organize by rows and columns."""
    table_data = []
    # Organize cells into rows based on their vertical alignment
    rows = {}
    for (x, y, w, h) in detected_cells:
        row_key = y // h  # This groups cells into rows by their y-coordinate
        if row_key in rows:
            rows[row_key].append((x, y, w, h))
        else:
            rows[row_key] = [(x, y, w, h)]

    # Sort rows and within each row sort cells by their x-coordinate to maintain left-to-right order
    sorted_rows = sorted(rows.items(), key=lambda item: item[0])
    for _, cells in sorted_rows:
        row_data = []
        for (x, y, w, h) in sorted(cells, key=lambda cell: cell[0]):
            cell_image = image[y:y+h, x:x+w]
            cell_text = perform_ocr_on_cell(cell_image)
            row_data.append(cell_text)
        table_data.append(row_data)

    return table_data

def clean_text(text):
    """Clean text by removing non-printable characters and trimming whitespace."""
    cleaned_text = text.strip().replace(u'\xa0', u' ')
    printable = set(string.printable)
    cleaned_text = ''.join(filter(lambda x: x in printable, cleaned_text))
    return cleaned_text

def save_to_excel(table_data, base_filename="output"):
    """Save the table data to an Excel file, ensuring no unintended header row."""
    cleaned_data = []
    for row in table_data:
        cleaned_row = [clean_text(str(cell)) for cell in row]  # Convert all cells to strings after cleaning
        cleaned_data.append(cleaned_row)

    print("Cleaned Data for Excel:")
    for row in cleaned_data:
        print(row)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_filename}_{current_time}.xlsx"
    df = pd.DataFrame(cleaned_data)
    print("DataFrame Head:", df.head())

    try:
        df.to_excel(filename, index=False, header=False)
        print(f"Data exported to Excel file {filename}")
    except Exception as e:
        print(f"Error writing to Excel: {e}")

# Test exporting a simple DataFrame to ensure Excel compatibility
def test_excel_output():
    print("Testing basic DataFrame export...")
    df_test = pd.DataFrame({'Numbers': [1, 2, 3], 'Letters': ['A', 'B', 'C']})
    try:
        df_test.to_excel("test_output.xlsx", index=False)
        print("Test Excel file created successfully.")
    except Exception as e:
        print("Failed to create test Excel file:", e)

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
    #test_excel_output()
