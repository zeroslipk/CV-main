import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from crop import crop_barcode

def display_image(img, title):
    """
    Displays an image using Matplotlib.
    """
    if img is None or img.size == 0:
        print(f"Cannot display '{title}'. Image is empty or None.")
        return
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_image(image_path):
    """
    Preprocess the image: apply noise reduction, resize, binarize, detect obstructions, and reconstruct barcode.
    """
    # Step 1: Load the image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Step 2: Apply a median filter to reduce noise
    kernel = np.ones((7, 1), np.float32) / 5
    median_blurred_img = cv.filter2D(img, -1, kernel)

    # Step 3: Resize the image for better feature visibility
    scale_factor = 4
    resized_img = cv.resize(
        median_blurred_img,
        (median_blurred_img.shape[1] * scale_factor, median_blurred_img.shape[0] * scale_factor)
    )

    # Step 4: Use adaptive thresholding for robust binarization
    binary_img = cv.adaptiveThreshold(
        resized_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2
    )

    # Step 5: Detect edges to identify the barcode region
    edges = cv.Canny(binary_img, 50, 150)

    # Step 6: Detect and mask obstructions
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_img)
    obstruction_detected = False
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if h > resized_img.shape[0] // 4:  # Assume obstructions are tall compared to the barcode
            cv.drawContours(mask, [contour], -1, 255, -1)
            obstruction_detected = True

    if obstruction_detected:
        print("Obstruction detected. Removing obstruction...")
        # Remove obstruction by interpolating lines
        barcode_lines = cv.bitwise_and(binary_img, cv.bitwise_not(mask))
        reconstructed_img = reconstruct_barcode_lines(barcode_lines, mask)
    else:
        reconstructed_img = binary_img

    # Step 7: Apply morphological operations to clean the barcode
    kernel = np.ones((1, 5), np.uint8)
    cleaned_img = cv.morphologyEx(reconstructed_img, cv.MORPH_CLOSE, kernel)

    return cleaned_img

def reconstruct_barcode_lines(barcode_lines, mask):
    """
    Reconstruct the missing barcode lines by interpolating across the obstruction.
    """
    # Find non-obstructed regions
    dilated_mask = cv.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    inverted_mask = cv.bitwise_not(dilated_mask)

    # Use horizontal projection to identify missing lines
    horizontal_projection = np.sum(barcode_lines, axis=1)
    avg_line_spacing = np.mean(np.diff(np.where(horizontal_projection > 0)[0]))

    # Interpolate missing lines in the masked region
    for i in range(barcode_lines.shape[0]):
        if np.any(mask[i, :] > 0):  # If there's obstruction in this row
            barcode_lines[i, :] = barcode_lines[i - 1, :]  # Copy from the previous line

    return barcode_lines

def process_test_cases(image_folder):
    """
    Process all images in the given folder and display results.
    """
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".jpg"):
            print(f"Processing: {file_name}")
            image_path = os.path.join(image_folder, file_name)

            # Preprocess the image
            cleaned_img = preprocess_image(image_path)

            # Debugging: Display the preprocessed image
            display_image(cleaned_img, "Preprocessed Image")

            # Crop the barcode region
            cropped_img = crop_barcode(cleaned_img)

            # Display the images
            display_image(cv.imread(image_path, cv.IMREAD_GRAYSCALE), "Original Image")
            display_image(cropped_img, "Cropped Barcode")

if __name__ == "__main__":
    # Replace with the path to your test cases folder
    image_folder = "C:\\Users\\A\\Desktop\\CV-main\\Test Cases-20241123"

    process_test_cases(image_folder)
