import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from crop import crop_barcode

def display_image(img, title):
    """
    Displays an image using Matplotlib.
    """
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_image(image_path):
    """
    Optimized image preprocessing: balanced contrast adjustment, noise reduction,
    and precise binarization for barcode detection.
    """
    # Step 1: Read the image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Step 2: Apply a slight blur to reduce noise
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)

    # Step 3: Normalize brightness and contrast
    normalized_img = cv.normalize(blurred_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Binarize the image using a threshold
    _, binary_img = cv.threshold(resized_img, 128, 255, cv.THRESH_BINARY_INV)

    # Morphological Opening
    kernel_height = 13
    kernel_width = 13
    kernel = np.zeros((kernel_height, kernel_width), np.uint8)
    kernel[:, kernel_width // 2] = 1
    opened_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    # Step 6: Morphological Closing to connect barcode lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))  # Small horizontal kernel
    closed_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)

    # Step 7: Further cleaning (optional)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))  # Small vertical kernel
    cleaned_img = cv.morphologyEx(closed_img, cv.MORPH_OPEN, kernel)

    return cleaned_img


def process_test_cases(image_folder):
    """
    Process all images in the given folder and display results.
    """
    # Get all .jpg files and sort them by the numerical prefix
    files = [file_name for file_name in os.listdir(image_folder) if file_name.endswith(".jpg")]
    files.sort(key=lambda x: int(x.split(' ')[0]))  # Extract the numeric part and sort

    for file_name in files:
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
    image_folder = "C:\\Users\\A\\Downloads\\Test Cases-20241123"
    process_test_cases(image_folder)

