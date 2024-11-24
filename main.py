import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def crop_barcode(img):
    """
    Crops the barcode region from the image using contours.
    """
    # Find contours in the binary image
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours detected! Skipping this image.")
        return img  # Return the original image if no contours are found

    # Find the horizontal limits of the barcode
    x_min = min([cv.boundingRect(contour)[0] for contour in contours])  # Leftmost x-coordinate
    x_max = max([cv.boundingRect(contour)[0] + cv.boundingRect(contour)[2] for contour in contours])  # Rightmost x-coordinate

    # Assume the largest contour corresponds to the vertical extent of the barcode
    largest_contour = max(contours, key=cv.contourArea)
    _, y, _, h = cv.boundingRect(largest_contour)  # Get the y and height (vertical cropping)

    # Crop the image so that only the barcode is visible
    cropped_img = img[y:y + h, x_min:x_max]

    return cropped_img

def display_image(img, title):
    """
    Displays an image using Matplotlib.
    """
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
# 1 2 4  8 5 6 10
# 3 7 9 11
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

    # Step 4: Adaptive Histogram Equalization (CLAHE) for contrast enhancement
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(normalized_img)

    # Step 5: Adaptive Thresholding for binarization
    # Use a smaller block size for finer details
    binary_img = cv.adaptiveThreshold(
        enhanced_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 8
    )

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
    image_folder = "/Users/mohamedwalid/Desktop/CV-main/Test Cases-20241123"
    process_test_cases(image_folder)

