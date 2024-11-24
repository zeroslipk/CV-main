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

def preprocess_image(image_path):
    """
    Preprocess the image: apply noise reduction, resize, and binarize.
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Apply Median Filter vertically only
    kernel = np.ones((7, 1), np.float32) / 5
    median_blurred_img = cv.filter2D(img, -1, kernel)

    # Resize the image to increase its dimensions
    scale_factor = 4  # Adjust the scale factor as needed
    resized_img = cv.resize(median_blurred_img, (median_blurred_img.shape[1] * scale_factor, median_blurred_img.shape[0] * scale_factor))

    # Binarize the image using a threshold
    _, binary_img = cv.threshold(resized_img, 128, 255, cv.THRESH_BINARY_INV)

    # Morphological Opening
    kernel_height = 13
    kernel_width = 13
    kernel = np.zeros((kernel_height, kernel_width), np.uint8)
    kernel[:, kernel_width // 2] = 1
    opened_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    # Morphological Closing
    kernel_height = 21
    kernel_width = 21
    kernel = np.zeros((kernel_height, kernel_width), np.uint8)
    kernel[:, kernel_width // 2] = 1
    closed_img = cv.morphologyEx(opened_img, cv.MORPH_CLOSE, kernel)

    # Further cleaning
    kernel_height = 1
    kernel_width = 5
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
    cleaned_img = cv.morphologyEx(closed_img, cv.MORPH_CLOSE, kernel)

    return cleaned_img

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
    image_folder = "C:\Users\A\Desktop\CV-main\Test Cases-20241123"
    process_test_cases(image_folder)
