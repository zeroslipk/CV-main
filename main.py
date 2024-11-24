import cv2 as cv  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for image display
import os  # Import OS library for file and folder operations

def crop_barcode(img):
    """
    Crops the barcode region from the image using contours.
    """
    # Find contours in the binary image (external contours only)
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Check if no contours are found; return the original image in such cases
    if not contours:
        print("No contours detected! Skipping this image.")
        return img  # Return the original image if no contours are found

    # Find the leftmost x-coordinate of the contours (minimum x value)
    x_min = min([cv.boundingRect(contour)[0] for contour in contours])
    
    # Find the rightmost x-coordinate of the contours (maximum x + width)
    x_max = max([cv.boundingRect(contour)[0] + cv.boundingRect(contour)[2] for contour in contours])

    # Identify the largest contour by area
    largest_contour = max(contours, key=cv.contourArea)
    
    # Get the vertical cropping limits (y and height) of the largest contour
    _, y, _, h = cv.boundingRect(largest_contour)

    # Crop the image using the calculated coordinates
    cropped_img = img[y:y + h, x_min:x_max]
    return cropped_img

def preprocess_image(image_path, special_image=False):
    """
    Unified preprocessing function.
    If special_image=True, applies advanced noise reduction and morphological operations (for 7th photo).
    """
    # Read the image from the specified path in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if special_image:  # Special preprocessing for the 7th photo
        # Apply a vertical median filter to reduce noise
        kernel = np.ones((7, 1), np.float32) / 5
        median_blurred_img = cv.filter2D(img, -1, kernel)
        
        # Upscale the image for better processing
        scale_factor = 4
        resized_img = cv.resize(median_blurred_img, (median_blurred_img.shape[1] * scale_factor, median_blurred_img.shape[0] * scale_factor))
        
        # Apply binary thresholding to binarize the image
        _, binary_img = cv.threshold(resized_img, 128, 255, cv.THRESH_BINARY_INV)
        
        # Morphological opening to remove small noise (vertical kernel)
        kernel = np.zeros((13, 13), np.uint8)
        kernel[:, kernel.shape[1] // 2] = 1  # Set the middle column to 1
        opened_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)
        
        # Morphological closing to close small gaps (vertical kernel)
        kernel = np.zeros((21, 21), np.uint8)
        kernel[:, kernel.shape[1] // 2] = 1  # Set the middle column to 1
        cleaned_img = cv.morphologyEx(opened_img, cv.MORPH_CLOSE, kernel)
        
        # Additional closing to further clean the image (small kernel)
        kernel = np.ones((1, 5), np.uint8)
        cleaned_img = cv.morphologyEx(cleaned_img, cv.MORPH_CLOSE, kernel)
    else:  # Standard preprocessing for all other photos
        # Apply Gaussian blur to reduce noise
        blurred_img = cv.GaussianBlur(img, (5, 5), 0)
        
        # Normalize brightness and contrast
        normalized_img = cv.normalize(blurred_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(normalized_img)
        
        # Apply adaptive thresholding to binarize the image
        binary_img = cv.adaptiveThreshold(enhanced_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 8)
        
        # Morphological closing to connect broken barcode lines
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        closed_img = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)
        
        # Morphological opening to remove small noise
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        cleaned_img = cv.morphologyEx(closed_img, cv.MORPH_OPEN, kernel)

    return cleaned_img

def process_test_cases(image_folder, special_image_index):
    """
    Process all images using unified preprocessing.
    Save both the preprocessed image and the cropped barcode image for each photo.
    """
    # Get all .jpg files in the folder and sort them by the numerical prefix
    files = [file_name for file_name in os.listdir(image_folder) if file_name.endswith(".jpg")]
    files.sort(key=lambda x: int(x.split(' ')[0]))  # Sort files by the numeric prefix
    
    # Create an output folder for saving processed images
    output_folder = os.path.join(image_folder, "Processed_Output")
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

    for i, file_name in enumerate(files):  # Iterate through each image file
        print(f"Processing: {file_name}")  # Log the current file being processed
        image_path = os.path.join(image_folder, file_name)  # Get the full file path

        # Determine if this is the special image (7th photo)
        special_image = (i + 1 == special_image_index)

        # Preprocess the image using the unified function
        cleaned_img = preprocess_image(image_path, special_image=special_image)

        # Save the preprocessed image
        preprocessed_path = os.path.join(output_folder, f"Preprocessed_{file_name}")
        cv.imwrite(preprocessed_path, cleaned_img)
        print(f"Saved Preprocessed Image: {preprocessed_path}")

        # Crop the barcode region from the preprocessed image
        cropped_img = crop_barcode(cleaned_img)

        # Save the cropped barcode image
        cropped_path = os.path.join(output_folder, f"Cropped_{file_name}")
        cv.imwrite(cropped_path, cropped_img)
        print(f"Saved Cropped Barcode: {cropped_path}")

if __name__ == "__main__":
    # Define the folder containing the test images
    image_folder = "C://Users//HD//Desktop//CV//CV-main//Test Cases-20241123"
    
    # Specify the index of the special image (7th photo)
    special_image_index = 7

    # Process all test images
    process_test_cases(image_folder, special_image_index)
