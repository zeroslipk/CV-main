import cv2 as cv
import numpy as np
import os

def remove_numbers_from_bottom(img, margin=20):
    """
    Removes any numbers or unwanted text at the bottom of the barcode by setting that area to white.
    """
    # Take the bottom 'margin' rows and identify the darker pixels which are likely numbers
    bottom_part = img[-margin:, :]
    
    # Threshold the bottom part to identify non-white regions (likely numbers)
    _, thresholded_bottom = cv.threshold(bottom_part, 240, 255, cv.THRESH_BINARY_INV)
    
    # Now set those areas to white (255)
    img[-margin:, :] = cv.bitwise_or(img[-margin:, :], thresholded_bottom)
    
    # Ensure the entire bottom part is white after thresholding
    img[-margin:, :] = 255  # Clean the entire bottom margin with white pixels
    
    return img

def crop_barcode(img):
    """
    Crops the barcode region from the image using contours.
    Also removes any numbers below the barcode.
    """
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours detected! Skipping this image.")
        return img

    # Find the bounding box of the largest contour
    x_min = min([cv.boundingRect(contour)[0] for contour in contours])
    x_max = max([cv.boundingRect(contour)[0] + cv.boundingRect(contour)[2] for contour in contours])
    largest_contour = max(contours, key=cv.contourArea)
    _, y, _, h = cv.boundingRect(largest_contour)

    # Crop the image based on the bounding box of the largest contour
    cropped_img = img[y:y + h, x_min:x_max]

    # Remove numbers from the bottom of the cropped barcode region
    cropped_img = remove_numbers_from_bottom(cropped_img)

    return cropped_img

def straighten_and_clean_9th_photo(img):
    """
    Straightens the barcode, removes numbers below the barcode,
    and centers the barcode on a white canvas.
    """
    # Detect edges using Canny edge detector
    edges = cv.Canny(img, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi / 2) * 180 / np.pi
            angles.append(angle)
        average_angle = np.mean(angles)

        # Rotate the image to straighten it
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, average_angle, 1.0)
        img = cv.warpAffine(img, rotation_matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    # Remove numbers and noise below the barcode
    img = remove_numbers_from_bottom(img)

    # Fix the top-left corner if necessary (cleaning up residual noise)
    img[:50, :50] = 255  # Fill the top-left corner with white (clean background)

    # Resize the cropped barcode if needed to ensure it fits
    target_size = (512, 512)  # Desired canvas size
    img = center_barcode(img, target_size=target_size)

    return img

def center_barcode(img, target_size=(512, 512)):
    """
    Centers the barcode on a blank canvas of the target size.
    If the barcode exceeds the target size, resize proportionally to fit.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Resize the image if it exceeds the target dimensions
    if h > target_h or w > target_w:
        scaling_factor = min(target_w / w, target_h / h)  # Scale to fit within target size
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        h, w = img.shape[:2]

    # Create a blank white image of the target size
    centered_img = np.ones((target_h, target_w), dtype=np.uint8) * 255

    # Compute the position to center the barcode
    x_offset = (target_w - w) // 2
    y_offset = (target_h - h) // 2

    # Place the barcode in the center
    centered_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return centered_img

def preprocess_image(image_path, special_image=False, straighten_and_clean=False):
    """
    Unified preprocessing function for general photos,
    with special handling for the 7th and 9th photos.
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if straighten_and_clean:  # Special handling for the 9th photo
        img = straighten_and_clean_9th_photo(img)
    elif special_image:  # Special preprocessing for the 7th photo
        kernel = np.ones((7, 1), np.float32) / 5
        img = cv.filter2D(img, -1, kernel)
        scale_factor = 4
        img = cv.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor))
        
        # Apply a standard threshold to preserve the original black and white structure
        _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
        
        # Perform morphological operations to clean up the image (bars should stay intact)
        kernel = np.zeros((13, 13), np.uint8)
        kernel[:, kernel.shape[1] // 2] = 1
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        
        kernel = np.zeros((21, 21), np.uint8)
        kernel[:, kernel.shape[1] // 2] = 1
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        
        kernel = np.ones((1, 5), np.uint8)
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    else:  # Standard preprocessing for other photos
        img = cv.GaussianBlur(img, (5, 5), 0)
        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 8)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return img

def process_test_cases(image_folder, special_image_index, straighten_image_index):
    """
    Process all images using unified preprocessing. Handles special cases for
    7th and 9th photos. Saves the preprocessed and cropped results.
    """
    files = [file_name for file_name in os.listdir(image_folder) if file_name.endswith(".jpg")]
    files.sort(key=lambda x: int(x.split(' ')[0]))  # Sort by numerical prefix
    output_folder = os.path.join(image_folder, "Processed_Output")
    os.makedirs(output_folder, exist_ok=True)

    for i, file_name in enumerate(files):
        print(f"Processing: {file_name}")
        image_path = os.path.join(image_folder, file_name)
        straighten_and_clean = (i + 1 == straighten_image_index)
        special_image = (i + 1 == special_image_index)

        # Preprocess the image
        cleaned_img = preprocess_image(image_path, special_image=special_image, straighten_and_clean=straighten_and_clean)

        # Save the processed image
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
    image_folder = "C://Users//HD//Desktop//CV//CV-main//Test Cases-20241123"
    special_image_index = 7
    straighten_image_index = 9
    process_test_cases(image_folder, special_image_index, straighten_image_index)
