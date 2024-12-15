import cv2 as cv
import numpy as np
import os

def remove_numbers_from_bottom(img, margin=20):
    """
    Removes any numbers or unwanted text at the bottom of the barcode by setting that area to white.
    """
    bottom_part = img[-margin:, :]
    _, thresholded_bottom = cv.threshold(bottom_part, 240, 255, cv.THRESH_BINARY_INV)
    img[-margin:, :] = cv.bitwise_or(img[-margin:, :], thresholded_bottom)
    img[-margin:, :] = 255  # Clean the entire bottom margin with white pixels
    return img

def crop_barcode(img):
    """
    Crops the barcode region from the image using contours.
    Also removes any numbers below the barcode.
    """
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, thresholded = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours detected! Skipping this image.")
        return img  # Return the original image if no contours are found

    x_min = min([cv.boundingRect(contour)[0] for contour in contours])
    x_max = max([cv.boundingRect(contour)[0] + cv.boundingRect(contour)[2] for contour in contours])
    largest_contour = max(contours, key=cv.contourArea)
    _, y, _, h = cv.boundingRect(largest_contour)

    cropped_img = img[y:y + h, x_min:x_max]
    cropped_img = remove_numbers_from_bottom(cropped_img)

    return cropped_img

def straighten_and_clean_9th_photo(img):
    """
    Straightens the barcode, removes numbers below the barcode, and centers the barcode on a white canvas.
    """
    edges = cv.Canny(img, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi / 2) * 180 / np.pi
            angles.append(angle)
        average_angle = np.mean(angles)

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, average_angle, 1.0)
        img = cv.warpAffine(img, rotation_matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    img = remove_numbers_from_bottom(img)
    target_size = (512, 512)
    img = center_barcode(img, target_size=target_size)

    return img

def center_barcode(img, target_size=(512, 512)):
    """
    Centers the barcode on a blank canvas of the target size.
    If the barcode exceeds the target size, resize proportionally to fit.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    if h > target_h or w > target_w:
        scaling_factor = min(target_w / w, target_h / h)  # Scale to fit within target size
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        h, w = img.shape[:2]

    centered_img = np.ones((target_h, target_w), dtype=np.uint8) * 255
    x_offset = (target_w - w) // 2
    y_offset = (target_h - h) // 2
    centered_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return centered_img

def remove_hand(image_path):
    """
    Removes the hand from the image using skin color segmentation and fills the removed area with the barcode.
    """
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at path {image_path}")
        return None

    # Convert image to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define lower and upper bounds for skin color (HSV range for skin tones)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin regions
    skin_mask = cv.inRange(hsv, lower_skin, upper_skin)

    # Invert the skin mask to focus on the background (the hand will be removed)
    _, skin_mask = cv.threshold(skin_mask, 127, 255, cv.THRESH_BINARY_INV)

    # Apply the mask to the original image to remove the hand
    hand_removed = cv.bitwise_and(img, img, mask=skin_mask)

    # Convert the image with the removed hand to grayscale
    hand_removed_gray = cv.cvtColor(hand_removed, cv.COLOR_BGR2GRAY)

    # Apply vertical closing to fill the gaps where the hand was
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 30))  # Vertical kernel for closing
    img_no_hand = cv.morphologyEx(hand_removed_gray, cv.MORPH_CLOSE, kernel)

    # Optionally, apply dilation and erosion to improve the result
    kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # Dilation to smoothen the result
    img_no_hand = cv.dilate(img_no_hand, kernel_dilate, iterations=1)

    kernel_erode = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # Erosion to remove noise
    img_no_hand = cv.erode(img_no_hand, kernel_erode, iterations=1)

    return img_no_hand

def preprocess_image(image_path, special_image=False, straighten_and_clean=False, remove_hand_image=False):
    """
    Unified preprocessing function for general photos, with special handling for the 7th and 9th photos,
    and hand removal for specific images.
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if remove_hand_image:  # Special handling for the 3rd photo to remove the hand
        img = remove_hand(image_path)
    elif straighten_and_clean:  # Special handling for the 9th photo
        img = straighten_and_clean_9th_photo(img)
    elif special_image:  # Special preprocessing for the 7th photo
        kernel = np.ones((7, 1), np.float32) / 5
        img = cv.filter2D(img, -1, kernel)
        scale_factor = 4
        img = cv.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor))
        
        _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
        
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

def process_test_cases(image_folder, special_image_index, straighten_image_index, remove_hand_index):
    """
    Process all images using unified preprocessing. Handles special cases for
    7th, 9th photos and removes hand from 3rd photo. Saves the preprocessed and cropped results.
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
        remove_hand_image = (i + 1 == remove_hand_index)

        # Preprocess the image
        cleaned_img = preprocess_image(image_path, special_image=special_image, straighten_and_clean=straighten_and_clean, remove_hand_image=remove_hand_image)

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
    remove_hand_index = 3  # Remove hand from the 3rd photo
    process_test_cases(image_folder, special_image_index, straighten_image_index, remove_hand_index)
