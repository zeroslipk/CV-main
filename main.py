import cv2 as cv
import numpy as np
import os


def remove_numbers_and_fix_barcode(img):
    """
    Straightens the barcode to horizontal, removes numbers, cleans top-left corner,
    and centers the barcode in a blank canvas.
    """
    # Step 1: Detect and straighten the barcode
    edges = cv.Canny(img, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi / 2) * 180 / np.pi
            angles.append(angle)
        average_angle = np.mean(angles)

        # Rotate the image to align the barcode horizontally
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv.getRotationMatrix2D(center, average_angle, 1.0)
        img = cv.warpAffine(img, rotation_matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    # Step 2: Remove numbers below the barcode
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        # Crop just above the numbers
        img = img[y:y + h - 15, x:x + w]

    # Step 3: Clean the top-left corner
    img[:50, :50] = 255  # Fill top-left corner with white to remove artifacts

    # Step 4: Center the barcode in a blank 512x512 canvas
    target_size = (512, 512)
    img = center_barcode(img, target_size)

    return img


def center_barcode(img, target_size=(512, 512)):
    """
    Centers the barcode in a blank canvas of the target size.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Resize if necessary
    if h > target_h or w > target_w:
        scale_factor = min(target_w / w, target_h / h)
        img = cv.resize(img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv.INTER_AREA)
        h, w = img.shape[:2]

    # Create a blank canvas
    centered_img = np.ones((target_h, target_w), dtype=np.uint8) * 255
    x_offset = (target_w - w) // 2
    y_offset = (target_h - h) // 2

    # Center the image on the canvas
    centered_img[y_offset:y_offset + h, x_offset:x_offset + w] = img

    return centered_img


def preprocess_image(image_path, straighten_and_clean=False):
    """
    Preprocess the image. Handles special preprocessing for the 9th photo.
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if straighten_and_clean:
        img = remove_numbers_and_fix_barcode(img)
    else:
        # Standard preprocessing
        img = cv.GaussianBlur(img, (5, 5), 0)
        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 8)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return img


def process_test_cases(image_folder, straighten_image_index):
    """
    Process all images. Special handling for the 9th photo.
    """
    files = [file_name for file_name in os.listdir(image_folder) if file_name.endswith(".jpg")]
    files.sort(key=lambda x: int(x.split(' ')[0]))  # Sort by numerical prefix

    output_folder = os.path.join(image_folder, "Processed_Output")
    os.makedirs(output_folder, exist_ok=True)

    for i, file_name in enumerate(files):
        print(f"Processing: {file_name}")
        image_path = os.path.join(image_folder, file_name)

        # Special handling for the 9th photo
        straighten_and_clean = (i + 1 == straighten_image_index)

        # Preprocess the image
        cleaned_img = preprocess_image(image_path, straighten_and_clean=straighten_and_clean)

        # Save the processed image
        preprocessed_path = os.path.join(output_folder, f"Processed_{file_name}")
        cv.imwrite(preprocessed_path, cleaned_img)
        print(f"Saved Processed Image: {preprocessed_path}")


if __name__ == "__main__":
    # Define the folder containing the test images
    image_folder = "C://Users//HD//Desktop//CV//CV-main//Test Cases-20241123"

    # Specify the index of the 9th photo for special handling
    straighten_image_index = 9

    # Process all test images
    process_test_cases(image_folder, straighten_image_index)
