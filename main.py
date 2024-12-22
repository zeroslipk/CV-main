# %%
import os
from typing import Any
import cv2 as cv  # Correct module for OpenCV
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for array operations

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to make it ready for contouring.

    Args:
        image (np.ndarray): Original image without any filters applied on it.

    Returns:
        np.ndarray: Preprocessed binary image ready for contouring.
    """
    # Step 1: Copy image and convert to grayscale and HSV
    img = image.copy()
    
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Step 2: Detect if the image contains colors
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Create masks for black, gray, and white
    black_mask = cv.inRange(img_hsv, lower_black, upper_black)
    gray_mask = cv.inRange(img_hsv, lower_gray, upper_gray)
    white_mask = cv.inRange(img_hsv, lower_white, upper_white)

    # Combine the masks
    combined_mask = cv.bitwise_or(black_mask, gray_mask)
    combined_mask = cv.bitwise_or(combined_mask, white_mask)

    # Invert the mask to get non-gray colors
    non_gray_mask = cv.bitwise_not(combined_mask)

    # Step 3: Check if there are non-gray pixels and inpaint if colors are detected
    if non_gray_mask.any():
        print("Image contains colors")
        lower_color = np.array([0, 30, 60])
        upper_color = np.array([20, 150, 255])
        mask = cv.inRange(img_hsv, lower_color, upper_color)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        inpainted_image = cv.inpaint(img, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
        img_gray = cv.cvtColor(inpainted_image, cv.COLOR_BGR2GRAY)
        _, img_gray = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY)

    # Step 4: Blur and denoise
    img_blurred = cv.medianBlur(cv.blur(img_gray, (3, 3)), 3)
    _, img_denoised = cv.threshold(
        img_blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )

    # Step 5: Detect sine wave noise
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 1))
    img_dilated = cv.morphologyEx(img_denoised, cv.MORPH_DILATE, kernel)
    if len(np.unique(img_dilated)) > 1:
        print("Image contains sine wave noise")
        img_denoised = cv.adaptiveThreshold(
            img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
        )

    return img_denoised


# %%
from typing import Optional, Tuple
import numpy as np


def find_barcode_contours(image: np.ndarray) -> Optional[Tuple[np.ndarray, int, int]]:
    """
    Attempt to find the contours of a barcode within the image.

    Args:
        image (np.ndarray): Input image in which to find the barcode contours.

    Returns:
        Optional[Tuple[np.ndarray, int, int]]: A tuple containing:
            - The contour of the barcode as a NumPy array.
            - The width of the bounding box.
            - The height of the bounding box.
            Returns None if no contours are found.
    """
    img = image.copy()

    # Preprocess image for contour detection
    img_edges = cv.Canny(img, 100, 200)  # Detect edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    img_closed = cv.morphologyEx(
        img_edges, cv.MORPH_CLOSE, kernel
    )  # Close the edges to obtain a mask for the barcode
    _, img_closed_binary = cv.threshold(
        img_closed, 50, 255, cv.THRESH_BINARY
    )  # Binarize the image

    # Find contours in the preprocessed image
    contours, _ = cv.findContours(
        img_closed_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        # Get the largest contour, assuming it represents the barcode
        largest_contour = max(contours, key=cv.contourArea)

        # Get the minimum area rotated bounding rectangle
        rect = cv.minAreaRect(largest_contour)

        # Get the four points of the bounding rectangle
        box = cv.boxPoints(rect)
        box = np.intp(box)  # Convert all coordinates to integers

        # Get the width and height of the bounding box
        width = int(rect[1][0])
        height = int(rect[1][1])

        return box, width, height

    print("No contours were found")
    return None

# Example usage
# image = cv.imread('path/to/image.jpg', cv.IMREAD_GRAYSCALE)
# result = find_barcode_contours(image)
# if result:
#     box, width, height = result
#     print("Bounding box:", box)
#     print("Width:", width, "Height:", height)


# %%
def vertical_erosion_dilation(image):
    """
    Perform vertical erosion and then vertical dilation on an input image.

    Args:
        image (numpy.ndarray): The input image (Mat-like object) as a NumPy array.

    Returns:
        numpy.ndarray: The image after vertical erosion and dilation.
    """
    if image is None:
        raise ValueError("Input image cannot be None")

    # Create a vertical kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))  
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (1, 1000))  


    # Perform vertical erosion
    eroded = cv.erode(image, kernel, iterations=3)

    # Perform vertical dilation
    dilated = cv.dilate(eroded, kernel, iterations=3)

    dilated2 = cv.dilate(dilated, kernel2, iterations=3)

    return dilated2

# %%
from typing import Optional
import cv2 as cv
import numpy as np

def extract_barcode(image: np.ndarray) -> Optional[np.ndarray]:

    # Preprocess the image
    cleaned_img = preprocess_image(image)  # Ensure this function is defined

    # Find the bounding box of the barcode
    result = find_barcode_contours(cleaned_img)  # Ensure this function is defined
    if result is not None:
        bounding_box, width, height = result

        # Define the destination points for perspective transformation
        destination_points = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )

        # Get the perspective transform matrix
        M = cv.getPerspectiveTransform(
            bounding_box.astype("float32"), destination_points
        )
        # Warp the image to get a top-down view of the barcode
        warped = cv.warpPerspective(cleaned_img, M, (width, height))
        # Rotate the warped image if needed
        warped_rotated = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)

        # Perform vertical erosion and dilation
        eroded_and_dilated = vertical_erosion_dilation(
            warped_rotated
        )  # Ensure this function is defined

        return warped_rotated

    print("No contour was found")
    return None


# %%
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

def plot_image(
    file_name: str, original_image: np.ndarray, barcode_image: Optional[np.ndarray] = None
) -> None:
    """
    Plot the original image and the barcode extracted from it.

    Args:
        file_name (str): Name of the image file.
        original_image (np.ndarray): Original image without any preprocessing.
        barcode_image (Optional[np.ndarray], optional): Extracted barcode image if found. Defaults to None.
    """
    plt.figure(figsize=(15, 20))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.title(f"Processed {file_name}")
    plt.axis("off")
    plt.imshow(original_image, cmap="gray")

    # Plot the extracted barcode image (if found)
    plt.subplot(1, 2, 2)
    plt.title(f"{file_name} Extracted Barcode")
    plt.axis("off")
    if barcode_image is not None:
        plt.imshow(barcode_image, cmap="gray")
    else:
        plt.text(0.5, 0.5, "No barcode detected", ha="center", va="center", fontsize=12)
        plt.axis("off")

    plt.show()


# %%
def extract_barcodes(
    images_dir: str, output_dir: str = r".\processed_barcodes"
) -> None:
    """Extract a barcode from each image in the images directory and saves the result as an image in output directory

    Args:
        images_dir (str): Path to the directory that contains the images to extract barcodes from
        output_dir (str, optional): Path to the directory to save the images of barcodes in. Defaults to r".\processed_barcodes".
    """

    IMAGE_FILES_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
    }  # List of valid image extensions for OpenCV

    for file_name in os.listdir(images_dir):  # Iterate over files in the directory
        ext = os.path.splitext(file_name)[1].lower()  # Get the file extension
        if ext in IMAGE_FILES_EXTENSIONS:  # Check if the file is an image
            print(f"Processing: {file_name}")
            image_path = os.path.join(images_dir, file_name)
            image = cv.imread(image_path)
            barcode = extract_barcode(image)  # Extract barcode
            plot_image(
                file_name, image, barcode
            )  # Plot original image and extracted barcode
        else:
            print(f"{file_name} is not an image file.")

# %%
# Define the folder containing the test images
IMAGES_DIR = "/Users/mohamedwalid/Desktop/Semester 7/Computer vision/CV-main/Test Cases-20241123"

# Specify the output folder where you want to save the extracted barcodes
OUTPUT_DIR = os.path.join(IMAGES_DIR, "processed_barcodes")

# Ensure the output folder exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Process all test images
extract_barcodes(IMAGES_DIR, OUTPUT_DIR)


