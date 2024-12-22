import os
from typing import Optional
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def clean(img: np.ndarray, margin: int = 20) -> np.ndarray:
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, thresholded = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    thresholded = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = w / h
        if cv.contourArea(c) > 100 and aspect_ratio < 2:
            filtered_contours.append(c)
    
    if not filtered_contours:
        return img

    x_min = min(cv.boundingRect(c)[0] for c in filtered_contours)
    x_max = max(cv.boundingRect(c)[0] + cv.boundingRect(c)[2] for c in filtered_contours)
    y_min = min(cv.boundingRect(c)[1] for c in filtered_contours)
    y_max = max(cv.boundingRect(c)[1] + cv.boundingRect(c)[3] for c in filtered_contours)

    barcode_height = y_max - y_min
    y_max = int(y_max - 0.15 * barcode_height)

    cropped_img = img[y_min:y_max, x_min:x_max]

    if margin > 0:
        padded_img = cv.copyMakeBorder(cropped_img, 0, margin, 0, 0, cv.BORDER_CONSTANT, value=255)
    else:
        padded_img = cropped_img

    return padded_img

def preprocess_image(image: np.ndarray) -> np.ndarray:
    img = image.copy()

    if len(img.shape) == 3:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img_gray = img

    img_blurred = cv.medianBlur(cv.blur(img_gray, (3, 3)), 3)
    _, img_denoised = cv.threshold(img_blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))
    img_denoised = cv.morphologyEx(img_denoised, cv.MORPH_CLOSE, closing_kernel, iterations=2)

    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
    # upper_skin1 = np.array([25, 170, 255], dtype=np.uint8)
    # lower_skin2 = np.array([160, 20, 50], dtype=np.uint8)
    # upper_skin2 = np.array([180, 170, 255], dtype=np.uint8)

    # mask1 = cv.inRange(hsv, lower_skin1, upper_skin1)
    # mask2 = cv.inRange(hsv, lower_skin2, upper_skin2)
    # skin_mask = cv.bitwise_or(mask1, mask2)

    # if cv.countNonZero(skin_mask) > 1000:
    #     img_denoised = remove_hand(img)
        
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 1))
    img_dilated = cv.morphologyEx(img_denoised, cv.MORPH_DILATE, kernel)
    if len(np.unique(img_dilated)) > 1:
        print("Image contains sine wave noise")
        img_denoised = cv.adaptiveThreshold(
            img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
        )    

    return img_denoised

def remove_hand(img: np.ndarray) -> Optional[np.ndarray]:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin1 = np.array([25, 170, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 20, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 170, 255], dtype=np.uint8)

    mask1 = cv.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv.bitwise_or(mask1, mask2)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv.findContours(skin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  

    largest_contour = max(contours, key=cv.contourArea)

    hand_mask = np.zeros_like(skin_mask)
    cv.drawContours(hand_mask, [largest_contour], -1, 255, thickness=cv.FILLED)

    hand_mask_inv = cv.bitwise_not(hand_mask)

    img_no_hand = cv.inpaint(img, hand_mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

    gray = cv.cvtColor(img_no_hand, cv.COLOR_BGR2GRAY)

    erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 50))
    binary_erode = cv.erode(gray, erosion_kernel, iterations=2)

    dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 60))
    binary_opened = cv.dilate(binary_erode, dilation_kernel, iterations=3)

    dilationkernel2 = cv.getStructuringElement(cv.MORPH_RECT, (1, 10))
    binary_final = cv.dilate(binary_opened, dilationkernel2, iterations=2)

    kernel = np.ones((30, 1), np.float32) / 30
    vertical_mean_filtered = cv.filter2D(binary_final, -1, kernel)

    img = cv.normalize(vertical_mean_filtered, None, 0, 255, cv.NORM_MINMAX)

    return img

def display_before_after(original_img: np.ndarray, cropped_img: np.ndarray, title_before: str = "Original Image", title_after: str = "Cropped Barcode"):
    plt.figure(figsize=(12, 6))
    images = [(original_img, title_before), (cropped_img, title_after)]
    for i, (img, title) in enumerate(images):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img if len(img.shape) == 2 else cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis("off")
    plt.show()

def process_test_cases(image_folder: str, special_image_index: Optional[int] = None):
    files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    files.sort(key=lambda x: int(os.path.splitext(x)[0].split()[0]))
    output_folder = os.path.join(image_folder, "Processed_Output")
    os.makedirs(output_folder, exist_ok=True)

    for i, file_name in enumerate(files):
        image_path = os.path.join(image_folder, file_name)
        original_img = cv.imread(image_path)
        if original_img is None:
            print(f"Could not read image {file_name}. Skipping...")
            continue

        cleaned_img = preprocess_image(original_img)
        cropped_img = clean(cleaned_img)
        cv.imwrite(os.path.join(output_folder, f"Cropped_{file_name}"), cropped_img)
        display_before_after(original_img, cropped_img)

if __name__ == "__main__":
    folder = "/Users/mohamedwalid/Desktop/Semester 7/Computer vision/CV-main/Test Cases-20241123"
    process_test_cases(folder, special_image_index=7)