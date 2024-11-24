import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """
    Preprocessing function to remove the hand from the image and extract a cleaned barcode.
    """
    # Step 1: Read the image
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Step 2: Thresholding to segment the image
    _, binary_img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

    # Step 3: Detect edges for contour detection
    edges = cv.Canny(binary_img, 50, 150)

    # Step 4: Find contours to isolate regions
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Step 5: Create a mask to remove large unwanted regions like the hand
    mask = np.ones_like(img, dtype=np.uint8) * 255
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        # Identify the hand/nose region by size and position
        if area > 1000 and w > 50 and h > 50:  # Adjust thresholds as needed
            cv.drawContours(mask, [contour], -1, 0, thickness=cv.FILLED)

    # Step 6: Apply the mask to the image
    cleaned_img = cv.bitwise_and(img, mask)

    # Step 7: Fill removed regions with white
    cleaned_img[mask == 0] = 255

    # Step 8: Further clean the barcode region (dilate to merge broken parts)
    kernel = np.ones((1, 50), np.uint8)
    dilated_img = cv.dilate(cleaned_img, kernel, iterations=1)

    # Step 9: Extract the largest contour (barcode)
    contours, _ = cv.findContours(dilated_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    barcode_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(barcode_contour)
    cropped_barcode = cleaned_img[y:y+h, x:x+w]

    # Step 10: Final binarization for the cleaned barcode
    _, final_img = cv.threshold(cropped_barcode, 128, 255, cv.THRESH_BINARY)

    return final_img

def save_and_display_image(img, output_path, title="Image"):
    """
    Utility function to save and display an image.
    """
    # Save the image
    cv.imwrite(output_path, img)
    print(f"Image saved at: {output_path}")

    # Display the image
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Test the function
# Replace with the correct paths
image_path = "/Users/mohamedwalid/Desktop/CV-main/03 - eda ya3am ew3a soba3ak mathazarsh.jpg"  # Test input image
output_path = "/Users/mohamedwalid/Desktop/CV-main/Cleaned_Barcode_Result.jpg"  # Path to save the cleaned barcode
result_img = preprocess_image(image_path)
save_and_display_image(result_img, output_path, title="Cleaned Barcode")
