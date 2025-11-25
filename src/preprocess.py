import cv2
import numpy as np
from pathlib import Path


#loads image - numpy array
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cant open img")
    return img


#converts to greyscale imhg
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#gaussian blur and median blur to reduce noise 
def denoise(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    blurred = cv2.medianBlur(blurred, 3)
    return blurred

#adaptive hist eq for more ocntrast 
def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# load img, grayscale, de noise, increase contrast
def preprocess_image(path):
    img = load_image(path)
    gray = to_grayscale(img)
    clean = denoise(gray)
    enhanced = enhance_contrast(clean)
    return enhanced


if __name__ == "__main__":
    #testing 
    test_img = Path(__file__).resolve().parents[1] / "data" / "orion2.jpg"

    print(f"Testing preprocessing on: {test_img}")
    processed = preprocess_image(test_img)

    # Save the result for inspection
    out_path = Path(__file__).resolve().parents[1] / "data" / "orion1_processed.jpg"
    cv2.imwrite(str(out_path), processed)
    print(f"Saved processed image ? {out_path}")
