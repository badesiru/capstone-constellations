import cv2
import numpy as np
from pathlib import Path


#detect starts in grayscale, and x,y coordinates returend adaptive threshold and contour
def detect_stars(processed_img):


    #adaptive threshold
    thresh = cv2.adaptiveThreshold(
        processed_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2   
    )

    #removing small noise 
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    star_points = []

    for c in contours:
        area = cv2.contourArea(c)

        #filters noise and tiny blobs
        if area < 5 or area > 200:
            continue

        #star center computed 
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        #get brightness at that point 
        brightness = processed_img[cy, cx]

        #extra filtering, keep only bright enough stars
        if brightness < 180:
            continue

        star_points.append((cx, cy))

    return star_points



if __name__ == "__main__":
    #test detection in one image
    test_img = Path(__file__).resolve().parents[1] / "data" / "orion1_processed.jpg"

    img = cv2.imread(str(test_img), cv2.IMREAD_GRAYSCALE)
    stars = detect_stars(img)

    print(f"Detected {len(stars)} stars")
    #prints first 10
    print(stars[:10])
