import cv2
import numpy as np
from pathlib import Path





def detect_stars(processed_img, synthetic=False):

        _, thresh = cv2.threshold(processed_img, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        star_points = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 3:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            star_points.append((cx, cy))

        return star_points

    thresh = cv2.adaptiveThreshold(
        processed_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    star_points = []
    star_brightness = []


    bright_thresh = np.percentile(processed_img, 99)

    for c in contours:
        area = cv2.contourArea(c)


        if area < 20 or area > 2000:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        b = processed_img[cy, cx]
        if b < bright_thresh:
            continue

        star_points.append((cx, cy))
        star_brightness.append(b)

    if len(star_points) > 15:
        zipped = list(zip(star_points, star_brightness))
        zipped.sort(key=lambda x: x[1], reverse=True)
        star_points = [p for p, bright in zipped[:15]]

    return star_points


def detect_stars_synthetic(img):

    _, th = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    star_points = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > 5000:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        star_points.append((cx, cy))

    return star_points


if __name__ == "__main__":
    test_img = Path(__file__).resolve().parents[1] / "data" / "orion1_processed.jpg"
    img = cv2.imread(str(test_img), cv2.IMREAD_GRAYSCALE)

    stars = detect_stars(img)
    print(f"Detected {len(stars)} stars")
    print(stars)
