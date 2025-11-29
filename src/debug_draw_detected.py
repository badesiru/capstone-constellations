import cv2
from detect_stars import detect_stars_synthetic

img = cv2.imread("data/synthetic/simple/Ori.png", cv2.IMREAD_GRAYSCALE)
stars = detect_stars_synthetic(img)

print("Detected:", stars)

# Draw detected positions in red
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for (x, y) in stars:
    cv2.circle(vis, (x, y), 10, (0, 0, 255), 2)

cv2.imwrite("data/Ori_detected_overlay.png", vis)
print(" data/Ori_detected_overlay.png")
