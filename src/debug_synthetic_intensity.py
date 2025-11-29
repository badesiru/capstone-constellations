import cv2
import numpy as np

img = cv2.imread("data/synthetic/simple/Ori.png", cv2.IMREAD_GRAYSCALE)

print("Image loaded:", img.shape)
print("Min pixel:", img.min())
print("Max pixel:", img.max())

# Find bright pixels
bright = np.where(img > 180)
print("Number of bright pixels:", len(bright[0]))

# Show exact bright clusters
coords = list(zip(bright[1], bright[0]))  # (x,y)
coords_sample = coords[:50]
print("Sample bright pixel coords:", coords_sample)
