import cv2
import numpy as np
from pathlib import Path

from parse_stellarium import build_catalog_from_stellarium
from build_catalog import project_radec
from match_constellation import normalize_points

IMG_SIZE = 1024
SCALE = IMG_SIZE * 0.38
CX = IMG_SIZE // 2
CY = IMG_SIZE // 2

def main():

    stell = r"C:\Program Files\Stellarium\skycultures\modern_iau\index.json"
    hyg   = r"data/hyg/hyg_v38.csv.gz"

    catalog = build_catalog_from_stellarium(stell, hyg)
    ori = catalog["Ori"]

    hips = ori[:,0]
    ra   = ori[:,1]
    dec  = ori[:,2]
    mag  = ori[:,3]

    valid = ~np.isnan(mag)
    ra = ra[valid]
    dec = dec[valid]
    mag = mag[valid]

    order = np.argsort(mag)
    sel = order[:8]
    ra = ra[sel]
    dec = dec[sel]
    mag = mag[sel]


    coords = project_radec(ra, dec)
    coords = normalize_points(coords)


    pixels = []
    for (x, y) in coords:
        px = int(CX + x * SCALE)
        py = int(CY - y * SCALE)
        pixels.append((px, py))

    print("\nEXPECTED ORION PIXELS:")
    for p in pixels:
        print(p)


    imgpath = Path("data/synthetic/simple/Ori.png")
    img = cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Could not load image at", imgpath)
        return

    print("\nRun detect_stars_synthetic() manually next to compare.")

    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for p in pixels:
        cv2.circle(out, p, 5, (0,0,255), 2)

    cv2.imwrite("data/Ori_expected_overlay.png", out)
    print("Saved expected overlay -> data/Ori_expected_overlay.png")

if __name__ == "__main__":
    main()
