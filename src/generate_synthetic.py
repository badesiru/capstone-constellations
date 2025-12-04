import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from parse_stellarium import build_catalog_from_stellarium


IMG_SIZE = 1024
SCALE = int(IMG_SIZE * 0.38)
SIMPLE_TOP_N = 12


def mag_to_radius(mag):

    m = max(min(mag, 8), 0)
    return int(3 + (8 - m) * 0.8)

def mag_to_brightness(mag):
    m = max(min(mag, 8), 0)
    return int(150 + (8 - m) * 12)

def project_radec(ra, dec):
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)

    return np.column_stack((x, y))

def normalize_coords(coords):
    # Center and normalize to unit scale
    coords = coords - coords.mean(axis=0)
    maxd = np.max(np.linalg.norm(coords, axis=1))
    if maxd > 0:
        coords = coords / maxd
    return coords



def draw_constellation(coords, mags, out_path):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    scale = IMG_SIZE * 0.45  # Increased from 0.38 to spread stars more
    cx = IMG_SIZE // 2
    cy = IMG_SIZE // 2

    for (x, y), m in zip(coords, mags):
        px = int(cx + x * SCALE)
        py = int(cy - y * SCALE)  # Use larger Y scale for vertical stretch 

        brightness = mag_to_brightness(m)
        radius = mag_to_radius(m)

        color = (brightness, brightness, brightness)
        cv2.circle(img, (px, py), radius, color, -1)

    cv2.imwrite(str(out_path), img)


def main():
    print("Loading Stellarium + HYG catalog")
    catalog = build_catalog_from_stellarium(
        index_path=r"C:\\Program Files\\Stellarium\\skycultures\\modern_iau\\index.json",
        hyg_path="data/hyg/hyg_v38.csv.gz"
    )

    out_simple = Path("data/synthetic/simple")
    out_full = Path("data/synthetic/full")
    out_simple.mkdir(parents=True, exist_ok=True)
    out_full.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic star maps...")

    for const_name, coords_raw in catalog.items():

        ra = coords_raw[:, 1] * 15  # Convert hours to degrees (1 hour = 15°)
        dec = coords_raw[:, 2]
        mags = coords_raw[:, 3]

        full_coords = project_radec(ra, dec)
        full_coords = normalize_coords(full_coords)

        full_path = out_full / f"{const_name}.png"
        draw_constellation(full_coords, mags, full_path)

        order = np.argsort(mags)[:SIMPLE_TOP_N]

        ra_sel = ra[order]  # Already converted to degrees above
        dec_sel = dec[order]
        mags_sel = mags[order]

        simple_coords = project_radec(ra_sel, dec_sel)
        simple_coords = normalize_coords(simple_coords)

        simple_path = out_simple / f"{const_name}.png"
        draw_constellation(simple_coords, mags_sel, simple_path)

        print(f"Generated: {const_name}")

    print("\nDONE: Synthetic images created.")



if __name__ == "__main__":
    main()
