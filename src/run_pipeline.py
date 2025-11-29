import sys
from pathlib import Path
import cv2
import argparse

from preprocess import preprocess_image
from detect_stars import detect_stars, detect_stars_synthetic

from build_catalog import build_catalog_shapes
from match_constellation import match_constellation


def main(image_path, synthetic=False):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found {image_path}")

    print(f"Input image {image_path}")


    if synthetic:
        processed = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if processed is None:
            raise RuntimeError(f"Could not load synthetic image: {image_path}")
    else:
        processed = preprocess_image(image_path)
        if processed is None:
            raise RuntimeError("processed image is None")


    if synthetic:
        stars = detect_stars_synthetic(processed)
    else:
        stars = detect_stars(processed)

    print(f"Detected {len(stars)} stars")
    print("Star coordinates", stars)

    if len(stars) < 3:
        print("Not enough stars detected to perform matching.")
        return

    catalog = build_catalog_shapes()


    print("Catalog loaded. Constellations:", list(catalog.keys()))

    print("Matching constellation...")

    try:
        best_const, score = match_constellation(stars, catalog)
    except Exception as e:
        print("Error during matching:", e)
        return

    print("RESULT")
    print(f" Predicted Constellation: {best_const}")
    print(f" Matching Score: {score:.6f}")

    out_path = Path("data") / f"{image_path.stem}_processed_output.jpg"
    cv2.imwrite(str(out_path), processed)
    print(f"Saved processed image - {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic star detector")
    args = parser.parse_args()

    main(args.image_path, synthetic=args.synthetic)
