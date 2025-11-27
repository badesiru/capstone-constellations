import sys
from pathlib import Path
import cv2

from preprocess import preprocess_image
from detect_stars import detect_stars
from build_catalog import build_catalog_shapes
from match_constellation import match_constellation


def main(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found {image_path}")

    print(f"Input image {image_path}")


    processed = preprocess_image(image_path)


    #load preprocessed img as grayscale 
    if processed is None:
        raise RuntimeError("processed image is None")


    stars = detect_stars(processed)
    print(f"Detected {len(stars)} stars")
    print("Star coordinates", stars)

    if len(stars) < 3:
        print("Not enough stars detected to perform matching.")
        return


    catalog = build_catalog_shapes(max_mag=4.5)
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
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/run_pipeline.py data/<imagefile>.jpg")
        sys.exit(1)

    main(sys.argv[1])
