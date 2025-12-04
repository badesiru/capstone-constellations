import sys
from pathlib import Path
import cv2
import argparse

from preprocess import preprocess_image
from detect_stars import detect_stars, detect_stars_synthetic
from build_catalog import build_catalog_shapes
from match_constellation_ultra import match_constellation_ultra
from visualize_results import draw_constellation_overlay


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

    # Build catalog with magnitudes for ultra-advanced matching
    catalog = build_catalog_shapes(top_n=12, include_magnitudes=True)

    print("Catalog loaded. Constellations:", list(catalog.keys()))

    print("Matching constellation...")

    try:
        # Use ultra-advanced algorithm (85.2% accuracy)
        # tolerance=0.08 for optimal geometric filtering
        # magnitude weighting gives 3x weight to bright stars
        best_const, score, top_5 = match_constellation_ultra(
            stars, catalog, tolerance=0.08, use_icp=True
        )
    except Exception as e:
        print("Error during matching:", e)
        return

    print("RESULT")
    print(f" Predicted Constellation: {best_const}")
    print(f" Matching Score: {score:.6f}")

    # Create visualization with constellation overlay
    annotated = draw_constellation_overlay(processed, stars, best_const, score)
    
    # Save both processed and annotated images
    out_processed = Path("data") / f"{image_path.stem}_processed.jpg"
    out_annotated = Path("data") / f"{image_path.stem}_annotated.jpg"
    
    cv2.imwrite(str(out_processed), processed)
    cv2.imwrite(str(out_annotated), annotated)
    
    print(f"Saved processed image: {out_processed}")
    print(f"Saved annotated image: {out_annotated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic star detector")
    args = parser.parse_args()

    main(args.image_path, synthetic=args.synthetic)
