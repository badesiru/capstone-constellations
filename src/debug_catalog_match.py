
import numpy as np
from build_catalog import build_catalog_shapes
from match_constellation import match_constellation

def main():
    catalog = build_catalog_shapes(top_n=8)
    print("Catalog constellations:", list(catalog.keys()))

    test_consts = ["Ori", "Cas", "UMa", "Boo", "Car", "Pic", "Ari"]

    for cname in test_consts:
        if cname not in catalog:
            print(f"[WARN] {cname} not in catalog, skipping.")
            continue

        pts = catalog[cname]


        idx = np.arange(len(pts))
        np.random.shuffle(idx)
        pts_shuffled = pts[idx]

        pred, score = match_constellation(pts_shuffled, catalog)

        ok = "OK" if pred == cname else "NO"
        print(f"True: {cname:3s} ? Pred: {pred:3s}   score={score:.6f}   {ok}")

if __name__ == "__main__":
    main()
