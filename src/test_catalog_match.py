# src/test_catalog_match.py

import numpy as np
from parse_stellarium import build_catalog_from_stellarium
from match_constellation import match_constellation


def main():
    #building 88 constellation catalog from Stellarium + HYG
    stell_path = r"C:\Program Files\Stellarium\skycultures\modern_iau\index.json"
    hyg_path = r"data/hyg/hyg_v38.csv.gz"

    print("Building catalog from Stellarium + HYG")
    catalog = build_catalog_from_stellarium(stell_path, hyg_path)
    print(f"   Loaded {len(catalog)} constellations from catalog.\n")

    #test constelatioins
    test_consts = ["Ori", "Cas", "UMa", "Tau", "Sco", "Cyg"]

    total = len(test_consts)
    correct = 0


    for cname in test_consts:
        if cname not in catalog:
            print(f"  [WARN] {cname} not found in catalog keys, skipping.")
            continue

        pts = catalog[cname]

        #shuffle order
        idx = np.arange(len(pts))
        np.random.shuffle(idx)
        pts_shuffled = pts[idx]

        pred, score = match_constellation(pts_shuffled, catalog)

        ok = "ok" if pred == cname else "no"
        if pred == cname:
            correct += 1

        print(f"  True: {cname:3s}  ?  Pred: {pred:3s}   score={score:.6f}   {ok}")

    print("\n SUMMARY ")
    print(f"  Correct: {correct}/{total}")



if __name__ == "__main__":
    main()
