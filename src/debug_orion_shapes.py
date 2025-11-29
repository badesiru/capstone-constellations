# src/debug_orion_shapes.py

import numpy as np

from parse_stellarium import build_catalog_from_stellarium
from build_catalog import build_catalog_shapes, project_radec as cat_project
from match_constellation import normalize_points

def main():
    stell_path = r"C:\Program Files\Stellarium\skycultures\modern_iau\index.json"
    hyg_path   = r"data/hyg/hyg_v38.csv.gz"

    # 1) Load raw Stellarium+HYG catalog (HIP, ra, dec, mag)
    raw_catalog = build_catalog_from_stellarium(stell_path, hyg_path)


    catalog_shapes = build_catalog_shapes(top_n=8)

    # ---------- ORI FROM CATALOG_SHAPES ----------
    ori_cat = catalog_shapes["Ori"]     # shape (8, 2)
    print("Catalog Ori coords (normalized):")
    print(ori_cat)
    print("Catalog Ori shape:", ori_cat.shape)

    # ---------- ORI FROM RAW CATALOG (using SAME logic as generate_synthetic) ----------
    ori_raw = raw_catalog["Ori"]        # columns: [hip, ra, dec, mag]
    hips = ori_raw[:, 0]
    ra   = ori_raw[:, 1]
    dec  = ori_raw[:, 2]
    mag  = ori_raw[:, 3]

    # drop NaN mags
    valid = ~np.isnan(mag)
    hips = hips[valid]
    ra   = ra[valid]
    dec  = dec[valid]
    mag  = mag[valid]

    # select top-8 brightest
    order = np.argsort(mag)  # lower mag = brighter
    sel = order[:8]

    hips_sel = hips[sel]
    ra_sel   = ra[sel]
    dec_sel  = dec[sel]
    mag_sel  = mag[sel]

    print("\nRaw Ori top-8 (HIP, mag):")
    for h, m in zip(hips_sel, mag_sel):
        print(f"  HIP={int(h):6d}, mag={m:.3f}")

    # project + normalize using SAME functions as catalog
    coords_syn = cat_project(ra_sel, dec_sel)
    coords_syn = normalize_points(coords_syn)

    print("\nSynthetic-style Ori coords (normalized, top-8):")
    print(coords_syn)
    print("Synthetic-style Ori shape:", coords_syn.shape)

    # ---------- COMPARE ----------
    if ori_cat.shape == coords_syn.shape:
        diffs = np.linalg.norm(ori_cat - coords_syn, axis=1)
        print("\nPer-star coord diffs (catalog vs synthetic-style):")
        for i, d in enumerate(diffs):
            print(f"  star {i}: diff={d:.6e}")
        print("Mean diff:", diffs.mean())
    else:
        print("\nSHAPE MISMATCH between catalog Ori and synthetic-style Ori:")
        print("  catalog shape:", ori_cat.shape)
        print("  synthetic shape:", coords_syn.shape)

if __name__ == "__main__":
    main()
