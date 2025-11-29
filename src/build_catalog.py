import numpy as np
import pandas as pd

from parse_stellarium import build_catalog_from_stellarium


def project_radec(ra, dec):

    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)

    coords = np.column_stack((x, y))


    coords -= coords.mean(axis=0)


    dists = np.linalg.norm(coords, axis=1)
    maxd = np.max(dists) if dists.size > 0 else 0.0
    if maxd > 0:
        coords /= maxd

    return coords


def build_catalog_shapes(top_n=12):

    raw_catalog = build_catalog_from_stellarium(
        index_path=r"C:\Program Files\Stellarium\skycultures\modern_iau\index.json",
        hyg_path="data/hyg/hyg_v38.csv.gz"
    )

    catalog_shapes = {}

    for const_name, stars in raw_catalog.items():

        if stars.shape[0] == 0:
            continue

        hips = stars[:, 0]
        ra   = stars[:, 1]
        dec  = stars[:, 2]
        mags = stars[:, 3]


        valid = ~np.isnan(mags)
        if not np.any(valid):
            continue

        ra   = ra[valid]
        dec  = dec[valid]
        mags = mags[valid]

        if ra.size == 0:
            continue


        order = np.argsort(mags)
        k = min(top_n, order.size)
        sel = order[:k]

        ra_sel   = ra[sel]
        dec_sel  = dec[sel]
        mags_sel = mags[sel] 

        coords = project_radec(ra_sel, dec_sel)

        if coords.size == 0 or np.isnan(coords).any():
            continue

        catalog_shapes[const_name] = coords

    return catalog_shapes


if __name__ == "__main__":
    catalog = build_catalog_shapes()
    print("Built shapes for", len(catalog), "constellations")
