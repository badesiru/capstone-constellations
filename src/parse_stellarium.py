import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_stellarium_constellations(index_path):
    index_path = Path(index_path)
    data = json.loads(index_path.read_text(encoding="utf-8"))

    constellations = {}

    for entry in data["constellations"]:
        cid = entry["id"]             
        abbr = cid.split()[-1]        
        lines = entry["lines"]

        hips = set()
        for poly in lines:
            for hip in poly:
                hips.add(hip)

        constellations[abbr] = {
            "hip_list": sorted(list(hips)),
            "lines": lines,
            "name": entry["common_name"]["english"]
        }

    return constellations


def build_catalog_from_stellarium(index_path, hyg_path):
    consts = load_stellarium_constellations(index_path)

    df = pd.read_csv(hyg_path, compression="gzip")

    catalog = {}

    for abbr, info in consts.items():
        hips = info["hip_list"]

        stars = df[df["hip"].isin(hips)]

        if stars.empty:
            continue

        catalog[abbr] = np.vstack([
            stars["hip"].values,
            stars["ra"].values,
            stars["dec"].values,
            stars["mag"].values
        ]).T

    return catalog


if __name__ == "__main__":
    stell_path = r"C:\Program Files\Stellarium\skycultures\modern_iau\index.json"
    hyg_path = r"data/hyg/hyg_v38.csv.gz" 

    consts = load_stellarium_constellations(stell_path)
    print(f"Loaded {len(consts)} constellations")

    catalog = build_catalog_from_stellarium(stell_path, hyg_path)
    print(f"Catalog built: {len(catalog)} constellations")
