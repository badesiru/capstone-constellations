
import pandas as pd
from pathlib import Path
import numpy as np

from project_catalog import load_constellation, project_constellation
from parse_stellarium import build_catalog_from_stellarium


#keeps only the brightest n stars - lowest magnitude is the brightest
def filter_bright_stars(df, max_mag=4.5, top_n=8):

    df = df.dropna(subset=["mag"])
    df = df[df["mag"] <= max_mag]
    df = df.sort_values(by="mag")  
    return df.head(top_n)

def build_catalog_shapes(max_mag=None):
    catalog = build_catalog_from_stellarium(
        index_path=r"C:\Program Files\Stellarium\skycultures\modern_iau\index.json",
        hyg_path="data/hyg/hyg_v38.csv.gz"
    )
    return catalog

if __name__ == "__main__":
    build_catalog_shapes()
