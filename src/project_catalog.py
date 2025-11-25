import pandas as pd
import numpy as np
from pathlib import Path

#filtters out keeps only the britghtest stars
def filter_bright_stars(df, max_mag=4.5):
    df = df.dropna(subset=["mag"])
    df = df[df["mag"] <= max_mag]

    return df

#loads a filtered constellation csv 
def load_constellation(const_name):
    path = Path(__file__).resolve().parents[1] / "data" / "hyg" / f"{const_name}_stars.csv"
    df = pd.read_csv(path)
    return df

#converst to 2D plane and normalizes shape, for simplified shape for constellation matching 
def project_constellation(df):

    ra = df["ra"].values
    dec = df["dec"].values


    x = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = np.cos(np.radians(dec)) * np.sin(np.radians(ra))

    coords = np.column_stack((x, y))

    coords -= coords.mean(axis=0)

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    coords /= max_dist

    return coords


if __name__ == "__main__":
    df = load_constellation("Ori")


    df_bright = filter_bright_stars(df, max_mag=4.5)

    print("Bright stars remaining:", len(df_bright))

    coords = project_constellation(df_bright)

    print("\nProjected Orion bright-star shape:")
    print(coords[:10])
    print(f"Total bright stars in projected shape: {len(coords)}")

