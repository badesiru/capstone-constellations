import numpy as np
import pandas as pd
from pathlib import Path

def project_constellation(df):
  
    ra = df["ra"].values
    dec = df["dec"].values

    x = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = np.cos(np.radians(dec)) * np.sin(np.radians(ra))

    coords = np.column_stack((x, y))
    if coords.shape[0] == 0:
        return coords

    coords -= coords.mean(axis=0)

    dists = np.linalg.norm(coords, axis=1)
    max_dist = np.max(dists) if dists.size > 0 else 0.0

    if max_dist > 0:
        coords /= max_dist

    return coords

