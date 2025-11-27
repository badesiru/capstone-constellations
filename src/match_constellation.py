import numpy as np
from scipy.spatial import Delaunay

def normalize(points):
    pts = np.array(points, dtype=float)
    pts -= pts.mean(axis=0)
    maxd = np.max(np.linalg.norm(pts, axis=1))
    return pts / maxd if maxd > 0 else pts

def triangle_signature(points):
    pts = normalize(points)
    if len(pts) < 3:
        return np.zeros((0, 3))

    tri = Delaunay(pts)
    signatures = []

    for simplex in tri.simplices:
        p = pts[simplex]
        d01 = np.linalg.norm(p[0] - p[1])
        d12 = np.linalg.norm(p[1] - p[2])
        d20 = np.linalg.norm(p[2] - p[0])
        signatures.append(sorted([d01, d12, d20]))

    return np.array(signatures)

def signature_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf

    L = min(len(A), len(B))
    A = np.sort(A, axis=0)[:L]
    B = np.sort(B, axis=0)[:L]

    return np.mean(np.abs(A - B))

def match_constellation(image_points, catalog):
    sig_img = triangle_signature(image_points)

    best = None
    best_score = np.inf

    for cname, coords in catalog.items():
        sig_cat = triangle_signature(coords)
        score = signature_distance(sig_img, sig_cat)
        if score < best_score:
            best_score = score
            best = cname

    return best, best_score

