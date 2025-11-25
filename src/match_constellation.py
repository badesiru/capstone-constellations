import numpy as np


#normalize and x,y coordinates
def normalize_points(points):

    pts = np.array(points, dtype=float)
    pts -= pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist

    return pts

#computing the distance ratio for sets of normalized points
def compute_distance_ratios(points):

    pts = np.array(points)
    n = len(pts)

    ratios = []

    for i in range(n):
        for j in range(i+1, n):
            d1 = np.linalg.norm(pts[i] - pts[j])
            if d1 == 0:
                continue
            for k in range(j+1, n):
                d2 = np.linalg.norm(pts[i] - pts[k])
                if d2 == 0:
                    continue
                ratios.append(d1 / d2)

    return np.sort(ratios)


#compare two ratio with l1 differences, so lower the score the betterhte match
def compare_ratio_sets(r1, r2):


    L = min(len(r1), len(r2))
    r1 = r1[:L]
    r2 = r2[:L]

    return np.sum(np.abs(r1 - r2))



#dictionary of coordinates, returns best match 
def match_constellation(image_points, catalog_constellations):

    img_norm = normalize_points(image_points)
    img_ratios = compute_distance_ratios(img_norm)

    best_const = None
    best_score = float("inf")

    for cname, coords in catalog_constellations.items():
        cat_ratios = compute_distance_ratios(coords)
        score = compare_ratio_sets(img_ratios, cat_ratios)

        if score < best_score:
            best_score = score
            best_const = cname

    return best_const, best_score
