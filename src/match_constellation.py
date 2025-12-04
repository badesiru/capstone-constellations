import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull


def compute_shape_features(points):
    """Compute geometric features for shape discrimination."""
    pts = np.array(points, dtype=float)
    if pts.shape[0] < 3:
        return None
    
    # Normalize first
    pts = pts - pts.mean(axis=0)
    
    features = {}
    
    # Convex hull area ratio (area / bounding box area)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # 'volume' is area in 2D
    except:
        hull_area = 0.0
    
    # Bounding box
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    bbox_area = x_range * y_range if x_range > 0 and y_range > 0 else 1.0
    
    features['convex_ratio'] = hull_area / bbox_area if bbox_area > 0 else 0.0
    features['aspect_ratio'] = max(x_range, y_range) / max(min(x_range, y_range), 1e-9)
    
    # Compactness (how spread out the points are)
    dists = np.linalg.norm(pts, axis=1)
    features['spread'] = np.std(dists) if len(dists) > 0 else 0.0
    
    return features


def shapes_are_compatible(features1, features2, tolerance=0.3):
    """Check if two shapes are geometrically compatible."""
    if features1 is None or features2 is None:
        return True  # Can't filter, allow matching
    
    # Check aspect ratio similarity (elongated vs compact)
    aspect_diff = abs(np.log(features1['aspect_ratio']) - np.log(features2['aspect_ratio']))
    if aspect_diff > tolerance:
        return False
    
    # Check convex hull ratio similarity
    convex_diff = abs(features1['convex_ratio'] - features2['convex_ratio'])
    if convex_diff > tolerance:
        return False
    
    return True


def normalize_points(points):
    pts = np.array(points, dtype=float)
    if pts.shape[0] == 0:
        return pts
    # center
    pts -= pts.mean(axis=0)
    # scale
    maxd = np.max(np.linalg.norm(pts, axis=1))
    if maxd > 0:
        pts /= maxd
    return pts


def kabsch(P, Q):

    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    R = np.dot(V, Wt)
    return R


def rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))


def weighted_rmsd(P, Q, weights):
    """Calculate RMSD with weights (brighter stars = higher weight)"""
    squared_dists = np.sum((P - Q) ** 2, axis=1)
    weighted_mean = np.sum(weights * squared_dists) / np.sum(weights)
    return np.sqrt(weighted_mean)


def best_rmsd_between(P, Q, size_penalty=0.10):

    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    nP = P.shape[0]
    nQ = Q.shape[0]

    if nP < 3 or nQ < 3:
        return np.inf
    
    Pn = normalize_points(P)
    Qn = normalize_points(Q)

    dist_matrix = np.linalg.norm(Pn[:, None, :] - Qn[None, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    Pm = Pn[row_ind]
    Qm = Qn[col_ind]

    Qm_flipped = Qm.copy()
    Qm_flipped[:, 1] *= -1.0


    R1 = kabsch(Pm, Qm)
    P_aligned_1 = Pm @ R1
    score1 = rmsd(P_aligned_1, Qm)
    R2 = kabsch(Pm, Qm_flipped)
    P_aligned_2 = Pm @ R2
    score2 = rmsd(P_aligned_2, Qm_flipped)

    base_score = min(score1, score2)
    size_mismatch = abs(nP - nQ)
    final_score = base_score + size_penalty * size_mismatch

    return final_score


def match_constellation(image_points, catalog):

    scores = []

    P = np.array(image_points, dtype=float)
    if P.shape[0] < 3:
        return None, float("inf"), []
    
    # Compute shape features for detected constellation
    P_features = compute_shape_features(P)

    for cname, coords in catalog.items():
        Q = np.array(coords, dtype=float)
        if Q.shape[0] < 3:
            continue
        
        # Compute catalog shape features and filter incompatible shapes
        Q_features = compute_shape_features(Q)
        if not shapes_are_compatible(P_features, Q_features, tolerance=0.22):
            continue  # Skip geometrically incompatible candidates

        try:
            score = best_rmsd_between(P, Q)
        except Exception:
            continue

        scores.append((cname, score, len(Q)))

    if not scores:
        return None, float("inf"), []

    scores.sort(key=lambda x: x[1])
    best_const, best_score, best_n = scores[0]

    print("Top 5 matches:")
    for cname, s, n in scores[:5]:
        print(f"  {cname:3s}  score={s:.6f}  n_cat_stars={n}")


    for target in ["Ori", "UMa", "Boo", "Ser", "Sge", "Cru"]:
        for cname, s, n in scores:
            if cname == target:
                print(f"  [{target}] score={s:.6f}, n_cat_stars={n}")
                break

    # Return best match, score, and top 5 list (name, score pairs)
    top_5 = [(cname, score) for cname, score, n in scores[:5]]
    return best_const, best_score, top_5


def match_constellation_multiscale(image_points, catalogs):
    """
    Match against multiple catalog scales (8, 10, 12 stars).
    Choose the scale closest to detected star count, then pick best match.
    """
    P = np.array(image_points, dtype=float)
    n_detected = P.shape[0]
    
    if n_detected < 3:
        return None, float("inf"), []
    
    # Choose catalog scale closest to detected stars
    available_scales = sorted(catalogs.keys())
    best_scale = min(available_scales, key=lambda s: abs(s - n_detected))
    
    # Match using the best scale
    return match_constellation(image_points, catalogs[best_scale])
