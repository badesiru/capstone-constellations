"""
Ultra-advanced constellation matching with magnitude weighting and all optimizations.
Target: 75-80% accuracy
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import wasserstein_distance


def compute_magnitude_weights(mags):
    """
    Compute weights from magnitudes. Brighter stars (lower magnitude) get higher weight.
    """
    if mags is None or len(mags) == 0:
        return None
    
    # Normalize magnitudes to [0, 1] range within this constellation
    mag_array = np.array(mags, dtype=float)
    mag_min = np.min(mag_array)
    mag_max = np.max(mag_array)
    
    if mag_max - mag_min < 0.01:
        # All similar brightness
        return np.ones(len(mags))
    
    # Normalize: 0 = brightest (lowest mag), 1 = dimmest (highest mag)
    norm_mags = (mag_array - mag_min) / (mag_max - mag_min)
    
    # Convert to weights: brightest gets weight ~3, dimmest gets weight ~1
    weights = 3.0 - 2.0 * norm_mags
    
    return weights


def compute_advanced_features(points, mags=None):
    """Compute advanced geometric features with optional magnitude info."""
    pts = np.array(points, dtype=float)
    if pts.shape[0] < 3:
        return None
    
    # Normalize first
    pts = pts - pts.mean(axis=0)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    
    features = {}
    
    # 1. Convex hull features
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        hull_perimeter = 0
        for simplex in hull.simplices:
            hull_perimeter += np.linalg.norm(pts[simplex[0]] - pts[simplex[1]])
        features['hull_area'] = hull_area
        features['hull_perimeter'] = hull_perimeter
        features['compactness'] = 4 * np.pi * hull_area / (hull_perimeter ** 2) if hull_perimeter > 0 else 0
    except:
        features['hull_area'] = 0.0
        features['hull_perimeter'] = 0.0
        features['compactness'] = 0.0
    
    # 2. Bounding box features
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    bbox_area = x_range * y_range if x_range > 0 and y_range > 0 else 1e-9
    
    features['aspect_ratio'] = max(x_range, y_range) / max(min(x_range, y_range), 1e-9)
    features['convex_ratio'] = features['hull_area'] / bbox_area if bbox_area > 0 else 0.0
    
    # 3. Distance distribution features
    dist_mat = distance_matrix(pts, pts)
    upper_tri = dist_mat[np.triu_indices_from(dist_mat, k=1)]
    if len(upper_tri) > 0:
        features['mean_distance'] = np.mean(upper_tri)
        features['std_distance'] = np.std(upper_tri)
        
        # Distance histogram (10 bins) - normalized
        hist, _ = np.histogram(upper_tri, bins=10, range=(0, 1.5))
        features['distance_histogram'] = hist / (np.sum(hist) + 1e-9)
    else:
        features['mean_distance'] = 0
        features['std_distance'] = 0
        features['distance_histogram'] = np.zeros(10)
    
    # 4. Radial distribution
    dists_from_center = np.linalg.norm(pts, axis=1)
    features['radial_std'] = np.std(dists_from_center)
    features['radial_mean'] = np.mean(dists_from_center)
    
    # 5. Magnitude-weighted centroid offset (if magnitudes available)
    if mags is not None and len(mags) == len(pts):
        weights = compute_magnitude_weights(mags)
        weighted_centroid = np.average(pts, axis=0, weights=weights)
        features['magnitude_centroid_offset'] = np.linalg.norm(weighted_centroid)
        features['has_magnitudes'] = True
    else:
        features['magnitude_centroid_offset'] = 0.0
        features['has_magnitudes'] = False
    
    return features


def advanced_shape_compatibility(feat1, feat2, tolerance=0.25):
    """Advanced geometric compatibility check."""
    if feat1 is None or feat2 is None:
        return True, 0.0
    
    incompatibility_score = 0.0
    
    # 1. Aspect ratio check (log space)
    aspect_diff = abs(np.log(feat1['aspect_ratio']) - np.log(feat2['aspect_ratio']))
    if aspect_diff > tolerance:
        return False, aspect_diff
    incompatibility_score += aspect_diff
    
    # 2. Convex ratio check
    convex_diff = abs(feat1['convex_ratio'] - feat2['convex_ratio'])
    if convex_diff > tolerance:
        return False, convex_diff
    incompatibility_score += convex_diff
    
    # 3. Compactness check
    compact_diff = abs(feat1['compactness'] - feat2['compactness'])
    if compact_diff > tolerance * 1.5:
        return False, compact_diff
    incompatibility_score += compact_diff * 0.5
    
    # 4. Distance histogram similarity
    hist_distance = wasserstein_distance(
        np.arange(10), np.arange(10),
        feat1['distance_histogram'], feat2['distance_histogram']
    )
    if hist_distance > tolerance * 10:
        return False, hist_distance
    incompatibility_score += hist_distance * 0.1
    
    # 5. Radial distribution check
    radial_diff = abs(feat1['radial_std'] - feat2['radial_std'])
    if radial_diff > tolerance:
        return False, radial_diff
    incompatibility_score += radial_diff * 0.3
    
    # 6. Magnitude centroid check (if both have magnitudes)
    if feat1['has_magnitudes'] and feat2['has_magnitudes']:
        mag_centroid_diff = abs(feat1['magnitude_centroid_offset'] - feat2['magnitude_centroid_offset'])
        if mag_centroid_diff > tolerance * 0.5:
            return False, mag_centroid_diff
        incompatibility_score += mag_centroid_diff * 0.2
    
    return True, incompatibility_score


def normalize_points(points):
    pts = np.array(points, dtype=float)
    if pts.shape[0] == 0:
        return pts
    
    pts = pts - pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    max_d = np.max(dists)
    if max_d > 0:
        pts /= max_d
    return pts


def kabsch(P, Q):
    C = P.T @ Q
    U, _, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))


def weighted_rmsd(P, Q, weights):
    """RMSD with per-point weights (for magnitude weighting)."""
    if weights is None or len(weights) != len(P):
        return rmsd(P, Q)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize
    
    sq_dists = np.sum((P - Q) ** 2, axis=1)
    return np.sqrt(np.sum(weights * sq_dists))


def icp_refinement(P, Q, weights=None, max_iterations=5, tolerance=1e-6):
    """ICP with optional magnitude weighting."""
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)
    
    if P.shape[0] < 3 or Q.shape[0] < 3:
        return P, Q, np.inf
    
    P_aligned = P.copy()
    prev_error = np.inf
    
    for iteration in range(max_iterations):
        # Find nearest neighbors
        dist_matrix = np.linalg.norm(P_aligned[:, None, :] - Q[None, :, :], axis=2)
        nearest_idx = np.argmin(dist_matrix, axis=1)
        
        # Align using Kabsch
        Q_matched = Q[nearest_idx]
        R = kabsch(P_aligned, Q_matched)
        P_aligned = P_aligned @ R
        
        # Calculate error (weighted if magnitudes available)
        if weights is not None:
            error = weighted_rmsd(P_aligned, Q_matched, weights)
        else:
            error = rmsd(P_aligned, Q_matched)
        
        # Check convergence
        if abs(prev_error - error) < tolerance:
            break
        
        prev_error = error
    
    return P_aligned, Q_matched, error


def ultra_advanced_matching(P, Q, mags_P=None, mags_Q=None, size_penalty=0.07, use_icp=True):
    """
    Ultra-advanced RMSD matching with magnitude weighting.
    """
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    nP = P.shape[0]
    nQ = Q.shape[0]

    if nP < 3 or nQ < 3:
        return np.inf
    
    Pn = normalize_points(P)
    Qn = normalize_points(Q)
    
    # Compute magnitude weights if available
    weights_P = compute_magnitude_weights(mags_P) if mags_P is not None else None
    weights_Q = compute_magnitude_weights(mags_Q) if mags_Q is not None else None

    best_score = np.inf
    
    # Strategy 1: Standard Hungarian with magnitude weighting
    dist_matrix = np.linalg.norm(Pn[:, None, :] - Qn[None, :, :], axis=2)
    
    # Weight the distance matrix by magnitudes if available
    if weights_P is not None and weights_Q is not None:
        # Brighter stars should be matched more carefully
        weight_matrix = np.outer(weights_P, weights_Q)
        dist_matrix = dist_matrix / (weight_matrix + 0.5)
    
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    Pm = Pn[row_ind]
    Qm = Qn[col_ind]
    
    # Get corresponding weights
    if weights_P is not None:
        weights_matched = weights_P[row_ind]
    else:
        weights_matched = None
    
    # Try normal and flipped orientations
    for flip in [False, True]:
        Qm_test = Qm.copy()
        if flip:
            Qm_test[:, 1] *= -1.0
        
        # Initial Kabsch alignment
        R = kabsch(Pm, Qm_test)
        P_aligned = Pm @ R
        
        # Calculate score with magnitude weighting
        if weights_matched is not None:
            score = weighted_rmsd(P_aligned, Qm_test, weights_matched)
        else:
            score = rmsd(P_aligned, Qm_test)
        
        # Apply ICP refinement for promising matches
        if use_icp and score < 0.5:
            P_refined, Q_refined, icp_score = icp_refinement(
                P_aligned, Qm_test, weights=weights_matched
            )
            if icp_score < score:
                score = icp_score
        
        if score < best_score:
            best_score = score
    
    # Apply size penalty (reduced since we're more confident now)
    size_mismatch = abs(nP - nQ)
    final_score = best_score + size_penalty * size_mismatch
    
    return final_score


def match_constellation_ultra(image_points, catalog_with_mags, tolerance=0.13, use_icp=True):
    """
    Ultra-advanced matching with all optimizations including magnitude weighting.
    
    Args:
        image_points: Detected star coordinates
        catalog_with_mags: Dict with structure {name: {'coords': array, 'mags': array}}
        tolerance: Geometric compatibility tolerance
        use_icp: Whether to use ICP refinement
    """
    scores = []
    
    P = np.array(image_points, dtype=float)
    if P.shape[0] < 3:
        return None, float("inf"), []
    
    # Compute features for detected constellation (no magnitudes for detected stars)
    P_features = compute_advanced_features(P, mags=None)
    
    for cname, cat_data in catalog_with_mags.items():
        # Handle both dict format (with mags) and array format (backward compatible)
        if isinstance(cat_data, dict):
            Q = np.array(cat_data['coords'], dtype=float)
            mags_Q = cat_data.get('mags', None)
        else:
            Q = np.array(cat_data, dtype=float)
            mags_Q = None
        
        if Q.shape[0] < 3:
            continue
        
        # Compute catalog shape features
        Q_features = compute_advanced_features(Q, mags=mags_Q)
        
        # Check geometric compatibility
        compatible, incompat_score = advanced_shape_compatibility(
            P_features, Q_features, tolerance=tolerance
        )
        
        if not compatible:
            continue
        
        try:
            rmsd_score = ultra_advanced_matching(
                P, Q, mags_P=None, mags_Q=mags_Q, use_icp=use_icp
            )
        except Exception:
            continue
        
        # Combine scores
        combined_score = rmsd_score + incompat_score * 0.08
        
        scores.append((cname, combined_score, rmsd_score, len(Q)))
    
    if not scores:
        return None, float("inf"), []
    
    # Sort by combined score
    scores.sort(key=lambda x: x[1])
    best_const, best_combined, best_rmsd, best_n = scores[0]
    
    # Debug output (can be disabled for production)
    print("Top 3 matches:")
    for cname, combined, rmsd_s, n in scores[:3]:
        print(f"  {cname:3s}  combined={combined:.6f}  rmsd={rmsd_s:.6f}  n={n}")
    
    # Return best match, RMSD score, and top 5
    top_5 = [(cname, rmsd_s) for cname, combined, rmsd_s, n in scores[:5]]
    return best_const, best_rmsd, top_5


if __name__ == "__main__":
    # Test
    from build_catalog import build_catalog_shapes
    
    catalog = build_catalog_shapes(top_n=12, include_magnitudes=True)
    
    # Test with Orion
    test_data = catalog["Ori"]
    test_points = test_data['coords']
    
    # Shuffle to test
    idx = np.arange(len(test_points))
    np.random.shuffle(idx)
    test_points = test_points[idx]
    
    pred, score, top5 = match_constellation_ultra(test_points, catalog)
    print(f"\nTest result: {pred} with score {score:.6f}")
