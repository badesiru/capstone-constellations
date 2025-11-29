import numpy as np
from scipy.optimize import linear_sum_assignment


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


def best_rmsd_between(P, Q, size_penalty=0.05):

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
        return None, float("inf")

    for cname, coords in catalog.items():
        Q = np.array(coords, dtype=float)
        if Q.shape[0] < 3:
            continue

        try:
            score = best_rmsd_between(P, Q)
        except Exception:
            continue

        scores.append((cname, score, len(Q)))

    if not scores:
        return None, float("inf")

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

    return best_const, best_score
