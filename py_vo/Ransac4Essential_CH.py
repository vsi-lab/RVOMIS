from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from .ComputeEssentialMatrix import ComputeEssentialMatrix


def _score_single(seed: int, idx_pool: np.ndarray, matchImg1: np.ndarray, matchImg2: np.ndarray, K: np.ndarray, thresh: float):
    rng = np.random.RandomState(seed)
    subset = rng.choice(idx_pool, size=5, replace=False)
    gamma1 = matchImg1[:2, subset]
    gamma2 = matchImg2[:2, subset]
    Es = ComputeEssentialMatrix(gamma1, gamma2, K)
    if not Es:
        return None

    K_inv = np.linalg.inv(K)
    best = (0, None, None)
    for E in Es:
        if E.shape != (3, 3):
            continue
        calE = K_inv.T @ E @ K_inv
        A = calE[0] @ matchImg1
        B = calE[1] @ matchImg1
        C = calE[2] @ matchImg1
        numer = np.abs(A * matchImg2[0] + B * matchImg2[1] + C)
        denom = np.sqrt(A * A + B * B) + 1e-12
        dist = numer / denom
        inliers = np.where(dist < thresh)[0]
        if inliers.size > best[0]:
            best = (inliers.size, E, inliers)
    return best


def Ransac4Essential_CH(PARAMS, matchImg1: np.ndarray, matchImg2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    top_n = max(5, int(round(PARAMS.TOP_N_RATIO_RANK_ORDERED_LIST * matchImg1.shape[1])))
    idx_pool = np.arange(top_n)

    # Random seeds per run
    rng_master = np.random.default_rng()
    seeds = rng_master.integers(0, 2**32 - 1, size=PARAMS.RANSAC_ITERATIONS, dtype=np.uint64)
    best_overall = (0, None, None)

    if getattr(PARAMS, "N_JOBS", 1) > 1:
        print(f"[RANSAC] Using {PARAMS.N_JOBS} workers for essential matrix.")
        with ProcessPoolExecutor(max_workers=PARAMS.N_JOBS) as ex:
            futures = {ex.submit(_score_single, int(seed), idx_pool, matchImg1, matchImg2, K, PARAMS.INLIER_THRESH): seed for seed in seeds}
            for fut in as_completed(futures):
                res = fut.result()
                if res and res[0] > best_overall[0]:
                    best_overall = res
    else:
        for seed in seeds:
            res = _score_single(int(seed), idx_pool, matchImg1, matchImg2, K, PARAMS.INLIER_THRESH)
            if res and res[0] > best_overall[0]:
                best_overall = res

    _, finalE, inlierIndx = best_overall
    return finalE, inlierIndx
