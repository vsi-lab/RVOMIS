from __future__ import annotations

import numpy as np

from .ComputeEssentialMatrix_matlab import ComputeEssentialMatrix
from .matlab_semantics import matlab_round


def Ransac4Essential_CH(PARAMS, matchImg1, matchImg2, K, rng, frame_id=1, return_diagnostics=False):
    inlier_num_max = 0
    final_e = None
    final_inliers = None
    selected_iteration = -1
    selected_candidate = -1
    top_n = matlab_round(PARAMS.TOP_N_RATIO_RANK_ORDERED_LIST * matchImg1.shape[1])
    if top_n < 5:
        raise ValueError(f"MATLAB top-ranked list has only {top_n} entries")
    inv_k = np.linalg.inv(K)
    for iteration in range(PARAMS.RANSAC_ITERATIONS):
        indices = rng.sample_without_replacement(
            top_n, 5, stage="essential", frame_id=frame_id, iteration=iteration
        )
        candidates = ComputeEssentialMatrix(matchImg1[:2, indices], matchImg2[:2, indices], K)
        for candidate_index, essential in enumerate(candidates):
            cal_e = inv_k.T @ essential @ inv_k
            A = cal_e[0, :] @ matchImg1
            B = cal_e[1, :] @ matchImg1
            C = cal_e[2, :] @ matchImg1
            numerator = np.abs(A * matchImg2[0, :] + B * matchImg2[1, :] + C)
            denominator = np.sqrt(A*A + B*B)
            with np.errstate(divide="ignore", invalid="ignore"):
                distance = numerator / denominator
            inliers = np.flatnonzero(distance < PARAMS.INLIER_THRESH)
            if inliers.size > inlier_num_max:
                inlier_num_max = int(inliers.size)
                final_e = essential.copy()
                final_inliers = inliers
                selected_iteration = iteration
                selected_candidate = candidate_index
    if final_e is None:
        raise RuntimeError("MATLAB-compatible essential RANSAC found no candidate with positive support")
    diagnostics = {
        "top_n": int(top_n),
        "selected_iteration": int(selected_iteration),
        "selected_candidate": int(selected_candidate),
        "inlier_count": int(inlier_num_max),
    }
    if return_diagnostics:
        return final_e, final_inliers, diagnostics
    return final_e, final_inliers
