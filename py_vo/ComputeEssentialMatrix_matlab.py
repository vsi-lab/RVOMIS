from __future__ import annotations

import numpy as np

from .fivePointAlgorithmSelf_matlab import five_point_algorithm_self


def ComputeEssentialMatrix(pixels1: np.ndarray, pixels2: np.ndarray, K: np.ndarray):
    """Current MATLAB source semantics: local five-point solver, no fallback."""
    if pixels1.shape[1] != 5 or pixels2.shape[1] != 5:
        raise ValueError("MATLAB ComputeEssentialMatrix expects exactly five correspondences")
    inv_k = np.linalg.inv(K)
    matches = np.zeros((5, 3, 2), dtype=np.float64)
    for index in range(5):
        matches[index, :, 0] = inv_k @ np.r_[pixels1[:2, index], 1.0]
        matches[index, :, 1] = inv_k @ np.r_[pixels2[:2, index], 1.0]
    values = five_point_algorithm_self(matches)
    return [np.asarray(value, dtype=np.float64) for value in values]
