"""
ComputeEssentialMatrix
Mirrors MATLAB signature: takes 2x5 pixel arrays and intrinsic K, returns list of candidate Es.
Prefers local five-point implementation; falls back to OpenCV if unavailable/fails.
"""
from typing import List
import numpy as np
import cv2

try:
    from .fivePointAlgorithmSelf import five_point_algorithm_self  # type: ignore
    _HAS_FIVEPOINT = True
except Exception:
    _HAS_FIVEPOINT = False


def ComputeEssentialMatrix(pixels1: np.ndarray, pixels2: np.ndarray, K: np.ndarray) -> List[np.ndarray]:
    solver_used = None
    pts1 = pixels1[:2, :].T.astype(np.float64)
    pts2 = pixels2[:2, :].T.astype(np.float64)

    # Prefer local five-point implementation (use first 5 correspondences).
    if _HAS_FIVEPOINT and pixels1.shape[1] >= 5:
        try:
            K_inv = np.linalg.inv(K)
            p1 = np.vstack([pixels1[:2, :], np.ones((1, pixels1.shape[1]))])
            p2 = np.vstack([pixels2[:2, :], np.ones((1, pixels2.shape[1]))])
            matches = np.zeros((5, 3, 2), dtype=float)
            for i in range(5):
                m1 = K_inv @ p1[:, i]
                m2 = K_inv @ p2[:, i]
                matches[i, :, 0] = m1
                matches[i, :, 1] = m2
            Es = five_point_algorithm_self(matches)
            if Es:
                return [np.array(E, dtype=float) for E in Es]
        except Exception:
            solver_used = None

    pts1 = pixels1[:2, :].T.astype(np.float64)
    pts2 = pixels2[:2, :].T.astype(np.float64)
    # Fallback: OpenCV five-point solver with minimal internal RANSAC.
    if hasattr(cv2, "USAC_NISTER"):
        E, _ = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=K,
            method=cv2.USAC_NISTER,
            prob=0.999,
            threshold=1e-12,
            maxIters=1,
        )
    else:
        # Fallback for older OpenCV: still minimize internal RANSAC influence by limiting iterations
        E, _ = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1e-12,
            maxIters=1,
        )
    if E is None:
        return []
    candidates: List[np.ndarray] = []
    if E.ndim == 1 and E.size == 9:
        E = E.reshape(3, 3)
    if E.ndim == 2:
        if E.shape == (3, 3):
            candidates.append(E)
        elif E.shape[0] % 3 == 0 and E.shape[1] == 3:
            num = E.shape[0] // 3
            for i in range(num):
                block = E[3 * i : 3 * (i + 1), :]
                if block.shape == (3, 3):
                    candidates.append(block)
    elif E.ndim == 3:
        for i in range(E.shape[0]):
            block = E[i]
            if block.shape == (3, 3):
                candidates.append(block)
    return candidates
