from __future__ import annotations

import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .P3P_LambdaTwist_matlab import P3P_LambdaTwist


def rotation_matrix_to_euler_zyx(matrix):
    return Rotation.from_matrix(matrix).as_euler("ZYX")


def euler_zyx_to_rotation_matrix(euler):
    return Rotation.from_euler("ZYX", euler).as_matrix()


def reprojection_error(rotation, translation, points3d, points2d, K):
    projected = K @ (rotation @ points3d + translation)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = projected / projected[2:3, :]
    return np.linalg.norm(projected - points2d, axis=0)


def Msac4AbsolutePose_CHM(
    PARAMS,
    Points3D,
    Points2D,
    K,
    rng,
    frame_id,
    p3p_solver=P3P_LambdaTwist,
    return_diagnostics=False,
):
    min_total_cost = np.inf
    best_r = np.eye(3)
    best_t = np.zeros((3, 1))
    best_inliers = np.empty(0, dtype=np.int64)
    selected_iteration = -1
    selected_candidate = -1
    points2d_metric = np.linalg.inv(K) @ Points2D
    sampled_triplets = np.empty((PARAMS.RANSAC_ITERATIONS, 3), dtype=np.int64)
    for iteration in range(PARAMS.RANSAC_ITERATIONS):
        indices = rng.sample_without_replacement(
            Points2D.shape[1], 3, stage="p3p", frame_id=frame_id, iteration=iteration
        )
        sampled_triplets[iteration] = indices
        rotations, translations = p3p_solver(points2d_metric[:, indices], Points3D[:, indices])
        for candidate_index in range(rotations.shape[2]):
            rotation = rotations[:, :, candidate_index]
            translation = translations[:, candidate_index:candidate_index+1]
            errors = reprojection_error(rotation, translation, Points3D, Points2D, K)
            total_cost = float(np.sum(np.minimum(errors, PARAMS.INLIER_THRESH)))
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_r = rotation.copy()
                best_t = translation.copy()
                best_inliers = np.flatnonzero(errors < PARAMS.INLIER_THRESH)
                selected_iteration = iteration
                selected_candidate = candidate_index
    before = np.hstack([best_r, best_t])
    initial_lm_cost = np.nan
    final_lm_cost = np.nan
    nfev = 0
    status = 0
    message = "no inliers"
    start = time.perf_counter()
    if best_inliers.size:
        p3 = Points3D[:, best_inliers]
        p2 = Points2D[:, best_inliers]
        x0 = np.r_[rotation_matrix_to_euler_zyx(best_r), best_t.ravel()]

        def residuals(x):
            rotation = euler_zyx_to_rotation_matrix(x[:3])
            translation = x[3:].reshape(3, 1)
            projected = K @ (rotation @ p3 + translation)
            projected /= projected[2:3, :]
            # MATLAB fills [x1,y1,x2,y2,...].
            return (p2[:2, :] - projected[:2, :]).T.ravel()

        initial_vector = residuals(x0)
        initial_lm_cost = float(initial_vector @ initial_vector)
        result = least_squares(residuals, x0, method="lm")
        best_r = euler_zyx_to_rotation_matrix(result.x[:3])
        best_t = result.x[3:].reshape(3, 1)
        final_vector = residuals(result.x)
        final_lm_cost = float(final_vector @ final_vector)
        nfev, status, message = int(result.nfev), int(result.status), str(result.message)
    elapsed = time.perf_counter() - start
    after = np.hstack([best_r, best_t])
    diagnostics = {
        "selected_iteration": int(selected_iteration),
        "selected_candidate": int(selected_candidate),
        "selected_msac_cost": float(min_total_cost),
        "sampled_triplets": sampled_triplets,
        "pose_before_refinement": before,
        "pose_after_refinement": after,
        "initial_lm_cost": initial_lm_cost,
        "final_lm_cost": final_lm_cost,
        "lm_nfev": nfev,
        "lm_status": status,
        "lm_message": message,
        "final_reprojection_error": reprojection_error(best_r, best_t, Points3D, Points2D, K),
    }
    if return_diagnostics:
        return best_r, best_t, best_inliers, elapsed, diagnostics
    return best_r, best_t, best_inliers, elapsed
