import time
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .P3P_LambdaTwist import P3P_LambdaTwist


def _msac_score_chunk(seed, iterations, Points3D, Points2D_metric, Points2D, K, T_thresh):
    rng = np.random.RandomState(seed)
    local_best = (np.inf, None, None, None)
    for _ in range(iterations):
        idx = rng.choice(Points2D.shape[1], size=3, replace=False)
        gamma = Points2D_metric[:, idx]
        Gamma = Points3D[:, idx]
        Rs, Ts = P3P_LambdaTwist(gamma, Gamma)
        for ci in range(Rs.shape[2]):
            R_ = Rs[:, :, ci]
            T_ = Ts[:, ci : ci + 1]
            Reproj = K @ (R_ @ Points3D + T_)
            Reproj /= Reproj[2, :]
            Reproj_Error = np.linalg.norm(Reproj - Points2D, axis=0)
            costs = np.minimum(Reproj_Error, T_thresh)
            current_total_cost = np.sum(costs)
            if current_total_cost < local_best[0]:
                local_best = (
                    current_total_cost,
                    R_,
                    T_,
                    np.where(Reproj_Error < T_thresh)[0],
                )
    return local_best


def Msac4AbsolutePose_CHM(PARAMS, Points3D: np.ndarray, Points2D: np.ndarray, K: np.ndarray):
    min_total_cost = np.inf
    R_max_support = np.eye(3)
    T_max_support = np.zeros((3, 1))
    inlier_index_max_support = np.array([], dtype=int)

    Points2D_metric = np.linalg.inv(K) @ Points2D
    T_thresh = PARAMS.INLIER_THRESH

    total_iters = PARAMS.RANSAC_ITERATIONS
    n_jobs = getattr(PARAMS, "N_JOBS", 1)
    if n_jobs > 1:
        chunk = math.ceil(total_iters / n_jobs)
        rng_master = np.random.default_rng()
        seeds = rng_master.integers(0, 2**32 - 1, size=n_jobs, dtype=np.uint64)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {
                ex.submit(
                    _msac_score_chunk,
                    int(seed),
                    chunk,
                    Points3D,
                    Points2D_metric,
                    Points2D,
                    K,
                    T_thresh,
                ): seed
                for seed in seeds
            }
            for fut in as_completed(futs):
                cost, R_, T_, inliers = fut.result()
                if cost < min_total_cost:
                    min_total_cost = cost
                    R_max_support = R_
                    T_max_support = T_
                    inlier_index_max_support = inliers
    else:
        rng = np.random.default_rng()
        for _ in range(total_iters):
            idx = rng.choice(Points2D.shape[1], size=3, replace=False)
            gamma = Points2D_metric[:, idx]
            Gamma = Points3D[:, idx]
            Rs, Ts = P3P_LambdaTwist(gamma, Gamma)
            for ci in range(Rs.shape[2]):
                R_ = Rs[:, :, ci]
                T_ = Ts[:, ci : ci + 1]
                Reproj = K @ (R_ @ Points3D + T_)
                Reproj /= Reproj[2, :]
                Reproj_Error = np.linalg.norm(Reproj - Points2D, axis=0)
                costs = np.minimum(Reproj_Error, T_thresh)
                current_total_cost = np.sum(costs)
                if current_total_cost < min_total_cost:
                    min_total_cost = current_total_cost
                    R_max_support = R_
                    T_max_support = T_
                    inlier_index_max_support = np.where(Reproj_Error < T_thresh)[0]

    Abs_R = R_max_support
    Abs_T = T_max_support
    inlierIndx = inlier_index_max_support

    if inlierIndx.size == 0:
        return Abs_R, Abs_T, inlierIndx, 0.0

    p33 = Points3D[:, inlierIndx]
    p22 = Points2D[:, inlierIndx]
    x0 = np.zeros(6)
    x0[:3] = rotation_matrix_to_euler(Abs_R)
    x0[3:] = Abs_T.ravel()

    def residuals(x):
        R1 = euler_to_rotation_matrix(x[:3])
        T1 = x[3:].reshape(3, 1)
        proj = K @ (R1 @ p33 + T1)
        proj /= proj[2, :]
        err = proj[:2, :] - p22[:2, :]
        return err.ravel()

    t0 = time.perf_counter()
    result = least_squares(residuals, x0, method="lm")
    t_opt = time.perf_counter() - t0
    Abs_R_y = euler_to_rotation_matrix(result.x[:3])
    Abs_T_y = result.x[3:].reshape(3, 1)
    return Abs_R_y, Abs_T_y, inlierIndx, t_opt


def rotation_matrix_to_euler(rot):
    # Match MATLAB rotm2eul default ('ZYX')
    return R.from_matrix(rot).as_euler("zyx")


def euler_to_rotation_matrix(euler):
    return R.from_euler("zyx", euler).as_matrix()
