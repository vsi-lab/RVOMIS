import numpy as np


def Get_Veridical_RT_from_E(E: np.ndarray, inliers_Img1: np.ndarray, inliers_Img2: np.ndarray, K: np.ndarray):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    T1 = U[:, 2]
    T2 = -U[:, 2]

    if np.linalg.det(R1) < 0 or np.linalg.det(R2) < 0:
        E = -E
        U, _, Vt = np.linalg.svd(E)
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        T1 = U[:, 2]
        T2 = -U[:, 2]

    Rall = [R1, R1, R2, R2]
    Tall = [T1, T2, T1, T2]
    e1 = np.array([1.0, 0.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    invK = np.linalg.inv(K)
    Gamma_1 = invK @ inliers_Img1
    Gamma_2 = invK @ inliers_Img2
    num_positive_depths = []

    for R12, T12 in zip(Rall, Tall):
        rho1 = (e1 @ T12 - (e3 @ T12) * (e1 @ Gamma_2)) / (
            (e3 @ (R12 @ Gamma_1)) * (e1 @ Gamma_2) - (e1 @ (R12 @ Gamma_1))
        )
        rho2 = ((e1 @ T12) * (e3 @ (R12 @ Gamma_1)) - (e3 @ T12) * (e1 @ (R12 @ Gamma_1))) / (
            (e3 @ (R12 @ Gamma_1)) * (e1 @ Gamma_2) - (e1 @ (R12 @ Gamma_1))
        )
        num_positive_depths.append(np.sum(rho1 > 0) + np.sum(rho2 > 0))

    max_idx = int(np.argmax(num_positive_depths))
    Rel_R = Rall[max_idx]
    Rel_T = Tall[max_idx].reshape(3, 1)
    return Rel_R, Rel_T
