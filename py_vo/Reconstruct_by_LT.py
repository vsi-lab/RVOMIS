import numpy as np


def Reconstruct_by_LT(Rs, Ts, N, inliers, K):
    K_inv = np.linalg.inv(K)
    skew = lambda T: np.array([[0, -T[2, 0], T[1, 0]], [T[2, 0], 0, -T[0, 0]], [-T[1, 0], T[0, 0], 0]])

    Rel_Rs = [np.eye(3) for _ in range(N)]
    Rel_Ts = [np.zeros((3, 1)) for _ in range(N)]
    for i in range(1, N):
        Rel_Rs[i] = Rs[:, :, i - 1]
        Rel_Ts[i] = Ts[:, i - 1 : i]

    Gamma_w = np.zeros((3, inliers.shape[1]))
    for i in range(inliers.shape[1]):
        A_rows = []
        offset = 0
        for v in range(N):
            gamma_i = K_inv @ np.vstack([inliers[offset : offset + 2, i : i + 1], [[1.0]]])
            offset += 2
            A_rows.append(skew(gamma_i) @ np.hstack([Rel_Rs[v], Rel_Ts[v]]))
        A = np.vstack(A_rows)
        _, _, Vt = np.linalg.svd(A)
        p3d = Vt[-1, :]
        p3d = p3d / p3d[-1]
        Gamma_w[:, i] = p3d[:3]
    return Gamma_w
