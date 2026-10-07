from __future__ import annotations

import numpy as np
from scipy.linalg import eig, svd

from .five_point_coefficients import compute_coefficients


def _polyeig4(A0, A1, A2, A3, E1, E2, E3, E4):
    n, p = A0.shape[0], 3
    A = np.eye(n * p, dtype=A0.dtype)
    A[:n, :n] = A0
    B = np.zeros_like(A)
    # MATLAB's column-major linear indexing creates shifted identity blocks.
    B[n:2*n, :n] = np.eye(n)
    B[2*n:3*n, n:2*n] = np.eye(n)
    B[:n, :] = -np.hstack([A1, A2, A3])
    values, vectors = eig(A, B)
    candidates = []
    for index, value in enumerate(values):
        if not np.isfinite(value) or abs(value.imag) >= 1e-8:
            continue
        e = float(value.real)
        V = vectors[:, index].real.reshape(n, p, order="F")
        residual = (A0 + e*A1 + e*e*A2 + e*e*e*A3) @ V
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.sum(np.abs(residual), axis=0) / np.sum(np.abs(V), axis=0)
        column = int(np.nanargmin(normalized))
        v = V[[7, 8, 9], column]
        v = v / np.linalg.norm(v)
        essential = v[0]*E1 + v[1]*E2 + (v[2]*e)*E3 + v[2]*E4
        if np.all(np.isfinite(essential)) and np.linalg.matrix_rank(essential) >= 2:
            candidates.append(essential)
    return candidates


def five_point_algorithm_self(matches):
    q1, q2 = matches[:, :, 0], matches[:, :, 1]
    Q = np.column_stack([
        q2[:, 0]*q1[:, 0], q2[:, 0]*q1[:, 1], q2[:, 0]*q1[:, 2],
        q2[:, 1]*q1[:, 0], q2[:, 1]*q1[:, 1], q2[:, 1]*q1[:, 2],
        q2[:, 2]*q1[:, 0], q2[:, 2]*q1[:, 1], q2[:, 2]*q1[:, 2],
    ])
    _, _, vh = svd(Q)
    basis = vh[-4:].T
    E1, E2, E3, E4 = [basis[:, i].reshape(3, 3) for i in range(4)]
    C = compute_coefficients(E1, E2, E3, E4)
    C1 = C[:, np.r_[0:4, 10:13, 16:18, 19]]
    C2 = np.hstack([np.zeros((10, 4)), C[:, np.r_[4:7, 13:15, 18]]])
    C3 = np.hstack([np.zeros((10, 7)), C[:, np.r_[7:9, 15]]])
    C4 = np.hstack([np.zeros((10, 9)), C[:, [9]]])
    return _polyeig4(C1, C2, C3, C4, E1, E2, E3, E4)
