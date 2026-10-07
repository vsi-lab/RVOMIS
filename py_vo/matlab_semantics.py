from __future__ import annotations

import math
import numpy as np


def matlab_round(value):
    """MATLAB round: ties go away from zero."""
    array = np.asarray(value, dtype=np.float64)
    result = np.sign(array) * np.floor(np.abs(array) + 0.5)
    if np.ndim(value) == 0:
        return int(result)
    return result.astype(np.int64)


def matlab_intersect_rows(a: np.ndarray, b: np.ndarray):
    """Sorted unique row intersection with first-occurrence ia/ib indices.

    This matches modern MATLAB ``intersect(A,B,'rows')`` ordering, including
    lexicographic sorting. Exact float equality is intentional: RVO intersects
    keypoints extracted independently from the same image.
    """
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Inputs must be 2D arrays with equal column counts")
    if a.dtype != b.dtype:
        dtype = np.result_type(a.dtype, b.dtype)
        a, b = a.astype(dtype), b.astype(dtype)
    a_first: dict[tuple, int] = {}
    b_first: dict[tuple, int] = {}
    for index, row in enumerate(a):
        a_first.setdefault(tuple(row.tolist()), index)
    for index, row in enumerate(b):
        b_first.setdefault(tuple(row.tolist()), index)
    keys = sorted(set(a_first).intersection(b_first))
    if not keys:
        return np.empty((0, a.shape[1]), dtype=a.dtype), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    common = np.asarray(keys, dtype=a.dtype)
    ia = np.asarray([a_first[key] for key in keys], dtype=np.int64)
    ib = np.asarray([b_first[key] for key in keys], dtype=np.int64)
    assert np.array_equal(common, a[ia]) and np.array_equal(common, b[ib])
    return common, ia, ib
