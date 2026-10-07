"""Compatibility wrapper for validated linear triangulation."""

from .rvo_closed_loop_replay import reconstruct_by_lt


def Reconstruct_by_LT(Rs, Ts, N, inliers, K):
    return reconstruct_by_lt(Rs, Ts, N, inliers, K)


__all__ = ["Reconstruct_by_LT"]
