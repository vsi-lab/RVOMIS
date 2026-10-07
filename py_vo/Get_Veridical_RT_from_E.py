"""Compatibility wrapper for the validated relative-pose recovery routine."""

from .rvo_closed_loop_replay import recover_relative_pose


def Get_Veridical_RT_from_E(E, inliers_Img1, inliers_Img2, K):
    return recover_relative_pose(E, inliers_Img1, inliers_Img2, K)


__all__ = ["Get_Veridical_RT_from_E"]
