from dataclasses import dataclass
import os


def _default_jobs():
    try:
        c = os.cpu_count() or 1
        return max(1, c - 1)
    except Exception:
        return 1


@dataclass
class RansacParams:
    INLIER_THRESH: float = 2.0
    RANSAC_ITERATIONS: int = 3000
    TOP_N_RATIO_RANK_ORDERED_LIST: float = 0.8
    NUM_OF_FRAMES_FROM_LAST_KF: int = 15
    RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF: float = 0.55
    N_JOBS: int = 8  # parallel workers; set to 2 to avoid excessive memory use


DEFAULT_PARAMS = RansacParams()
