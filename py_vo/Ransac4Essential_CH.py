"""Public compatibility wrapper for validated Essential-matrix RANSAC."""

from .Ransac4Essential_CH_matlab import Ransac4Essential_CH as _validated_ransac
from .matlab_exact_rng import MatlabExactRNG


def Ransac4Essential_CH(
    PARAMS,
    matchImg1,
    matchImg2,
    K,
    rng=None,
    frame_id=1,
    return_diagnostics=False,
):
    """Run validated RANSAC, retaining the legacy four-argument entry point."""
    if rng is None:
        rng = MatlabExactRNG(seed=0)
    return _validated_ransac(
        PARAMS,
        matchImg1,
        matchImg2,
        K,
        rng,
        frame_id=frame_id,
        return_diagnostics=return_diagnostics,
    )

__all__ = ["Ransac4Essential_CH"]
