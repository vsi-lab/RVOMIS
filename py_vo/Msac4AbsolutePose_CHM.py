"""Public compatibility wrapper for validated P3P/MSAC pose estimation."""

from .Msac4AbsolutePose_CHM_matlab import (
    Msac4AbsolutePose_CHM as _validated_msac,
    euler_zyx_to_rotation_matrix,
    reprojection_error,
    rotation_matrix_to_euler_zyx,
)
from .matlab_exact_rng import MatlabExactRNG


def Msac4AbsolutePose_CHM(
    PARAMS,
    Points3D,
    Points2D,
    K,
    rng=None,
    frame_id=1,
    p3p_solver=None,
    return_diagnostics=False,
):
    """Run validated MSAC, retaining the legacy four-argument entry point."""
    if rng is None:
        rng = MatlabExactRNG(seed=0)
    kwargs = {"return_diagnostics": return_diagnostics}
    if p3p_solver is not None:
        kwargs["p3p_solver"] = p3p_solver
    return _validated_msac(
        PARAMS,
        Points3D,
        Points2D,
        K,
        rng,
        frame_id,
        **kwargs,
    )


# Legacy helper names used by downstream scripts.
rotation_matrix_to_euler = rotation_matrix_to_euler_zyx
euler_to_rotation_matrix = euler_zyx_to_rotation_matrix

__all__ = [
    "Msac4AbsolutePose_CHM",
    "euler_zyx_to_rotation_matrix",
    "euler_to_rotation_matrix",
    "reprojection_error",
    "rotation_matrix_to_euler",
    "rotation_matrix_to_euler_zyx",
]
