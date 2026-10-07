"""Public parameter aliases for the validated serial backend."""

from .rvo_closed_loop_replay import ClosedLoopParams

RansacParams = ClosedLoopParams
DEFAULT_PARAMS = ClosedLoopParams()

__all__ = ["ClosedLoopParams", "RansacParams", "DEFAULT_PARAMS"]
