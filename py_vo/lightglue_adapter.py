"""
LightGlue-based feature matcher.
Keeps the same API as the MATLAB-Python bridge: returns two 3xN homogeneous point arrays.
Now supports using an externally configured lg.py if provided.
"""
from typing import Tuple
import os
import sys
import numpy as np

# Set CUSTOM_LG_PATH to your absolute folder containing lg.py, or set env LG_PY_PATH.
# Default to the local LightGlue repo used by MATLAB.
CUSTOM_LG_PATH = "/data/zwang570/LightGlue"

_use_external_lg = False
_lg = None

# Try to import external lg.py if path is given.
_ext_path = os.getenv("LG_PY_PATH") or CUSTOM_LG_PATH
if _ext_path:
    sys.path.insert(0, _ext_path)
    try:
        import lg as _lg  # type: ignore

        _use_external_lg = True
    except ImportError:
        _use_external_lg = False

# Lazy globals for internal LightGlue to avoid re-init per call
_extractor = None
_matcher = None
_device_used = None

if not _use_external_lg:
    import torch
    from lightglue import LightGlue, SuperPoint, SIFT
    from lightglue.utils import rbd, load_image


def match_features(
    img1_path: str, img2_path: str, device: str = "cuda", max_keypoints: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        points0_homo: 3xN
        points1_homo: 3xN
    """
    if _use_external_lg and _lg is not None:
        return _lg.match_features(img1_path, img2_path, device=device, max_keypoints=max_keypoints)

    global _extractor, _matcher, _device_used
    if _extractor is None or _matcher is None or _device_used != device:
        _extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)
        _matcher = LightGlue(features="sift").eval().to(device)
        _device_used = device

    image0 = load_image(img1_path).to(device)
    image1 = load_image(img2_path).to(device)

    feats0 = _extractor.extract(image0)
    feats1 = _extractor.extract(image1)

    matches01 = _matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matches = matches01["matches"]
    scores = matches01["scores"]

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    _ = sorted_scores  # keep API parity even if unused
    sorted_matches = matches[sorted_indices]

    points0 = feats0["keypoints"][sorted_matches[:, 0]].cpu().numpy()
    points1 = feats1["keypoints"][sorted_matches[:, 1]].cpu().numpy()

    points0_homo = np.vstack([points0.T, np.ones((1, points0.shape[0]))])
    points1_homo = np.vstack([points1.T, np.ones((1, points1.shape[0]))])
    return points0_homo, points1_homo
