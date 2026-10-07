"""Portable SIFT + LightGlue frontend used by the public RVO-MIS runner.

The geometric backend is MATLAB-compatible, but raw frontend results can vary
across operating-system, CUDA, and package stacks.  This module deliberately
keeps the audited SIFT/LightGlue configuration unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

MAX_KEYPOINTS = 4096


def _load_external_lg(path_value: str):
    candidate = Path(path_value).expanduser().resolve()
    module_path = candidate if candidate.is_file() else candidate / "lg.py"
    if module_path.is_file():
        spec = importlib.util.spec_from_file_location("rvomis_external_lg", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load lg.py from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "match_features"):
            raise ImportError(f"{module_path} does not define match_features")
        return "external lg.py", str(module_path), module

    package_root = candidate / "LightGlue" if (candidate / "LightGlue").is_dir() else candidate
    if (package_root / "lightglue").is_dir():
        sys.path.insert(0, str(package_root))
        return "external LightGlue package", str(package_root), None
    raise FileNotFoundError(
        f"LG_PY_PATH={path_value!r} contains neither lg.py nor a LightGlue checkout"
    )


def _resolve_frontend():
    configured = os.getenv("LG_PY_PATH")
    if configured:
        return _load_external_lg(configured)

    repo_root = Path(__file__).resolve().parents[1]
    bundled_root = repo_root / "Matlab" / "lightglue" / "LightGlue"
    if (bundled_root / "lightglue").is_dir():
        sys.path.insert(0, str(bundled_root))
        return "repository-local LightGlue package", str(bundled_root), None

    bundled_wrapper = repo_root / "Matlab" / "lightglue" / "lg.py"
    if bundled_wrapper.is_file():
        return _load_external_lg(str(bundled_wrapper))

    try:
        spec = importlib.util.find_spec("lightglue")
    except (ImportError, AttributeError, ValueError):
        spec = None
    if spec is not None:
        return "installed lightglue package", str(spec.origin or "installed"), None

    raise ImportError(
        "LightGlue was not found. Set LG_PY_PATH to lg.py or a LightGlue checkout, "
        "initialize Matlab/lightglue/LightGlue, or install the lightglue package."
    )


class LightGlueFrontend:
    """Lazy matcher with deterministic on-disk pair caching."""

    def __init__(self, cache_dir, device="cuda", max_keypoints=MAX_KEYPOINTS):
        if int(max_keypoints) != MAX_KEYPOINTS:
            raise ValueError(f"The audited frontend requires max_keypoints={MAX_KEYPOINTS}")
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = str(device)
        self.max_keypoints = MAX_KEYPOINTS
        self._extractor = None
        self._matcher = None
        self._external_module = None
        self._selection = None
        self.manifest = []

    def _initialize(self):
        if self._selection is not None:
            return
        kind, location, external_module = _resolve_frontend()
        self._selection = {"kind": kind, "location": location}
        self._external_module = external_module
        print(f"[Frontend] Selected {kind}: {location}", flush=True)
        if external_module is not None:
            return

        import torch
        from lightglue import LightGlue, SIFT

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        self._extractor = SIFT(max_num_keypoints=MAX_KEYPOINTS).eval().to(self.device)
        self._matcher = LightGlue(features="sift").eval().to(self.device)

    @property
    def selection(self):
        self._initialize()
        return dict(self._selection)

    def _compute(self, image0, image1):
        self._initialize()
        if self._external_module is not None:
            result = self._external_module.match_features(
                str(image0),
                str(image1),
                device=self.device,
                max_keypoints=MAX_KEYPOINTS,
                return_metadata=True,
            )
            points0, points1, metadata = result
            scores = np.asarray(metadata["scores"], dtype=float).reshape(-1)
            return np.asarray(points0, dtype=float), np.asarray(points1, dtype=float), scores

        import torch
        from lightglue.utils import load_image, rbd

        image0_tensor = load_image(str(image0)).to(self.device)
        image1_tensor = load_image(str(image1)).to(self.device)
        with torch.inference_mode():
            feats0 = self._extractor.extract(image0_tensor)
            feats1 = self._extractor.extract(image1_tensor)
            matches01 = self._matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(value) for value in (feats0, feats1, matches01)]
        order = torch.argsort(matches01["scores"], descending=True)
        matches = matches01["matches"][order]
        scores = matches01["scores"][order].detach().cpu().numpy()
        xy0 = feats0["keypoints"][matches[:, 0]].detach().cpu().numpy()
        xy1 = feats1["keypoints"][matches[:, 1]].detach().cpu().numpy()
        points0 = np.vstack([xy0.T, np.ones((1, len(xy0)))])
        points1 = np.vstack([xy1.T, np.ones((1, len(xy1)))])
        return points0, points1, scores

    def match(self, image0, image1, frame0, frame1):
        path = self.cache_dir / f"frame_{frame0:06d}_{frame1:06d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                points0 = np.asarray(data["points0"], dtype=float)
                points1 = np.asarray(data["points1"], dtype=float)
                scores = np.asarray(data["scores"], dtype=float)
            source = "cache"
            self._initialize()
        else:
            points0, points1, scores = self._compute(image0, image1)
            np.savez_compressed(path, points0=points0, points1=points1, scores=scores)
            source = "computed"

        if points0.shape != points1.shape or points0.ndim != 2 or points0.shape[0] != 3:
            raise AssertionError("Frontend points must be aligned 3xN homogeneous arrays")
        if scores.shape != (points0.shape[1],):
            raise AssertionError("Frontend scores are not aligned with returned matches")
        self.manifest.append({
            "frame0": int(frame0),
            "frame1": int(frame1),
            "matches": int(points0.shape[1]),
            "source": source,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": str(path),
            "frontend_kind": self._selection["kind"],
            "frontend_location": self._selection["location"],
        })
        return points0.copy(), points1.copy(), scores.copy()

    def write_manifest(self, path):
        fields = [
            "frame0", "frame1", "matches", "source", "sha256", "path",
            "frontend_kind", "frontend_location",
        ]
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.manifest)


_FUNCTION_FRONTENDS = {}


def match_features(img1_path, img2_path, device="cuda", max_keypoints=MAX_KEYPOINTS, return_metadata=False):
    """Backward-compatible stateless API; the public runner uses ``LightGlueFrontend``."""
    key = (str(device), int(max_keypoints))
    frontend = _FUNCTION_FRONTENDS.get(key)
    if frontend is None:
        frontend = LightGlueFrontend(Path(".rvomis_match_cache"), device, max_keypoints)
        _FUNCTION_FRONTENDS[key] = frontend
    points0, points1, scores = frontend._compute(img1_path, img2_path)
    if return_metadata:
        return points0, points1, {"scores": scores}
    return points0, points1


__all__ = ["LightGlueFrontend", "MAX_KEYPOINTS", "match_features"]
