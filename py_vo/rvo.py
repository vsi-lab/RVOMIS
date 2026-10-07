"""Portable CLI for the validated MATLAB-compatible RVO-MIS backend."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from .lightglue_adapter import LightGlueFrontend
from .matlab_exact_rng import MatlabExactRNG
from .rvo_closed_loop_replay import ClosedLoopParams, run_closed_loop


def load_intrinsics(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        labeled = []
        plain = []
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, values = line.split(":", 1)
                array = np.fromstring(values.replace(",", " "), sep=" ")
                if array.size in (9, 12):
                    labeled.append((key.strip().upper(), array))
            else:
                array = np.fromstring(line.replace(",", " "), sep=" ")
                if array.size:
                    plain.append(array)
        if labeled:
            order = ("P2", "P_RECT_02", "K_02", "K", "P0", "P1", "P3")
            labeled.sort(key=lambda item: next(
                (index for index, prefix in enumerate(order) if item[0].startswith(prefix)), 999
            ))
            values = labeled[0][1]
            return values.reshape(3, -1)[:, :3].astype(float)
        if len(plain) == 1 and plain[0].size in (9, 12):
            return plain[0].reshape(3, -1)[:, :3].astype(float)
        if len(plain) == 3 and all(row.size in (3, 4) for row in plain):
            return np.vstack(plain)[:, :3].astype(float)
        raise ValueError(f"Cannot parse a 3x3 intrinsic matrix from {path}")

    if suffix in (".yaml", ".yml"):
        import cv2

        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise ValueError(f"Cannot open intrinsic YAML file: {path}")
        try:
            for key in ("M1", "K", "IntrinsicMatrix", "intrinsicMatrix", "intrinsics"):
                node = storage.getNode(key)
                if not node.empty():
                    matrix = node.mat()
                    if matrix is not None:
                        return np.asarray(matrix, dtype=float).reshape(3, -1)[:, :3]
        finally:
            storage.release()
        raise ValueError(f"Cannot find an intrinsic matrix in {path}")

    if suffix == ".mat":
        data = sio.loadmat(path)
        for key in ("K", "IntrinsicMatrix", "intrinsicMatrix", "intrinsics"):
            if key in data:
                matrix = np.squeeze(np.asarray(data[key], dtype=float))
                if matrix.shape in ((3, 3), (3, 4)):
                    return matrix[:, :3]
        raise ValueError(f"Cannot find an intrinsic matrix in {path}")
    raise ValueError(f"Unsupported intrinsic file type: {path.suffix}")


def load_gt_poses(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        data = sio.loadmat(path)
        values = None
        for key in ("GT_Poses", "GTPoses", "poses", "Abs_Poses_GT", "Abs_Poses"):
            if key in data:
                values = np.asarray(data[key], dtype=float)
                break
        if values is None:
            raise ValueError(f"Cannot find GT poses in {path}")
        if values.ndim == 3 and values.shape[:2] in ((3, 4), (4, 4)):
            return np.moveaxis(values[:3, :4, :], 2, 0)
        if values.ndim == 3 and values.shape[1:] in ((3, 4), (4, 4)):
            return values[:, :3, :4]
        raise ValueError(f"Unsupported GT pose array shape {values.shape} in {path}")

    values = np.loadtxt(path, dtype=float)
    values = np.atleast_2d(values)
    if values.shape[1] == 12:
        return values.reshape(-1, 3, 4)
    if values.shape[1] == 16:
        return values.reshape(-1, 4, 4)[:, :3, :4]
    raise ValueError(f"Expected 12 or 16 values per GT row in {path}")


def _matlab_scale_and_metrics(raw_poses: np.ndarray, gt_poses: np.ndarray):
    count = min(len(raw_poses), len(gt_poses))
    if count < 2:
        raise ValueError("At least two GT and estimated poses are required")
    gt = gt_poses[:count]
    estimate = raw_poses[:count].copy()
    center0 = -gt[0, :3].T @ gt[0, 3]
    center1 = -gt[1, :3].T @ gt[1, 3]
    scale = float(np.linalg.norm(center0 - center1))
    estimate[:, :, 3] *= scale
    rotation_sq = []
    translation_sq = []
    for reference, predicted in zip(gt, estimate):
        cosine = np.clip((np.trace(reference[:, :3].T @ predicted[:, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotation_sq.append(float(np.arccos(cosine) ** 2))
        translation_sq.append(float(np.sum((reference[:, 3] - predicted[:, 3]) ** 2)))
    return estimate, {
        "matlab_first_baseline_scale": scale,
        "translation_rmse": float(np.sqrt(np.mean(translation_sq))),
        "rotation_rmse_radians": float(np.sqrt(np.mean(rotation_sq))),
        "evaluated_frames": count,
    }


def run(args) -> dict:
    images = sorted(glob.glob(args.image_glob))
    if args.max_frames is not None:
        images = images[: args.max_frames]
    if len(images) < 2:
        raise RuntimeError(f"Need at least two images; matched {len(images)} with {args.image_glob!r}")
    if args.seed != 0:
        raise ValueError("The validated public backend supports MATLAB rng(0,'twister') only; use --seed 0")
    if args.n_jobs != 1:
        raise ValueError("The validated shared MATLAB RNG stream requires --n_jobs 1")

    intrinsics = load_intrinsics(args.intrinsic_path)
    output_dir = Path(args.output_dir).resolve()
    backend_dir = output_dir / "backend"
    cache_dir = output_dir / "match_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend = LightGlueFrontend(cache_dir, device=args.device, max_keypoints=4096)
    rng = MatlabExactRNG(seed=0)
    params = ClosedLoopParams(
        INLIER_THRESH=args.inlier_thresh,
        RANSAC_ITERATIONS=args.ransac_iterations,
        TOP_N_RATIO_RANK_ORDERED_LIST=args.top_n_ratio,
        NUM_OF_FRAMES_FROM_LAST_KF=args.num_frames_from_last_kf,
        RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF=args.ratio_covisible,
        SEED=0,
        N_JOBS=1,
    )
    result = run_closed_loop(
        intrinsics,
        len(images),
        frontend,
        rng,
        backend_dir,
        params=params,
        images=images,
    )
    rng.write_csv(output_dir / "sample_schedule.csv")

    raw_path = backend_dir / "trajectory_raw.txt"
    raw_poses = np.loadtxt(raw_path, dtype=float).reshape(-1, 3, 4)
    output_poses = raw_poses
    metrics = None
    if args.gt_path:
        gt_poses = load_gt_poses(args.gt_path)
        output_poses, metrics = _matlab_scale_and_metrics(raw_poses, gt_poses)
        print(f"Translation RMSE (MATLAB first-baseline scale): {metrics['translation_rmse']:.12g}")
        print(f"Rotation RMSE (radians): {metrics['rotation_rmse_radians']:.12g}")

    output_path = output_dir / args.output_filename
    np.savetxt(output_path, output_poses.reshape(len(output_poses), 12), fmt="%.17g")
    if args.visualize:
        if not args.gt_path:
            raise ValueError("--visualize requires --gt_path")
        from .Visualize_Trajectory import Visualize_Trajectory

        Visualize_Trajectory(
            np.moveaxis(load_gt_poses(args.gt_path), 0, 2),
            np.moveaxis(output_poses, 0, 2),
            [],
        )

    summary = {
        "backend": "validated MATLAB-compatible geometry and exact rng(0,'twister') sampling",
        "frontend": frontend.selection,
        "frontend_platform_note": "Raw SIFT/LightGlue correspondences may vary across platform stacks.",
        "input_frames": len(images),
        "output_path": str(output_path),
        "raw_backend_trajectory": str(raw_path),
        "metrics": metrics,
        "run": result,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def rvo(
    data_root="Matlab/MyData",
    device="cuda",
    params=None,
    visualize=False,
    intrinsic_path=None,
    gt_path=None,
    image_glob=None,
    output_kitti_path=None,
    output_dir=None,
    output_filename="estimated_poses.txt",
    max_frames=None,
    **_unused,
):
    """Backward-compatible programmatic entry point using the validated backend."""
    defaults = ClosedLoopParams() if params is None else params
    if output_kitti_path:
        output_kitti_path = Path(output_kitti_path)
        output_dir = output_kitti_path.parent or Path(".")
        output_filename = output_kitti_path.name
    output_dir = Path(output_dir or "experiments/repro_run")
    args = argparse.Namespace(
        image_glob=image_glob or str(Path(data_root) / "fr2_desk" / "*.png"),
        intrinsic_path=intrinsic_path or str(Path(data_root) / "IntrinsicMatrix.mat"),
        gt_path=(str(Path(data_root) / "GT_Poses.mat") if gt_path is None else gt_path),
        output_dir=str(output_dir),
        output_filename=output_filename,
        device=device,
        visualize=bool(visualize),
        inlier_thresh=float(defaults.INLIER_THRESH),
        ransac_iterations=int(defaults.RANSAC_ITERATIONS),
        top_n_ratio=float(defaults.TOP_N_RATIO_RANK_ORDERED_LIST),
        num_frames_from_last_kf=int(defaults.NUM_OF_FRAMES_FROM_LAST_KF),
        ratio_covisible=float(defaults.RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF),
        n_jobs=int(defaults.N_JOBS),
        seed=int(defaults.SEED),
        max_frames=max_frames,
    )
    summary = run(args)
    raw = np.loadtxt(summary["raw_backend_trajectory"], dtype=float).reshape(-1, 3, 4)
    estimated = np.loadtxt(summary["output_path"], dtype=float).reshape(-1, 3, 4)
    metrics = summary["metrics"] or {}
    return {
        "Estimated_Poses": np.moveaxis(estimated, 0, 2),
        "Abs_Poses": np.moveaxis(raw, 0, 2),
        "RMSET": metrics.get("translation_rmse"),
        "RMSER": metrics.get("rotation_rmse_radians"),
        "KeyFrame_Indx": [],
        "kitti_path": summary["output_path"],
        "backend_output_dir": str(Path(summary["raw_backend_trajectory"]).parent),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_glob", default="Matlab/MyData/fr2_desk/*.png")
    parser.add_argument("--intrinsic_path", default="Matlab/MyData/IntrinsicMatrix.mat")
    parser.add_argument("--gt_path", default="Matlab/MyData/GT_Poses.mat", help="Use an empty string to disable GT evaluation")
    parser.add_argument("--output_dir", default="experiments/repro_run")
    parser.add_argument("--output_filename", default="estimated_poses.txt")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--inlier_thresh", type=float, default=2.0)
    parser.add_argument("--ransac_iterations", type=int, default=3000)
    parser.add_argument("--top_n_ratio", type=float, default=0.8)
    parser.add_argument("--num_frames_from_last_kf", type=int, default=15)
    parser.add_argument("--ratio_covisible", type=float, default=0.55)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_frames", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not Path(args.intrinsic_path).is_file():
        raise FileNotFoundError(f"Intrinsic matrix path does not exist: {args.intrinsic_path}")
    if args.gt_path and not Path(args.gt_path).is_file():
        raise FileNotFoundError(f"GT path does not exist: {args.gt_path}")
    run(args)


if __name__ == "__main__":
    main()
