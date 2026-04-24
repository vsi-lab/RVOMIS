import os
import glob
import time
import argparse
import numpy as np
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor
from .params import DEFAULT_PARAMS, RansacParams


def _row_intersect(a: np.ndarray, b: np.ndarray):
    """
    Intersection of rows for 2D arrays, returning common rows and indices in a and b.
    """
    a2 = a.view([("", a.dtype)] * a.shape[1])
    b2 = b.view([("", b.dtype)] * b.shape[1])
    common, ia, ib = np.intersect1d(a2.ravel(), b2.ravel(), return_indices=True)
    if common.size == 0:
        return np.empty((a.shape[1], 0), dtype=a.dtype), np.array([], dtype=int), np.array([], dtype=int)
    return a[ia], ia, ib


def load_K(path):
    if path.lower().endswith(".txt"):
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        labeled = []
        plain_rows = []
        for ln in lines:
            if ln.startswith("#"):
                continue
            if ":" in ln:
                key, values = ln.split(":", 1)
                arr = np.fromstring(values.replace(",", " "), sep=" ", dtype=float)
                if arr.size in (9, 12):
                    labeled.append((key.strip().upper(), arr))
            else:
                row = np.fromstring(ln.replace(",", " "), sep=" ", dtype=float)
                if row.size > 0:
                    plain_rows.append(row)

        # Prefer KITTI-like keys first if present.
        if labeled:
            preferred_prefixes = ("P2", "P_RECT_02", "K_02", "K", "P0", "P1", "P3")
            ordered = sorted(
                labeled,
                key=lambda kv: next((i for i, p in enumerate(preferred_prefixes) if kv[0].startswith(p)), 999),
            )
            arr = ordered[0][1]
            if arr.size == 12:
                return arr.reshape(3, 4)[:, :3]
            return arr.reshape(3, 3)

        # Fallback: plain numeric text matrix/vector.
        if plain_rows:
            if len(plain_rows) == 1 and plain_rows[0].size in (9, 12):
                arr = plain_rows[0]
                if arr.size == 12:
                    return arr.reshape(3, 4)[:, :3]
                return arr.reshape(3, 3)
            if len(plain_rows) == 3 and plain_rows[0].size in (3, 4):
                mat = np.vstack(plain_rows)
                if mat.shape == (3, 4):
                    return mat[:, :3]
                if mat.shape == (3, 3):
                    return mat

        raise ValueError(f"Cannot parse intrinsic matrix from txt file: {path}")

    if path.lower().endswith((".yaml", ".yml")):
        # Try OpenCV FileStorage for YAML
        try:
            import cv2

            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            if fs.isOpened():
                for key in ["M1", "K", "IntrinsicMatrix"]:
                    node = fs.getNode(key)
                    if node.empty():
                        continue
                    K = node.mat()
                    fs.release()
                    return np.array(K, dtype=float)
                fs.release()
        except Exception:
            pass
        # Fallback: naive parse for data: [ ... ]
        import re

        with open(path, "r") as f:
            txt = f.read()
        m = re.search(r"M1:.*?data:\\s*\\[([^\\]]+)\\]", txt, re.S)
        if m:
            vals = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
            if len(vals) >= 9:
                K = np.array(vals[:9], dtype=float).reshape(3, 3)
                return K
        raise ValueError(f"Cannot find intrinsic matrix in {path}")
    mat = sio.loadmat(path)
    for key in ["K", "IntrinsicMatrix", "intrinsicMatrix", "intrinsics"]:
        if key in mat:
            K = mat[key]
            if K.shape[-1] == 1 and K.shape[-2] == 1:
                continue
            return np.array(K, dtype=float)
    raise ValueError(f"Cannot find intrinsic matrix in {path}")


def load_GT(path):
    mat = sio.loadmat(path)
    for key in ["GT_Poses", "GTPoses", "poses", "Abs_Poses_GT", "Abs_Poses"]:
        if key in mat:
            return np.array(mat[key], dtype=float)
    raise ValueError(f"Cannot find GT poses in {path}")


def rvo(
    data_root="Matlab/MyData",
    device="cuda",
    params=DEFAULT_PARAMS,
    visualize=False,
    intrinsic_path=None,
    gt_path=None,
    image_glob=None,
    output_kitti_path=None,
):
    from .lightglue_adapter import match_features
    from .Ransac4Essential_CH import Ransac4Essential_CH
    from .Get_Veridical_RT_from_E import Get_Veridical_RT_from_E
    from .Reconstruct_by_LT import Reconstruct_by_LT
    from .Msac4AbsolutePose_CHM import Msac4AbsolutePose_CHM
    from .Visualize_Trajectory import Visualize_Trajectory

    assert params is not None, "params must not be None."
    assert params.INLIER_THRESH > 0, "INLIER_THRESH must be > 0."
    assert params.RANSAC_ITERATIONS > 0, "RANSAC_ITERATIONS must be > 0."
    assert 0 < params.TOP_N_RATIO_RANK_ORDERED_LIST <= 1.0, "TOP_N_RATIO_RANK_ORDERED_LIST must be in (0, 1]."
    assert params.NUM_OF_FRAMES_FROM_LAST_KF >= 1, "NUM_OF_FRAMES_FROM_LAST_KF must be >= 1."
    assert 0 < params.RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF <= 1.0, \
        "RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF must be in (0, 1]."
    assert params.N_JOBS >= 1, "N_JOBS must be >= 1."

    # Report device/GPU info once for debugging
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            gpu_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            gpu_count = torch.cuda.device_count()
            print(f"[Device] Using CUDA device {gpu_idx} ({gpu_name}), total GPUs: {gpu_count}")
        else:
            print(f"[Device] Using device: {device}")
    except Exception:
        print(f"[Device] Using device: {device}")

    rng = np.random.default_rng(0)
    K_path = intrinsic_path or os.path.join(data_root, "IntrinsicMatrix.mat")
    assert os.path.exists(K_path), f"Intrinsic matrix file not found: {K_path}"
    K = load_K(K_path)
    GT_Poses = None
    if gt_path is None:
        default_gt = os.path.join(data_root, "GT_Poses.mat")
        if os.path.exists(default_gt):
            GT_Poses = load_GT(default_gt)
    else:
        if os.path.exists(gt_path):
            GT_Poses = load_GT(gt_path)
    if image_glob is None:
        image_glob = os.path.join(data_root, "fr2_desk", "*.png")
    image_list = sorted(glob.glob(image_glob))
    assert len(image_list) > 0, f"No images matched image_glob: {image_glob}"
    if len(image_list) < 2:
        raise RuntimeError("Not enough images found.")

    PARAMS = params
    Prev_Abs_R = np.eye(3)
    Prev_Abs_T = np.zeros((3, 1))
    Abs_Poses = np.zeros((3, 4, len(image_list)))
    Abs_Poses[:, :, 0] = np.hstack([Prev_Abs_R, Prev_Abs_T])
    Abs_Cam_Center = np.zeros((3, len(image_list)))
    Abs_Cam_Center[:, 0] = (-Prev_Abs_R.T @ Prev_Abs_T).ravel()

    KeyFrame_Indx = []
    Points3D_Cloud = []
    invK = np.linalg.inv(K)

    t_match = 0.0
    t_ransac = 0.0
    t_triang = 0.0
    t_opt = 0.0

    prev_img_path = image_list[0]
    Prev_f_KF_ranked = None
    Prev_f_KF_HavePts3D = None
    Points3D_Cam_Last_KF = None
    Last_Keyframe_Index = 0
    Abs_R_KF = Prev_Abs_R
    Abs_T_KF = Prev_Abs_T

    # Optional process pool for parallel RANSAC if N_JOBS > 1
    pool = ProcessPoolExecutor(max_workers=params.N_JOBS) if getattr(params, "N_JOBS", 1) and params.N_JOBS > 1 else None

    for mi in range(1, len(image_list)):
        curr_img_path = image_list[mi]
        if mi == 1:
            t0 = time.perf_counter()
            f1_ranked_12, f2_ranked_12 = match_features(prev_img_path, curr_img_path, device=device)
            t_match += time.perf_counter() - t0

            Prev_f_KF_ranked = f2_ranked_12

            t1 = time.perf_counter()
            E, inlierIdx = Ransac4Essential_CH(PARAMS, f1_ranked_12, f2_ranked_12, K)
            inliers_Img1 = f1_ranked_12[:, inlierIdx]
            inliers_Img2 = f2_ranked_12[:, inlierIdx]
            Rel_R, Rel_T = Get_Veridical_RT_from_E(E, inliers_Img1, inliers_Img2, K)
            t_ransac += time.perf_counter() - t1
            Prev_f_KF_HavePts3D = inliers_Img2

            Rs = np.zeros((3, 3, 1))
            Ts = np.zeros((3, 1))
            Rs[:, :, 0] = Rel_R
            Ts[:, 0:1] = Rel_T
            inliers = np.vstack([inliers_Img1[:2, :], inliers_Img2[:2, :]])
            t2 = time.perf_counter()
            Points3D_Cam_Last_KF = Reconstruct_by_LT(Rs, Ts, 2, inliers, K)
            t_triang += time.perf_counter() - t2

            Abs_Poses[:, :, mi] = np.hstack([Rel_R, Rel_T])
            Abs_Cam_Center[:, mi] = (-Rel_R.T @ Rel_T).ravel()
            Abs_R_KF = Rel_R
            Abs_T_KF = Rel_T

            Last_Keyframe_Index = 1
            KeyFrame_Indx.extend([0, 1])
            Points3D_Cloud.append(Points3D_Cam_Last_KF)
        else:
            t0 = time.perf_counter()
            f_KF_ranked, f_CF_ranked = match_features(image_list[Last_Keyframe_Index], curr_img_path, device=device)
            t_match += time.perf_counter() - t0

            _, CovIndx_KF_ranked, CovIndx_CF = _row_intersect(Prev_f_KF_ranked.T, f_KF_ranked.T)
            f_KF_ranked_ = f_KF_ranked[:, CovIndx_CF]
            f_CF_ranked_ = f_CF_ranked[:, CovIndx_CF]
            _, Prev_KF_Indx_HavePts3D, f_CF_Indx_HavePts3D = _row_intersect(Prev_f_KF_HavePts3D.T, f_KF_ranked_.T)
            f_KF_HavePts3D = f_KF_ranked_[:, f_CF_Indx_HavePts3D]
            f_CF_HavePts3D = f_CF_ranked_[:, f_CF_Indx_HavePts3D]

            Points3D_Last_KF_Cam1 = Points3D_Cam_Last_KF[:, Prev_KF_Indx_HavePts3D]
            t1 = time.perf_counter()
            Abs_R_CF, Abs_T_CF, inlierRansacIndx, opt_time = Msac4AbsolutePose_CHM(
                PARAMS, Points3D_Last_KF_Cam1, f_CF_HavePts3D, K
            )
            t_ransac += time.perf_counter() - t1
            t_opt += opt_time
            Abs_Poses[:, :, mi] = np.hstack([Abs_R_CF, Abs_T_CF])
            Abs_Cam_Center[:, mi] = (-Abs_R_CF.T @ Abs_T_CF).ravel()
            f_CF_HavePts3D = f_CF_HavePts3D[:, inlierRansacIndx]
            Points3D_Last_KF_Cam1 = Points3D_Last_KF_Cam1[:, inlierRansacIndx]

            Num_Of_Frames_From_Last_Keyframe = mi - Last_Keyframe_Index
            Ratio_Of_Covisible_Points = len(Prev_KF_Indx_HavePts3D) / max(1, Points3D_Cam_Last_KF.shape[1])

            if (
                Num_Of_Frames_From_Last_Keyframe >= PARAMS.NUM_OF_FRAMES_FROM_LAST_KF
                or Ratio_Of_Covisible_Points < PARAMS.RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF
            ):
                _, f_CF_Indx_HavePts3D, _ = _row_intersect(f_CF_ranked.T, f_CF_HavePts3D.T)
                f_CF_ranked_transpose = np.delete(f_CF_ranked.T, f_CF_Indx_HavePts3D, axis=0)
                f_KF_ranked_transpose = np.delete(f_KF_ranked.T, f_CF_Indx_HavePts3D, axis=0)
                f_CF_ranked_HaveNoPts3D = f_CF_ranked_transpose.T
                f_KF_ranked_HaveNoPts3D = f_KF_ranked_transpose.T

                Rel_R_wrt_Last_KF = Abs_R_CF @ Abs_R_KF.T
                Rel_T_wrt_Last_KF = Abs_T_CF - Abs_R_CF @ Abs_R_KF.T @ Abs_T_KF

                E = skew(Rel_T_wrt_Last_KF) @ Rel_R_wrt_Last_KF
                F = invK.T @ E @ invK
                Apixel = F[0] @ f_KF_ranked_HaveNoPts3D
                Bpixel = F[1] @ f_KF_ranked_HaveNoPts3D
                Cpixel = F[2] @ f_KF_ranked_HaveNoPts3D
                A_xi = Apixel * f_CF_ranked_HaveNoPts3D[0, :]
                B_eta = Bpixel * f_CF_ranked_HaveNoPts3D[1, :]
                numerOfDist = np.abs(A_xi + B_eta + Cpixel)
                denomOfDist = np.sqrt(Apixel ** 2 + Bpixel ** 2) + 1e-12
                dist2EL = numerOfDist / denomOfDist
                inlier_Indx_KC = np.where(dist2EL <= PARAMS.INLIER_THRESH)[0]

                f_CF_Inliers_HaveNoPts3D = f_CF_ranked_HaveNoPts3D[:, inlier_Indx_KC]
                f_KF_Inliers_HaveNoPts3D = f_KF_ranked_HaveNoPts3D[:, inlier_Indx_KC]

                Rs = np.zeros((3, 3, 1))
                Ts = np.zeros((3, 1))
                Rs[:, :, 0] = Rel_R_wrt_Last_KF
                Ts[:, 0:1] = Rel_T_wrt_Last_KF
                KF_CF_Inliers = np.vstack([f_KF_Inliers_HaveNoPts3D[:2, :], f_CF_Inliers_HaveNoPts3D[:2, :]])
                t2 = time.perf_counter()
                Points3D_Cam_Last_KF = Reconstruct_by_LT(Rs, Ts, 2, KF_CF_Inliers, K)
                t_triang += time.perf_counter() - t2

                Points3D_Cam1_New = Abs_R_KF.T @ (Points3D_Cam_Last_KF - Abs_T_KF)
                Points3D_Cam_Last_KF = np.concatenate([Points3D_Cam1_New, Points3D_Last_KF_Cam1], axis=1)
                Points2D_CF = np.concatenate([f_CF_Inliers_HaveNoPts3D, f_CF_HavePts3D], axis=1)
                Points3D_Cloud.append(Points3D_Cam_Last_KF)

                KeyFrame_Indx.append(mi)
                Last_Keyframe_Index = mi
                Abs_R_KF = Abs_R_CF
                Abs_T_KF = Abs_T_CF
                Prev_f_KF_ranked = f_CF_ranked
                Prev_f_KF_HavePts3D = Points2D_CF

        if mi % 5 == 0:
            print(".", end="", flush=True)
        if mi % 50 == 0:
            print()
    print()

    Estimated_Poses = None
    RMSER = None
    RMSET_standard = None
    if GT_Poses is not None:
        GT_c1 = -GT_Poses[:, :3, 0].T @ GT_Poses[:, 3, 0]
        GT_c2 = -GT_Poses[:, :3, 1].T @ GT_Poses[:, 3, 1]
        scale = np.linalg.norm(GT_c1 - GT_c2)
        Estimated_Poses = np.zeros_like(Abs_Poses)
        for ci in range(Abs_Poses.shape[2]):
            Estimated_Poses[:, :, ci] = np.hstack([Abs_Poses[:, :3, ci], Abs_Poses[:, 3, ci:ci + 1] * scale])

        RMSER = 0.0
        for i in range(GT_Poses.shape[2]):
            r_est = Estimated_Poses[:, :3, i]
            r_gt = GT_Poses[:, :3, i]
            val = 0.5 * (np.trace(r_gt.T @ r_est) - 1)
            val = np.clip(val, -1.0, 1.0)
            RMSER += (np.arccos(val)) ** 2
        RMSER = np.sqrt(RMSER / GT_Poses.shape[2])

        RMSET_standard = 0.0
        for i in range(GT_Poses.shape[2]):
            t_est = Estimated_Poses[:, 3, i]
            t_gt = GT_Poses[:, 3, i]
            RMSET_standard += np.sum((t_gt - t_est) ** 2)
        RMSET_standard = np.sqrt(RMSET_standard / GT_Poses.shape[2])

        print(f"Total match time: {t_match:.3f}s, essential/pose time: {t_ransac:.3f}s, triangulation: {t_triang:.3f}s, LM opt: {t_opt:.3f}s")
        print("Standard RMSE for translations (Euclidean distance) (without KA):")
        print(RMSET_standard)
        print("RMSE for rotations (without KA):")
        print(RMSER)

        if visualize:
            Visualize_Trajectory(GT_Poses, Estimated_Poses, KeyFrame_Indx)
    else:
        print(f"Total match time: {t_match:.3f}s, essential/pose time: {t_ransac:.3f}s, triangulation: {t_triang:.3f}s, LM opt: {t_opt:.3f}s")
        Estimated_Poses = Abs_Poses

    # Optionally save poses in KITTI format: each line 12 values (row-major 3x4 matrix).
    if output_kitti_path:
        out_dir = os.path.dirname(output_kitti_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        poses_to_save = Estimated_Poses if Estimated_Poses is not None else Abs_Poses
        kitti_mats = poses_to_save.transpose(2, 0, 1).reshape(poses_to_save.shape[2], -1)
        np.savetxt(output_kitti_path, kitti_mats, fmt="%.9f")

    return {
        "Estimated_Poses": Estimated_Poses,
        "Abs_Poses": Abs_Poses,
        "RMSET": RMSET_standard,
        "RMSER": RMSER,
        "KeyFrame_Indx": KeyFrame_Indx,
        "kitti_path": output_kitti_path,
    }


def skew(T):
    return np.array([[0, -T[2, 0], T[1, 0]], [T[2, 0], 0, -T[0, 0]], [-T[1, 0], T[0, 0], 0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RVOMIS with command-line configuration.")
    parser.add_argument("--image_glob", default="Matlab/MyData/fr2_desk/*.png", help="Input image glob pattern.")
    parser.add_argument("--intrinsic_path", default="Matlab/MyData/IntrinsicMatrix.mat", help="Path to intrinsic matrix file.")
    parser.add_argument(
        "--gt_path",
        default="Matlab/MyData/GT_Poses.mat",
        help="Path to GT poses file. Set to empty string to disable GT.",
    )
    parser.add_argument("--output_dir", default="experiments/repro_run", help="Directory for output pose file.")
    parser.add_argument("--output_filename", default="estimated_poses.txt", help="Output pose filename.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device.")
    parser.add_argument("--visualize", action="store_true", help="Enable trajectory visualization.")
    parser.add_argument("--inlier_thresh", type=float, default=2.0)
    parser.add_argument("--ransac_iterations", type=int, default=3000)
    parser.add_argument("--top_n_ratio", type=float, default=0.8)
    parser.add_argument("--num_frames_from_last_kf", type=int, default=15)
    parser.add_argument("--ratio_covisible", type=float, default=0.55)
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    params = RansacParams(
        INLIER_THRESH=args.inlier_thresh,
        RANSAC_ITERATIONS=args.ransac_iterations,
        TOP_N_RATIO_RANK_ORDERED_LIST=args.top_n_ratio,
        NUM_OF_FRAMES_FROM_LAST_KF=args.num_frames_from_last_kf,
        RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF=args.ratio_covisible,
        N_JOBS=args.n_jobs,
    )

    gt_path = None if args.gt_path == "" else args.gt_path

    assert os.path.exists(args.intrinsic_path), f"Intrinsic matrix path does not exist: {args.intrinsic_path}"
    assert len(glob.glob(args.image_glob)) >= 2, f"Need at least 2 images. Current image_glob matched: {args.image_glob}"
    if gt_path is not None:
        assert os.path.exists(gt_path), f"GT path does not exist: {gt_path}"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)

    rvo(
        data_root=".",
        device=args.device,
        params=params,
        visualize=args.visualize,
        intrinsic_path=args.intrinsic_path,
        gt_path=gt_path,
        image_glob=args.image_glob,
        output_kitti_path=output_path,
    )
