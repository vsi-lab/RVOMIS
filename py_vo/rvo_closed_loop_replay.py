"""Closed-loop MATLAB-compatible RVO-MIS replay.

The runner owns every pose, map point, association, keyframe decision, and map
update.  In exact-replay mode the MATLAB audit bundle contributes only frozen
2D matches and sampled minimal-set indices.  MATLAB states are read after each
Python stage solely by the optional comparator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .Msac4AbsolutePose_CHM_matlab import Msac4AbsolutePose_CHM
from .Ransac4Essential_CH_matlab import Ransac4Essential_CH
from .matlab_semantics import matlab_intersect_rows


@dataclass(frozen=True)
class ClosedLoopParams:
    INLIER_THRESH: float = 2.0
    RANSAC_ITERATIONS: int = 3000
    TOP_N_RATIO_RANK_ORDERED_LIST: float = 0.8
    NUM_OF_FRAMES_FROM_LAST_KF: int = 15
    RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF: float = 0.55
    SEED: int = 0
    N_JOBS: int = 1


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def rotation_difference_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
    value = np.clip((np.trace(reference.T @ estimate) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def recover_relative_pose(
    essential: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of MATLAB ``Get_Veridical_RT_from_E`` without legacy imports."""
    u, _, vt = np.linalg.svd(essential)
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r1, r2 = u @ w @ vt, u @ w.T @ vt
    t1, t2 = u[:, 2], -u[:, 2]
    if np.linalg.det(r1) < 0.0 or np.linalg.det(r2) < 0.0:
        u, _, vt = np.linalg.svd(-essential)
        r1, r2 = u @ w @ vt, u @ w.T @ vt
        t1, t2 = u[:, 2], -u[:, 2]

    gamma0 = np.linalg.inv(intrinsics) @ points0
    gamma1 = np.linalg.inv(intrinsics) @ points1
    e1 = np.array([1.0, 0.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    support = []
    candidates = ((r1, t1), (r1, t2), (r2, t1), (r2, t2))
    for rotation, translation in candidates:
        rg = rotation @ gamma0
        denominator = (e3 @ rg) * (e1 @ gamma1) - (e1 @ rg)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho0 = (e1 @ translation - (e3 @ translation) * (e1 @ gamma1)) / denominator
            rho1 = ((e1 @ translation) * (e3 @ rg) - (e3 @ translation) * (e1 @ rg)) / denominator
        support.append(int(np.sum(rho0 > 0.0) + np.sum(rho1 > 0.0)))
    rotation, translation = candidates[int(np.argmax(support))]
    return rotation.copy(), translation.reshape(3, 1).copy()


def reconstruct_by_lt(
    rotations: np.ndarray,
    translations: np.ndarray,
    view_count: int,
    observations: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Port of MATLAB ``Reconstruct_by_LT`` without legacy imports."""
    inv_k = np.linalg.inv(intrinsics)
    relative_rotations = [np.eye(3)]
    relative_translations = [np.zeros((3, 1))]
    for index in range(1, view_count):
        relative_rotations.append(rotations[:, :, index - 1])
        relative_translations.append(translations[:, index - 1:index])

    points = np.zeros((3, observations.shape[1]), dtype=float)
    for point_index in range(observations.shape[1]):
        rows = []
        offset = 0
        for view_index in range(view_count):
            pixel = np.r_[observations[offset:offset + 2, point_index], 1.0]
            ray = inv_k @ pixel
            projection = np.hstack([
                relative_rotations[view_index], relative_translations[view_index]
            ])
            rows.append(_skew(ray) @ projection)
            offset += 2
        _, _, vt = np.linalg.svd(np.vstack(rows))
        homogeneous = vt[-1]
        points[:, point_index] = homogeneous[:3] / homogeneous[3]
    return points


class AuditFrontend:
    """Read only the frozen MATLAB frontend matches, never MATLAB VO state."""

    def __init__(self, audit_dir: str | Path):
        self.audit_dir = Path(audit_dir)
        self.records: list[dict] = []

    def match(self, _image0, _image1, frame0: int, frame1: int):
        path = self.audit_dir / "frontend" / f"pair_{frame0 + 1:04d}_{frame1 + 1:04d}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen MATLAB frontend pair is unavailable: {path.name}")
        data = loadmat(path, squeeze_me=True)
        points0 = np.asarray(data["points0_homo"], dtype=float)
        points1 = np.asarray(data["points1_homo"], dtype=float)
        if points0.shape != points1.shape or points0.shape[0] != 3:
            raise AssertionError(f"Invalid frontend arrays in {path}")
        self.records.append({
            "frame0": frame0,
            "frame1": frame1,
            "matches": points0.shape[1],
            "path": str(path),
        })
        return points0.copy(), points1.copy(), np.full(points0.shape[1], np.nan)

    def write_manifest(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame0", "frame1", "matches", "path"])
            writer.writeheader()
            writer.writerows(self.records)


class AuditSchedule:
    """Lazy exact MATLAB Essential/P3P minimal-set schedule."""

    def __init__(self, audit_dir: str | Path):
        self.audit_dir = Path(audit_dir)
        self._essential = self._load(
            self.audit_dir / "essential" / "essential_samples_zero_based.csv", 5
        )
        self._p3p: dict[int, np.ndarray] = {}
        self.records: list[dict] = []

    @staticmethod
    def _load(path: Path, width: int) -> np.ndarray:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
        values = np.atleast_2d(values)[:, 1:]
        if values.shape != (3000, width):
            raise AssertionError(f"Unexpected schedule shape {values.shape}: {path}")
        return values

    def sample_without_replacement(
        self, n: int, k: int, *, stage: str, frame_id: int, iteration: int
    ) -> np.ndarray:
        if stage == "essential":
            sample = self._essential[iteration]
        elif stage == "p3p":
            matlab_frame = frame_id + 1
            if matlab_frame not in self._p3p:
                path = self.audit_dir / "p3p" / f"frame_{matlab_frame:04d}_samples_zero_based.csv"
                self._p3p[matlab_frame] = self._load(path, 3)
            sample = self._p3p[matlab_frame][iteration]
        else:
            raise KeyError(stage)
        if sample.shape != (k,) or np.any(sample < 0) or np.any(sample >= n):
            raise AssertionError(
                f"Exact schedule incompatible with Python factors: stage={stage}, "
                f"frame={frame_id}, iteration={iteration}, n={n}, sample={sample.tolist()}"
            )
        return sample.copy()


class AuditComparator:
    """Post-stage comparison against reference states excluded from runtime input."""

    def __init__(self, audit_dir: str | Path):
        self.audit_dir = Path(audit_dir)
        self.first_numerical_divergence: dict | None = None
        self.first_structural_divergence: dict | None = None

    @staticmethod
    def _indices(data: dict, key: str) -> np.ndarray:
        return np.atleast_1d(data[key]).astype(np.int64) - 1

    def note(self, category: str, frame: int, stage: str, detail: str, *, structural: bool) -> None:
        value = {
            "category": category,
            "frame_id_matlab_1based": int(frame),
            "frame_id_python_0based": int(frame - 1),
            "stage": stage,
            "detail": detail,
        }
        if structural and self.first_structural_divergence is None:
            self.first_structural_divergence = value
        if not structural and self.first_numerical_divergence is None:
            self.first_numerical_divergence = value

    def state(self, frame: int) -> dict:
        return loadmat(
            self.audit_dir / "frames" / f"frame_{frame:04d}_state.mat", squeeze_me=True
        )


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_trajectory(path: Path, poses: np.ndarray) -> None:
    np.savetxt(path, poses.reshape(len(poses), 12), fmt="%.17g")


def run_closed_loop(
    intrinsics: np.ndarray,
    frame_count: int,
    frontend,
    rng,
    output_dir: str | Path,
    *,
    params: ClosedLoopParams = ClosedLoopParams(),
    comparator: AuditComparator | None = None,
    stop_on_structural_divergence: bool = True,
    images: list[str] | None = None,
) -> dict:
    """Run a complete state-owning RVO-MIS trajectory."""
    if params.N_JOBS != 1:
        raise AssertionError("Closed-loop MATLAB-compatible execution must be serial")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    k = np.asarray(intrinsics, dtype=float)
    poses = np.full((frame_count, 3, 4), np.nan)
    poses[0] = np.c_[np.eye(3), np.zeros(3)]
    rows: list[dict] = []
    terminated = False

    reference = comparator.state(1) if comparator else None
    rows.append({
        "frame_id_matlab_1based": 1,
        "frame_id_python_0based": 0,
        "active_keyframe_before_python": 1,
        "active_keyframe_before_matlab": int(reference["active_keyframe_before_frame"]) if reference else "",
        "matches_python": 0,
        "matches_matlab": int(reference["number_of_matches"]) if reference else "",
        "covisible_python": 0,
        "covisible_matlab": int(reference["number_of_covisible_points"]) if reference else "",
        "factors_python": 0,
        "factors_matlab": int(reference["number_of_2d3d_factors"]) if reference else "",
        "factor_2d_sha256": "",
        "factor_3d_sha256": "",
        "association_indices_equal": True if comparator else "",
        "selected_iteration_python_1based": "",
        "selected_iteration_matlab_1based": "",
        "selected_candidate_python_1based": "",
        "selected_candidate_matlab_1based": "",
        "inliers_equal": True if comparator else "",
        "inlier_indices_0based": "",
        "post_lm_rotation_difference_deg": 0.0 if comparator else "",
        "post_lm_translation_difference": 0.0 if comparator else "",
        "keyframe_python": 1,
        "keyframe_matlab": int(reference["is_keyframe"]) if reference else "",
        "active_keyframe_after_python": 1,
        "active_keyframe_after_matlab": int(reference["active_keyframe_after_frame"]) if reference else "",
        "map_size_python": 0,
        "map_size_matlab": int(reference["map_size"]) if reference else "",
    })

    # Initialization: only frozen 2D matches and Essential samples enter Python.
    if images is not None and len(images) != frame_count:
        raise ValueError("images length must equal frame_count")
    image0 = images[0] if images is not None else None
    image1 = images[1] if images is not None else None
    points0, points1, _ = frontend.match(image0, image1, 0, 1)
    essential, initial_inliers, essential_diag = Ransac4Essential_CH(
        params, points0, points1, k, rng, frame_id=1, return_diagnostics=True
    )
    inlier0, inlier1 = points0[:, initial_inliers], points1[:, initial_inliers]
    relative_r, relative_t = recover_relative_pose(essential, inlier0, inlier1, k)
    points3d = reconstruct_by_lt(
        relative_r[:, :, None], relative_t, 2,
        np.vstack([inlier0[:2], inlier1[:2]]), k,
    )
    poses[1] = np.c_[relative_r, relative_t]
    active_keyframe = 1
    abs_r_keyframe, abs_t_keyframe = relative_r, relative_t
    previous_ranked = points1
    previous_with_3d = inlier1
    accumulated_map_size = points3d.shape[1]

    init_rotation_difference = init_translation_difference = init_point_max = ""
    init_point_median = ""
    reference = comparator.state(2) if comparator else None
    essential_iteration_matlab = essential_candidate_matlab = ""
    essential_inliers_equal = ""
    if comparator:
        essential_winner = loadmat(comparator.audit_dir / "essential/winner.mat", squeeze_me=True)
        essential_iteration_matlab = int(essential_winner["selected_iteration"])
        essential_candidate_matlab = int(essential_winner["selected_candidate"])
        essential_inliers_equal = np.array_equal(
            initial_inliers, np.atleast_1d(essential_winner["inlierIndx"]).astype(np.int64) - 1
        )
        init_pose = loadmat(comparator.audit_dir / "essential/initial_relative_pose.mat", squeeze_me=True)
        init_map = loadmat(comparator.audit_dir / "essential/initial_triangulation.mat", squeeze_me=True)["Points3D_Cam_Last_KF"]
        init_rotation_difference = rotation_difference_deg(np.asarray(init_pose["Rel_R"]), relative_r)
        init_translation_difference = float(np.linalg.norm(np.asarray(init_pose["Rel_T"]).reshape(3, 1) - relative_t))
        delta = np.abs(np.asarray(init_map) - points3d)
        init_point_max, init_point_median = float(np.max(delta)), float(np.median(delta))
        if not np.allclose(points3d, init_map, atol=1e-8, rtol=1e-8):
            comparator.note(
                "TRIANGULATION", 2, "initial_triangulation",
                f"max_abs={init_point_max:.9g}, median_abs={init_point_median:.9g}", structural=False,
            )
    rows.append({
        "frame_id_matlab_1based": 2,
        "frame_id_python_0based": 1,
        "active_keyframe_before_python": 1,
        "active_keyframe_before_matlab": int(reference["active_keyframe_before_frame"]) if reference else "",
        "matches_python": points0.shape[1],
        "matches_matlab": int(reference["number_of_matches"]) if reference else "",
        "covisible_python": len(initial_inliers),
        "covisible_matlab": int(reference["number_of_covisible_points"]) if reference else "",
        "factors_python": points3d.shape[1],
        "factors_matlab": int(reference["number_of_2d3d_factors"]) if reference else "",
        "factor_2d_sha256": _array_sha256(inlier1),
        "factor_3d_sha256": _array_sha256(points3d),
        "association_indices_equal": True if comparator else "",
        "selected_iteration_python_1based": essential_diag["selected_iteration"] + 1,
        "selected_iteration_matlab_1based": essential_iteration_matlab,
        "selected_candidate_python_1based": essential_diag["selected_candidate"] + 1,
        "selected_candidate_matlab_1based": essential_candidate_matlab,
        "sampled_triplets_equal": True if comparator else "",
        "inliers_equal": essential_inliers_equal,
        "inlier_indices_0based": " ".join(map(str, initial_inliers)),
        "post_lm_rotation_difference_deg": init_rotation_difference,
        "post_lm_translation_difference": init_translation_difference,
        "triangulation_max_abs_difference": init_point_max,
        "triangulation_median_abs_difference": init_point_median,
        "keyframe_python": 1,
        "keyframe_matlab": int(reference["is_keyframe"]) if reference else "",
        "active_keyframe_after_python": 2,
        "active_keyframe_after_matlab": int(reference["active_keyframe_after_frame"]) if reference else "",
        "map_size_python": points3d.shape[1],
        "map_size_matlab": int(reference["map_size"]) if reference else "",
    })

    for frame_id in range(2, frame_count):
        matlab_frame = frame_id + 1
        reference = comparator.state(matlab_frame) if comparator else None
        keyframe_before = active_keyframe
        if comparator and keyframe_before + 1 != int(reference["active_keyframe_before_frame"]):
            comparator.note(
                "KEYFRAME_BOUNDARY", matlab_frame, "before_frontend",
                f"python={keyframe_before + 1}, matlab={int(reference['active_keyframe_before_frame'])}",
                structural=True,
            )
            terminated = True
            break

        key_image = images[active_keyframe] if images is not None else None
        current_image = images[frame_id] if images is not None else None
        key_points, current_points, _ = frontend.match(
            key_image, current_image, active_keyframe, frame_id
        )
        _, _, current_covisible = matlab_intersect_rows(previous_ranked.T, key_points.T)
        key_covisible = key_points[:, current_covisible]
        current_covisible_points = current_points[:, current_covisible]
        _, map_indices, covisible_indices = matlab_intersect_rows(previous_with_3d.T, key_covisible.T)
        current_2d = current_covisible_points[:, covisible_indices]
        map_3d = points3d[:, map_indices]

        association_equal = ""
        factor_point_max = ""
        if comparator:
            association = loadmat(
                comparator.audit_dir / "frames" / f"frame_{matlab_frame:04d}_association.mat",
                squeeze_me=True,
            )
            exact_parts = [
                np.array_equal(current_covisible, comparator._indices(association, "CovIndx_KF_ranked")),
                np.array_equal(map_indices, comparator._indices(association, "Prev_KF_Indx_HavePts3D")),
                np.array_equal(covisible_indices, comparator._indices(association, "f_CF_Indx_HavePts3D")),
            ]
            association_equal = bool(all(exact_parts))
            msac_input = loadmat(
                comparator.audit_dir / "frames" / f"frame_{matlab_frame:04d}_msac_input.mat",
                squeeze_me=True,
            )
            matlab_points2d = np.asarray(msac_input["Points2D"])
            matlab_points3d = np.asarray(msac_input["Points3D"])
            same_factor_count = current_2d.shape == matlab_points2d.shape
            same_factor_order = same_factor_count and np.array_equal(current_2d, matlab_points2d)
            if map_3d.shape == matlab_points3d.shape:
                factor_point_max = float(np.max(np.abs(map_3d - matlab_points3d)))
            if not association_equal:
                comparator.note("ASSOCIATION_ORDER", matlab_frame, "row_intersection", str(exact_parts), structural=True)
            elif not same_factor_count:
                comparator.note(
                    "FACTOR_COUNT", matlab_frame, "msac_input",
                    f"python={current_2d.shape[1]}, matlab={matlab_points2d.shape[1]}", structural=True,
                )
            elif not same_factor_order:
                comparator.note("ASSOCIATION_ORDER", matlab_frame, "factor_order", "2D factor arrays differ", structural=True)
            if comparator.first_structural_divergence and stop_on_structural_divergence:
                terminated = True
                break

        try:
            abs_r, abs_t, msac_inliers, _, diagnostics = Msac4AbsolutePose_CHM(
                params, map_3d, current_2d, k, rng, frame_id,
                return_diagnostics=True,
            )
        except AssertionError as error:
            if comparator:
                comparator.note("P3P", matlab_frame, "sample_schedule", str(error), structural=True)
            terminated = True
            break
        poses[frame_id] = np.c_[abs_r, abs_t]

        selected_iteration_matlab = selected_candidate_matlab = ""
        inliers_equal = post_rotation_difference = post_translation_difference = ""
        pre_rotation_difference = pre_translation_difference = ""
        matlab_msac_cost = msac_cost_difference = ""
        if comparator:
            winner = loadmat(
                comparator.audit_dir / "frames" / f"frame_{matlab_frame:04d}_msac_winner.mat",
                squeeze_me=True,
            )
            post = loadmat(comparator.audit_dir / "lm" / f"frame_{matlab_frame:04d}_post.mat", squeeze_me=True)
            selected_iteration_matlab = int(winner["winner_iteration"])
            selected_candidate_matlab = int(winner["winner_candidate_index"])
            inliers_equal = np.array_equal(msac_inliers, comparator._indices(winner, "inlier_indices"))
            pre_pose = diagnostics["pose_before_refinement"]
            pre_rotation_difference = rotation_difference_deg(np.asarray(winner["R_before_LM"]), pre_pose[:, :3])
            pre_translation_difference = float(np.linalg.norm(
                np.asarray(winner["T_before_LM"]).reshape(3, 1) - pre_pose[:, 3:4]
            ))
            matlab_msac_cost = float(winner["winner_cost"])
            msac_cost_difference = abs(diagnostics["selected_msac_cost"] - matlab_msac_cost)
            post_rotation_difference = rotation_difference_deg(np.asarray(post["final_R"]), abs_r)
            post_translation_difference = float(np.linalg.norm(np.asarray(post["final_T"]).reshape(3, 1) - abs_t))
            if diagnostics["selected_iteration"] + 1 != selected_iteration_matlab or diagnostics["selected_candidate"] + 1 != selected_candidate_matlab:
                comparator.note(
                    "MSAC", matlab_frame, "winner",
                    f"python={diagnostics['selected_iteration'] + 1}/{diagnostics['selected_candidate'] + 1}, "
                    f"matlab={selected_iteration_matlab}/{selected_candidate_matlab}", structural=True,
                )
            elif not inliers_equal:
                comparator.note("MSAC", matlab_frame, "inliers", "inlier index arrays differ", structural=True)
            elif post_rotation_difference >= 1e-3 or post_translation_difference >= 1e-4:
                comparator.note(
                    "LM", matlab_frame, "post_refinement",
                    f"rotation_deg={post_rotation_difference:.9g}, translation={post_translation_difference:.9g}",
                    structural=False,
                )

        inlier_current = current_2d[:, msac_inliers]
        inlier_map = map_3d[:, msac_inliers]
        frames_since_keyframe = frame_id - active_keyframe
        covisible_ratio = len(map_indices) / points3d.shape[1]
        inserted = (
            frames_since_keyframe >= params.NUM_OF_FRAMES_FROM_LAST_KF
            or covisible_ratio < params.RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF
        )
        if comparator and bool(reference["is_keyframe"]) != inserted:
            comparator.note(
                "KEYFRAME_BOUNDARY", matlab_frame, "decision",
                f"python={int(inserted)}, matlab={int(reference['is_keyframe'])}", structural=True,
            )

        map_update_points_max = map_update_points2d_equal = map_update_ranked_equal = ""
        if inserted:
            _, known_current_indices, _ = matlab_intersect_rows(current_points.T, inlier_current.T)
            unknown_current = np.delete(current_points.T, known_current_indices, axis=0).T
            unknown_key = np.delete(key_points.T, known_current_indices, axis=0).T
            relative_r_keyframe = abs_r @ abs_r_keyframe.T
            relative_t_keyframe = abs_t - relative_r_keyframe @ abs_t_keyframe
            fundamental = np.linalg.inv(k).T @ (_skew(relative_t_keyframe) @ relative_r_keyframe) @ np.linalg.inv(k)
            line = fundamental @ unknown_key
            numerator = np.abs(np.sum(unknown_current * line, axis=0))
            denominator = np.sqrt(line[0] ** 2 + line[1] ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                epipolar_distance = numerator / denominator
            new_indices = np.flatnonzero(epipolar_distance <= params.INLIER_THRESH)
            new_key, new_current = unknown_key[:, new_indices], unknown_current[:, new_indices]
            new_local = reconstruct_by_lt(
                relative_r_keyframe[:, :, None], relative_t_keyframe, 2,
                np.vstack([new_key[:2], new_current[:2]]), k,
            )
            new_world = abs_r_keyframe.T @ (new_local - abs_t_keyframe)
            points3d = np.concatenate([new_world, inlier_map], axis=1)
            previous_with_3d = np.concatenate([new_current, inlier_current], axis=1)
            previous_ranked = current_points
            active_keyframe = frame_id
            abs_r_keyframe, abs_t_keyframe = abs_r, abs_t
            accumulated_map_size += points3d.shape[1]

            if comparator:
                update = loadmat(
                    comparator.audit_dir / "frames" / f"frame_{matlab_frame:04d}_keyframe_update.mat",
                    squeeze_me=True,
                )
                reference_points = np.asarray(update["Points3D_Cam_Last_KF"])
                map_update_points_max = float(np.max(np.abs(points3d - reference_points))) if points3d.shape == reference_points.shape else float("inf")
                map_update_points2d_equal = np.array_equal(previous_with_3d, np.asarray(update["Prev_f_KF_HavePts3D"]))
                map_update_ranked_equal = np.array_equal(previous_ranked, np.asarray(update["Prev_f_KF_ranked"]))
                if not map_update_points2d_equal or not map_update_ranked_equal:
                    comparator.note("MAP_UPDATE", matlab_frame, "keyframe_feature_state", "2D keyframe state differs", structural=True)

        if comparator:
            if int(reference["active_keyframe_after_frame"]) != active_keyframe + 1:
                comparator.note("KEYFRAME_BOUNDARY", matlab_frame, "after_update", "active keyframe differs", structural=True)
            if int(reference["map_size"]) != points3d.shape[1]:
                comparator.note(
                    "MAP_UPDATE", matlab_frame, "map_size",
                    f"python={points3d.shape[1]}, matlab={int(reference['map_size'])}", structural=True,
                )

        rows.append({
            "frame_id_matlab_1based": matlab_frame,
            "frame_id_python_0based": frame_id,
            "active_keyframe_before_python": keyframe_before + 1,
            "active_keyframe_before_matlab": int(reference["active_keyframe_before_frame"]) if reference else "",
            "matches_python": key_points.shape[1],
            "matches_matlab": int(reference["number_of_matches"]) if reference else "",
            "covisible_python": key_covisible.shape[1],
            "covisible_matlab": int(reference["number_of_covisible_points"]) if reference else "",
            "factors_python": current_2d.shape[1],
            "factors_matlab": int(reference["number_of_2d3d_factors"]) if reference else "",
            "factor_2d_sha256": _array_sha256(current_2d),
            "factor_3d_sha256": _array_sha256(map_3d),
            "association_indices_equal": association_equal,
            "factor_point_max_abs_difference": factor_point_max,
            "selected_iteration_python_1based": diagnostics["selected_iteration"] + 1,
            "selected_iteration_matlab_1based": selected_iteration_matlab,
            "selected_candidate_python_1based": diagnostics["selected_candidate"] + 1,
            "selected_candidate_matlab_1based": selected_candidate_matlab,
            "sampled_triplets_equal": True if comparator else "",
            "inliers_equal": inliers_equal,
            "inlier_indices_0based": " ".join(map(str, msac_inliers)),
            "pre_lm_rotation_difference_deg": pre_rotation_difference,
            "pre_lm_translation_difference": pre_translation_difference,
            "post_lm_rotation_difference_deg": post_rotation_difference,
            "post_lm_translation_difference": post_translation_difference,
            "keyframe_python": int(inserted),
            "keyframe_matlab": int(reference["is_keyframe"]) if reference else "",
            "active_keyframe_after_python": active_keyframe + 1,
            "active_keyframe_after_matlab": int(reference["active_keyframe_after_frame"]) if reference else "",
            "map_size_python": points3d.shape[1],
            "map_size_matlab": int(reference["map_size"]) if reference else "",
            "selected_msac_cost_python": diagnostics["selected_msac_cost"],
            "selected_msac_cost_matlab": matlab_msac_cost,
            "selected_msac_cost_abs_difference": msac_cost_difference,
            "inlier_count_python": len(msac_inliers),
            "map_update_points_max_abs_difference": map_update_points_max,
            "map_update_points2d_equal": map_update_points2d_equal,
            "map_update_ranked_equal": map_update_ranked_equal,
        })
        print(f"[closed-loop] frame {matlab_frame}/{frame_count}", flush=True)
        if comparator and comparator.first_structural_divergence and stop_on_structural_divergence:
            terminated = True
            break

    completed = int(np.flatnonzero(np.isfinite(poses[:, 0, 0]))[-1] + 1)
    _save_trajectory(output / "trajectory_raw.txt", poses[:completed])
    _write_csv(output / "frame_state.csv", rows)
    frontend.write_manifest(output / "frontend_manifest.csv")
    metadata = {
        "completed_frames": completed,
        "requested_frames": frame_count,
        "terminated_on_structural_divergence": terminated,
        "params": asdict(params),
        "first_numerical_divergence": comparator.first_numerical_divergence if comparator else None,
        "first_structural_divergence": comparator.first_structural_divergence if comparator else None,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
