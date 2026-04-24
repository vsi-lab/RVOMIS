import numpy as np
from py_vo.rvo import rvo


def load_tum_poses(path: str) -> np.ndarray:
    """
    Load poses from a TUM-format pose.txt: ts tx ty tz qx qy qz qw.
    Returns (3,4,N) extrinsics [R|t].
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    assert data.shape[1] == 8, "Expected TUM format: ts tx ty tz qx qy qz qw"
    poses = np.zeros((3, 4, data.shape[0]))
    for i, row in enumerate(data):
        tx, ty, tz, qx, qy, qz, qw = row[1:]
        # Quaternion to rotation matrix (qw, qx, qy, qz)
        R = quat_to_rot(np.array([qw, qx, qy, qz], dtype=float))
        poses[:, :3, i] = R
        poses[:, 3, i] = np.array([tx, ty, tz], dtype=float)
    return poses


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = q / n
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    return R


def angle_between_R(R1: np.ndarray, R2: np.ndarray) -> float:
    val = 0.5 * (np.trace(R2.T @ R1) - 1.0)
    val = np.clip(val, -1.0, 1.0)
    return float(np.arccos(val))


def main():
    gt_path = "pose.txt"
    GT = load_tum_poses(gt_path)

    res = rvo(visualize=False)
    est = res["Estimated_Poses"]
    n = min(GT.shape[2], est.shape[2])
    t_err = []
    r_err = []
    for i in range(n):
        t_err.append(np.linalg.norm(GT[:, 3, i] - est[:, 3, i]))
        r_err.append(angle_between_R(est[:, :3, i], GT[:, :3, i]))

    t_err = np.array(t_err)
    r_err = np.array(r_err)

    # Find first frame where translation error jumps above 2x median of first 5 frames
    base_med = np.median(t_err[:5]) if n >= 5 else np.median(t_err)
    jump_idx = np.argmax(t_err > 2.0 * base_med)

    print(f"Frames compared: {n}")
    print(f"Translation error stats (m): min {t_err.min():.4f}, med {np.median(t_err):.4f}, max {t_err.max():.4f}")
    print(f"Rotation error stats (rad): min {r_err.min():.4f}, med {np.median(r_err):.4f}, max {r_err.max():.4f}")
    print(f"Max translation at frame {int(np.argmax(t_err))}: {t_err.max():.4f} m")
    print(f"Max rotation at frame {int(np.argmax(r_err))}: {r_err.max():.4f} rad")
    print(f"First jump (>2x median of first5) at frame {int(jump_idx)} with {t_err[jump_idx]:.4f} m")


if __name__ == "__main__":
    main()
