import os
import scipy.io as sio
import numpy as np


def Organize_Data(abs_poses_path: str, gt_pose_list_path: str, image_dir: str, out_dir: str):
    Abs_Poses = sio.loadmat(abs_poses_path)["Abs_Poses"]
    image_sequence = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    os.makedirs(out_dir, exist_ok=True)
    est_path = os.path.join(out_dir, "Estimations.txt")
    gt_path = os.path.join(out_dir, "GroundTruths.txt")
    GT_Poses = np.loadtxt(gt_pose_list_path)

    with open(est_path, "w") as est_f, open(gt_path, "w") as gt_f:
        for ci, name in enumerate(image_sequence):
            timestamp = os.path.splitext(name)[0]
            R = Abs_Poses[:, :3, ci]
            Q = rotm2quat(R)
            T = Abs_Poses[:, 3, ci : ci + 1]
            C = -R.T @ T
            C_str = " ".join(map(str, C.ravel()))
            Q_str = " ".join(map(str, [Q[3], Q[0], Q[1], Q[2]]))
            est_f.write(f"{timestamp} {C_str} {Q_str}\n")

            gt_row = " ".join(map(str, GT_Poses[ci, :]))
            gt_f.write(f"{timestamp} {gt_row}\n")

    Prev_Abs_R = quat2rotm([GT_Poses[0, 6], GT_Poses[0, 3], GT_Poses[0, 4], GT_Poses[0, 5]])
    Prev_Abs_C = GT_Poses[0, :3].reshape(3, 1)

    Prev_Abs_R_ = np.eye(3)
    Prev_Abs_T_ = np.zeros((3, 1))
    Abs_Poses_GT = np.zeros((3, 4, GT_Poses.shape[0]))
    Abs_Cam_Center_GT = np.zeros((3, GT_Poses.shape[0]))

    for mi in range(GT_Poses.shape[0]):
        Curr_Abs_R = quat2rotm([GT_Poses[mi, 6], GT_Poses[mi, 3], GT_Poses[mi, 4], GT_Poses[mi, 5]])
        Curr_Abs_C = GT_Poses[mi, :3].reshape(3, 1)
        Rel_R = Curr_Abs_R.T @ Prev_Abs_R
        Rel_T = Curr_Abs_R.T @ (Prev_Abs_C - Curr_Abs_C)

        Curr_Abs_R_ = Rel_R @ Prev_Abs_R_
        Curr_Abs_T_ = Rel_R @ Prev_Abs_T_ + Rel_T
        Abs_Poses_GT[:, :, mi] = np.hstack([Curr_Abs_R_, Curr_Abs_T_])
        Abs_Cam_Center_GT[:, mi] = -Curr_Abs_R_.T @ Curr_Abs_T_

        Prev_Abs_R_ = Curr_Abs_R_
        Prev_Abs_T_ = Curr_Abs_T_
        Prev_Abs_R = Curr_Abs_R
        Prev_Abs_C = Curr_Abs_C

    return Abs_Poses_GT, Abs_Cam_Center_GT


def rotm2quat(R):
    qw = np.sqrt(1 + np.trace(R)) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qx, qy, qz, qw])


def quat2rotm(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
