import numpy as np
import matplotlib.pyplot as plt


def Visualize_Trajectory(GT_Poses: np.ndarray, Estimated_Poses: np.ndarray, KeyFrame_Indx):
    Abs_Cam_Center_GT = np.zeros((3, GT_Poses.shape[2]))
    for ci in range(GT_Poses.shape[2]):
        Abs_Cam_Center_GT[:, ci] = -GT_Poses[:3, :3, ci].T @ GT_Poses[:3, 3, ci]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Abs_Cam_Center_GT[0], Abs_Cam_Center_GT[1], Abs_Cam_Center_GT[2], "bo-")

    Abs_Cam_Center = np.zeros((3, Estimated_Poses.shape[2]))
    for ci in range(Estimated_Poses.shape[2]):
        Abs_Cam_Center[:, ci] = -Estimated_Poses[:3, :3, ci].T @ Estimated_Poses[:3, 3, ci]
    ax.plot(Abs_Cam_Center[0], Abs_Cam_Center[1], Abs_Cam_Center[2], "go-")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(["Ground Truth Trajectory", "Estimated Trajectory without keyponts adjustment"])
    ax.grid(True)
    plt.tight_layout()
    plt.show()
