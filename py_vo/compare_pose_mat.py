import os
import sys
import numpy as np
import scipy.io as sio

# Ensure package import works when running as script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from py_vo.rvo import rvo

def angle_between_R(R1: np.ndarray, R2: np.ndarray) -> float:
    val = 0.5 * (np.trace(R2.T @ R1) - 1.0)
    val = np.clip(val, -1.0, 1.0)
    return float(np.arccos(val))

def main():
    gt = sio.loadmat(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pose.mat')).get('Abs_Poses')
    if gt is None:
        raise FileNotFoundError('Abs_Poses not found in pose.mat')

    res = rvo(visualize=False)
    est = res['Estimated_Poses']

    n = min(gt.shape[2], est.shape[2])
    t_err = np.zeros(n)
    r_err = np.zeros(n)
    for i in range(n):
        t_err[i] = np.linalg.norm(gt[:, 3, i] - est[:, 3, i])
        r_err[i] = angle_between_R(est[:, :3, i], gt[:, :3, i])

    base_med = np.median(t_err[:5]) if n >= 5 else np.median(t_err)
    jumps = np.where(t_err > 2.0 * base_med)[0]
    first_jump = int(jumps[0]) if jumps.size else -1

    print(f"Frames compared: {n}")
    print(f"Translation RMSE: {np.sqrt(np.mean(t_err**2)):.6f}")
    print(f"Rotation RMSE (rad): {np.sqrt(np.mean(r_err**2)):.6f}")
    print(f"Translation error stats (m): min {t_err.min():.4f}, med {np.median(t_err):.4f}, max {t_err.max():.4f}")
    print(f"Rotation error stats (rad): min {r_err.min():.4f}, med {np.median(r_err):.4f}, max {r_err.max():.4f}")
    print(f"Max translation at frame {int(np.argmax(t_err))}: {t_err.max():.4f} m")
    print(f"Max rotation at frame {int(np.argmax(r_err))}: {r_err.max():.4f} rad")
    if first_jump >= 0:
        print(f"First jump (>2x median of first5) at frame {first_jump} with {t_err[first_jump]:.4f} m")
    else:
        print("No translation jump >2x median of first5 detected")

if __name__ == '__main__':
    main()
