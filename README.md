# RVO-MIS

RVO-MIS is a monocular visual odometry framework based on feature matching,
five-point essential-matrix estimation, and P3P/MSAC pose estimation. Beyond
minimally invasive surgery, it can also serve as a baseline for monocular VO
evaluation.

**Project Page:** [RVO-MIS](https://cho-wang001.github.io/rvomis/)<br>
**Paper:** [RVO-MIS: Robust Visual Odometry for Minimally Invasive Surgery](https://openreview.net/pdf?id=Gr3W3c5tz9)

## Installation

Python 3.10 is recommended. Create an environment and install the generic
runtime dependencies:

```bash
git clone --recurse-submodules https://github.com/vsi-lab/RVOMIS.git
cd RVOMIS
conda create -n rvomis python=3.10 -y
conda activate rvomis
python -m pip install --upgrade pip
python -m pip install -r req.txt
```

`req.txt` intentionally does not pin one platform-specific CUDA stack. Installing
it is sufficient to obtain dependencies, but does not guarantee bitwise
cross-platform reproduction. Strict numerical comparisons should use one of the
audited environments listed below.

### LightGlue resolution

The public frontend always uses the audited configuration:

- `SIFT(max_num_keypoints=4096)`
- `LightGlue(features="sift")`

It resolves LightGlue in this order:

1. `LG_PY_PATH`, pointing to `lg.py` or a LightGlue checkout.
2. The repository-local `Matlab/lightglue/LightGlue` submodule or `lg.py`.
3. An installed `lightglue` package.

The selected implementation and path are printed at runtime and recorded in
`frontend_manifest.csv`. The matcher implementation is never changed silently.

## Running Python RVO-MIS

Run from the repository root:

```bash
python -m py_vo.rvo \
  --image_glob "Matlab/MyData/fr2_desk/*.png" \
  --intrinsic_path "Matlab/MyData/IntrinsicMatrix.mat" \
  --gt_path "Matlab/MyData/GT_Poses.mat" \
  --output_dir "experiments/repro_run" \
  --output_filename "estimated_poses.txt" \
  --device cuda
```

Supported public options include:

- `--image_glob`
- `--intrinsic_path`
- `--gt_path` (pass `""` to disable GT evaluation)
- `--output_dir` and `--output_filename`
- `--device {cuda,cpu}`
- `--visualize`
- the original geometric thresholds and RANSAC iteration count

The validated backend uses MATLAB `rng(0,'twister')` and exact compatible
`randperm(n,k)` sampling. Consequently, `--seed` must remain `0` and `--n_jobs`
must remain `1`; parallel independent RNG streams would change the validated
sampling order.

`--device cuda` uses the CUDA-visible device exposed to PyTorch. To select a
physical GPU on Linux, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m py_vo.rvo --device cuda ...
```

GPU 0 is not a requirement; choose the desired device through
`CUDA_VISIBLE_DEVICES`.

### Input and output formats

Camera intrinsics can be loaded from `.mat`, `.txt`, `.yaml`, or `.yml` files.
Text input accepts a plain 3x3/3x4 matrix or common KITTI labels such as `P2`,
`P_rect_02`, and `K_02`.

The backend writes its unscaled world-to-camera trajectory to
`OUTPUT_DIR/backend/trajectory_raw.txt`. The requested output file contains one
row per frame with 12 row-major values. If GT is provided, that public output
uses the original MATLAB demo's first-baseline monocular scale convention; GT is
used only after VO inference for scaling and evaluation. With an empty GT path,
the requested output remains in the internal monocular scale.

## Validated Backend

The corrected `py_vo` package uses the MATLAB-compatible implementations of:

- five-point essential-matrix generation and Essential RANSAC;
- LambdaTwist P3P and exact MSAC truncated-residual scoring;
- LM pose refinement;
- MATLAB row-intersection, keyframe, triangulation, and map-update semantics;
- MATLAB `rng(0,'twister')` and two-input `randperm` sampling order.

Under frozen or otherwise matched frontend correspondences, the Python backend
reproduced the MATLAB geometric pipeline's Essential and P3P sampling schedules,
winning hypotheses, inlier sets, keyframe schedule, map-update behavior, and
final frozen-frontend trajectory metrics.

## Reproducibility and Platform Notes

> **Warning:** The Python geometric backend was validated against MATLAB under
> matched frontend correspondences and sampling. Raw-image Python/MATLAB bitwise
> equivalence is not claimed; the live SIFT/LightGlue frontend can differ across
> operating-system, CUDA, software, and GPU stacks.

The CUDA audit used identical image bytes, decoded pixels, `lg.py`,
LightGlue source, and model weights on Windows and Linux. Audited pairs had the
same SIFT keypoint counts, but the first structural difference occurred in the
SIFT keypoint coordinates. Scales, orientations, descriptors, LightGlue output,
and final matches then differed. Linux CUDA was internally deterministic across
three clean-process repetitions.

In that audit, the Windows CUDA frontend was closer to the MATLAB reference than
the audited Linux CUDA environment. This is an observation about those two
recorded platform stacks, not a claim that one operating system is generally
more accurate. At the live trajectory level, the first divergence was
`frontend_matches` at MATLAB frame 2 for both `fr2` and `D1K1`. Cross-platform
trajectory differences therefore must not automatically be attributed to the
Essential/P3P/MSAC/LM backend.

### Audited environments

| Component | Windows CUDA reference | Linux CUDA audit |
|---|---|---|
| OS | Windows 10 10.0.22631 SP0 | Linux 5.15.0-194-generic x86_64, glibc 2.35 |
| GPU | NVIDIA GeForce RTX 3050 4GB Laptop GPU | NVIDIA RTX A6000 |
| NVIDIA driver | not recorded | 615.71.09 |
| PyTorch CUDA runtime | 11.8 | 12.8 |
| Python | 3.10.13 | 3.10.19 |
| PyTorch | 2.6.0+cu118 | 2.9.1+cu128 |
| torchvision | 0.21.0+cu118 | 0.24.1+cu128 |
| NumPy | 1.26.4 | 2.2.6 |
| OpenCV | 4.11.0 | 4.12.0 |
| Kornia | 0.8.0 | 0.8.2 |

## MATLAB

The MATLAB implementation is under `Matlab/`. Configure its local Python
environment and data paths before running it. The Python release update does not
modify MATLAB source or MATLAB experiment artifacts.

## Utility Conversion

`tools/mat_to_kitti_txt.py` converts supported `.mat` matrices to plain text:

```bash
python tools/mat_to_kitti_txt.py \
  --input-dir Matlab/MyData \
  --output-dir Matlab/MyData/kitti_txt
```

## Citation

I would greatly appreciate it if you could cite this project:

```bibtex
@inproceedings{wang2026rvomis,
  title={{RVO}-{MIS}: Robust Visual Odometry for Minimally Invasive Surgery},
  author={Zhuo Wang and Chiang-Heng Chien and Eungjoo Lee},
  booktitle={Medical Imaging with Deep Learning},
  year={2026},
  url={https://openreview.net/forum?id=Gr3W3c5tz9}
}
```

## Contributors

Zhuo Wang ([zwang570@arizona.edu](mailto:zwang570@arizona.edu))  
Chiang-Heng Chien ([chiang-heng_chien@brown.edu](mailto:chiang-heng_chien@brown.edu))
