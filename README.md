# RVO-MIS

RVO-MIS is a monocular visual odometry (VO) framework based on feature matching, five-point essential matrix estimation, and P3P/MSAC pose estimation. Beyond its primary application in minimally invasive surgery (MIS), it generalizes well and can be readily deployed as a reliable baseline for outdoor visual odometry benchmarking.

## 1. Environment Setup

Python 3.9+ is recommended.

Create and activate a Conda environment:

```bash
cd /RVOMIS
conda create -n rvomis python=3.10 -y
conda activate rvomis
```

Install dependencies from `req.txt`:

```bash
pip install --upgrade pip
pip install -r req.txt
```

### LightGlue Dependency Notes

`py_vo/lightglue_adapter.py` supports two modes:

1. External `lg.py` (preferred by default)
- Set `LG_PY_PATH` to a directory containing `lg.py`.
- Or edit `CUSTOM_LG_PATH` in `py_vo/lightglue_adapter.py`.

2. Internal `lightglue` package
- If external `lg.py` is not loaded, it falls back to `from lightglue import LightGlue, SIFT`.
- You must install `lightglue` and its dependencies yourself.

For the MATLAB pipeline (`Matlab/lightglue` and `Matlab/rvo.m`), this repository includes LightGlue as a git submodule at:

- `Matlab/lightglue/LightGlue`

The wrapper file `Matlab/lightglue/lg.py` is kept in this repo and imports from the submodule path above.

Clone this repository with submodules:

```bash
git clone --recurse-submodules <your_rvomis_repo_url>
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

## 2. Reproducible Configuration (CLI)

This project is configured via command-line arguments.

Main path/config arguments:

- `--image_glob`: input image glob pattern
- `--intrinsic_path`: camera intrinsic file path
- `--gt_path`: GT pose path (pass empty string `""` to disable GT)
- `--output_dir`: output folder
- `--output_filename`: output pose filename
- `--device`: `cuda` or `cpu`
- `--visualize`: enable trajectory visualization (flag)

Hyper-parameters:

- `--inlier_thresh`
- `--ransac_iterations`
- `--top_n_ratio`
- `--num_frames_from_last_kf`
- `--ratio_covisible`
- `--n_jobs`

## 3. Run

From the repository root:

```bash
python -m py_vo.rvo
```

Full example:

```bash
cd /data/zwang570/RVOMIS
python -m py_vo.rvo \
  --image_glob "Matlab/MyData/fr2_desk/*.png" \
  --intrinsic_path "Matlab/MyData/IntrinsicMatrix.mat" \
  --gt_path "Matlab/MyData/GT_Poses.mat" \
  --output_dir "experiments/repro_run" \
  --output_filename "estimated_poses.txt" \
  --device cuda \
  --inlier_thresh 2.0 \
  --ransac_iterations 3000 \
  --top_n_ratio 0.8 \
  --num_frames_from_last_kf 15 \
  --ratio_covisible 0.55 \
  --n_jobs 8
```

Disable GT example:

```bash
python -m py_vo.rvo --gt_path ""
```

Without GT, the pipeline still runs normally and writes estimated poses.
The only difference is that `RMSET` and `RMSER` are not computed/printed.

## 4. Intrinsic File Formats

`--intrinsic_path` currently supports:

- `.mat` (keys like `K`, `IntrinsicMatrix`, `intrinsicMatrix`, `intrinsics`)
- `.yaml` / `.yml`
- `.txt` (including KITTI-style calibration text)

For KITTI-style `.txt`, the loader supports common entries such as `P2`, `P_rect_02`, `K_02`, and also plain numeric `3x3` / `3x4` formats.

## 5. Convert `.mat` to KITTI-style `.txt`

This repo provides a helper script:

- `tools/mat_to_kitti_txt.py`

It scans `.mat` files and exports numeric arrays into KITTI-style text files that can be directly consumed by the Python pipeline.

Supported conversions:

- `3x3` matrix -> `3x3` txt (camera intrinsics)
- `3x4xN` or `Nx3x4` tensor -> `Nx12` txt (trajectory rows)
- `Nx12` matrix -> saved as-is

Example (convert all `.mat` under a directory):

```bash
cd /data/zwang570/RVOMIS
python3 tools/mat_to_kitti_txt.py \
  --input-dir Matlab/MyData \
  --output-dir Matlab/MyData/kitti_txt
```

Example (single `.mat`):

```bash
python3 tools/mat_to_kitti_txt.py \
  --input-file Matlab/MyData/IntrinsicMatrix.mat \
  --output-dir Matlab/MyData/kitti_txt
```

Then point `--intrinsic_path` to the generated intrinsic txt file, for example:

```bash
python -m py_vo.rvo \
  --image_glob "Matlab/MyData/fr2_desk/*.png" \
  --intrinsic_path "Matlab/MyData/kitti_txt/IntrinsicMatrix_K.txt"
```

## 6. Output

- Output pose file: `OUTPUT_DIR/OUTPUT_FILENAME`
- Format: one row per frame, 12 values (3x4 row-major, KITTI-style)
- If GT is provided, `RMSET` and `RMSER` are printed to terminal

## 7. Assertions and Common Errors

The program validates paths and hyper-parameters before running.

Typical assertion failures:

- Missing intrinsic file: `Intrinsic matrix path does not exist`
- Not enough images: `Need at least 2 images`
- Missing GT file (when `--gt_path` is not empty)
- Invalid hyper-parameters (for example `RANSAC_ITERATIONS <= 0` or ratio arguments outside `(0, 1]`)

Minimal checklist when reproducing on a new dataset:

1. `--image_glob` matches at least 2 images
2. `--intrinsic_path` can be parsed into a valid 3x3 intrinsic matrix
3. LightGlue path/dependency is correctly set up (external `lg.py` or internal `lightglue`)

## 8. Reproducibility Note (Python vs MATLAB)

The MATLAB version uses a fixed random seed (`rng(0)`), while the current Python pipeline uses non-fixed random sampling in RANSAC/MSAC by default.

If you need strict reproducibility for paper-level result matching, please use the MATLAB version with fixed seed.

When running the MATLAB version, make sure to update the Python environment path in the main MATLAB script (`Matlab/rvo.m`) to your local absolute path (the `pyenv('Version', '...')` setting).

## 9. Citation

If you use this project, please cite:

```bibtex
@inproceedings{wang2026rvomis,
  title={{RVO}-{MIS}: Robust Visual Odometry for Minimally Invasive Surgery},
  author={Zhuo Wang and Chiang-Heng Chien and Eungjoo Lee},
  booktitle={Medical Imaging with Deep Learning},
  year={2026},
  url={https://openreview.net/forum?id=Gr3W3c5tz9}
}
```

## 10. Contributors

Zhuo Wang ([zwang570@arizona.edu](mailto:zwang570@arizona.edu))  
Chiang-Heng Chien ([chiang-heng_chien@brown.edu](mailto:chiang-heng_chien@brown.edu))
