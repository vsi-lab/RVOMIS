#!/usr/bin/env python3
"""Convert numeric arrays in .mat files to KITTI-style txt files.

Supported conversions:
- 3x3 matrix            -> saved as 3 rows x 3 cols txt (for K)
- Nx12 matrix           -> saved as-is (KITTI trajectory style)
- 3x4xN or Nx3x4 tensor -> converted to Nx12 row-major

Usage examples:
  python tools/mat_to_kitti_txt.py \
    --input-dir Matlab/MyData/Problem2 \
    --output-dir Matlab/MyData/Problem2/txt

  python tools/mat_to_kitti_txt.py \
    --input-file Matlab/MyData/Problem2/IntrinsicMatrix.mat \
    --output-dir Matlab/MyData/Problem2/txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import scipy.io as sio


def _iter_mat_files(input_file: Path | None, input_dir: Path | None) -> Iterable[Path]:
    if input_file is not None:
        yield input_file
        return
    assert input_dir is not None
    yield from sorted(input_dir.rglob("*.mat"))


def _to_kitti_rows(arr: np.ndarray) -> np.ndarray | None:
    """Return array ready for np.savetxt, or None if unsupported shape."""
    if not np.issubdtype(arr.dtype, np.number):
        return None

    data = np.asarray(arr)

    if data.ndim == 2:
        # K matrix or already KITTI rows
        if data.shape == (3, 3):
            return data.astype(float)
        if data.shape[1] == 12:
            return data.astype(float)
        if data.shape == (3, 4):
            return data.reshape(1, 12).astype(float)
        return None

    if data.ndim == 3:
        # 3x4xN (MATLAB style in this project)
        if data.shape[0] == 3 and data.shape[1] == 4:
            # (3,4,N) -> (N,3,4) -> (N,12)
            return np.transpose(data, (2, 0, 1)).reshape(data.shape[2], 12).astype(float)
        # Nx3x4
        if data.shape[1] == 3 and data.shape[2] == 4:
            return data.reshape(data.shape[0], 12).astype(float)
        return None

    return None


def _save_array(out_path: Path, rows: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, rows, fmt="%.9f")


def convert_mat_file(mat_path: Path, output_dir: Path) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    mat = sio.loadmat(str(mat_path))

    for key, value in mat.items():
        if key.startswith("__"):
            continue

        rows = _to_kitti_rows(value)
        if rows is None:
            continue

        out_name = f"{mat_path.stem}_{key}.txt"
        out_path = output_dir / out_name
        _save_array(out_path, rows)
        result.append((key, out_path))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .mat arrays to KITTI-style txt files.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-file", type=Path, help="Single .mat file to convert")
    g.add_argument("--input-dir", type=Path, help="Directory to recursively scan for .mat files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for txt files")
    args = parser.parse_args()

    input_file = args.input_file.resolve() if args.input_file else None
    input_dir = args.input_dir.resolve() if args.input_dir else None
    output_dir = args.output_dir.resolve()

    if input_file is not None:
        assert input_file.exists(), f"input-file does not exist: {input_file}"
        assert input_file.suffix.lower() == ".mat", f"input-file must be .mat: {input_file}"
    if input_dir is not None:
        assert input_dir.exists(), f"input-dir does not exist: {input_dir}"

    converted = 0
    generated = 0
    for mat_path in _iter_mat_files(input_file, input_dir):
        pairs = convert_mat_file(mat_path, output_dir)
        if not pairs:
            continue
        converted += 1
        for key, out_path in pairs:
            generated += 1
            print(f"[OK] {mat_path.name}:{key} -> {out_path}")

    print(f"Done. Converted files: {converted}, generated txt files: {generated}")


if __name__ == "__main__":
    main()
