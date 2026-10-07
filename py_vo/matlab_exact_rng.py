"""Exact MATLAB R2025b ``rng(0,'twister')`` and ``randperm(n,k)``.

The implementation is independent of saved sample schedules.  Schedules are
used only by the Day-34 validation tool.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampleRecord:
    stage: str
    frame_id: int
    iteration: int
    indices: tuple[int, ...]


class MatlabExactRNG:
    """MATLAB's factory MT19937 stream and two-input ``randperm``.

    MATLAB's saved seed-0 ``twister`` state is the canonical MT19937 state
    initialized with 5489.  MATLAB's two-input ``randperm(n,k)`` uses a
    partial Fisher-Yates draw from the shrinking unsampled prefix.  Each
    selected value consumes one uniform double from the shared stream.
    """

    MATLAB_TYPE = "twister"
    MATLAB_SEED = 0
    _MT_INITIALIZATION_SEED = 5489

    def __init__(self, seed: int = 0):
        if int(seed) != 0:
            raise ValueError("Day-34 validates only MATLAB rng(0,'twister')")
        self.seed = 0
        self._rng = np.random.RandomState(self._MT_INITIALIZATION_SEED)
        self.records: list[SampleRecord] = []
        self.uniform_draw_count = 0

    @classmethod
    def seed_zero(cls) -> "MatlabExactRNG":
        return cls(seed=0)

    @property
    def exact_matlab_values(self) -> bool:
        return True

    def rand(self, *size: int):
        shape = size if size else None
        value = self._rng.random_sample(shape)
        self.uniform_draw_count += int(np.size(value))
        return value

    def randperm(self, n: int, k: int | None = None) -> np.ndarray:
        """Return MATLAB-style 1-based indices in generated order."""
        n = int(n)
        if k is None:
            if n < 0:
                raise ValueError(f"Invalid randperm argument: n={n}")
            # MATLAB preserves the historical one-input random-key-sort
            # branch. It intentionally differs from randperm(n,n).
            return np.argsort(np.asarray(self.rand(n)), kind="stable").astype(np.int64) + 1
        k = int(k)
        if n < 0 or k < 0 or k > n:
            raise ValueError(f"Invalid randperm arguments: n={n}, k={k}")
        if k == 0:
            return np.empty(0, dtype=np.int64)

        # Sparse partial Fisher-Yates. ``mapping`` represents swaps without
        # allocating 1:n for the k << n branch used by RVO-MIS.
        mapping: dict[int, int] = {}
        result = np.empty(k, dtype=np.int64)
        uniforms = np.asarray(self.rand(k), dtype=np.float64).reshape(-1)
        for draw, uniform in enumerate(uniforms):
            remaining = n - draw
            selected_slot = int(np.floor(uniform * remaining))
            selected_value = mapping.get(selected_slot, selected_slot)
            last_slot = remaining - 1
            last_value = mapping.get(last_slot, last_slot)
            if selected_slot != last_slot:
                mapping[selected_slot] = last_value
            mapping.pop(last_slot, None)
            result[draw] = selected_value + 1
        return result

    def sample_without_replacement(
        self,
        n: int,
        k: int,
        *,
        stage: str,
        frame_id: int,
        iteration: int,
    ) -> np.ndarray:
        sample = self.randperm(n, k) - 1
        key = (str(stage), int(frame_id), int(iteration))
        if sample.shape != (k,) or len(np.unique(sample)) != k:
            raise AssertionError(f"Invalid sample for {key}: {sample}")
        self.records.append(SampleRecord(*key, tuple(int(x) for x in sample)))
        return sample.copy()

    def get_state(self) -> dict:
        algorithm, keys, position, has_gauss, cached_gaussian = self._rng.get_state()
        if algorithm != "MT19937" or has_gauss:
            raise AssertionError("Unexpected NumPy MT19937 state")
        matlab_state = np.concatenate([
            np.asarray(keys, dtype=np.uint32),
            np.asarray([position], dtype=np.uint32),
        ])
        return {
            "Type": self.MATLAB_TYPE,
            "Seed": self.MATLAB_SEED,
            "State": matlab_state,
            "CachedGaussian": float(cached_gaussian),
            "UniformDrawCount": int(self.uniform_draw_count),
        }

    def set_state(self, state) -> None:
        raw = state["State"] if isinstance(state, dict) else state
        values = np.asarray(raw, dtype=np.uint32).reshape(-1)
        if values.size != 625:
            raise ValueError(f"Expected 625 MATLAB state words, got {values.size}")
        position = int(values[-1])
        if not 0 <= position <= 624:
            raise ValueError(f"Invalid MT19937 position: {position}")
        self._rng.set_state(("MT19937", values[:624], position, 0, 0.0))
        self.uniform_draw_count = 0

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["stage", "frame_id", "iteration", "index_base", "indices"],
            )
            writer.writeheader()
            for row in self.records:
                writer.writerow({
                    "stage": row.stage,
                    "frame_id": row.frame_id,
                    "iteration": row.iteration,
                    "index_base": 0,
                    "indices": " ".join(map(str, row.indices)),
                })
