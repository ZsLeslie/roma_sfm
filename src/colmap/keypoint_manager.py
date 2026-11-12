"""Image-global RoMa keypoints, using deterministic spatial quantization.

Different pair predictions may refer to nearly the same location in one image.
For scalability, bin coordinates into square cells, aggregate their positions,
and reuse one COLMAP keypoint ID for all coordinates falling in that cell.

This is *not* Euclidean nearest-neighbor clustering, nor proof that two
observations depict the same physical 3D point. Cell boundaries are strict.
COLMAP's geometric verification and mapper should handle remaining conflicts.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

_COORD_DTYPE = np.dtype([("y", "<i8"), ("x", "<i8")])


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Keypoints must be an Nx2 array, got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("Non-finite keypoint passed to KeypointManager")
    return np.ascontiguousarray(pts)


def quantize(points: np.ndarray, bin_size: float) -> np.ndarray:
    """Quantize pixel locations into (y, x) integer grid cells."""
    if not 0 < bin_size <= 10:
        raise ValueError("bin_size must be within (0, 10] pixels")
    pts = _as_points(points)
    # A fixed origin means the same coordinate is assigned independently of
    # the order in which image pairs are processed. np.rint uses ties-to-even.
    cells = np.rint(pts.astype(np.float64) / bin_size).astype(np.int64)
    cells = np.ascontiguousarray(cells[:, [1, 0]])
    return cells.view(_COORD_DTYPE).reshape(-1)


@dataclass
class ImageKeypoints:
    positions: np.ndarray  # float32 [K,2], means of candidate pixel locations
    keys: np.ndarray       # structured (y,x), sorted and unique


class KeypointManager:
    """Two-pass, order-independent image-global indexer.

    First pass: add all valid pair endpoints. Second pass: build unique indices.
    Later, lookup returns a stable COLMAP keypoint index for every endpoint.
    """

    def __init__(self, bin_size: float = 1.0):
        if not 0 < bin_size <= 10:
            raise ValueError("bin_size must be within (0, 10] pixels")
        self.bin_size = float(bin_size)
        self._pending: dict[str, list[np.ndarray]] = defaultdict(list)
        self._images: dict[str, ImageKeypoints] = {}
        self._built = False

    def add(self, image_name: str, points: np.ndarray) -> None:
        if self._built:
            raise RuntimeError("KeypointManager is finalized")
        pts = _as_points(points)
        if len(pts):
            self._pending[image_name].append(pts.copy())

    def build(self) -> None:
        if self._built:
            raise RuntimeError("Already built")
        for name in sorted(self._pending):
            chunks = self._pending.pop(name)
            pts = np.concatenate(chunks, axis=0)
            cells = quantize(pts, self.bin_size)
            keys, inverse, counts = np.unique(
                cells, return_inverse=True, return_counts=True
            )
            sum_x = np.bincount(inverse, weights=pts[:, 0], minlength=len(keys))
            sum_y = np.bincount(inverse, weights=pts[:, 1], minlength=len(keys))
            positions = np.column_stack((sum_x / counts, sum_y / counts)).astype(np.float32)
            self._images[name] = ImageKeypoints(positions=positions, keys=keys)
        self._pending.clear()
        self._built = True

    def get(self, image_name: str) -> np.ndarray:
        if not self._built:
            raise RuntimeError("Call build() first")
        data = self._images.get(image_name)
        return data.positions if data is not None else np.empty((0, 2), dtype=np.float32)

    def lookup(self, image_name: str, points: np.ndarray) -> np.ndarray:
        if not self._built:
            raise RuntimeError("Call build() first")
        cells = quantize(points, self.bin_size)
        if not len(cells):
            return np.empty((0,), dtype=np.uint32)
        data = self._images.get(image_name)
        if data is None:
            raise KeyError(f"No keypoints registered for {image_name}")
        indices = np.searchsorted(data.keys, cells)
        ok = indices < len(data.keys)
        ok[ok] &= data.keys[indices[ok]] == cells[ok]
        if not ok.all():
            raise ValueError(f"Keypoint not found in global index for {image_name}")
        if len(data.keys) >= 2**32:
            raise OverflowError("Too many keypoints for COLMAP's uint32 indexing")
        return indices.astype(np.uint32)

    def counts(self) -> dict[str, int]:
        if not self._built:
            raise RuntimeError("Call build() first")
        return {name: len(data.positions) for name, data in self._images.items()}


def resolve_one_to_one(indices0: np.ndarray, indices1: np.ndarray,
                       confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep highest-confidence matches, one-to-one within each image pair.

    Returns [M,2] uint32 matches, and the selected original row positions.
    Does not enforce graph-wide (multi-view) track consistency.
    """
    a = np.asarray(indices0, dtype=np.uint32).reshape(-1)
    b = np.asarray(indices1, dtype=np.uint32).reshape(-1)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if not (len(a) == len(b) == len(conf)):
        raise ValueError("indices and confidence lengths differ")
    if not np.isfinite(conf).all():
        raise ValueError("Non-finite match confidence")
    ranking = np.argsort(-conf, kind="stable")
    used_a, used_b = set(), set()
    selected = []
    for j in ranking:
        x, y = int(a[j]), int(b[j])
        if x in used_a or y in used_b:
            continue
        used_a.add(x)
        used_b.add(y)
        selected.append(int(j))
    if not selected:
        return np.empty((0, 2), dtype=np.uint32), np.empty((0,), dtype=np.int64)
    keep = np.asarray(selected, dtype=np.int64)
    return np.ascontiguousarray(np.column_stack((a[keep], b[keep])), dtype=np.uint32), keep
