"""RoMa inference and confidence-preserving point filtering."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class MatchResult:
    keypoints0: np.ndarray
    keypoints1: np.ndarray
    confidence: np.ndarray
    sampled: int
    after_confidence: int
    after_mask: int | None = None

    def __post_init__(self):
        if self.after_mask is None:
            self.after_mask = self.after_confidence

    @property
    def after_dedup(self):
        return len(self.confidence)


def _dedup_by_both_images(pts0: np.ndarray, pts1: np.ndarray,
                          scores: np.ndarray, radius: float):
    """Keep highest-confidence match when either image has nearby keypoints."""
    if radius <= 0 or len(scores) == 0:
        return pts0, pts1, scores

    radius_sq = radius * radius
    grids = ({}, {})
    keep = []
    # score is kept unchanged; merges are evaluated in ORIGINAL pixel coordinates.
    for idx in np.argsort(-scores, kind="stable"):
        positions = (pts0[idx], pts1[idx])
        bad = False
        cell_keys = []
        for image_idx, position in enumerate(positions):
            x, y = map(float, position)
            cx, cy = floor(x / radius), floor(y / radius)
            cell_keys.append((cx, cy))
            grid = grids[image_idx]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for prev in grid.get((cx + dx, cy + dy), ()):
                        other = pts0[prev] if image_idx == 0 else pts1[prev]
                        if float(np.sum((position - other) ** 2)) < radius_sq:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    break
            if bad:
                break
        if bad:
            continue
        keep.append(int(idx))
        for image_idx, cell in enumerate(cell_keys):
            grids[image_idx].setdefault(cell, []).append(int(idx))
    # Highest-confidence first; stable for equal scores.
    selected = np.asarray(keep, dtype=np.int64)
    return pts0[selected], pts1[selected], scores[selected]


def match_pair(model, image_path0: Path, image_path1: Path, *, device: str,
               num_matches: int, min_confidence: float,
               dedup_radius: float, mask_path0: Path | None = None,
               mask_path1: Path | None = None, mask_threshold: int = 128,
               mask_erode: int = 0) -> MatchResult:
    import torch

    # Match PIL images directly to avoid redundant full-resolution disk reads.
    with Image.open(image_path0) as f:
        im0 = f.convert("RGB")
    with Image.open(image_path1) as f:
        im1 = f.convert("RGB")
    w0, h0 = im0.size
    w1, h1 = im1.size
    try:
        with torch.inference_mode():
            warp, certainty = model.match(im0, im1, device=device)
            matches, scores = model.sample(warp, certainty, num=num_matches)
            k0, k1 = model.to_pixel_coordinates(matches, h0, w0, h1, w1)
            pts0 = k0.float().cpu().numpy().astype(np.float32, copy=False)
            pts1 = k1.float().cpu().numpy().astype(np.float32, copy=False)
            conf = scores.float().cpu().numpy().astype(np.float32, copy=False)
            del warp, certainty, matches, scores, k0, k1
    finally:
        im0.close()
        im1.close()

    sampled = len(conf)
    valid = (np.isfinite(pts0).all(axis=1) & np.isfinite(pts1).all(axis=1)
             & np.isfinite(conf) & (conf >= min_confidence)
             & (pts0[:, 0] >= 0) & (pts0[:, 0] < w0)
             & (pts0[:, 1] >= 0) & (pts0[:, 1] < h0)
             & (pts1[:, 0] >= 0) & (pts1[:, 0] < w1)
             & (pts1[:, 1] >= 0) & (pts1[:, 1] < h1))
    pts0, pts1, conf = pts0[valid], pts1[valid], conf[valid]
    after_confidence = len(conf)
    if (mask_path0 is None) != (mask_path1 is None):
        raise ValueError("Both image masks must be provided or neither")
    if mask_path0 is not None:
        from mask_filter import mask_correspondences
        pts0, pts1, conf = mask_correspondences(
            pts0, pts1, conf, mask_path0, mask_path1, (w0, h0), (w1, h1),
            threshold=mask_threshold, erode=mask_erode,
        )
    after_mask = len(conf)
    pts0, pts1, conf = _dedup_by_both_images(pts0, pts1, conf, dedup_radius)
    return MatchResult(pts0, pts1, conf, sampled, after_confidence, after_mask)
