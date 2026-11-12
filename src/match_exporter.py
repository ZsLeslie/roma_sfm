"""Atomic per-pair NPZ output and lightweight optional visualizations."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from pair_generator import ImagePair
from roma_matcher import MatchResult


def pair_output_path(output_dir: Path, pair: ImagePair) -> Path:
    return output_dir / "matches" / f"{pair.pair_id}.npz"


def save_npz_atomic(path: Path, pair: ImagePair, result: MatchResult,
                    inliers: np.ndarray, elapsed_s: float,
                    peak_cuda_gb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temporary file stays in the destination directory for atomic os.replace.
    fd, name = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(
                f,
                image0=np.str_(pair.image0),
                image1=np.str_(pair.image1),
                keypoints0=result.keypoints0.astype(np.float32),
                keypoints1=result.keypoints1.astype(np.float32),
                confidence=result.confidence.astype(np.float32),
                inliers=inliers.astype(np.bool_),
                sampled=np.int32(result.sampled),
                after_confidence=np.int32(result.after_confidence),
                after_mask=np.int32(result.after_mask),
                elapsed_s=np.float32(elapsed_s),
                peak_cuda_gb=np.float32(peak_cuda_gb),
            )
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.remove(name)


def save_visualization(path: Path, image_path0: Path, image_path1: Path,
                       result: MatchResult, inliers: np.ndarray,
                       max_lines: int = 150) -> None:
    with Image.open(image_path0) as f:
        left = f.convert("RGB")
    with Image.open(image_path1) as f:
        right = f.convert("RGB")
    orig_left, orig_right = left.size, right.size
    left.thumbnail((960, 960), Image.Resampling.BILINEAR)
    right.thumbnail((960, 960), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(out)
    sx0, sy0 = left.width / orig_left[0], left.height / orig_left[1]
    sx1, sy1 = right.width / orig_right[0], right.height / orig_right[1]
    good_ids = np.flatnonzero(inliers)[:max_lines]
    for i in good_ids:
        xa, ya = result.keypoints0[i]
        xb, yb = result.keypoints1[i]
        p0 = (float(xa * sx0), float(ya * sy0))
        p1 = (float(xb * sx1 + left.width), float(yb * sy1))
        draw.line((p0, p1), fill=(60, 255, 110), width=1)
        draw.ellipse((p0[0] - 2, p0[1] - 2, p0[0] + 2, p0[1] + 2), fill=(255, 80, 80))
        draw.ellipse((p1[0] - 2, p1[1] - 2, p1[0] + 2, p1[1] + 2), fill=(255, 80, 80))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=88)
    left.close()
    right.close()
