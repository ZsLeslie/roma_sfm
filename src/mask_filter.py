"""Binary valid-region masks for RoMa correspondences (black=invalid)."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_mask_path(masks_dir: Path, image_name: str) -> Path:
    """Find a mask for an image, retaining any relative subdirectory.

    Supported in priority order:
       masks/sub/img.png, masks/sub/img.jpg.png, masks/sub/img.jpg.
    Mask paths cannot escape masks_dir, including via symlinks.
    """
    masks_dir = masks_dir.resolve()
    rel = Path(image_name)
    candidates = [rel.with_suffix(".png"), Path(f"{image_name}.png"), rel]
    seen = set()
    for candidate in candidates:
        path = (masks_dir / candidate).resolve()
        if not path.is_relative_to(masks_dir):
            raise ValueError(f"Mask path escapes mask directory: {image_name!r}")
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing binary mask for {image_name!r} in {masks_dir}. "
        "Expected relative image stem + '.png', image filename + '.png', "
        "or the original image filename."
    )


def resolve_pair_masks(masks_dir: Path | None, pairs) -> dict[str, Path]:
    """Preflight all requested pairs; missing masks cannot silently pass through."""
    if masks_dir is None:
        return {}
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {masks_dir}")
    names = {name for pair in pairs for name in (pair.image0, pair.image1)}
    return {name: resolve_mask_path(masks_dir, name) for name in sorted(names)}


def masks_manifest(masks_dir: Path | None) -> dict | None:
    """Fingerprint mask filenames and (size, mtime) to prevent stale NPZ reuse.

    Independent of --max-pairs, allowing a 30-pair test then full continuation.
    """
    if masks_dir is None:
        return None
    digest = sha256()
    count = 0
    for path in sorted(masks_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MASK_EXTS:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(masks_dir.resolve()):
            raise ValueError(f"Mask file escapes mask directory: {path}")
        st = path.stat()
        rel = path.relative_to(masks_dir).as_posix()
        digest.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode("utf-8"))
        count += 1
    return {"directory": str(masks_dir), "files": count, "sha256_stat": digest.hexdigest()}


def read_valid_mask(path: Path, image_size: tuple[int, int], *,
                    threshold: int = 128, erode: int = 0) -> np.ndarray:
    """Return HxW bool array; True means KEEP; black is excluded.

    image_size is (width,height). Nearest-neighbor resizing preserves discrete
    mask labels if source mask resolution differs from the original image.
    """
    with Image.open(path) as source:
        mask = source.convert("L")
        if mask.size != image_size:
            src_aspect = mask.width / mask.height
            dst_aspect = image_size[0] / image_size[1]
            if abs(src_aspect / dst_aspect - 1.0) > 0.01:
                raise ValueError(
                    f"Mask aspect ratio does not match the image: {path} "
                    f"mask={mask.size}, image={image_size}. "
                    "Masks must cover the SAME uncropped image area."
                )
            mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
        valid = np.asarray(mask) >= threshold

    if erode:
        import cv2
        # Shrinks the allowed (white) region so sky/foreground boundaries
        # do not contribute uncertain matches. Radius is in IMAGE pixels.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)
        )
        valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1) != 0
    return valid


def mask_correspondences(
    points0: np.ndarray, points1: np.ndarray, confidence: np.ndarray,
    mask_path0: Path, mask_path1: Path,
    image_size0: tuple[int, int], image_size1: tuple[int, int], *,
    threshold: int = 128, erode: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retain a correspondence only when BOTH endpoint pixels are valid."""
    if len(confidence) == 0:
        return points0, points1, confidence
    masks = (
        read_valid_mask(mask_path0, image_size0, threshold=threshold, erode=erode),
        read_valid_mask(mask_path1, image_size1, threshold=threshold, erode=erode),
    )
    keep = np.ones(len(confidence), dtype=bool)
    for points, valid in zip((points0, points1), masks):
        # match_pair has already rejected nonfinite/out-of-bounds points.
        x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, valid.shape[1] - 1)
        y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, valid.shape[0] - 1)
        keep &= valid[y, x]
    return points0[keep], points1[keep], confidence[keep]
