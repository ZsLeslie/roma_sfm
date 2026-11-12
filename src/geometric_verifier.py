"""Pairwise fundamental-matrix verification (not COLMAP track building)."""
from __future__ import annotations

import cv2
import numpy as np


def verify_fundamental(kpts0: np.ndarray, kpts1: np.ndarray,
                       threshold: float, confidence: float,
                       max_iters: int) -> np.ndarray:
    """Return a boolean mask with the same length as the input matches."""
    count = len(kpts0)
    if count < 8:
        return np.zeros(count, dtype=np.bool_)
    # USAC_MAGSAC is available in OpenCV 4.x builds such as Orin's 4.10.
    try:
        F, mask = cv2.findFundamentalMat(
            np.ascontiguousarray(kpts0, dtype=np.float32),
            np.ascontiguousarray(kpts1, dtype=np.float32),
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=threshold,
            confidence=confidence,
            maxIters=max_iters,
        )
    except cv2.error:
        return np.zeros(count, dtype=np.bool_)
    if F is None or mask is None or mask.size != count:
        return np.zeros(count, dtype=np.bool_)
    return mask.reshape(-1).astype(np.bool_)
