"""Load full RoMa Outdoor exactly once; do not reinstall or replace Jetson torch."""
from __future__ import annotations

import gc
from pathlib import Path


def load_model(weights_dir: Path, device: str, coarse_res: int, upsample_res: int):
    import torch
    from romatch import roma_outdoor

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment")
    torch.set_float32_matmul_precision("highest")  # required by current RoMa code

    path_roma = weights_dir / "roma_outdoor.pth"
    path_dino = weights_dir / "dinov2_vitl14_pretrain.pth"
    for path in (path_roma, path_dino):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    weights = torch.load(path_roma, map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(path_dino, map_location="cpu", weights_only=True)
    model = roma_outdoor(
        device=device,
        weights=weights,
        dinov2_weights=dinov2_weights,
        coarse_res=coarse_res,
        upsample_res=upsample_res,
        amp_dtype=torch.float16,
        symmetric=False,  # reduce computation on Orin NX
        use_custom_corr=False,  # no optional fused-local-corr extension
        upsample_preds=True,
        do_compile=False,
    )
    # RoMa's sample() tests for substrings "threshold" and "balanced".
    # The default "threshold_balanced" overwrites scores above sample_thresh
    # with 1.0; "weighted" keeps true sigmoid probabilities and skips KDE.
    model.sample_mode = "weighted"
    model.eval()
    del weights, dinov2_weights
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return model
