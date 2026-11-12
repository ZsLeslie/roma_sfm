#!/usr/bin/env python3
"""Run RoMa Outdoor once for a sequence of image pairs on Jetson Orin NX.

Example:
    python batch_match.py --images-dir /home/nvidia/colmap_data/DjiData_1367/images \
        --max-pairs 30 --coarse-res 448 --upsample-res 672

This stage produces NPZ correspondences; it does NOT modify a COLMAP database.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter

import numpy as np

from config import parse_args
from pair_generator import collect_pairs, safe_image_path
from match_exporter import pair_output_path, save_npz_atomic, save_visualization
from geometric_verifier import verify_fundamental
from mask_filter import resolve_pair_masks, masks_manifest

LOG = logging.getLogger("roma_sfm")
FIELDS = (
    "pair_id", "image0", "image1", "status", "sampled", "after_confidence",
    "after_mask", "after_dedup", "inliers", "inlier_ratio", "elapsed_s", "peak_cuda_gb", "error"
)


def _write_atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _checkpoint_info(path: Path):
    path = path.resolve()
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _config_signature(args) -> dict:
    """Settings that affect pair outputs; max-pairs/overlap are intentionally excluded."""
    return {
        "images_dir": str(args.images_dir),
        "roma_checkpoint": _checkpoint_info(args.weights_dir / "roma_outdoor.pth"),
        "dinov2_checkpoint": _checkpoint_info(args.weights_dir / "dinov2_vitl14_pretrain.pth"),
        "device": args.device,
        "coarse_res": args.coarse_res,
        "upsample_res": args.upsample_res,
        "num_matches": args.num_matches,
        "min_confidence": args.min_confidence,
        "dedup_radius": args.dedup_radius,
        "masks": masks_manifest(args.masks_dir),
        "mask_threshold": args.mask_threshold,
        "mask_erode": args.mask_erode,
        "ransac_threshold": args.ransac_threshold,
        "ransac_confidence": args.ransac_confidence,
        "ransac_max_iters": args.ransac_max_iters,
        "symmetric": False,
        "use_custom_corr": False,
        "amp_dtype": "float16",
        "sampling": "weighted_without_threshold_or_kde",
        "seed": args.seed,
        "pipeline_version": 2,
    }


def _prepare_output(args, pairs):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = args.output_dir / "run_config.json"
    signature = _config_signature(args)
    if signature_path.is_file():
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise ValueError(
                "Run-critical settings or weight files differ from run_config.json. "
                "Use a NEW --output-dir; never mix NPZ outputs made with different settings."
            )
    else:
        # An existing match directory without manifest must not be silently adopted.
        match_dir = args.output_dir / "matches"
        if match_dir.is_dir() and any(match_dir.glob("*.npz")):
            raise ValueError("Found NPZ files without run_config.json; use a new output directory")
        _write_atomic_text(signature_path, json.dumps(signature, indent=2, ensure_ascii=False) + "\n")
    pair_lines = "".join(f"{p.image0} {p.image1}\n" for p in pairs)
    _write_atomic_text(args.output_dir / "pairs_used.txt", pair_lines)
    (args.output_dir / "matches").mkdir(exist_ok=True)


class StatsJournal:
    """Append each finished pair immediately, compact duplicate rows when done."""
    def __init__(self, csv_path: Path):
        self.path = csv_path
        self.rows: dict[str, dict] = {}
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("pair_id"):
                        self.rows[row["pair_id"]] = {k: row.get(k, "") for k in FIELDS}

    def record(self, row: dict):
        record = {key: row.get(key, "") for key in FIELDS}
        first_write = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if first_write:
                writer.writeheader()
            writer.writerow(record)
            f.flush()
        self.rows[record["pair_id"]] = record

    def compact(self):
        fd, tmp = tempfile.mkstemp(prefix=".tmp_stats_", suffix=".csv", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(self.rows.values())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def _recover_stat(path: Path, pair):
    """Recover CSV metadata after a crash between atomic NPZ save and CSV append."""
    with np.load(path, allow_pickle=False) as d:
        if str(d["image0"]) != pair.image0 or str(d["image1"]) != pair.image1:
            raise ValueError(f"NPZ pair identity mismatch: {path}")
        n = len(d["inliers"])
        nin = int(d["inliers"].sum())
        return {
            "pair_id": pair.pair_id, "image0": pair.image0, "image1": pair.image1,
            "status": "ok", "sampled": int(d["sampled"]),
            "after_confidence": int(d["after_confidence"]),
            "after_mask": int(d["after_mask"]) if "after_mask" in d else int(d["after_confidence"]),
            "after_dedup": n, "inliers": nin,
            "inlier_ratio": f"{nin / n:.6f}" if n else "0.000000",
            "elapsed_s": f"{float(d['elapsed_s']):.3f}",
            "peak_cuda_gb": f"{float(d['peak_cuda_gb']):.3f}", "error": "",
        }


def _is_oom(exc: Exception):
    return "out of memory" in str(exc).lower() or "cuda error: out of memory" in str(exc).lower()


def main(argv=None) -> int:
    args = parse_args(argv)
    pairs = collect_pairs(args.images_dir, args.pairs_file, args.overlap, args.max_pairs)
    if not pairs:
        print("No image pairs found", file=sys.stderr)
        return 1
    masks_by_name = resolve_pair_masks(args.masks_dir, pairs)
    print(f"Images directory : {args.images_dir}")
    print(f"Total pairs      : {len(pairs)}")
    print(f"Mask directory   : {args.masks_dir if args.masks_dir else '(disabled)'}")
    if masks_by_name:
        print(f"Validated masks  : {len(masks_by_name)}")
    print(f"Output directory : {args.output_dir}")
    if args.dry_run:
        for pair in pairs[:12]:
            print(f"  {pair.image0}  {pair.image1}")
        if len(pairs) > 12:
            print(f"  ... and {len(pairs) - 12} more")
        return 0

    _prepare_output(args, pairs)
    # Use a dedicated logger; close handlers on repeated invocations in one process.
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    for old_handler in list(LOG.handlers):
        LOG.removeHandler(old_handler)
        old_handler.close()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(args.output_dir / "batch_match.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
    stats = StatsJournal(args.output_dir / "statistics.csv")
    pending = []
    skipped = 0
    for pair in pairs:
        npz_path = pair_output_path(args.output_dir, pair)
        if npz_path.is_file() and not args.overwrite:
            skipped += 1
            # If process died just after NPZ write, reconstruct the missing CSV row.
            if stats.rows.get(pair.pair_id, {}).get("status") != "ok":
                try:
                    stats.record(_recover_stat(npz_path, pair))
                except Exception as exc:
                    LOG.warning("Invalid existing NPZ; will recompute %s: %s", pair.pair_id, exc)
                    pending.append(pair)
                    skipped -= 1
        else:
            pending.append(pair)

    LOG.info("Planned=%d, skipped completed=%d, pending=%d", len(pairs), skipped, len(pending))
    if not pending:
        stats.compact()
        LOG.info("All requested image pairs are already complete")
        return 0

    import torch
    from model_loader import load_model
    from roma_matcher import match_pair

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    LOG.info("Loading RoMa Outdoor once: coarse=%d, upsample=%d", args.coarse_res, args.upsample_res)
    model = load_model(args.weights_dir, args.device, args.coarse_res, args.upsample_res)
    completed, failed, aborted_oom = 0, 0, False
    try:
        for index, pair in enumerate(pending, start=1):
            # Make sampled correspondences reproducible even after a resume,
            # regardless of how many earlier pairs were skipped.
            pair_seed = (args.seed ^ int(pair.pair_id[:8], 16)) & 0x7FFFFFFF
            torch.manual_seed(pair_seed)
            import cv2
            cv2.setRNGSeed(pair_seed)
            im0 = safe_image_path(args.images_dir, pair.image0)
            im1 = safe_image_path(args.images_dir, pair.image1)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = perf_counter()
            try:
                result = match_pair(
                    model, im0, im1, device=args.device,
                    num_matches=args.num_matches,
                    min_confidence=args.min_confidence,
                    dedup_radius=args.dedup_radius,
                    mask_path0=masks_by_name.get(pair.image0),
                    mask_path1=masks_by_name.get(pair.image1),
                    mask_threshold=args.mask_threshold,
                    mask_erode=args.mask_erode,
                )
                inliers = verify_fundamental(
                    result.keypoints0, result.keypoints1,
                    threshold=args.ransac_threshold,
                    confidence=args.ransac_confidence,
                    max_iters=args.ransac_max_iters,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                else:
                    peak = 0.0
                elapsed = perf_counter() - t0
                # NPZ must be committed BEFORE the CSV status, so interrupted
                # runs can safely recover stats from an existing NPZ.
                save_npz_atomic(pair_output_path(args.output_dir, pair), pair,
                                result, inliers, elapsed, peak)
                nin = int(inliers.sum())
                ratio = nin / result.after_dedup if result.after_dedup else 0.0
                stats.record({
                    "pair_id": pair.pair_id, "image0": pair.image0,
                    "image1": pair.image1, "status": "ok",
                    "sampled": result.sampled,
                    "after_confidence": result.after_confidence,
                    "after_mask": result.after_mask,
                    "after_dedup": result.after_dedup,
                    "inliers": nin, "inlier_ratio": f"{ratio:.6f}",
                    "elapsed_s": f"{elapsed:.3f}",
                    "peak_cuda_gb": f"{peak:.3f}", "error": "",
                })
                completed += 1
                LOG.info("[%d/%d] %s | confident=%d mask_kept=%d dedup=%d "
                         "inliers=%d (%.1f%%) time=%.2fs peak=%.2fGiB",
                         index, len(pending), pair.pair_id,
                         result.after_confidence, result.after_mask,
                         result.after_dedup, nin, 100 * ratio, elapsed, peak)
                if args.visualize_every and completed % args.visualize_every == 0:
                    try:
                        save_visualization(args.output_dir / "visualizations" / f"{pair.pair_id}.jpg",
                                           im0, im1, result, inliers)
                    except Exception as exc:
                        LOG.warning("Visualization failed for %s: %s", pair.pair_id, exc)
            except Exception as exc:
                failed += 1
                elapsed = perf_counter() - t0
                LOG.exception("[%d/%d] Pair failed: %s %s", index, len(pending), pair.image0, pair.image1)
                stats.record({
                    "pair_id": pair.pair_id, "image0": pair.image0,
                    "image1": pair.image1, "status": "error",
                    "elapsed_s": f"{elapsed:.3f}", "error": str(exc)[:500],
                })
                if _is_oom(exc):
                    LOG.error("CUDA OOM: stopping. Try --coarse-res 336 --upsample-res 448 "
                              "with a NEW --output-dir (configuration must remain consistent).")
                    aborted_oom = True
                    break
                if args.fail_fast:
                    break
    except KeyboardInterrupt:
        LOG.warning("Interrupted; completed NPZ files can be resumed")
        return 130
    finally:
        stats.compact()
    LOG.info("Finished: completed this run=%d, skipped=%d, failed=%d, OOM=%s",
             completed, skipped, failed, aborted_oom)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
