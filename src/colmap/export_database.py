#!/usr/bin/env python3
"""Convert RoMa batch V2 NPZ matches into a native COLMAP feature database.

Run from the project root:
    python src/colmap/export_database.py --help

The output contains cameras, images, keypoints, and raw matches. You MUST run
COLMAP's `geometric_verifier` afterwards to populate two_view_geometries before
running `colmap mapper`. RoMa/MAGSAC's F-only inlier mask is not a calibrated
COLMAP two-view geometry estimate and is deliberately NOT written as one.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
from PIL import Image

# Support both `python src/colmap/export_database.py` and module invocation.
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from colmap.database_writer import COLMAPDatabaseWriter, load_camera_specs
from colmap.keypoint_manager import KeypointManager, resolve_one_to_one
from pair_generator import list_images, safe_image_path


@dataclass(frozen=True)
class PairRecord:
    path: Path
    image0: str
    image1: str


@dataclass
class PairData:
    image0: str
    image1: str
    points0: np.ndarray
    points1: np.ndarray
    confidence: np.ndarray
    n_original: int


def _pair_hash(a: str, b: str) -> str:
    return sha1((a + "\0" + b).encode("utf-8")).hexdigest()[:20]


def _scalar_name(array, path: Path) -> str:
    arr = np.asarray(array)
    if arr.ndim != 0 or arr.dtype.kind not in "US":
        raise ValueError(f"Invalid image name in {path}")
    return str(arr.item())


def _npz_identity(path: Path) -> tuple[str, str]:
    with np.load(path, allow_pickle=False) as npz:
        a, b = _scalar_name(npz["image0"], path), _scalar_name(npz["image1"], path)
    if a == b or _pair_hash(a, b) != path.stem:
        raise ValueError(f"NPZ filename/hash does not match its image names: {path}")
    return a, b


def find_pairs(match_output_dir: Path, max_pairs: int = 0) -> tuple[list[PairRecord], int]:
    match_dir = match_output_dir / "matches"
    if not match_dir.is_dir():
        raise FileNotFoundError(f"Missing RoMa matches directory: {match_dir}")
    existing = {p.stem: p for p in match_dir.glob("*.npz")}
    manifest = match_output_dir / "pairs_used.txt"
    records = []
    missing = 0
    if manifest.is_file():
        seen = set()
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed pairs_used.txt at line {line_no}")
            a, b = parts
            if a == b:
                continue
            unordered = tuple(sorted((a, b)))
            if unordered in seen:
                continue
            seen.add(unordered)
            path = existing.pop(_pair_hash(a, b), None)
            if path is None:
                missing += 1
            else:
                x, y = _npz_identity(path)
                if (a, b) != (x, y):
                    raise ValueError(f"NPZ pair mismatch at {path}")
                records.append(PairRecord(path, a, b))
            if max_pairs and len(records) >= max_pairs:
                break
        # Unlisted files are not automatically mixed into an existing run.
        # For limited exports there will naturally be unvisited files.
        if not max_pairs and existing:
            raise ValueError(f"{len(existing)} NPZ files are not in pairs_used.txt")
    else:
        seen = set()
        for path in sorted(existing.values()):
            a, b = _npz_identity(path)
            unordered = tuple(sorted((a, b)))
            if unordered in seen:
                raise ValueError(f"Duplicate unordered image pair: {a}, {b}")
            seen.add(unordered)
            records.append(PairRecord(path, a, b))
            if max_pairs and len(records) >= max_pairs:
                break
    if not records:
        raise ValueError("No completed RoMa NPZ files found")
    return records, missing


def read_pair(rec: PairRecord, image_sizes: dict[str, tuple[int, int]],
              selection: str) -> PairData:
    with np.load(rec.path, allow_pickle=False) as d:
        a = _scalar_name(d["image0"], rec.path)
        b = _scalar_name(d["image1"], rec.path)
        if (a, b) != (rec.image0, rec.image1):
            raise ValueError(f"Unexpected image names in {rec.path}")
        p0 = np.asarray(d["keypoints0"], dtype=np.float32)
        p1 = np.asarray(d["keypoints1"], dtype=np.float32)
        conf = np.asarray(d["confidence"], dtype=np.float32)
        inliers = np.asarray(d["inliers"], dtype=np.bool_)
    if p0.ndim != 2 or p0.shape[1] != 2 or p1.shape != p0.shape:
        raise ValueError(f"Invalid keypoints in {rec.path}: {p0.shape}, {p1.shape}")
    n = len(p0)
    if conf.shape != (n,) or inliers.shape != (n,):
        raise ValueError(f"Invalid confidence/inliers lengths in {rec.path}")
    if a not in image_sizes or b not in image_sizes:
        raise ValueError(f"Image missing from image directory: {a if a not in image_sizes else b}")
    w0, h0 = image_sizes[a]
    w1, h1 = image_sizes[b]
    keep = (
        np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
        & np.isfinite(conf)
        & (p0[:, 0] >= 0) & (p0[:, 0] < w0)
        & (p0[:, 1] >= 0) & (p0[:, 1] < h0)
        & (p1[:, 0] >= 0) & (p1[:, 0] < w1)
        & (p1[:, 1] >= 0) & (p1[:, 1] < h1)
    )
    if selection == "inliers":
        keep &= inliers
    return PairData(a, b, p0[keep], p1[keep], conf[keep], n)


def image_dimensions(images_dir: Path) -> dict[str, tuple[int, int]]:
    sizes = {}
    for name in list_images(images_dir):
        path = safe_image_path(images_dir, name)
        with Image.open(path) as img:
            sizes[name] = img.size
    return sizes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--match-output-dir", required=True, type=Path,
                   help="RoMa V2 output directory containing matches/*.npz")
    p.add_argument("--database-path", required=True, type=Path,
                   help="New COLMAP database output path")
    camera = p.add_argument_group("Camera intrinsics (do not guess)")
    camera.add_argument("--camera-model", type=str, default=None,
                        help="e.g. PINHOLE, OPENCV; all images must share calibration")
    camera.add_argument("--camera-params", type=str, default=None,
                        help="comma-separated COLMAP params, e.g. fx,fy,cx,cy")
    camera.add_argument("--camera-config", type=Path, default=None,
                        help="JSON with cameras and per-image camera assignment")
    p.add_argument("--selection", choices=("all", "inliers"), default="all",
                   help="all V2 prefiltered matches, or only its MAGSAC inliers")
    p.add_argument("--keypoint-bin-size", type=float, default=1.0,
                   help="Pixel grid cell width (NOT Euclidean radius), default 1.0")
    p.add_argument("--min-pair-matches", type=int, default=15,
                   help="Skip pair after one-to-one dedup if fewer than N correspondences")
    p.add_argument("--max-pairs", type=int, default=0,
                   help="Export first N completed pairs (0 = all)")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing database")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs without writing DB")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_pairs < 0 or args.min_pair_matches < 1:
        raise ValueError("--max-pairs must be >=0 and --min-pair-matches >=1")
    if not 0 < args.keypoint_bin_size <= 10:
        raise ValueError("--keypoint-bin-size must be within (0,10]")
    images_dir = args.images_dir.expanduser().resolve()
    match_dir = args.match_output_dir.expanduser().resolve()
    db_path = args.database_path.expanduser().resolve()
    if db_path.is_dir():
        raise ValueError("--database-path must be a file, not a folder")
    if db_path.exists() and not args.overwrite:
        raise FileExistsError(f"Database already exists: {db_path}; choose another path or --overwrite")
    if not (match_dir / "run_config.json").is_file():
        print("Warning: RoMa run_config.json absent; verify these NPZ files belong together.", file=sys.stderr)

    records, missing = find_pairs(match_dir, args.max_pairs)
    print(f"NPZ pairs         : {len(records)} (missing in manifest: {missing})", flush=True)
    sizes = image_dimensions(images_dir)
    print(f"Images registered : {len(sizes)}", flush=True)
    cameras, assignments = load_camera_specs(
        image_sizes=sizes, camera_config=args.camera_config,
        camera_model=args.camera_model, camera_params=args.camera_params,
    )
    print(f"Cameras           : {len(cameras)}", flush=True)

    manager = KeypointManager(bin_size=args.keypoint_bin_size)
    endpoints, original, invalid = 0, 0, 0
    for i, rec in enumerate(records, 1):
        data = read_pair(rec, sizes, args.selection)
        manager.add(data.image0, data.points0)
        manager.add(data.image1, data.points1)
        endpoints += len(data.points0)
        original += data.n_original
        invalid += data.n_original - len(data.points0)
        if i % 500 == 0:
            print(f"[index pass] {i}/{len(records)}", flush=True)
    manager.build()
    keypoint_count = sum(manager.counts().values())
    print(f"Candidate matches : {original} -> {endpoints} after selection/coordinate checks")
    print(f"Unique keypoints  : {keypoint_count} (grid {args.keypoint_bin_size:g} px)")
    if args.dry_run:
        return 0

    pair_rows = []
    total_written = 0
    written_pairs = 0
    # The DB stays temporary until the whole job succeeds. Read only one NPZ
    # pair at a time in the second pass; no full match matrix is held in memory.
    with COLMAPDatabaseWriter(db_path, overwrite=args.overwrite) as writer:
        id_map = writer.add_cameras_images(cameras, assignments)
        written_keypoints = writer.add_keypoints(id_map, manager)
        assert written_keypoints == keypoint_count
        for i, rec in enumerate(records, 1):
            data = read_pair(rec, sizes, args.selection)
            idx0 = manager.lookup(data.image0, data.points0)
            idx1 = manager.lookup(data.image1, data.points1)
            matched, chosen = resolve_one_to_one(idx0, idx1, data.confidence)
            if len(matched) >= args.min_pair_matches:
                writer.add_matches(id_map[data.image0], id_map[data.image1], matched)
                written_pairs += 1
                total_written += len(matched)
                status = "written"
            else:
                status = "too_few"
            pair_rows.append({
                "image0": data.image0, "image1": data.image1,
                "npz_matches": data.n_original, "selected": len(data.points0),
                "unique_matches": len(matched), "conflict_removed": len(data.points0)-len(matched),
                "status": status,
            })
            if i % 500 == 0:
                print(f"[database pass] {i}/{len(records)}; pairs={written_pairs}", flush=True)

    report_path = db_path.with_suffix(".export_report.csv")
    with report_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["image0", "image1", "npz_matches", "selected", "unique_matches",
                "conflict_removed", "status"]
        writer_csv = csv.DictWriter(f, fieldnames=cols)
        writer_csv.writeheader()
        writer_csv.writerows(pair_rows)
    print(f"Database saved     : {db_path}")
    print(f"Export report      : {report_path}")
    print(f"Matched image pairs: {written_pairs}")
    print(f"Stored matches     : {total_written}")
    print("NEXT: colmap geometric_verifier --database_path " + str(db_path))
    print("Then: colmap mapper --database_path DB --image_path IMAGES --output_path SPARSE_DIR")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
