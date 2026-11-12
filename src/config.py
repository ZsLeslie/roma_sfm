"""CLI configuration for Jetson-friendly RoMa batch matching."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch RoMa Outdoor matching; save per-pair NPZ for a future COLMAP importer."
    )
    parser.add_argument("--images-dir", "--image-dir", type=Path, required=True,
                        help="Image root. Pair names are relative to this directory.")
    parser.add_argument("--masks-dir", type=Path, default=None,
                        help="Optional valid-region masks; black=exclude, white=keep. "
                             "When set, every requested image MUST have a mask.")
    parser.add_argument("--mask-threshold", type=int, default=128,
                        help="Grayscale >= threshold is valid (default 128).")
    parser.add_argument("--mask-erode", type=int, default=0,
                        help="Erode WHITE allowed region by this radius in original image pixels.")
    parser.add_argument("--pairs-file", "--pairs", type=Path, default=None,
                        help="Optional two-column pairs.txt (image0 image1); otherwise sequential pairs.")
    parser.add_argument("--overlap", type=int, default=5,
                        help="When --pairs-file is absent, match each image to its next N images.")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Process only first N pairs; 0 means all pairs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--weights-dir", type=Path, default=Path.home() / "models" / "RoMa")
    parser.add_argument("--coarse-res", type=int, default=448)
    parser.add_argument("--upsample-res", type=int, default=672)
    parser.add_argument("--num-matches", type=int, default=3000)
    parser.add_argument("--min-confidence", type=float, default=0.05,
                        help="RoMa sigmoid certainty cutoff (true score; not thresholded to 1).")
    parser.add_argument("--dedup-radius", type=float, default=2.0,
                        help="Minimum separation in BOTH original-resolution images (pixels).")
    parser.add_argument("--ransac-threshold", type=float, default=2.0,
                        help="MAGSAC reprojection threshold in original-resolution pixels.")
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-max-iters", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize-every", type=int, default=0,
                        help="Save an inlier visualisation every N newly processed pairs; 0 disables.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute selected pairs. Required if changing run-critical settings.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on the first unexpected input/pair error (OOM always stops).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List planned pairs without loading RoMa or writing output.")
    args = parser.parse_args(argv)

    args.images_dir = args.images_dir.expanduser().resolve()
    args.weights_dir = args.weights_dir.expanduser().resolve()
    if args.masks_dir is not None:
        args.masks_dir = args.masks_dir.expanduser().resolve()
    if args.pairs_file is not None:
        args.pairs_file = args.pairs_file.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.images_dir.parent / "roma_matches"
    args.output_dir = args.output_dir.expanduser().resolve()

    if not 1 <= args.mask_threshold <= 255:
        parser.error("mask-threshold must be in [1, 255]")
    if args.mask_erode < 0:
        parser.error("mask-erode must be >= 0")
    if args.masks_dir is None and args.mask_erode:
        parser.error("--mask-erode requires --masks-dir")
    if args.overlap <= 0 or args.max_pairs < 0 or args.num_matches <= 0:
        parser.error("overlap and num-matches must be positive, max-pairs must be >= 0")
    if args.coarse_res <= 0 or args.coarse_res % 14:
        parser.error("coarse-res must be a positive multiple of 14")
    if args.upsample_res <= 0:
        parser.error("upsample-res must be positive")
    if not 0 <= args.min_confidence <= 1:
        parser.error("min-confidence must be in [0, 1]")
    if args.dedup_radius < 0 or args.ransac_threshold <= 0:
        parser.error("dedup-radius must be >= 0 and ransac-threshold > 0")
    if not 0 < args.ransac_confidence < 1 or args.ransac_max_iters <= 0:
        parser.error("invalid RANSAC settings")
    if args.visualize_every < 0:
        parser.error("visualize-every must be >= 0")
    return args
