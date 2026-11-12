"""Generate sequential pairs or parse the HLoc/COLMAP-style pairs.txt."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImagePair:
    image0: str  # POSIX path relative to images_dir
    image1: str

    @property
    def pair_id(self) -> str:
        return sha1((self.image0 + "\0" + self.image1).encode("utf-8")).hexdigest()[:20]


def safe_image_path(images_dir: Path, name: str) -> Path:
    """Never allow pair-list entries to read outside images_dir."""
    images_dir = images_dir.resolve()
    path = (images_dir / name).resolve()
    if not path.is_relative_to(images_dir):
        raise ValueError(f"Image name escapes image root: {name!r}")
    return path


def list_images(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    names = [p.relative_to(images_dir).as_posix()
             for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    names.sort()
    if not names:
        raise ValueError(f"No supported images in {images_dir}")
    if any(any(ch.isspace() for ch in name) for name in names):
        raise ValueError("Whitespace in an image filename is unsupported by two-column pairs.txt")
    return names


def generate_sequential(names: list[str], overlap: int) -> list[ImagePair]:
    return [ImagePair(names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, min(len(names), i + overlap + 1))]


def load_pairs(pairs_path: Path, images_dir: Path) -> list[ImagePair]:
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    pairs = []
    seen = set()
    for line_no, line in enumerate(pairs_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names = line.split()
        if len(names) != 2:
            raise ValueError(f"{pairs_path}:{line_no}: expected exactly 2 image names")
        a, b = names
        if a == b:
            continue
        for name in (a, b):
            p = safe_image_path(images_dir, name)
            if not p.is_file():
                raise FileNotFoundError(f"{pairs_path}:{line_no}: missing image: {p}")
        # Reverse duplicates are equivalent; preserve the direction of first occurrence.
        undirected_key = tuple(sorted((a, b)))
        if undirected_key not in seen:
            pairs.append(ImagePair(a, b))
            seen.add(undirected_key)
    return pairs


def collect_pairs(images_dir: Path, pairs_file: Path | None, overlap: int,
                  max_pairs: int) -> list[ImagePair]:
    if pairs_file is not None:
        pairs = load_pairs(pairs_file, images_dir)
    else:
        pairs = generate_sequential(list_images(images_dir), overlap)
    return pairs[:max_pairs] if max_pairs else pairs
