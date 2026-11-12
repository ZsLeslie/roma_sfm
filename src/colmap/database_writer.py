"""Create a fresh COLMAP database using the native pycolmap Database API.

Do not hand-write COLMAP's SQLite schema: COLMAP 4.x includes rigs/frames,
and camera storage evolves between versions. Writes are made to a temporary
new database, which replaces the target only when every step has succeeded.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .keypoint_manager import KeypointManager


@dataclass(frozen=True)
class CameraSpec:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


def _parse_parameters(value: str) -> tuple[float, ...]:
    try:
        params = tuple(float(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise ValueError("Camera params must be comma-separated numbers") from exc
    if not params or not np.isfinite(params).all() or params[0] <= 0:
        raise ValueError("Invalid camera parameters")
    return params


def load_camera_specs(*, image_sizes: Mapping[str, tuple[int, int]],
                      camera_config: Path | None,
                      camera_model: str | None,
                      camera_params: str | None) -> tuple[list[CameraSpec], dict[str, int]]:
    """Return camera specifications and exact image-to-camera assignments.

    The source dimensions must agree with every camera: no silent resizing or
    guessed image intrinsics are ever applied.
    """
    if not image_sizes:
        raise ValueError("No images to calibrate")
    if camera_config is not None:
        if camera_model is not None or camera_params is not None:
            raise ValueError("Use either --camera-config or --camera-model/--camera-params")
        data = json.loads(camera_config.read_text(encoding="utf-8"))
        raw_cameras = data.get("cameras", [])
        if not raw_cameras:
            raise ValueError("Camera JSON must contain nonempty 'cameras'")
        specs = []
        for c in raw_cameras:
            specs.append(CameraSpec(
                camera_id=int(c["id"]), model=str(c["model"]).upper(),
                width=int(c["width"]), height=int(c["height"]),
                params=tuple(float(x) for x in c["params"]),
            ))
        explicit = {str(n): int(cid) for n, cid in data.get("image_cameras", {}).items()}
        default_cam = data.get("default_camera_id")
        default_cam = int(default_cam) if default_cam is not None else None
        assignments = {}
        for name in image_sizes:
            if name in explicit:
                assignments[name] = explicit[name]
            elif default_cam is not None:
                assignments[name] = default_cam
            else:
                raise ValueError(f"Missing camera assignment for image: {name}")
    else:
        if not camera_model or not camera_params:
            raise ValueError("Specify --camera-config or BOTH --camera-model and --camera-params")
        distinct_dims = set(image_sizes.values())
        if len(distinct_dims) != 1:
            raise ValueError(
                "Images have different dimensions; supply --camera-config with camera IDs"
            )
        width, height = next(iter(distinct_dims))
        specs = [CameraSpec(1, camera_model.upper(), width, height,
                            _parse_parameters(camera_params))]
        assignments = {name: 1 for name in image_sizes}
    by_id = {c.camera_id: c for c in specs}
    if len(by_id) != len(specs):
        raise ValueError("Duplicate camera ID")
    for cam in specs:
        if cam.camera_id <= 0 or cam.camera_id >= 2147483647:
            raise ValueError(f"Invalid camera ID: {cam.camera_id}")
        if cam.width <= 0 or cam.height <= 0:
            raise ValueError(f"Invalid dimensions for camera {cam.camera_id}")
        if not cam.params or not np.isfinite(cam.params).all() or cam.params[0] <= 0:
            raise ValueError(f"Invalid intrinsics for camera {cam.camera_id}")
    for name, actual_size in image_sizes.items():
        cid = assignments[name]
        if cid not in by_id:
            raise ValueError(f"Image {name} refers to unknown camera ID {cid}")
        cam = by_id[cid]
        if actual_size != (cam.width, cam.height):
            raise ValueError(
                f"Image size / intrinsics mismatch for {name}: "
                f"image={actual_size} camera={cam.width}x{cam.height} (ID {cid})"
            )
    return specs, assignments


def create_pycolmap_camera(pycolmap, spec: CameraSpec):
    """Create and validate a pycolmap Camera without model-ID assumptions."""
    camera = pycolmap.Camera.create_from_model_name(
        spec.camera_id, spec.model, spec.params[0], spec.width, spec.height
    )
    if not camera.set_params_from_string(",".join(f"{v:.17g}" for v in spec.params)):
        raise ValueError(f"Camera model {spec.model} rejected {len(spec.params)} parameters")
    if not camera.verify_params():
        raise ValueError(f"Camera ID {spec.camera_id} has incompatible intrinsics")
    camera.has_prior_focal_length = True
    if hasattr(camera, "source") and hasattr(pycolmap, "CameraSource"):
        camera.source = pycolmap.CameraSource.USER
    return camera


class COLMAPDatabaseWriter:
    def __init__(self, db_path: Path, overwrite: bool = False):
        self.db_path = Path(db_path).expanduser().resolve()
        self.overwrite = overwrite
        self._temp: Path | None = None
        self._db = None
        self._pycolmap = None

    def __enter__(self):
        if self.db_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Database exists: {self.db_path} (use --overwrite explicitly)"
            )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pycolmap
        except ImportError as exc:
            raise RuntimeError(
                "pycolmap not found; use the same environment as COLMAP 4.x "
                "or install a compatible pycolmap build"
            ) from exc
        self._pycolmap = pycolmap
        fd, name = tempfile.mkstemp(prefix=".roma_colmap_", suffix=".db", dir=self.db_path.parent)
        os.close(fd)
        self._temp = Path(name)
        self._temp.unlink()  # Let pycolmap initialize a new empty DB + schema.
        try:
            self._db = pycolmap.Database.open(str(self._temp))
        except Exception:
            self.abort()
            raise
        return self

    def add_cameras_images(self, specs: list[CameraSpec],
                           assignments: Mapping[str, int]) -> dict[str, int]:
        db, lib = self._db, self._pycolmap
        # Each independent camera gets its own trivial rig (the camera is
        # the reference sensor). Each image gets a trivial frame. Without
        # these relations, recent COLMAP mappers may not be able to build a
        # consistent reconstruction even though cameras/images tables exist.
        sensors = {}
        for cam in sorted(specs, key=lambda s: s.camera_id):
            db.write_camera(create_pycolmap_camera(lib, cam), use_camera_id=True)
            sensor = lib.sensor_t(lib.SensorType.CAMERA, cam.camera_id)
            rig = lib.Rig()
            rig.rig_id = cam.camera_id
            rig.add_ref_sensor(sensor)
            db.write_rig(rig, use_rig_id=True)
            sensors[cam.camera_id] = sensor
        id_map = {}
        for image_id, name in enumerate(sorted(assignments), start=1):
            if image_id >= 2147483647:
                raise OverflowError("Too many COLMAP images")
            camid = int(assignments[name])
            data_id = lib.data_t(sensors[camid], image_id)
            frame = lib.Frame()
            frame.frame_id = image_id
            frame.rig_id = camid
            frame.add_data_id(data_id)
            db.write_frame(frame, use_frame_id=True)
            image = lib.Image(name=name, camera_id=camid, image_id=image_id)
            image.frame_id = image_id
            # image.data_id = data_id
            returned = db.write_image(image, use_image_id=True)
            if int(returned) != image_id:
                raise RuntimeError(f"COLMAP assigned unexpected image ID {returned} for {name}")
            id_map[name] = image_id
        return id_map

    def add_keypoints(self, id_map: Mapping[str, int], manager: KeypointManager) -> int:
        count = 0
        # Descriptors are deliberately empty: RoMa provides correspondences,
        # not SIFT descriptors. An empty entry keeps the DB schema complete;
        # no zero-filled fake descriptors are attached to the keypoints.
        desc = self._pycolmap.FeatureDescriptors(
            self._pycolmap.FeatureExtractorType.SIFT,
            np.empty((0, 128), dtype=np.uint8),
        )
        for name in sorted(id_map):
            pts = np.ascontiguousarray(manager.get(name), dtype=np.float32)
            self._db.write_keypoints(id_map[name], pts)
            self._db.write_descriptors(id_map[name], desc)
            count += len(pts)
        return count

    def add_matches(self, id0: int, id1: int, matches: np.ndarray) -> None:
        pairs = np.asarray(matches, dtype=np.uint32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"Matches should be Nx2, got {pairs.shape}")
        if id0 == id1:
            raise ValueError("Same-image matches are invalid")
        if len(pairs):
            self._db.write_matches(id0, id1, np.ascontiguousarray(pairs))

    def close(self, success: bool):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._temp is not None:
            if success:
                os.replace(self._temp, self.db_path)
            else:
                self._temp.unlink(missing_ok=True)
            self._temp = None

    def abort(self):
        self.close(False)

    def __exit__(self, exc_type, exc, tb):
        self.close(success=exc_type is None)
        return False
