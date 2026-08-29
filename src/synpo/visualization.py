from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import numpy as np
import psutil
import tifffile
import zarr
from skimage.measure import marching_cubes

from .detection import detection_cache_path
from .models import ProgressCallback
from .preprocessing import ProcessingCancelled
from .review import review_cache_path


@dataclass(frozen=True)
class ProjectionData:
    raw: np.ndarray
    dendrites: np.ndarray
    spines: np.ndarray
    clusters: np.ndarray


@dataclass(frozen=True)
class ContextVolume:
    xy: ProjectionData
    xz: ProjectionData
    yz: ProjectionData
    points_um: np.ndarray
    point_kinds: np.ndarray
    point_object_ids: np.ndarray
    xy_um_per_pixel: float
    z_step_um: float
    source_mode: str
    corrected: bool
    z_count: int
    roi_xy: tuple[int, int, int, int] | None = None
    mesh_vertices_um: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    mesh_faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    mesh_face_kinds: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8)
    )


def context_signature(
    manifest: dict[str, object], specimen_index: int, *, corrected: bool
) -> tuple[object, ...]:
    specimen = manifest["specimens"][specimen_index]
    detection_signature = specimen["checkpoints"]["detection"].get(
        "settings_signature"
    )
    if not corrected:
        return ("automatic", detection_signature)
    history = specimen["review"].get("history", [])
    latest = tuple(
        (entry.get("action_id"), bool(entry.get("undone", False)))
        for entry in history[-100:]
    )
    return (
        "review",
        detection_signature,
        specimen["checkpoints"]["review"].get("edit_count", 0),
        latest,
    )


def _cancel_if_requested(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled("3D-context generation was cancelled.")


def _mask_groups(
    manifest: dict[str, object], specimen_index: int, corrected: bool
) -> tuple[zarr.Group, zarr.Group, bool]:
    key = f"specimens/{specimen_index:04d}"
    detection_root = zarr.open_group(str(detection_cache_path(manifest)), mode="r")
    detection_group = detection_root[key]
    if not corrected or not review_cache_path(manifest).exists():
        return detection_group, detection_group, False
    review_root = zarr.open_group(str(review_cache_path(manifest)), mode="r")
    if key not in review_root:
        return detection_group, detection_group, False
    review_group = review_root[key]
    if not bool(review_group.attrs.get("initialized", False)):
        return detection_group, detection_group, False
    if review_group.attrs.get("detection_signature") != detection_group.attrs.get(
        "settings_signature"
    ):
        return detection_group, detection_group, False
    return review_group, detection_group, True


def _enforce_ram_policy(
    manifest: dict[str, object],
    shape: tuple[int, int, int],
    mesh_shape: tuple[int, int, int] | None = None,
) -> None:
    z, y, x = shape
    projection_pixels = (y * x + z * x + z * y) * 4
    mesh_workspace = int(np.prod(mesh_shape, dtype=np.int64)) * 12 if mesh_shape else 0
    estimated_extra = projection_pixels * 18 + mesh_workspace + 384 * 1024 * 1024
    fraction = min(
        0.8, float(manifest["resource_policy"].get("maximum_ram_fraction", 0.8))
    )
    if not 0 < fraction <= 0.8:
        raise ValueError("Maximum RAM fraction must be above zero and at most 0.8.")
    if psutil.Process().memory_info().rss + estimated_extra > int(
        psutil.virtual_memory().total * fraction
    ):
        raise MemoryError(
            "Generating the orthogonal and 3D views would exceed Synpo's RAM limit."
        )


def _volume_points(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint32)
    return np.argwhere(labels > 0).astype(np.int32, copy=False)


def _limit_points(
    points: list[np.ndarray],
    object_ids: list[np.ndarray],
    kinds: list[np.ndarray],
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.uint32),
        )
    all_points = np.concatenate(points, axis=0)
    all_ids = np.concatenate(object_ids, axis=0)
    all_kinds = np.concatenate(kinds, axis=0)
    if len(all_points) > maximum_points:
        selected = np.linspace(
            0, len(all_points) - 1, maximum_points, dtype=np.int64
        )
        all_points = all_points[selected]
        all_ids = all_ids[selected]
        all_kinds = all_kinds[selected]
    return all_points, all_kinds, all_ids


def _interpolated_meshes(
    masks: list[np.ndarray],
    *,
    x0: int,
    y0: int,
    xy_size: float,
    z_step: float,
    maximum_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    kind_chunks: list[np.ndarray] = []
    faces_per_kind = max(2000, maximum_faces // max(1, len(masks)))
    vertex_offset = 0
    for kind, mask in enumerate(masks):
        z_indices = np.flatnonzero(np.any(mask, axis=(1, 2)))
        if not len(z_indices):
            continue
        y_indices = np.flatnonzero(np.any(mask, axis=(0, 2)))
        x_indices = np.flatnonzero(np.any(mask, axis=(0, 1)))
        lower = np.asarray(
            (z_indices[0], y_indices[0], x_indices[0]), dtype=np.int32
        )
        upper = np.asarray(
            (z_indices[-1] + 1, y_indices[-1] + 1, x_indices[-1] + 1),
            dtype=np.int32,
        )
        cropped = mask[
            lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]
        ]
        padded = np.pad(cropped, 1, mode="constant", constant_values=False)
        step_size = 1
        while True:
            vertices_zyx, faces, _normals, _values = marching_cubes(
                padded.astype(np.uint8, copy=False),
                level=0.5,
                spacing=(z_step, xy_size, xy_size),
                step_size=step_size,
                allow_degenerate=False,
            )
            if len(faces) <= faces_per_kind or step_size >= 4:
                break
            step_size += 1
        origin_zyx = np.asarray(
            (
                lower[0] * z_step - z_step,
                (lower[1] + y0) * xy_size - xy_size,
                (lower[2] + x0) * xy_size - xy_size,
            ),
            dtype=np.float32,
        )
        vertices_zyx += origin_zyx
        vertices_xyz = vertices_zyx[:, (2, 1, 0)].astype(np.float32, copy=False)
        faces = faces.astype(np.int32, copy=False)
        vertex_chunks.append(vertices_xyz)
        face_chunks.append(faces + vertex_offset)
        kind_chunks.append(np.full(len(faces), kind, dtype=np.uint8))
        vertex_offset += len(vertices_xyz)
    if not vertex_chunks:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty(0, dtype=np.uint8),
        )
    return (
        np.concatenate(vertex_chunks),
        np.concatenate(face_chunks),
        np.concatenate(kind_chunks),
    )


def generate_context_volume(
    manifest: dict[str, object],
    specimen_index: int,
    background_channel: str,
    *,
    corrected: bool,
    include_3d: bool = True,
    roi_xy: tuple[int, int, int, int] | None = None,
    maximum_3d_points: int = 60000,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> ContextVolume:
    specimen = manifest["specimens"][specimen_index]
    metadata_shape = tuple(
        int(value)
        for value in specimen["channels"][background_channel]["metadata"]["shape"]
    )
    shape = (1, *metadata_shape) if len(metadata_shape) == 2 else metadata_shape
    z_count, y_count, x_count = shape
    if roi_xy is None:
        x0, y0, x1, y1 = 0, 0, x_count, y_count
    else:
        x0, y0, x1, y1 = (int(value) for value in roi_xy)
        x0, x1 = sorted(
            (min(x_count, max(0, x0)), min(x_count, max(0, x1)))
        )
        y0, y1 = sorted(
            (min(y_count, max(0, y0)), min(y_count, max(0, y1)))
        )
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("Select a larger rectangle for the 3D view.")
        roi_xy = (x0, y0, x1, y1)
    _enforce_ram_policy(
        manifest,
        shape,
        (z_count, y1 - y0, x1 - x0) if include_3d else None,
    )
    editable_group, detection_group, actually_corrected = _mask_groups(
        manifest, specimen_index, corrected
    )
    source = (
        Path(str(manifest["source_directory"]))
        / specimen["channels"][background_channel]["filename"]
    )
    raw_xy = np.zeros((y_count, x_count), dtype=np.uint16)
    raw_xz = np.zeros((z_count, x_count), dtype=np.uint16)
    raw_yz = np.zeros((z_count, y_count), dtype=np.uint16)
    projections: dict[str, list[np.ndarray]] = {
        name: [
            np.zeros((y_count, x_count), dtype=np.uint32),
            np.zeros((z_count, x_count), dtype=np.uint32),
            np.zeros((z_count, y_count), dtype=np.uint32),
        ]
        for name in ("dendrite_labels", "spine_labels", "cluster_labels")
    }
    point_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []
    kind_chunks: list[np.ndarray] = []
    mesh_masks = (
        [
            np.zeros((z_count, y1 - y0, x1 - x0), dtype=bool)
            for _kind in range(3)
        ]
        if include_3d
        else []
    )
    xy_size = float(manifest["calibration"]["xy_um_per_pixel"])
    z_step = float(manifest["calibration"]["z_step_um"])
    per_plane_limit = max(100, maximum_3d_points // max(1, z_count))
    phase = "Generating cropped 3D context" if include_3d else "Generating projections"

    with tifffile.TiffFile(source) as tiff:
        series = tiff.series[0]
        for z_index in range(z_count):
            _cancel_if_requested(cancel_event)
            if progress:
                progress(
                    phase,
                    z_index,
                    z_count,
                    f"Reading Z {z_index + 1}/{z_count}",
                )
            raw = np.squeeze(
                np.asarray(series.asarray() if z_count == 1 else series.asarray(key=z_index))
            ).astype(np.uint16, copy=False)
            np.maximum(raw_xy, raw, out=raw_xy)
            raw_xz[z_index] = raw.max(axis=0)
            raw_yz[z_index] = raw.max(axis=1)
            for kind, name in enumerate(
                ("dendrite_labels", "spine_labels", "cluster_labels")
            ):
                group = detection_group if name == "cluster_labels" else editable_group
                labels = np.asarray(group[name][z_index], dtype=np.uint32)
                xy_projection, xz_projection, yz_projection = projections[name]
                np.maximum(xy_projection, labels, out=xy_projection)
                xz_projection[z_index] = labels.max(axis=0)
                yz_projection[z_index] = labels.max(axis=1)
                if not include_3d:
                    continue
                mesh_masks[kind][z_index] = labels[y0:y1, x0:x1] > 0
                # Retain samples throughout each mask rather than only its 2D
                # outline. The renderer depth-tests these samples into a shaded
                # surface, producing a closed, solid-looking 3D object.
                coordinates = _volume_points(labels[y0:y1, x0:x1])
                if len(coordinates) > per_plane_limit:
                    coordinates = coordinates[
                        np.linspace(
                            0,
                            len(coordinates) - 1,
                            per_plane_limit,
                            dtype=np.int64,
                        )
                    ]
                if len(coordinates):
                    ids = labels[
                        coordinates[:, 0] + y0, coordinates[:, 1] + x0
                    ]
                    points = np.column_stack(
                        (
                            (coordinates[:, 1].astype(np.float32) + x0) * xy_size,
                            (coordinates[:, 0].astype(np.float32) + y0) * xy_size,
                            np.full(len(coordinates), z_index * z_step, dtype=np.float32),
                        )
                    )
                    point_chunks.append(points)
                    id_chunks.append(ids)
                    kind_chunks.append(np.full(len(ids), kind, dtype=np.uint8))

    points, point_kinds, point_ids = _limit_points(
        point_chunks, id_chunks, kind_chunks, maximum_3d_points
    )
    if include_3d:
        mesh_vertices, mesh_faces, mesh_face_kinds = _interpolated_meshes(
            mesh_masks,
            x0=x0,
            y0=y0,
            xy_size=xy_size,
            z_step=z_step,
            maximum_faces=maximum_3d_points,
        )
    else:
        mesh_vertices = np.empty((0, 3), dtype=np.float32)
        mesh_faces = np.empty((0, 3), dtype=np.int32)
        mesh_face_kinds = np.empty(0, dtype=np.uint8)
    if progress:
        progress(phase, z_count, z_count, "Views ready")
    return ContextVolume(
        xy=ProjectionData(
            raw_xy,
            projections["dendrite_labels"][0],
            projections["spine_labels"][0],
            projections["cluster_labels"][0],
        ),
        xz=ProjectionData(
            raw_xz,
            projections["dendrite_labels"][1],
            projections["spine_labels"][1],
            projections["cluster_labels"][1],
        ),
        yz=ProjectionData(
            raw_yz,
            projections["dendrite_labels"][2],
            projections["spine_labels"][2],
            projections["cluster_labels"][2],
        ),
        points_um=points,
        point_kinds=point_kinds,
        point_object_ids=point_ids,
        xy_um_per_pixel=xy_size,
        z_step_um=z_step,
        source_mode="review" if corrected else "automatic detection",
        corrected=actually_corrected,
        z_count=z_count,
        roi_xy=roi_xy,
        mesh_vertices_um=mesh_vertices,
        mesh_faces=mesh_faces,
        mesh_face_kinds=mesh_face_kinds,
    )
