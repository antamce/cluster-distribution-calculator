from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np
import psutil
import tifffile
import zarr
from numcodecs import Blosc
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import closing, dilation, disk, skeletonize
from skimage.segmentation import watershed

from .models import ProgressCallback
from .preprocessing import ProcessingCancelled, project_cache_path
from .project import channel_source_path, save_project


ALGORITHM_VERSION = 2
AUTOMATIC_MEMORY_MODE = "automatic"
ALWAYS_LOW_MEMORY_MODE = "always_low_memory"
VALID_MEMORY_MODES = {AUTOMATIC_MEMORY_MODE, ALWAYS_LOW_MEMORY_MODE}
_MIB = 1024 * 1024


class InsufficientDetectionDiskSpace(OSError):
    def __init__(self, required_bytes: int, free_bytes: int) -> None:
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        super().__init__(
            "Low-memory detection needs "
            f"{_format_bytes(self.required_bytes)} of free temporary space, but only "
            f"{_format_bytes(self.free_bytes)} is available."
        )


def _format_bytes(value: int) -> str:
    amount = float(max(0, value))
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or suffix == "TiB":
            return f"{amount:.1f} {suffix}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


@dataclass(frozen=True)
class DetectionSettings:
    dendrite_sensitivity: float = 1.25
    cluster_sensitivity: float = 0.65
    spine_branch_length_um: float = 3.0
    minimum_dendrite_length_um: float = 4.0
    minimum_spine_projection_pixels: int = 6
    minimum_cluster_voxels: int = 25

    def validate(self) -> None:
        if not 0.25 <= self.dendrite_sensitivity <= 10.0:
            raise ValueError("Dendrite sensitivity must be between 0.25 and 10.0.")
        if not 0.25 <= self.cluster_sensitivity <= 10.0:
            raise ValueError("Cluster sensitivity must be between 0.25 and 10.0.")
        if not 0.5 <= self.spine_branch_length_um <= 10.0:
            raise ValueError("Spine branch length must be between 0.5 and 10 µm.")
        if not 0.5 <= self.minimum_dendrite_length_um <= 1000:
            raise ValueError("Minimum dendrite length must be at least 0.5 µm.")
        if self.minimum_spine_projection_pixels < 1:
            raise ValueError("Minimum spine projection area must be positive.")
        if self.minimum_cluster_voxels < 1:
            raise ValueError("Minimum cluster voxel count must be positive.")

    def to_dict(self) -> dict[str, float | int]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "DetectionSettings":
        result = cls(
            dendrite_sensitivity=float(value.get("dendrite_sensitivity", 1.25)),
            cluster_sensitivity=float(value.get("cluster_sensitivity", 0.65)),
            spine_branch_length_um=float(value.get("spine_branch_length_um", 3.0)),
            minimum_dendrite_length_um=float(
                value.get("minimum_dendrite_length_um", 4.0)
            ),
            minimum_spine_projection_pixels=int(
                value.get("minimum_spine_projection_pixels", 6)
            ),
            minimum_cluster_voxels=int(value.get("minimum_cluster_voxels", 25)),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class DetectionSummary:
    specimen_index: int
    dendrite_count: int
    spine_count: int
    cluster_count: int
    flagged_spine_count: int
    flagged_cluster_count: int
    elapsed_seconds: float
    skipped: bool = False
    processing_mode: str = "standard"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionSlice:
    raw: np.ndarray
    dendrites: np.ndarray
    spines: np.ndarray
    clusters: np.ndarray
    z_index: int
    z_count: int


def detection_cache_path(manifest: dict[str, object]) -> Path:
    return project_cache_path(manifest).parent / "detection.zarr"


def detection_signature(
    manifest: dict[str, object], specimen_index: int, settings: DetectionSettings
) -> str:
    specimen = manifest["specimens"][specimen_index]
    preprocessing_checkpoint = specimen["checkpoints"]["preprocessing"]
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "settings": settings.to_dict(),
        "preprocessing": {
            channel: value.get("settings_signature")
            for channel, value in preprocessing_checkpoint.get("channels", {}).items()
        },
        "channel_roles": manifest["channel_roles"],
        "calibration": manifest["calibration"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cancel_if_requested(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled(
            "Detection was cancelled. Completed specimen checkpoints remain available."
        )


def _role_channels(manifest: dict[str, object]) -> tuple[str, str]:
    by_role = {role: channel for channel, role in manifest["channel_roles"].items()}
    try:
        return by_role["dendrite_spines"], by_role["protein_clusters"]
    except KeyError as exc:
        raise ValueError("The project must assign one dendrite/spine and one protein channel.") from exc


def _validate_ram_fraction(maximum_ram_fraction: float) -> None:
    if not 0 < maximum_ram_fraction <= 0.8:
        raise ValueError("Maximum RAM fraction must be greater than zero and no more than 0.8.")


def _standard_detection_extra_bytes(shape: tuple[int, int, int]) -> int:
    voxels = int(np.prod(shape))
    return voxels * 12 + shape[1] * shape[2] * 48 + 256 * _MIB


def _standard_detection_fits(
    shape: tuple[int, int, int], maximum_ram_fraction: float
) -> bool:
    _validate_ram_fraction(maximum_ram_fraction)
    estimated_extra = _standard_detection_extra_bytes(shape)
    limit = int(psutil.virtual_memory().total * maximum_ram_fraction)
    return psutil.Process().memory_info().rss + estimated_extra <= limit


def _enforce_low_memory_projection_policy(
    shape: tuple[int, int, int], maximum_ram_fraction: float
) -> None:
    _validate_ram_fraction(maximum_ram_fraction)
    # The scientific dendrite/spine algorithm operates on a 2-D projection. The
    # source volume is streamed, but these 2-D working arrays must still fit.
    estimated_extra = shape[1] * shape[2] * 64 + 192 * _MIB
    limit = int(psutil.virtual_memory().total * maximum_ram_fraction)
    if psutil.Process().memory_info().rss + estimated_extra > limit:
        raise MemoryError(
            "Even low-memory detection cannot safely hold this stack's XY projection "
            "within Synpo's RAM limit. Close other Synpo views and try again."
        )


def _low_memory_required_disk_bytes(shape: tuple[int, int, int]) -> int:
    # Three final uint32 masks plus one provisional uint32 component volume,
    # with a conservative allowance for metadata and poorly compressible data.
    return int(np.prod(shape)) * 16 + 512 * _MIB


def _preflight_low_memory_disk(path: Path, shape: tuple[int, int, int]) -> tuple[int, int]:
    path.mkdir(parents=True, exist_ok=True)
    required = _low_memory_required_disk_bytes(shape)
    free = int(shutil.disk_usage(path).free)
    if free < required:
        raise InsufficientDetectionDiskSpace(required, free)
    return required, free


def _memory_mode(manifest: dict[str, object]) -> str:
    mode = str(manifest.get("detection", {}).get("memory_mode", AUTOMATIC_MEMORY_MODE))
    if mode not in VALID_MEMORY_MODES:
        raise ValueError(f"Unknown detection memory mode: {mode!r}.")
    return mode


def _percentile_projection_streamed(
    data: zarr.Array,
    percentile: float,
    *,
    cancel_event: Event | None,
) -> np.ndarray:
    z_count, y_count, x_count = (int(value) for value in data.shape)
    projection = np.empty((y_count, x_count), dtype=np.float32)
    # Bound the source block near 64 MiB. np.percentile may make an additional
    # work copy, so deliberately leave most of the low-memory budget untouched.
    bytes_per_row = max(1, z_count * x_count * int(np.dtype(data.dtype).itemsize))
    rows_per_block = max(1, min(y_count, (64 * _MIB) // bytes_per_row))
    for y0 in range(0, y_count, rows_per_block):
        _cancel_if_requested(cancel_event)
        y1 = min(y_count, y0 + rows_per_block)
        block = np.asarray(data[:, y0:y1, :])
        projection[y0:y1] = np.percentile(block, percentile, axis=0).astype(
            np.float32, copy=False
        )
    return projection


def _remove_small_labels(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]


def _relabel_selected(labels: np.ndarray, selected: np.ndarray) -> np.ndarray:
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.uint32)
    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[selected > 0]
    lookup[selected] = np.arange(1, len(selected) + 1, dtype=np.uint32)
    return lookup[labels]


def _segment_dendrites_and_spines(
    data: zarr.Array,
    settings: DetectionSettings,
    *,
    xy_um_per_pixel: float,
    cancel_event: Event | None,
    low_memory: bool = False,
    phase_callback: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    _cancel_if_requested(cancel_event)
    if low_memory:
        projection = _percentile_projection_streamed(
            data, 95, cancel_event=cancel_event
        )
    else:
        stack = np.asarray(data)
        projection = np.percentile(stack, 95, axis=0).astype(np.float32)
        del stack
    if phase_callback:
        phase_callback("projection")
    segmentation_projection = ndimage.gaussian_filter(projection, sigma=2.0)
    positive = segmentation_projection[segmentation_projection > 0]
    adaptive_threshold = float(threshold_otsu(positive)) if positive.size else 0.0
    applied_threshold = adaptive_threshold / settings.dendrite_sensitivity
    foreground = closing(segmentation_projection >= applied_threshold, disk(2))
    foreground = _remove_small_labels(
        foreground, max(20, settings.minimum_spine_projection_pixels)
    )
    if phase_callback:
        phase_callback("foreground")

    minimum_length_pixels = settings.minimum_dendrite_length_um / xy_um_per_pixel
    foreground_labels, foreground_count = ndimage.label(
        foreground, structure=np.ones((3, 3), dtype=bool)
    )
    attached_components: list[int] = []
    for label_id, bounds in enumerate(ndimage.find_objects(foreground_labels), start=1):
        if bounds is None:
            continue
        bounding_length = max(item.stop - item.start for item in bounds)
        if bounding_length >= minimum_length_pixels:
            attached_components.append(label_id)
    if not attached_components and foreground_count:
        sizes = np.bincount(foreground_labels.ravel())
        attached_components = [int(np.argmax(sizes[1:]) + 1)]
    dendrite_field = np.isin(foreground_labels, attached_components)
    skeleton = skeletonize(dendrite_field)
    trunk_skeleton = skeleton.copy()
    branch_length_pixels = max(
        1, int(round(settings.spine_branch_length_um / xy_um_per_pixel))
    )
    neighborhood = np.ones((3, 3), dtype=np.uint8)
    neighborhood[1, 1] = 0
    for _ in range(branch_length_pixels):
        neighbor_count = ndimage.convolve(
            trunk_skeleton.astype(np.uint8), neighborhood, mode="constant", cval=0
        )
        endpoints = trunk_skeleton & (neighbor_count <= 1)
        if not endpoints.any():
            break
        trunk_skeleton[endpoints] = False
    # Do not let pruning erase an entire short dendrite candidate.
    field_components, field_count = ndimage.label(
        dendrite_field, structure=np.ones((3, 3), dtype=bool)
    )
    for component_id in range(1, field_count + 1):
        component = field_components == component_id
        if not np.any(trunk_skeleton & component):
            trunk_skeleton |= skeleton & component
    terminal_skeleton = skeleton & ~trunk_skeleton
    terminal_labels, _ = ndimage.label(
        terminal_skeleton, structure=np.ones((3, 3), dtype=bool)
    )
    terminal_sizes = np.bincount(terminal_labels.ravel())
    minimum_terminal_pixels = max(4, int(round(0.25 / xy_um_per_pixel)))
    retained_terminal_ids = np.flatnonzero(terminal_sizes >= minimum_terminal_pixels)
    retained_terminal_ids = retained_terminal_ids[retained_terminal_ids > 0]
    terminal_skeleton = np.isin(terminal_labels, retained_terminal_ids)
    if terminal_skeleton.any():
        distance_to_trunk = ndimage.distance_transform_edt(~trunk_skeleton)
        distance_to_terminal = ndimage.distance_transform_edt(~terminal_skeleton)
        shaft_projection = (
            dendrite_field & (distance_to_trunk <= distance_to_terminal)
        )
    else:
        shaft_projection = dendrite_field.copy()
    if phase_callback:
        phase_callback("shaft core")
    dendrite_labels_2d, _ = ndimage.label(
        shaft_projection, structure=np.ones((3, 3), dtype=bool)
    )

    residual = dendrite_field & ~shaft_projection
    residual = _remove_small_labels(
        residual, settings.minimum_spine_projection_pixels
    )
    if phase_callback:
        phase_callback("spine residual")
    distance = ndimage.distance_transform_edt(residual)
    markers, marker_count = ndimage.label(
        terminal_skeleton & residual, structure=np.ones((3, 3), dtype=bool)
    )
    if phase_callback:
        phase_callback("spine maxima")
    residual_components, residual_count = ndimage.label(
        residual, structure=np.ones((3, 3), dtype=bool)
    )
    marked_components = set(int(value) for value in np.unique(residual_components[markers > 0]))
    marked_components.discard(0)
    residual &= np.isin(residual_components, list(marked_components))
    spine_labels_2d = watershed(
        -distance, markers, mask=residual, watershed_line=True
    ).astype(np.uint32)
    if phase_callback:
        phase_callback("spine watershed")
    adjacency = dilation(shaft_projection, disk(2)) & residual
    touching_labels = np.unique(spine_labels_2d[adjacency])
    spine_labels_2d = _relabel_selected(spine_labels_2d, touching_labels)

    projection_areas = np.bincount(spine_labels_2d.ravel())
    possible_filopodia: list[int] = []
    possible_dendrite_ends: list[int] = []
    edge_labels = set(
        int(value)
        for value in np.unique(
            np.concatenate(
                [
                    spine_labels_2d[0],
                    spine_labels_2d[-1],
                    spine_labels_2d[:, 0],
                    spine_labels_2d[:, -1],
                ]
            )
        )
        if value > 0
    )
    for label_id, bounds in enumerate(ndimage.find_objects(spine_labels_2d), start=1):
        if bounds is None:
            continue
        major = max(item.stop - item.start for item in bounds) * xy_um_per_pixel
        area = int(projection_areas[label_id])
        if major >= 2.0 and area < 1200:
            possible_filopodia.append(label_id)
        if area >= 2500 or label_id in edge_labels:
            possible_dendrite_ends.append(label_id)

    metadata = {
        "projection_percentile": 95,
        "projection_mask_gaussian_sigma_pixels": 2.0,
        "projection_otsu_threshold": adaptive_threshold,
        "projection_applied_threshold": applied_threshold,
        "spine_branch_length_pixels": branch_length_pixels,
        "minimum_terminal_branch_pixels": minimum_terminal_pixels,
        "possible_filopodia_ids": possible_filopodia,
        "possible_dendrite_end_ids": possible_dendrite_ends,
        "unassigned_projection_component_count": max(
            0, int(foreground_labels.max()) - len(attached_components)
        ),
    }
    return dendrite_labels_2d.astype(np.uint32), spine_labels_2d, metadata


def _compressor() -> Blosc:
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _create_mask_dataset(
    group: zarr.Group, name: str, shape: tuple[int, int, int]
) -> zarr.Array:
    return group.create_dataset(
        name,
        shape=shape,
        chunks=(1, min(512, shape[1]), min(512, shape[2])),
        dtype="uint32",
        compressor=_compressor(),
        overwrite=True,
    )


def _write_dendrite_and_spine_volumes(
    processed: zarr.Array,
    group: zarr.Group,
    dendrite_labels_2d: np.ndarray,
    spine_labels_2d: np.ndarray,
    settings: DetectionSettings,
    *,
    voxel_threshold: float,
    progress: ProgressCallback | None,
    progress_offset: int,
    progress_total: int,
    cancel_event: Event | None,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = tuple(int(value) for value in processed.shape)
    dendrite_output = _create_mask_dataset(group, "dendrite_labels", shape)
    spine_output = _create_mask_dataset(group, "spine_labels", shape)
    threshold = float(voxel_threshold)
    spine_count = int(spine_labels_2d.max())
    dendrite_count = int(dendrite_labels_2d.max())
    spine_voxels = np.zeros(spine_count + 1, dtype=np.int64)
    spine_first_z = np.full(spine_count + 1, shape[0], dtype=np.int32)
    spine_last_z = np.full(spine_count + 1, -1, dtype=np.int32)
    dendrite_voxels = np.zeros(dendrite_count + 1, dtype=np.int64)
    for z_index in range(shape[0]):
        _cancel_if_requested(cancel_event)
        foreground = np.asarray(processed[z_index]) >= threshold
        dendrites = np.where(foreground, dendrite_labels_2d, 0).astype(np.uint32)
        spines = np.where(foreground, spine_labels_2d, 0).astype(np.uint32)
        dendrite_output[z_index] = dendrites
        spine_output[z_index] = spines
        dendrite_voxels += np.bincount(
            dendrites.ravel(), minlength=dendrite_count + 1
        )
        counts = np.bincount(spines.ravel(), minlength=spine_count + 1)
        spine_voxels += counts
        present = np.flatnonzero(counts[1:]) + 1
        spine_first_z[present] = np.minimum(spine_first_z[present], z_index)
        spine_last_z[present] = z_index
        if progress:
            progress(
                "Detecting dendrites and spines",
                progress_offset + z_index + 1,
                progress_total,
                f"writing Z {z_index + 1}/{shape[0]}",
            )
    return (
        dendrite_count,
        spine_count,
        dendrite_voxels,
        spine_voxels,
        spine_first_z,
        spine_last_z,
    )


def _detect_clusters(
    processed: zarr.Array,
    group: zarr.Group,
    settings: DetectionSettings,
    *,
    progress: ProgressCallback | None,
    progress_offset: int,
    progress_total: int,
    cancel_event: Event | None,
) -> tuple[int, list[int], np.ndarray, np.ndarray, np.ndarray]:
    _cancel_if_requested(cancel_event)
    data = np.asarray(processed)
    threshold = (
        float(processed.attrs["statistics"]["applied_threshold"])
        / settings.cluster_sensitivity
    )
    mask = data >= threshold
    del data
    mask = ndimage.binary_closing(
        mask, structure=np.ones((1, 3, 3), dtype=bool), iterations=1
    )
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    del mask
    sizes = np.bincount(labels.ravel())
    selected = np.flatnonzero(sizes >= settings.minimum_cluster_voxels)
    selected = selected[selected > 0]
    labels = _relabel_selected(labels, selected)
    cluster_count = int(labels.max())
    cluster_voxels = np.bincount(labels.ravel(), minlength=cluster_count + 1)
    first_z = np.full(cluster_count + 1, labels.shape[0], dtype=np.int32)
    last_z = np.full(cluster_count + 1, -1, dtype=np.int32)
    output = _create_mask_dataset(group, "cluster_labels", tuple(labels.shape))
    for z_index in range(labels.shape[0]):
        _cancel_if_requested(cancel_event)
        plane = labels[z_index].astype(np.uint32, copy=False)
        output[z_index] = plane
        present = np.unique(plane)
        present = present[present > 0]
        first_z[present] = np.minimum(first_z[present], z_index)
        last_z[present] = z_index
        if progress:
            progress(
                "Detecting protein clusters",
                progress_offset + z_index + 1,
                progress_total,
                f"writing Z {z_index + 1}/{labels.shape[0]}",
            )
    spans = last_z - first_z + 1
    flagged = [
        label_id
        for label_id in range(1, cluster_count + 1)
        if cluster_voxels[label_id] < 100 or spans[label_id] < 3
    ]
    return cluster_count, flagged, cluster_voxels, first_z, last_z


def _union_component_pair(
    parent: np.ndarray, rank: np.ndarray, first: int, second: int
) -> None:
    while parent[first] != first:
        parent[first] = parent[parent[first]]
        first = int(parent[first])
    while parent[second] != second:
        parent[second] = parent[parent[second]]
        second = int(parent[second])
    if first == second:
        return
    if rank[first] < rank[second]:
        first, second = second, first
    parent[second] = first
    if rank[first] == rank[second]:
        rank[first] += 1


def _merge_slab_boundary_components(
    parent: np.ndarray,
    rank: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
) -> None:
    """Union all 26-connected provisional labels across one Z seam."""
    height, width = previous.shape
    for dy in (-1, 0, 1):
        previous_y0 = max(0, -dy)
        previous_y1 = min(height, height - dy)
        current_y0 = previous_y0 + dy
        current_y1 = previous_y1 + dy
        for dx in (-1, 0, 1):
            previous_x0 = max(0, -dx)
            previous_x1 = min(width, width - dx)
            current_x0 = previous_x0 + dx
            current_x1 = previous_x1 + dx
            first = previous[
                previous_y0:previous_y1, previous_x0:previous_x1
            ]
            second = current[current_y0:current_y1, current_x0:current_x1]
            valid = (first > 0) & (second > 0) & (first != second)
            if not np.any(valid):
                continue
            keys = (
                first[valid].astype(np.uint64) << np.uint64(32)
            ) | second[valid].astype(np.uint64)
            for key in np.unique(keys):
                _union_component_pair(
                    parent,
                    rank,
                    int(key >> np.uint64(32)),
                    int(key & np.uint64(0xFFFFFFFF)),
                )


def _low_memory_slab_depth(
    shape: tuple[int, int, int], maximum_ram_fraction: float
) -> int:
    limit = int(psutil.virtual_memory().total * maximum_ram_fraction)
    current_rss = psutil.Process().memory_info().rss
    bytes_per_plane = max(1, shape[1] * shape[2] * 10)
    if current_rss + bytes_per_plane + 64 * _MIB > limit:
        raise MemoryError(
            "Low-memory cluster detection cannot safely fit even one Z plane "
            "within Synpo's RAM limit."
        )
    available = limit - current_rss
    target = min(256 * _MIB, max(32 * _MIB, available // 4))
    return max(1, min(shape[0], target // bytes_per_plane))


def _detect_clusters_low_memory(
    processed: zarr.Array,
    group: zarr.Group,
    settings: DetectionSettings,
    *,
    maximum_ram_fraction: float,
    progress: ProgressCallback | None,
    progress_offset: int,
    progress_total: int,
    cancel_event: Event | None,
) -> tuple[int, list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Label clusters in Z slabs and reconcile objects crossing slab seams."""
    _cancel_if_requested(cancel_event)
    shape = tuple(int(value) for value in processed.shape)
    slab_depth = _low_memory_slab_depth(shape, maximum_ram_fraction)
    threshold = (
        float(processed.attrs["statistics"]["applied_threshold"])
        / settings.cluster_sensitivity
    )
    if "_cluster_work" in group:
        del group["_cluster_work"]
    work = group.require_group("_cluster_work")
    provisional = work.create_dataset(
        "provisional_labels",
        shape=shape,
        chunks=(1, min(512, shape[1]), min(512, shape[2])),
        dtype="uint32",
        compressor=_compressor(),
        overwrite=True,
    )
    component_sizes: list[np.ndarray] = []
    total_components = 0
    slab_starts = list(range(0, shape[0], slab_depth))

    for slab_number, z0 in enumerate(slab_starts, start=1):
        _cancel_if_requested(cancel_event)
        z1 = min(shape[0], z0 + slab_depth)
        data = np.asarray(processed[z0:z1, :, :])
        mask = data >= threshold
        del data
        mask = ndimage.binary_closing(
            mask, structure=np.ones((1, 3, 3), dtype=bool), iterations=1
        )
        labels, count = ndimage.label(
            mask, structure=np.ones((3, 3, 3), dtype=bool)
        )
        del mask
        if total_components + count > np.iinfo(np.uint32).max:
            raise MemoryError("This stack contains too many cluster components to label.")
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(
            np.int64, copy=False
        )
        component_sizes.append(sizes)
        if total_components:
            foreground = labels > 0
            labels[foreground] += total_components
        provisional[z0:z1, :, :] = labels.astype(np.uint32, copy=False)
        total_components += int(count)
        if progress:
            current = progress_offset + max(
                1, int(round((z1 / shape[0]) * shape[0] / 3))
            )
            progress(
                "Detecting protein clusters (low memory)",
                current,
                progress_total,
                f"labeling slab {slab_number}/{len(slab_starts)}",
            )

    reconciliation_bytes = total_components * 28 + 128 * _MIB
    memory_limit = int(psutil.virtual_memory().total * maximum_ram_fraction)
    if psutil.Process().memory_info().rss + reconciliation_bytes > memory_limit:
        raise MemoryError(
            "This stack contains too many provisional protein-cluster components "
            "to reconcile safely within Synpo's RAM limit."
        )
    parent = np.arange(total_components + 1, dtype=np.uint32)
    rank = np.zeros(total_components + 1, dtype=np.uint8)
    seam_count = max(0, len(slab_starts) - 1)
    for seam_number, z0 in enumerate(slab_starts[1:], start=1):
        _cancel_if_requested(cancel_event)
        _merge_slab_boundary_components(
            parent,
            rank,
            np.asarray(provisional[z0 - 1], dtype=np.uint32),
            np.asarray(provisional[z0], dtype=np.uint32),
        )
        if progress:
            current = progress_offset + shape[0] // 3
            if seam_count:
                current += int(round((seam_number / seam_count) * shape[0] / 3))
            progress(
                "Detecting protein clusters (low memory)",
                current,
                progress_total,
                f"merging slab seam {seam_number}/{seam_count}",
            )

    # Compress the union forest vectorially. Union-by-rank keeps this loop short.
    while True:
        compressed = parent[parent]
        if np.array_equal(compressed, parent):
            break
        parent = compressed
    provisional_sizes = np.zeros(total_components + 1, dtype=np.int64)
    if component_sizes:
        provisional_sizes[1:] = np.concatenate(component_sizes)
    root_sizes = np.zeros(total_components + 1, dtype=np.int64)
    np.add.at(root_sizes, parent, provisional_sizes)
    selected_roots = np.flatnonzero(
        root_sizes >= settings.minimum_cluster_voxels
    )
    selected_roots = selected_roots[selected_roots > 0]
    cluster_count = len(selected_roots)
    final_for_root = np.zeros(total_components + 1, dtype=np.uint32)
    final_for_root[selected_roots] = np.arange(
        1, cluster_count + 1, dtype=np.uint32
    )
    provisional_to_final = final_for_root[parent]
    cluster_voxels = np.zeros(cluster_count + 1, dtype=np.int64)
    if cluster_count:
        cluster_voxels[1:] = root_sizes[selected_roots]
    first_z = np.full(cluster_count + 1, shape[0], dtype=np.int32)
    last_z = np.full(cluster_count + 1, -1, dtype=np.int32)
    output = _create_mask_dataset(group, "cluster_labels", shape)

    for z_index in range(shape[0]):
        _cancel_if_requested(cancel_event)
        plane = provisional_to_final[
            np.asarray(provisional[z_index], dtype=np.uint32)
        ]
        output[z_index] = plane
        present = np.unique(plane)
        present = present[present > 0]
        first_z[present] = np.minimum(first_z[present], z_index)
        last_z[present] = z_index
        if progress:
            current = progress_offset + (2 * shape[0]) // 3
            current += int(round(((z_index + 1) / shape[0]) * shape[0] / 3))
            progress(
                "Detecting protein clusters (low memory)",
                min(progress_offset + shape[0], current),
                progress_total,
                f"writing Z {z_index + 1}/{shape[0]}",
            )

    spans = last_z - first_z + 1
    flagged = [
        label_id
        for label_id in range(1, cluster_count + 1)
        if cluster_voxels[label_id] < 100 or spans[label_id] < 3
    ]
    del group["_cluster_work"]
    return cluster_count, flagged, cluster_voxels, first_z, last_z


def detect_specimen(
    manifest: dict[str, object],
    specimen_index: int,
    settings: DetectionSettings,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> DetectionSummary:
    started = time.monotonic()
    settings.validate()
    specimen = manifest["specimens"][specimen_index]
    preprocessing = specimen["checkpoints"]["preprocessing"]
    if preprocessing.get("state") != "complete":
        raise ValueError(
            f"{specimen['specimen_id']} must finish preprocessing before detection."
        )
    dendrite_channel, cluster_channel = _role_channels(manifest)
    preprocessing_root = zarr.open_group(str(project_cache_path(manifest)), mode="r")
    dendrite_key = preprocessing["channels"][dendrite_channel]["dataset_key"]
    cluster_key = preprocessing["channels"][cluster_channel]["dataset_key"]
    dendrite_data = preprocessing_root[dendrite_key]
    cluster_data = preprocessing_root[cluster_key]
    shape = tuple(int(value) for value in dendrite_data.shape)
    if tuple(cluster_data.shape) != shape:
        raise ValueError("Registered channel cache shapes differ.")
    signature = detection_signature(manifest, specimen_index, settings)
    root = zarr.open_group(str(detection_cache_path(manifest)), mode="a")
    group_key = f"specimens/{specimen_index:04d}"
    if group_key in root:
        existing = root[group_key]
        if (
            bool(existing.attrs.get("complete", False))
            and existing.attrs.get("settings_signature") == signature
        ):
            saved = existing.attrs["summary"]
            return DetectionSummary(
                specimen_index=specimen_index,
                dendrite_count=int(saved["dendrite_count"]),
                spine_count=int(saved["spine_count"]),
                cluster_count=int(saved["cluster_count"]),
                flagged_spine_count=int(saved["flagged_spine_count"]),
                flagged_cluster_count=int(saved["flagged_cluster_count"]),
                elapsed_seconds=time.monotonic() - started,
                skipped=True,
                processing_mode=str(
                    existing.attrs.get(
                        "processing_mode", saved.get("processing_mode", "standard")
                    )
                ),
            )
        del root[group_key]

    maximum_ram_fraction = min(
        0.8, float(manifest["resource_policy"].get("maximum_ram_fraction", 0.8))
    )
    requested_mode = _memory_mode(manifest)
    use_low_memory = requested_mode == ALWAYS_LOW_MEMORY_MODE or not (
        _standard_detection_fits(shape, maximum_ram_fraction)
    )
    processing_mode = "low_memory" if use_low_memory else "standard"
    if use_low_memory:
        _enforce_low_memory_projection_policy(shape, maximum_ram_fraction)
        _preflight_low_memory_disk(detection_cache_path(manifest).parent, shape)

    group = root.require_group(group_key)
    group.attrs.update(
        {
            "complete": False,
            "settings_signature": signature,
            "settings": settings.to_dict(),
            "processing_mode": processing_mode,
        }
    )
    try:
        xy = float(manifest["calibration"]["xy_um_per_pixel"])
        total_progress = shape[0] * 2 + 2
        if progress:
            detail = (
                "large stack detected; building a streamed projection"
                if use_low_memory
                else "building projection"
            )
            progress("Detecting dendrite projection", 0, total_progress, detail)
        dendrite_2d, spine_2d, projection_metadata = _segment_dendrites_and_spines(
            dendrite_data,
            settings,
            xy_um_per_pixel=xy,
            cancel_event=cancel_event,
            low_memory=use_low_memory,
            phase_callback=(
                (
                    lambda detail: progress(
                        "Detecting dendrite projection", 0, total_progress, detail
                    )
                )
                if progress
                else None
            ),
        )
        if progress:
            progress(
                "Detecting dendrite projection",
                1,
                total_progress,
                "shaft and spine candidates separated",
            )
        (
            dendrite_count,
            spine_count,
            dendrite_voxels,
            spine_voxels,
            spine_first_z,
            spine_last_z,
        ) = _write_dendrite_and_spine_volumes(
            dendrite_data,
            group,
            dendrite_2d,
            spine_2d,
            settings,
            voxel_threshold=float(
                projection_metadata["projection_applied_threshold"]
            ),
            progress=progress,
            progress_offset=1,
            progress_total=total_progress,
            cancel_event=cancel_event,
        )
        del dendrite_2d, spine_2d
        if use_low_memory:
            cluster_result = _detect_clusters_low_memory(
                cluster_data,
                group,
                settings,
                maximum_ram_fraction=maximum_ram_fraction,
                progress=progress,
                progress_offset=shape[0] + 1,
                progress_total=total_progress,
                cancel_event=cancel_event,
            )
        else:
            cluster_result = _detect_clusters(
                cluster_data,
                group,
                settings,
                progress=progress,
                progress_offset=shape[0] + 1,
                progress_total=total_progress,
                cancel_event=cancel_event,
            )
        (
            cluster_count,
            flagged_clusters,
            cluster_voxels,
            cluster_first_z,
            cluster_last_z,
        ) = cluster_result

        spine_spans = spine_last_z - spine_first_z + 1
        flagged_spines = set(projection_metadata["possible_filopodia_ids"])
        flagged_spines.update(projection_metadata["possible_dendrite_end_ids"])
        flagged_spines.update(
            label_id
            for label_id in range(1, spine_count + 1)
            if spine_voxels[label_id] < 20 or spine_spans[label_id] < 2
        )
        summary = DetectionSummary(
            specimen_index=specimen_index,
            dendrite_count=dendrite_count,
            spine_count=spine_count,
            cluster_count=cluster_count,
            flagged_spine_count=len(flagged_spines),
            flagged_cluster_count=len(flagged_clusters),
            elapsed_seconds=time.monotonic() - started,
            processing_mode=processing_mode,
        )
        group.attrs.update(
            {
                "complete": True,
                "completed_at": time.time(),
                "summary": summary.to_dict(),
                "projection_metadata": projection_metadata,
                "flagged_spine_ids": sorted(flagged_spines),
                "flagged_cluster_ids": flagged_clusters,
                "candidate_status": "unreviewed",
            }
        )
        if progress:
            progress(
                "Detection complete",
                total_progress,
                total_progress,
                specimen["specimen_id"],
            )
        return summary
    except Exception:
        # A cancelled or failed specimen always restarts from its beginning.
        if group_key in root:
            del root[group_key]
        raise


def detect_project(
    manifest: dict[str, object],
    project_path: str | Path,
    *,
    progress: ProgressCallback | None = None,
    pair_completed: Callable[[int, dict[str, object]], None] | None = None,
    cancel_event: Event | None = None,
) -> dict[str, object]:
    settings = DetectionSettings.from_dict(manifest["detection"]["settings"])
    specimens = manifest["specimens"]
    eligible = [
        index
        for index, specimen in enumerate(specimens)
        if specimen["checkpoints"]["preprocessing"].get("state") == "complete"
    ]
    if not eligible:
        raise ValueError("No specimen pairs have completed preprocessing.")
    started = time.monotonic()
    summaries: list[dict[str, object]] = []
    dendrite_channel, _ = _role_channels(manifest)
    work_units: dict[int, int] = {}
    for index in eligible:
        shape = specimens[index]["channels"][dendrite_channel]["metadata"]["shape"]
        z_count = 1 if len(shape) == 2 else int(shape[0])
        work_units[index] = z_count * 2 + 2
    total_work = sum(work_units.values())
    completed_work = 0
    completed_pairs = 0
    reused_pairs = 0
    low_memory_pairs = 0
    skipped_pairs = 0
    failed_pairs = 0
    for specimen_index in eligible:
        _cancel_if_requested(cancel_event)
        specimen = specimens[specimen_index]
        checkpoint = specimen["checkpoints"]["detection"]
        checkpoint["state"] = "in_progress"

        def specimen_progress(phase: str, current: int, total: int, detail: str) -> None:
            if progress:
                overall = completed_work + current
                progress(
                    phase,
                    overall,
                    total_work,
                    f"{specimen['specimen_id']}: {detail}",
                )

        try:
            summary = detect_specimen(
                manifest,
                specimen_index,
                settings,
                progress=specimen_progress,
                cancel_event=cancel_event,
            )
        except ProcessingCancelled:
            checkpoint.update(
                {
                    "state": "not_started",
                    "updated_at": time.time(),
                    "reason": "Detection was cancelled; this specimen will restart.",
                }
            )
            save_project(project_path, manifest)
            raise
        except InsufficientDetectionDiskSpace as exc:
            outcome = {
                "outcome": "skipped",
                "specimen_index": specimen_index,
                "reason": str(exc),
                "required_bytes": exc.required_bytes,
                "free_bytes": exc.free_bytes,
            }
            checkpoint.update(
                {
                    "state": "skipped",
                    "updated_at": time.time(),
                    "reason": str(exc),
                    "required_bytes": exc.required_bytes,
                    "free_bytes": exc.free_bytes,
                }
            )
            save_project(project_path, manifest)
            summaries.append(outcome)
            skipped_pairs += 1
            completed_work += work_units[specimen_index]
            if progress:
                progress(
                    "Detection skipped",
                    completed_work,
                    total_work,
                    f"{specimen['specimen_id']}: {exc}",
                )
            if pair_completed:
                pair_completed(specimen_index, outcome)
            continue
        except Exception as exc:
            outcome = {
                "outcome": "failed",
                "specimen_index": specimen_index,
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
            checkpoint.update(
                {
                    "state": "failed",
                    "updated_at": time.time(),
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            save_project(project_path, manifest)
            summaries.append(outcome)
            failed_pairs += 1
            completed_work += work_units[specimen_index]
            if progress:
                progress(
                    "Detection failed",
                    completed_work,
                    total_work,
                    f"{specimen['specimen_id']}: {exc}",
                )
            if pair_completed:
                pair_completed(specimen_index, outcome)
            continue
        current_signature = detection_signature(manifest, specimen_index, settings)
        review_checkpoint = specimen["checkpoints"].get("review", {})
        if (
            review_checkpoint.get("state") not in {None, "not_started"}
            and review_checkpoint.get("detection_signature") != current_signature
        ):
            specimen["review"] = {
                "state": "needs_attention",
                "comment": "",
                "history": [],
                "object_status": {"dendrite": {}, "spine": {}},
            }
            specimen["checkpoints"]["review"] = {
                "state": "not_started",
                "updated_at": time.time(),
                "edit_count": 0,
            }
        checkpoint.update(
            {
                "state": "complete",
                "updated_at": time.time(),
                "settings_signature": current_signature,
                "cache_path": str(detection_cache_path(manifest)),
                "summary": summary.to_dict(),
                "processing_mode": summary.processing_mode,
            }
        )
        for stale_key in ("reason", "required_bytes", "free_bytes", "error_type"):
            checkpoint.pop(stale_key, None)
        if not summary.skipped:
            specimen["checkpoints"].setdefault("measurements", {}).update(
                {"state": "not_started", "updated_at": time.time()}
            )
        save_project(project_path, manifest)
        outcome = summary.to_dict()
        outcome["outcome"] = "complete"
        outcome["reused"] = summary.skipped
        summaries.append(outcome)
        completed_pairs += 1
        reused_pairs += int(summary.skipped)
        low_memory_pairs += int(
            not summary.skipped and summary.processing_mode == "low_memory"
        )
        if summary.skipped and progress:
            progress(
                "Detection checkpoint",
                completed_work + work_units[specimen_index],
                total_work,
                f"{specimen['specimen_id']}: unchanged result reused",
            )
        if pair_completed:
            pair_completed(specimen_index, outcome)
        completed_work += work_units[specimen_index]
    if progress:
        progress(
            "Detection complete",
            total_work,
            total_work,
            f"{completed_pairs} complete, {skipped_pairs} skipped, {failed_pairs} failed",
        )
    return {
        "eligible_pairs": len(eligible),
        "completed_pairs": completed_pairs,
        "reused_pairs": reused_pairs,
        "low_memory_pairs": low_memory_pairs,
        "skipped_pairs": skipped_pairs,
        "failed_pairs": failed_pairs,
        "elapsed_seconds": time.monotonic() - started,
        "summaries": summaries,
        "cache_path": str(detection_cache_path(manifest)),
    }


def load_detection_slice(
    manifest: dict[str, object], specimen_index: int, z_index: int, background_channel: str
) -> DetectionSlice:
    specimen = manifest["specimens"][specimen_index]
    shape = tuple(
        int(value)
        for value in specimen["channels"][background_channel]["metadata"]["shape"]
    )
    z_count = 1 if len(shape) == 2 else shape[0]
    if not 0 <= z_index < z_count:
        raise IndexError(f"Z slice {z_index} is outside 0..{z_count - 1}.")
    source = channel_source_path(
        manifest, specimen["channels"][background_channel]
    )
    with tifffile.TiffFile(source) as tiff:
        series = tiff.series[0]
        raw = np.squeeze(
            np.asarray(series.asarray() if z_count == 1 else series.asarray(key=z_index))
        )
    root = zarr.open_group(str(detection_cache_path(manifest)), mode="r")
    group_key = f"specimens/{specimen_index:04d}"
    if group_key not in root or not bool(root[group_key].attrs.get("complete", False)):
        raise KeyError("Detection is not complete for this specimen.")
    group = root[group_key]
    return DetectionSlice(
        raw=raw,
        dendrites=np.asarray(group["dendrite_labels"][z_index]),
        spines=np.asarray(group["spine_labels"][z_index]),
        clusters=np.asarray(group["cluster_labels"][z_index]),
        z_index=z_index,
        z_count=z_count,
    )
