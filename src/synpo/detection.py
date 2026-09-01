from __future__ import annotations

import hashlib
import json
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


def _enforce_ram_policy(
    shape: tuple[int, int, int], maximum_ram_fraction: float
) -> None:
    if not 0 < maximum_ram_fraction <= 0.8:
        raise ValueError("Maximum RAM fraction must be greater than zero and no more than 0.8.")
    voxels = int(np.prod(shape))
    estimated_extra = voxels * 12 + shape[1] * shape[2] * 48 + 256 * 1024 * 1024
    limit = int(psutil.virtual_memory().total * maximum_ram_fraction)
    if psutil.Process().memory_info().rss + estimated_extra > limit:
        raise MemoryError(
            "Detection would exceed Synpo's RAM safety limit for this stack. "
            "Close other Synpo views or process on a device with more RAM."
        )


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
    phase_callback: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    _cancel_if_requested(cancel_event)
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
    maximum_ram_fraction = min(
        0.8, float(manifest["resource_policy"].get("maximum_ram_fraction", 0.8))
    )
    _enforce_ram_policy(shape, maximum_ram_fraction)

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
            )
        del root[group_key]
    group = root.require_group(group_key)
    group.attrs.update(
        {
            "complete": False,
            "settings_signature": signature,
            "settings": settings.to_dict(),
        }
    )

    xy = float(manifest["calibration"]["xy_um_per_pixel"])
    total_progress = shape[0] * 2 + 2
    if progress:
        progress("Detecting dendrite projection", 0, total_progress, "building projection")
    dendrite_2d, spine_2d, projection_metadata = _segment_dendrites_and_spines(
        dendrite_data,
        settings,
        xy_um_per_pixel=xy,
        cancel_event=cancel_event,
        phase_callback=(
            (lambda detail: progress("Detecting dendrite projection", 0, total_progress, detail))
            if progress
            else None
        ),
    )
    if progress:
        progress("Detecting dendrite projection", 1, total_progress, "shaft and spine candidates separated")
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
        voxel_threshold=float(projection_metadata["projection_applied_threshold"]),
        progress=progress,
        progress_offset=1,
        progress_total=total_progress,
        cancel_event=cancel_event,
    )
    del dendrite_2d, spine_2d
    cluster_count, flagged_clusters, cluster_voxels, cluster_first_z, cluster_last_z = (
        _detect_clusters(
            cluster_data,
            group,
            settings,
            progress=progress,
            progress_offset=shape[0] + 1,
            progress_total=total_progress,
            cancel_event=cancel_event,
        )
    )

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
        progress("Detection complete", total_progress, total_progress, specimen["specimen_id"])
    return summary


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

        summary = detect_specimen(
            manifest,
            specimen_index,
            settings,
            progress=specimen_progress,
            cancel_event=cancel_event,
        )
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
            }
        )
        if not summary.skipped:
            specimen["checkpoints"].setdefault("measurements", {}).update(
                {"state": "not_started", "updated_at": time.time()}
            )
        save_project(project_path, manifest)
        summaries.append(summary.to_dict())
        if summary.skipped and progress:
            progress(
                "Detection checkpoint",
                completed_work + work_units[specimen_index],
                total_work,
                f"{specimen['specimen_id']}: unchanged result reused",
            )
        if pair_completed:
            pair_completed(specimen_index, summary.to_dict())
        completed_work += work_units[specimen_index]
    if progress:
        progress(
            "Detection complete",
            total_work,
            total_work,
            f"{len(eligible)} specimen pair(s) checkpointed",
        )
    return {
        "eligible_pairs": len(eligible),
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
