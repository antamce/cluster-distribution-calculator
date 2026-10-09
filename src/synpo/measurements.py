from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Literal

import numpy as np
import tifffile
import zarr
from scipy import ndimage
from skimage.morphology import skeletonize

from .detection import detection_cache_path
from .distribution import calculate_spine_distribution, distribution_row
from .models import ProgressCallback
from .preprocessing import ProcessingCancelled, project_cache_path
from .project import channel_source_path, save_project
from .review import review_cache_path
from .regions import normalize_rectangles, roi_id_for_mask


ClusterEndMethod = Literal["untrimmed", "fixed", "adaptive"]
ALGORITHM_VERSION = 4


@dataclass(frozen=True)
class MeasurementSettings:
    minimum_cluster_spine_overlap_percent: float = 80.0
    cluster_end_method: ClusterEndMethod = "adaptive"
    fixed_end_slices: int = 3
    adaptive_area_factor: float = 1.8
    minimum_retained_slices: int = 2
    maximum_centerline_gap_um: float = 1.0

    def validate(self) -> None:
        if not 0 <= self.minimum_cluster_spine_overlap_percent <= 100:
            raise ValueError("Cluster/spine overlap must be between 0% and 100%.")
        if self.cluster_end_method not in {"untrimmed", "fixed", "adaptive"}:
            raise ValueError("Unknown cluster-end measurement method.")
        if not 0 <= self.fixed_end_slices <= 20:
            raise ValueError("Fixed end trimming must be between 0 and 20 slices.")
        if not 1.0 <= self.adaptive_area_factor <= 10.0:
            raise ValueError("Adaptive area factor must be between 1 and 10.")
        if not 1 <= self.minimum_retained_slices <= 20:
            raise ValueError("At least one cluster slice must be retained.")
        if not 0.0 <= self.maximum_centerline_gap_um <= 10.0:
            raise ValueError("Maximum centerline gap must be between 0 and 10 µm.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "MeasurementSettings":
        settings = cls(
            minimum_cluster_spine_overlap_percent=float(
                value.get("minimum_cluster_spine_overlap_percent", 80.0)
            ),
            cluster_end_method=str(value.get("cluster_end_method", "adaptive")),  # type: ignore[arg-type]
            fixed_end_slices=int(value.get("fixed_end_slices", 3)),
            adaptive_area_factor=float(value.get("adaptive_area_factor", 1.8)),
            minimum_retained_slices=int(value.get("minimum_retained_slices", 2)),
            maximum_centerline_gap_um=float(value.get("maximum_centerline_gap_um", 1.0)),
        )
        settings.validate()
        return settings


@dataclass(frozen=True)
class MeasurementSummary:
    specimen_index: int
    spine_count: int
    included_cluster_count: int
    dendrite_count: int
    corrected_masks: bool
    elapsed_seconds: float
    skipped: bool = False


@dataclass(frozen=True)
class ClusterTrimPreview:
    raw_projection: np.ndarray
    counted_projection: np.ndarray
    discarded_projection: np.ndarray
    cluster_id: int
    retained_z_slices: tuple[int, ...]
    discarded_z_slices: tuple[int, ...]


@dataclass(frozen=True)
class DistributionPreview:
    dendrite_projection: np.ndarray
    protein_projection: np.ndarray
    spine_bins_projection: np.ndarray
    cluster_bins_projection: np.ndarray
    axis_xy: tuple[tuple[int, int], ...]
    dendrite_stack: np.ndarray
    spine_mask_stack: np.ndarray
    axis_points_local_zyx: tuple[tuple[int, int, int], ...]
    base_point_local_zyx: tuple[int, int, int] | None
    endpoint_local_zyx: tuple[int, int, int] | None
    bridge_points_local_zyx: tuple[tuple[int, int, int], ...]
    crop_origin_yx: tuple[int, int]
    spine_z_range: tuple[int, int]
    row: dict[str, object]


@dataclass(frozen=True)
class SpineReviewPreview:
    dendrite_projection: np.ndarray
    protein_projection: np.ndarray
    spine_projection: np.ndarray
    cluster_projection: np.ndarray
    crop_origin_yx: tuple[int, int]
    spine_z_range: tuple[int, int]
    row: dict[str, object]


@dataclass(frozen=True)
class MorphologyPreview:
    dendrite_stack: np.ndarray
    protein_stack: np.ndarray
    spine_mask_stack: np.ndarray
    head_mask_stack: np.ndarray
    cluster_mask_stack: np.ndarray
    axis_points_local_zyx: tuple[tuple[int, int, int], ...]
    base_point_local_zyx: tuple[int, int, int] | None
    tip_point_local_zyx: tuple[int, int, int] | None
    crop_origin_yx: tuple[int, int]
    spine_z_range: tuple[int, int]
    row: dict[str, object]


def measurement_cache_directory(manifest: dict[str, object]) -> Path:
    return project_cache_path(manifest).parent / "measurements"


def measurement_result_path(
    manifest: dict[str, object], specimen_index: int
) -> Path:
    return measurement_cache_directory(manifest) / f"specimen-{specimen_index:04d}.json.gz"


def _cancel_if_requested(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled(
            "Measurements were cancelled. Completed specimen checkpoints remain usable."
        )


def _mask_sources(
    manifest: dict[str, object], specimen_index: int
) -> tuple[zarr.Group, zarr.Group, bool, str]:
    key = f"specimens/{specimen_index:04d}"
    detection_root = zarr.open_group(str(detection_cache_path(manifest)), mode="r")
    if key not in detection_root or not bool(
        detection_root[key].attrs.get("complete", False)
    ):
        raise ValueError("Automatic detection is not complete for this specimen.")
    detection = detection_root[key]
    signature = str(detection.attrs.get("settings_signature", ""))
    if review_cache_path(manifest).exists():
        review_root = zarr.open_group(str(review_cache_path(manifest)), mode="r")
        if key in review_root:
            review = review_root[key]
            if bool(review.attrs.get("initialized", False)) and str(
                review.attrs.get("detection_signature", "")
            ) == signature:
                edit_count = int(
                    manifest["specimens"][specimen_index]["checkpoints"]["review"].get(
                        "edit_count", 0
                    )
                )
                return review, detection, True, f"{signature}:review:{edit_count}"
    return detection, detection, False, f"{signature}:automatic"


def _distribution_guidance(
    manifest: dict[str, object],
    specimen_index: int,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray | None:
    dendrite_channel = next(
        channel
        for channel, role in manifest["channel_roles"].items()
        if role == "dendrite_spines"
    )
    try:
        key = manifest["specimens"][specimen_index]["checkpoints"]["preprocessing"][
            "channels"
        ][dendrite_channel]["dataset_key"]
        root = zarr.open_group(str(project_cache_path(manifest)), mode="r")
        return np.asarray(root[key][:, y_slice, x_slice], dtype=np.float32)
    except (KeyError, OSError, ValueError):
        return None


def measurement_signature(
    manifest: dict[str, object], specimen_index: int, settings: MeasurementSettings
) -> str:
    _editable, _detection, _corrected, mask_signature = _mask_sources(
        manifest, specimen_index
    )
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "settings": settings.to_dict(),
        "mask_signature": mask_signature,
        "calibration": manifest["calibration"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cluster_keep_lookup(
    areas_by_z: np.ndarray, settings: MeasurementSettings
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    z_count, cluster_slots = areas_by_z.shape
    keep = areas_by_z > 0
    details: dict[int, dict[str, object]] = {}
    for cluster_id in range(1, cluster_slots):
        occupied = np.flatnonzero(areas_by_z[:, cluster_id] > 0)
        if not len(occupied):
            continue
        areas = areas_by_z[occupied, cluster_id]
        first_score = float(np.mean(areas[: min(2, len(areas))]))
        last_score = float(np.mean(areas[-min(2, len(areas)) :]))
        trim_from = "first" if first_score >= last_score else "last"
        discarded: list[int] = []
        maximum_trim = max(0, len(occupied) - settings.minimum_retained_slices)
        if settings.cluster_end_method == "fixed":
            trim_count = min(settings.fixed_end_slices, maximum_trim)
            if trim_count:
                removed = (
                    occupied[:trim_count]
                    if trim_from == "first"
                    else occupied[-trim_count:]
                )
                discarded = [int(value) for value in removed]
        elif settings.cluster_end_method == "adaptive" and maximum_trim:
            ordered = occupied if trim_from == "first" else occupied[::-1]
            stable_areas = np.sort(areas)[: max(1, len(areas) // 2)]
            stable_reference = max(1.0, float(np.median(stable_areas)))
            for z_index in ordered[:maximum_trim]:
                if areas_by_z[z_index, cluster_id] <= (
                    stable_reference * settings.adaptive_area_factor
                ):
                    break
                discarded.append(int(z_index))
        if discarded:
            keep[np.asarray(discarded, dtype=np.int32), cluster_id] = False
        details[cluster_id] = {
            "visible_z_slices": [int(value) for value in occupied],
            "slice_areas_voxels": [int(value) for value in areas],
            "larger_terminal_end": trim_from,
            "discarded_z_slices": discarded,
            "retained_z_slices": [
                int(value) for value in occupied if int(value) not in set(discarded)
            ],
        }
    return keep, details


def _dendrite_lengths(
    labels: np.ndarray, xy_um_per_pixel: float
) -> dict[int, float]:
    skeleton = skeletonize(labels > 0)
    skeleton_labels = np.where(skeleton, labels, 0).astype(np.uint32, copy=False)
    maximum = int(labels.max())
    lengths = np.zeros(maximum + 1, dtype=np.float64)
    for dy, dx, distance in (
        (0, 1, xy_um_per_pixel),
        (1, 0, xy_um_per_pixel),
        (1, 1, xy_um_per_pixel * np.sqrt(2.0)),
        (1, -1, xy_um_per_pixel * np.sqrt(2.0)),
    ):
        if dx >= 0:
            first = skeleton_labels[: labels.shape[0] - dy or None, : labels.shape[1] - dx or None]
            second = skeleton_labels[dy:, dx:]
        else:
            first = skeleton_labels[: labels.shape[0] - dy or None, -dx:]
            second = skeleton_labels[dy:, :dx]
        connected = (first > 0) & (first == second)
        if np.any(connected):
            lengths += np.bincount(
                first[connected], minlength=maximum + 1
            ) * distance
    isolated = np.flatnonzero(
        (np.bincount(skeleton_labels.ravel(), minlength=maximum + 1) > 0)
        & (lengths == 0)
    )
    lengths[isolated] = xy_um_per_pixel
    return {label_id: float(lengths[label_id]) for label_id in range(1, maximum + 1)}


def _assign_spines_to_dendrites(
    spine_projection: np.ndarray, dendrite_projection: np.ndarray
) -> dict[int, int]:
    assignments: dict[int, int] = {}
    maximum_spine = int(spine_projection.max())
    objects = ndimage.find_objects(spine_projection)
    nearest_labels: np.ndarray | None = None
    for spine_id in range(1, maximum_spine + 1):
        bounds = objects[spine_id - 1] if spine_id - 1 < len(objects) else None
        if bounds is None:
            continue
        expanded = tuple(
            slice(max(0, item.start - 3), min(limit, item.stop + 3))
            for item, limit in zip(bounds, spine_projection.shape)
        )
        local_spine = spine_projection[expanded] == spine_id
        contacts = dendrite_projection[expanded][ndimage.binary_dilation(local_spine, iterations=2)]
        contacts = contacts[contacts > 0]
        if len(contacts):
            counts = np.bincount(contacts)
            assignments[spine_id] = int(np.argmax(counts[1:]) + 1)
            continue
        if nearest_labels is None and np.any(dendrite_projection > 0):
            _distance, indices = ndimage.distance_transform_edt(
                dendrite_projection == 0, return_indices=True
            )
            nearest_labels = dendrite_projection[tuple(indices)]
        spine_pixels = spine_projection == spine_id
        candidates = (
            nearest_labels[spine_pixels] if nearest_labels is not None else np.empty(0)
        )
        candidates = candidates[candidates > 0]
        assignments[spine_id] = (
            int(np.argmax(np.bincount(candidates)[1:]) + 1) if len(candidates) else 0
        )
    return assignments


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(result, stream, separators=(",", ":"))
    os.replace(temporary, path)


def load_measurement_result(
    manifest: dict[str, object], specimen_index: int
) -> dict[str, object]:
    path = measurement_result_path(manifest, specimen_index)
    if not path.is_file():
        raise ValueError(
            "The saved measurement cache is missing at "
            f"{path}. The project JSON contains checkpoints, not the measurement "
            "tables; restore the project's .synpo-cache folder or import a full "
            "transfer ZIP."
        )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def recover_compatible_measurement_checkpoints(
    manifest: dict[str, object],
) -> list[int]:
    """Restore lost checkpoint metadata only for exact compatible cached results."""
    settings = MeasurementSettings.from_dict(manifest["measurements"]["settings"])
    recovered: list[int] = []
    for specimen_index, specimen in enumerate(manifest.get("specimens", [])):
        result_path = measurement_result_path(manifest, specimen_index)
        if not result_path.is_file():
            continue
        try:
            result = load_measurement_result(manifest, specimen_index)
            current_signature = measurement_signature(
                manifest, specimen_index, settings
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if int(result.get("algorithm_version", 0) or 0) != ALGORITHM_VERSION:
            continue
        if dict(result.get("settings", {})) != settings.to_dict():
            continue
        if str(result.get("settings_signature", "")) != current_signature:
            continue
        checkpoint = specimen.setdefault("checkpoints", {}).setdefault(
            "measurements", {}
        )
        if (
            checkpoint.get("state") == "complete"
            and str(checkpoint.get("settings_signature", "")) == current_signature
        ):
            continue
        specimen_rows = list(result.get("specimen_rows", []))
        if not specimen_rows:
            continue
        specimen_row = specimen_rows[0]
        timestamp = time.time()
        checkpoint.update(
            {
                "state": "complete",
                "updated_at": checkpoint.get("updated_at") or timestamp,
                "settings_signature": current_signature,
                "summary": {
                    "specimen_index": specimen_index,
                    "spine_count": int(specimen_row.get("spine_count") or 0),
                    "included_cluster_count": int(
                        specimen_row.get("included_cluster_count") or 0
                    ),
                    "dendrite_count": int(specimen_row.get("dendrite_count") or 0),
                    "corrected_masks": bool(result.get("corrected_masks", False)),
                    "elapsed_seconds": 0.0,
                    "skipped": True,
                },
                "recovered_at": timestamp,
                "reason": (
                    "Recovered an exact compatible saved measurement result "
                    "while opening the project."
                ),
            }
        )
        recovered.append(specimen_index)
    return recovered


def spine_volume_filter_settings(
    manifest: dict[str, object],
) -> tuple[bool, float]:
    """Return the saved post-measurement spine-volume filter settings."""
    settings = manifest.get("measurements", {}).get("settings", {})
    enabled = bool(settings.get("spine_volume_filter_enabled", False))
    cutoff = float(settings.get("spine_volume_filter_cutoff_um3", 0.0) or 0.0)
    return enabled, max(0.0, cutoff)


def specimen_volume_filter_overrides(
    manifest: dict[str, object], specimen_index: int
) -> set[int]:
    decisions = (
        manifest["specimens"][specimen_index]
        .get("distribution_review", {})
        .get("spines", {})
    )
    return {
        int(spine_id)
        for spine_id, decision in decisions.items()
        if isinstance(decision, dict)
        and bool(decision.get("volume_filter_force_keep", False))
    }


def apply_spine_volume_filter(
    result: dict[str, object],
    *,
    enabled: bool,
    cutoff_um3: float,
    force_keep_ids: set[int] | None = None,
    refresh_summaries: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Apply a reversible inclusion rule to a copy of a measurement result.

    The measured result on disk is never changed.  Strictly smaller volumes are
    excluded; a force-keep decision only overrides this volume rule and can
    never restore a manually invalid spine.
    """
    filtered = deepcopy(result)
    cutoff = max(0.0, float(cutoff_um3))
    force_keep = {int(value) for value in (force_keep_ids or set())}
    state: dict[int, tuple[bool, bool, bool]] = {}
    audit: list[dict[str, object]] = []
    for row in filtered.get("spine_rows", []):
        spine_id = int(row.get("spine_id") or 0)
        manual_valid = bool(row.get("manual_spine_valid", row.get("spine_valid", True)))
        keep = spine_id in force_keep
        below = float(row.get("volume_um3") or 0.0) < cutoff
        excluded = bool(enabled and manual_valid and below and not keep)
        row.update(
            {
                "manual_spine_valid": manual_valid,
                "volume_filter_enabled": bool(enabled),
                "volume_filter_cutoff_um3": cutoff,
                "volume_filter_below_cutoff": below,
                "volume_filter_force_keep": keep,
                "volume_filter_override_reason": (
                    "User selected Keep despite volume cutoff" if keep else ""
                ),
                "volume_filter_excluded": excluded,
                "spine_valid": manual_valid and not excluded,
            }
        )
        state[spine_id] = (manual_valid, keep, excluded)
        if excluded:
            audit.append(
                {
                    **row,
                    "exclusion_reason": "volume strictly below cutoff",
                    "volume_filter_override_applied": False,
                    "override_reason": "No Keep override selected",
                }
            )
    for collection_name in ("cluster_rows", "distribution_rows"):
        for row in filtered.get(collection_name, []):
            spine_id = int(row.get("spine_id") or 0)
            manual_valid, keep, excluded = state.get(
                spine_id,
                (bool(row.get("spine_valid", True)), False, False),
            )
            row.update(
                {
                    "manual_spine_valid": manual_valid,
                    "volume_filter_force_keep": keep,
                    "volume_filter_excluded": excluded,
                    "spine_valid": manual_valid and not excluded,
                }
            )
    if refresh_summaries:
        _refresh_result_summaries(filtered)
    return filtered, audit


def filtered_measurement_result(
    manifest: dict[str, object], specimen_index: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    enabled, cutoff = spine_volume_filter_settings(manifest)
    return apply_spine_volume_filter(
        load_measurement_result(manifest, specimen_index),
        enabled=enabled,
        cutoff_um3=cutoff,
        force_keep_ids=specimen_volume_filter_overrides(manifest, specimen_index),
    )


def _profile(values: list[dict[str, object]]) -> list[float | None]:
    return [
        _mean(
            [
                float(row[f"bin_{index:02d}_ratio"])
                for row in values
                if row.get(f"bin_{index:02d}_ratio") is not None
            ]
        )
        for index in range(1, 11)
    ]


def _refresh_result_summaries(result: dict[str, object]) -> None:
    spine_rows = list(result.get("spine_rows", []))
    cluster_rows = list(result.get("cluster_rows", []))
    distribution_rows = list(result.get("distribution_rows", []))
    validity = {
        int(row["spine_id"]): bool(row.get("spine_valid", True))
        for row in spine_rows
    }
    for row in cluster_rows:
        row["spine_valid"] = validity.get(int(row.get("spine_id") or 0), True)
    valid_spines = [row for row in spine_rows if bool(row.get("spine_valid", True))]
    manually_invalid = [
        row
        for row in spine_rows
        if not bool(row.get("manual_spine_valid", row.get("spine_valid", True)))
    ]
    volume_filtered = [
        row for row in spine_rows if bool(row.get("volume_filter_excluded", False))
    ]
    valid_clusters = [
        row
        for row in cluster_rows
        if row.get("row_type") == "individual_cluster"
        and bool(row.get("spine_valid", True))
    ]
    dendrite_rows = list(result.get("dendrite_rows", []))
    for dendrite in dendrite_rows:
        dendrite_id = int(dendrite["dendrite_id"])
        spines = [row for row in valid_spines if int(row["dendrite_id"]) == dendrite_id]
        clusters = [row for row in valid_clusters if int(row["dendrite_id"]) == dendrite_id]
        length = float(dendrite.get("length_um") or 0.0)
        dendrite.update(
            {
                "spine_count": len(spines),
                "spine_density_per_um": len(spines) / length if length else None,
                "average_spine_volume_um3": _mean([float(row["volume_um3"]) for row in spines]),
                "spines_with_clusters_percent": (
                    100.0 * sum(bool(row["has_protein_cluster"]) for row in spines) / len(spines)
                    if spines
                    else None
                ),
                "average_cluster_to_spine_volume_ratio": _mean(
                    [
                        float(row["cluster_to_spine_volume_ratio"])
                        for row in spines
                        if row.get("cluster_to_spine_volume_ratio") is not None
                    ]
                ),
                "average_cluster_volume_um3": _mean(
                    [float(row["volume_inside_spine_um3"]) for row in clusters]
                ),
            }
        )
        distribution = [
            row
            for row in distribution_rows
            if int(row["dendrite_id"]) == dendrite_id
            and bool(row.get("spine_valid", True))
            and bool(row.get("distribution_included", False))
        ]
        dendrite["average_protein_distribution"] = _profile(distribution)

    if not result.get("specimen_rows"):
        return
    specimen = result["specimen_rows"][0]
    included_distribution = [
        row
        for row in distribution_rows
        if bool(row.get("spine_valid", True))
        and bool(row.get("distribution_included", False))
    ]
    total_dendrite_length = sum(float(row.get("length_um") or 0.0) for row in dendrite_rows)
    specimen.update(
        {
            "spine_count": len(valid_spines),
            "included_cluster_count": len(valid_clusters),
            "average_spine_density_per_um": (
                len(valid_spines) / total_dendrite_length
                if total_dendrite_length
                else None
            ),
            "average_spine_volume_um3": _mean([float(row["volume_um3"]) for row in valid_spines]),
            "spines_with_clusters_percent": (
                100.0
                * sum(bool(row["has_protein_cluster"]) for row in valid_spines)
                / len(valid_spines)
                if valid_spines
                else None
            ),
            "average_cluster_to_spine_volume_ratio": _mean(
                [
                    float(row["cluster_to_spine_volume_ratio"])
                    for row in valid_spines
                    if row.get("cluster_to_spine_volume_ratio") is not None
                ]
            ),
            "average_cluster_volume_um3": _mean(
                [float(row["volume_inside_spine_um3"]) for row in valid_clusters]
            ),
            "average_protein_distribution": _profile(included_distribution),
            "invalid_spine_count": len(manually_invalid),
            "volume_filtered_spine_count": len(volume_filtered),
            "distribution_included_spine_count": len(included_distribution),
        }
    )
    for roi in result.get("roi_rows", []):
        roi_id = int(roi["roi_id"])
        roi_dendrites = [row for row in dendrite_rows if int(row.get("roi_id", 0)) == roi_id]
        roi_spines = [row for row in valid_spines if int(row.get("roi_id", 0)) == roi_id]
        roi_clusters = [row for row in valid_clusters if int(row.get("roi_id", 0)) == roi_id]
        roi_distributions = [
            row
            for row in distribution_rows
            if int(row.get("roi_id", 0)) == roi_id
            and bool(row.get("spine_valid", True))
            and bool(row.get("distribution_included", False))
        ]
        roi_length = sum(float(row.get("length_um") or 0.0) for row in roi_dendrites)
        roi.update(
            {
                "dendrite_count": len(roi_dendrites),
                "dendrite_length_um": roi_length,
                "spine_count": len(roi_spines),
                "included_cluster_count": len(roi_clusters),
                "spine_density_per_um": len(roi_spines) / roi_length if roi_length else None,
                "average_spine_volume_um3": _mean(
                    [float(row["volume_um3"]) for row in roi_spines]
                ),
                "spines_with_clusters_percent": (
                    100.0 * sum(bool(row["has_protein_cluster"]) for row in roi_spines) / len(roi_spines)
                    if roi_spines else None
                ),
                "average_cluster_to_spine_volume_ratio": _mean(
                    [
                        float(row["cluster_to_spine_volume_ratio"])
                        for row in roi_spines
                        if row.get("cluster_to_spine_volume_ratio") is not None
                    ]
                ),
                "average_cluster_volume_um3": _mean(
                    [float(row["volume_inside_spine_um3"]) for row in roi_clusters]
                ),
                "average_protein_distribution": _profile(roi_distributions),
            }
        )


def set_distribution_review(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    spine_id: int,
    *,
    distribution_included: bool,
    invalid_spine: bool,
    note: str = "",
) -> dict[str, object]:
    """Checkpoint a distribution/validity decision and refresh affected metrics."""
    specimen = manifest["specimens"][specimen_index]
    reviews = specimen.setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {})
    decision = reviews.setdefault(str(spine_id), {})
    decision.update(
        {
            "reviewed": True,
            "distribution_reviewed": True,
            "validity_reviewed": True,
            "review_kind": "cluster_positive",
            "distribution_included": bool(distribution_included and not invalid_spine),
            "invalid_spine": bool(invalid_spine),
            "note": str(note).strip(),
            "updated_at": time.time(),
        }
    )
    specimen["distribution_review"]["updated_at"] = decision["updated_at"]
    result = load_measurement_result(manifest, specimen_index)
    found = False
    for row in result.get("distribution_rows", []):
        if int(row["spine_id"]) == spine_id:
            row.update(
                {
                    "distribution_reviewed": True,
                    "distribution_included": decision["distribution_included"],
                    "spine_valid": not invalid_spine,
                    "review_note": decision["note"],
                }
            )
            found = True
    for row in result.get("spine_rows", []):
        if int(row["spine_id"]) == spine_id:
            row["spine_valid"] = not invalid_spine
            row["validity_note"] = decision["note"] if invalid_spine else ""
            row["validity_reviewed"] = True
            row["review_kind"] = "cluster_positive"
    if not found:
        raise ValueError("This spine has no cluster-positive distribution row.")
    _refresh_result_summaries(result)
    _write_result(measurement_result_path(manifest, specimen_index), result)
    specimen["checkpoints"].setdefault("measurements", {})["review_updated_at"] = time.time()
    save_project(project_path, manifest)
    return result


def distribution_profile_available(row: dict[str, object]) -> bool:
    """Return whether a distribution row contains any calculated profile data."""
    return any(
        row.get(f"bin_{index:02d}_ratio") is not None
        for index in range(1, 11)
    )


def accept_all_eligible_distribution_spines(
    manifest: dict[str, object],
    project_path: str | Path,
) -> dict[str, int]:
    """Accept every valid, retained spine that has a usable distribution path.

    Volume filtering remains reversible: eligibility is evaluated on filtered
    copies, while only review decisions are written back to cached measurement
    results.  Existing invalid-spine decisions and review notes are preserved.
    """
    counts = {
        "measured_specimen_count": 0,
        "eligible_count": 0,
        "accepted_count": 0,
        "already_reviewed_count": 0,
        "already_accepted_count": 0,
        "invalid_count": 0,
        "volume_filtered_count": 0,
        "unusable_path_count": 0,
    }
    changed = False
    for specimen_index, specimen in enumerate(manifest.get("specimens", [])):
        checkpoint = specimen.get("checkpoints", {}).get("measurements", {})
        if checkpoint.get("state") != "complete":
            continue
        counts["measured_specimen_count"] += 1
        result = load_measurement_result(manifest, specimen_index)
        filtered, _audit = filtered_measurement_result(manifest, specimen_index)
        filtered_rows = {
            int(row.get("spine_id") or 0): row
            for row in filtered.get("distribution_rows", [])
        }
        raw_rows = {
            int(row.get("spine_id") or 0): row
            for row in result.get("distribution_rows", [])
        }
        eligible_ids: set[int] = set()
        for spine_id, row in filtered_rows.items():
            if bool(row.get("volume_filter_excluded", False)):
                counts["volume_filtered_count"] += 1
                continue
            if not bool(row.get("spine_valid", True)):
                counts["invalid_count"] += 1
                continue
            if not distribution_profile_available(row):
                counts["unusable_path_count"] += 1
                continue
            counts["eligible_count"] += 1
            raw = raw_rows.get(spine_id)
            if raw is not None and bool(raw.get("distribution_reviewed", False)):
                counts["already_reviewed_count"] += 1
                if bool(raw.get("distribution_included", False)):
                    counts["already_accepted_count"] += 1
                continue
            eligible_ids.add(spine_id)

        if not eligible_ids:
            continue
        timestamp = time.time()
        reviews = specimen.setdefault(
            "distribution_review", {"spines": {}, "updated_at": None}
        ).setdefault("spines", {})
        for spine_id in eligible_ids:
            decision = reviews.setdefault(str(spine_id), {})
            decision.update(
                {
                    "reviewed": True,
                    "distribution_reviewed": True,
                    "validity_reviewed": True,
                    "review_kind": "cluster_positive",
                    "distribution_included": True,
                    "invalid_spine": False,
                    "updated_at": timestamp,
                }
            )
            decision.setdefault(
                "note", str(raw_rows.get(spine_id, {}).get("review_note", ""))
            )
            raw = raw_rows.get(spine_id)
            if raw is not None:
                raw.update(
                    {
                        "distribution_reviewed": True,
                        "distribution_included": True,
                        "spine_valid": True,
                        "review_note": str(decision.get("note", "")),
                    }
                )
        for row in result.get("spine_rows", []):
            if int(row.get("spine_id") or 0) in eligible_ids:
                row.update(
                    {
                        "spine_valid": True,
                        "validity_reviewed": True,
                        "review_kind": "cluster_positive",
                    }
                )
        _refresh_result_summaries(result)
        _write_result(measurement_result_path(manifest, specimen_index), result)
        specimen["distribution_review"]["updated_at"] = timestamp
        checkpoint["review_updated_at"] = timestamp
        counts["accepted_count"] += len(eligible_ids)
        changed = True

    if changed:
        save_project(project_path, manifest)
    return counts


def set_spine_quality_review(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    spine_id: int,
    *,
    invalid_spine: bool,
    note: str = "",
) -> dict[str, object]:
    """Checkpoint a validity decision for a spine without a distribution row."""
    specimen = manifest["specimens"][specimen_index]
    reviews = specimen.setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {})
    decision = reviews.setdefault(str(spine_id), {})
    decision.update(
        {
            "validity_reviewed": True,
            "review_kind": "cluster_less",
            "invalid_spine": bool(invalid_spine),
            "note": str(note).strip(),
            "updated_at": time.time(),
        }
    )
    specimen["distribution_review"]["updated_at"] = decision["updated_at"]
    result = load_measurement_result(manifest, specimen_index)
    found = False
    for row in result.get("spine_rows", []):
        if int(row["spine_id"]) == spine_id:
            if bool(row.get("has_protein_cluster", False)):
                raise ValueError(
                    "This spine is now cluster-positive; review it in the protein-positive queue."
                )
            row.update(
                {
                    "spine_valid": not invalid_spine,
                    "validity_reviewed": True,
                    "validity_note": decision["note"] if invalid_spine else decision["note"],
                    "review_kind": "cluster_less",
                }
            )
            found = True
            break
    if not found:
        raise ValueError("This spine is no longer present in the measurement result.")
    _refresh_result_summaries(result)
    _write_result(measurement_result_path(manifest, specimen_index), result)
    specimen["checkpoints"].setdefault("measurements", {})[
        "review_updated_at"
    ] = time.time()
    save_project(project_path, manifest)
    return result


def _recalculate_distribution_spine(
    manifest: dict[str, object],
    specimen_index: int,
    spine_id: int,
    result: dict[str, object],
) -> dict[str, object]:
    row = next(
        (item for item in result.get("distribution_rows", []) if int(item["spine_id"]) == spine_id),
        None,
    )
    if row is None:
        raise ValueError("This spine has no cluster-positive distribution row.")
    geometry = result.get("distribution_geometry", {}).get(str(spine_id))
    if geometry is None:
        raise ValueError("Saved distribution geometry is unavailable.")
    editable, detection, _corrected, _signature = _mask_sources(manifest, specimen_index)
    shape = tuple(int(value) for value in editable["spine_labels"].shape)
    bounds = geometry["bounds_zyx"]
    y_slice = slice(max(0, int(bounds[1][0])), min(shape[1], int(bounds[1][1])))
    x_slice = slice(max(0, int(bounds[2][0])), min(shape[2], int(bounds[2][1])))
    spine_labels = np.asarray(editable["spine_labels"][:, y_slice, x_slice], dtype=np.uint32)
    spine = spine_labels == spine_id
    if not np.any(spine):
        raise ValueError("The reviewed spine is no longer present in its saved region.")
    parent_id = int(row.get("dendrite_id") or 0)
    dendrites = np.asarray(editable["dendrite_labels"][:, y_slice, x_slice], dtype=np.uint32)
    parent = dendrites == parent_id if parent_id else dendrites > 0
    cluster_labels = np.asarray(detection["cluster_labels"][:, y_slice, x_slice], dtype=np.uint32)
    clusters = np.zeros(spine.shape, dtype=bool)
    included_ids = {
        int(item["cluster_id"])
        for item in result.get("cluster_rows", [])
        if item.get("row_type") == "individual_cluster"
        and int(item.get("spine_id") or 0) == spine_id
    }
    trim_details = result.get("cluster_trim_details", {})
    for cluster_id in included_ids:
        discarded = {
            int(value)
            for value in trim_details.get(str(cluster_id), {}).get("discarded_z_slices", [])
        }
        for z_index in range(shape[0]):
            if z_index not in discarded:
                clusters[z_index] |= (cluster_labels[z_index] == cluster_id) & spine[z_index]

    specimen = manifest["specimens"][specimen_index]
    decision = specimen.setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {}).setdefault(str(spine_id), {})
    hint_value = decision.get("centerline_endpoint_hint_zyx")
    hint = (
        tuple(int(value) for value in hint_value)
        if isinstance(hint_value, (list, tuple)) and len(hint_value) == 3
        else None
    )
    hint_valid = False
    if hint is not None:
        local_hint = (hint[0], hint[1] - y_slice.start, hint[2] - x_slice.start)
        hint_valid = (
            0 <= local_hint[0] < spine.shape[0]
            and 0 <= local_hint[1] < spine.shape[1]
            and 0 <= local_hint[2] < spine.shape[2]
            and bool(spine[local_hint])
        )
    if hint is not None and not hint_valid:
        history = decision.setdefault("centerline_hint_history", [])
        if not history or history[-1].get("action") != "invalidated_by_resegmentation":
            history.append(
                {
                    "action": "invalidated_by_resegmentation",
                    "point_zyx": list(hint),
                    "updated_at": time.time(),
                }
            )
        decision["reviewed"] = False
        decision["distribution_reviewed"] = False
    decision["centerline_endpoint_hint_valid"] = hint_valid
    xy_size = float(manifest["calibration"]["xy_um_per_pixel"])
    z_step = float(manifest["calibration"]["z_step_um"])
    calculated = calculate_spine_distribution(
        spine,
        parent,
        clusters,
        sampling_zyx_um=(z_step, xy_size, xy_size),
        global_offset_zyx=(0, y_slice.start, x_slice.start),
        endpoint_hint_zyx=hint if hint_valid else None,
        guidance_image=_distribution_guidance(
            manifest, specimen_index, y_slice, x_slice
        ),
        maximum_gap_um=float(
            manifest["measurements"]["settings"].get(
                "maximum_centerline_gap_um", 1.0
            )
        ),
    )
    recalculated = distribution_row(
        calculated,
        experimental_group=str(specimen["experimental_group"]),
        specimen_id=str(specimen["specimen_id"]),
        dendrite_id=parent_id,
        spine_id=spine_id,
        voxel_volume_um3=float(result["voxel_volume_um3"]),
    )
    recalculated.update(
        {
            "roi_id": row.get("roi_id", 0),
            "distribution_reviewed": bool(
                decision.get(
                    "distribution_reviewed",
                    decision.get("reviewed", row.get("distribution_reviewed", False)),
                )
            ),
            "distribution_included": bool(row.get("distribution_included", False)),
            "spine_valid": bool(row.get("spine_valid", True)),
            "review_note": str(decision.get("note", row.get("review_note", ""))),
            "centerline_endpoint_hint_valid": hint_valid,
            "centerline_endpoint_hint_present": hint is not None,
            "centerline_hint_history": list(decision.get("centerline_hint_history", [])),
        }
    )
    row.clear()
    row.update(recalculated)
    geometry.update(
        {
            "axis_points_zyx": [list(point) for point in calculated.axis_points_zyx],
            "base_point_zyx": list(calculated.base_point_zyx) if calculated.base_point_zyx else None,
            "endpoint_zyx": list(calculated.endpoint_zyx) if calculated.endpoint_zyx else None,
            "bridge_points_zyx": [list(point) for point in calculated.bridge_points_zyx],
            "bridge_length_um": calculated.bridge_length_um,
        }
    )
    for spine_row in result.get("spine_rows", []):
        if int(spine_row["spine_id"]) == spine_id:
            spine_row["protein_distribution_in_spine"] = [
                row.get(f"bin_{index:02d}_ratio") for index in range(1, 11)
            ]
    _refresh_result_summaries(result)
    _write_result(measurement_result_path(manifest, specimen_index), result)
    specimen["distribution_review"]["updated_at"] = time.time()
    specimen["checkpoints"].setdefault("measurements", {})["review_updated_at"] = time.time()
    return result


def set_centerline_endpoint_hint(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    spine_id: int,
    point_zyx: tuple[int, int, int],
) -> dict[str, object]:
    editable, _detection, _corrected, _signature = _mask_sources(manifest, specimen_index)
    shape = tuple(int(value) for value in editable["spine_labels"].shape)
    if any(value < 0 or value >= shape[axis] for axis, value in enumerate(point_zyx)):
        raise ValueError("The endpoint lies outside the stack.")
    if int(editable["spine_labels"][point_zyx]) != spine_id:
        raise ValueError("The endpoint must lie on the selected spine.")
    specimen = manifest["specimens"][specimen_index]
    decision = specimen.setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {}).setdefault(str(spine_id), {})
    previous = decision.get("centerline_endpoint_hint_zyx")
    action = "replaced" if previous is not None else "placed"
    decision["centerline_endpoint_hint_zyx"] = list(point_zyx)
    decision["centerline_endpoint_hint_valid"] = True
    decision.setdefault("centerline_hint_history", []).append(
        {
            "action": action,
            "point_zyx": list(point_zyx),
            "previous_point_zyx": previous,
            "updated_at": time.time(),
        }
    )
    result = _recalculate_distribution_spine(
        manifest, specimen_index, spine_id, load_measurement_result(manifest, specimen_index)
    )
    save_project(project_path, manifest)
    return result


def clear_centerline_endpoint_hint(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    spine_id: int,
) -> dict[str, object]:
    specimen = manifest["specimens"][specimen_index]
    decision = specimen.setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {}).setdefault(str(spine_id), {})
    previous = decision.pop("centerline_endpoint_hint_zyx", None)
    decision["centerline_endpoint_hint_valid"] = False
    decision.setdefault("centerline_hint_history", []).append(
        {
            "action": "cleared",
            "previous_point_zyx": previous,
            "updated_at": time.time(),
        }
    )
    result = _recalculate_distribution_spine(
        manifest, specimen_index, spine_id, load_measurement_result(manifest, specimen_index)
    )
    save_project(project_path, manifest)
    return result


def distribution_summary_rows(
    results: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    specimen_rows: list[dict[str, object]] = []
    for result in results:
        identity = result.get("specimen_rows", [{}])[0]
        included = [
            row
            for row in result.get("distribution_rows", [])
            if bool(row.get("spine_valid", True))
            and bool(row.get("distribution_included", False))
        ]
        if not included:
            continue
        row: dict[str, object] = {
            "experimental_group": identity.get("experimental_group", ""),
            "specimen_id": identity.get("specimen_id", ""),
            "included_spine_count": len(included),
        }
        for index in range(1, 11):
            values = [
                float(item[f"bin_{index:02d}_ratio"])
                for item in included
                if item.get(f"bin_{index:02d}_ratio") is not None
            ]
            row[f"bin_{index:02d}_mean"] = _mean(values)
            row[f"bin_{index:02d}_n"] = len(values)
        specimen_rows.append(row)

    group_rows: list[dict[str, object]] = []
    groups = sorted({str(row["experimental_group"]) for row in specimen_rows})
    for group in groups:
        members = [row for row in specimen_rows if row["experimental_group"] == group]
        row = {
            "experimental_group": group,
            "specimen_count": len(members),
            "included_spine_count": sum(int(item["included_spine_count"]) for item in members),
        }
        for index in range(1, 11):
            values = [
                float(item[f"bin_{index:02d}_mean"])
                for item in members
                if item.get(f"bin_{index:02d}_mean") is not None
            ]
            mean = _mean(values)
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
            row[f"bin_{index:02d}_mean"] = mean
            row[f"bin_{index:02d}_sd"] = sd
            row[f"bin_{index:02d}_sem"] = sd / np.sqrt(len(values)) if sd is not None and values else None
            row[f"bin_{index:02d}_n"] = len(values)
        group_rows.append(row)
    return specimen_rows, group_rows


def spine_volume_distribution_rows(
    results: list[dict[str, object]],
    *,
    filter_enabled: bool,
    cutoff_um3: float,
) -> list[dict[str, object]]:
    """Build pooled and specimen-weighted spine-volume distributions."""
    spines = [
        row
        for result in results
        for row in result.get("spine_rows", [])
        if bool(row.get("spine_valid", True))
    ]
    values = np.asarray([float(row.get("volume_um3") or 0.0) for row in spines])
    method = "no valid spines"
    if values.size == 0:
        edges = np.asarray([0.0, 1.0], dtype=np.float64)
    elif float(np.min(values)) == float(np.max(values)):
        center = float(values[0])
        padding = max(abs(center) * 0.01, 1e-9)
        edges = np.asarray([center - padding, center + padding], dtype=np.float64)
        method = "single bin (constant data)"
    else:
        q1, q3 = np.percentile(values, [25.0, 75.0])
        width = 2.0 * float(q3 - q1) * float(values.size) ** (-1.0 / 3.0)
        if values.size >= 4 and np.isfinite(width) and width > 0:
            bin_count = int(np.ceil((float(np.max(values)) - float(np.min(values))) / width))
            bin_count = min(100, max(1, bin_count))
            method = "Freedman-Diaconis (shared edges; capped at 100 bins)"
        else:
            bin_count = min(100, max(1, int(np.ceil(np.log2(values.size) + 1.0))))
            method = "Sturges fallback (small sample or zero IQR)"
        edges = np.linspace(float(np.min(values)), float(np.max(values)), bin_count + 1)

    output: list[dict[str, object]] = [
        {
            "row_type": "metadata",
            "filter_enabled": bool(filter_enabled),
            "cutoff_um3": float(cutoff_um3),
            "cutoff_rule": "exclude volume_um3 < cutoff_um3; equality is retained",
            "binning_method": method,
            "bin_edges_um3": [float(value) for value in edges],
            "valid_spine_count": len(spines),
        }
    ]
    if not spines:
        return output

    groups = sorted({str(row.get("experimental_group", "")) for row in spines})
    scopes = [("overall", "All", spines)] + [
        (
            "group",
            group,
            [row for row in spines if str(row.get("experimental_group", "")) == group],
        )
        for group in groups
    ]
    for scope_type, scope_name, members in scopes:
        scope_values = np.asarray([float(row.get("volume_um3") or 0.0) for row in members])
        output.append(
            {
                "row_type": "descriptive",
                "scope_type": scope_type,
                "scope": scope_name,
                "n_spines": len(members),
                "mean_um3": float(np.mean(scope_values)),
                "sd_um3": float(np.std(scope_values, ddof=1)) if len(members) > 1 else 0.0,
                "median_um3": float(np.median(scope_values)),
                "minimum_um3": float(np.min(scope_values)),
                "q1_um3": float(np.percentile(scope_values, 25.0)),
                "q3_um3": float(np.percentile(scope_values, 75.0)),
                "maximum_um3": float(np.max(scope_values)),
            }
        )
        specimen_keys = sorted(
            {
                (str(row.get("experimental_group", "")), str(row.get("specimen_id", "")))
                for row in members
            }
        )
        counts, _ = np.histogram(scope_values, bins=edges)
        specimen_percentages: list[np.ndarray] = []
        for group, specimen in specimen_keys:
            specimen_values = np.asarray(
                [
                    float(row.get("volume_um3") or 0.0)
                    for row in members
                    if str(row.get("experimental_group", "")) == group
                    and str(row.get("specimen_id", "")) == specimen
                ]
            )
            specimen_counts, _ = np.histogram(specimen_values, bins=edges)
            specimen_percentages.append(100.0 * specimen_counts / len(specimen_values))
        matrix = np.asarray(specimen_percentages, dtype=np.float64)
        for bin_index, count in enumerate(counts):
            percentages = matrix[:, bin_index]
            sd = float(np.std(percentages, ddof=1)) if len(percentages) > 1 else 0.0
            output.append(
                {
                    "row_type": "histogram_bin",
                    "scope_type": scope_type,
                    "scope": scope_name,
                    "bin_index": bin_index + 1,
                    "bin_left_um3": float(edges[bin_index]),
                    "bin_right_um3": float(edges[bin_index + 1]),
                    "right_edge_inclusive": bin_index == len(counts) - 1,
                    "pooled_spine_count": int(count),
                    "pooled_percent": 100.0 * float(count) / len(members),
                    "specimen_percentage_mean": float(np.mean(percentages)),
                    "specimen_percentage_sd": sd,
                    "specimen_percentage_sem": sd / np.sqrt(len(percentages)),
                    "n_specimens": len(percentages),
                }
            )
    return output


def cluster_end_comparison_rows(
    result: dict[str, object]
) -> list[dict[str, object]]:
    base = MeasurementSettings.from_dict(result["settings"])
    voxel_volume = float(result["voxel_volume_um3"])
    rows: list[dict[str, object]] = []
    for cluster_key, details in result.get("cluster_trim_details", {}).items():
        visible_z = [int(value) for value in details["visible_z_slices"]]
        areas = [int(value) for value in details["slice_areas_voxels"]]
        if not visible_z:
            continue
        matrix = np.zeros((max(visible_z) + 1, 2), dtype=np.int64)
        matrix[visible_z, 1] = areas
        fixed_settings = MeasurementSettings(
            minimum_cluster_spine_overlap_percent=base.minimum_cluster_spine_overlap_percent,
            cluster_end_method="fixed",
            fixed_end_slices=base.fixed_end_slices,
            adaptive_area_factor=base.adaptive_area_factor,
            minimum_retained_slices=base.minimum_retained_slices,
        )
        adaptive_settings = MeasurementSettings(
            minimum_cluster_spine_overlap_percent=base.minimum_cluster_spine_overlap_percent,
            cluster_end_method="adaptive",
            fixed_end_slices=base.fixed_end_slices,
            adaptive_area_factor=base.adaptive_area_factor,
            minimum_retained_slices=base.minimum_retained_slices,
        )
        fixed_keep, fixed_details = _cluster_keep_lookup(matrix, fixed_settings)
        adaptive_keep, adaptive_details = _cluster_keep_lookup(
            matrix, adaptive_settings
        )
        untrimmed_voxels = int(sum(areas))
        fixed_voxels = int(
            sum(matrix[z_index, 1] for z_index in visible_z if fixed_keep[z_index, 1])
        )
        adaptive_voxels = int(
            sum(
                matrix[z_index, 1]
                for z_index in visible_z
                if adaptive_keep[z_index, 1]
            )
        )
        rows.append(
            {
                "cluster_id": int(cluster_key),
                "untrimmed_candidate_volume_um3": untrimmed_voxels * voxel_volume,
                "fixed_candidate_volume_um3": fixed_voxels * voxel_volume,
                "fixed_discarded_z_slices": fixed_details[1]["discarded_z_slices"],
                "adaptive_candidate_volume_um3": adaptive_voxels * voxel_volume,
                "adaptive_discarded_z_slices": adaptive_details[1][
                    "discarded_z_slices"
                ],
            }
        )
    return rows


def load_cluster_trim_preview(
    manifest: dict[str, object],
    specimen_index: int,
    cluster_id: int,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> ClusterTrimPreview:
    result = load_measurement_result(manifest, specimen_index)
    details = result.get("cluster_trim_details", {}).get(str(cluster_id))
    if details is None:
        raise ValueError("The selected cluster is not present in this result.")
    _editable, detection, _corrected, _signature = _mask_sources(
        manifest, specimen_index
    )
    clusters = detection["cluster_labels"]
    z_count, y_count, x_count = (int(value) for value in clusters.shape)
    raw_projection = np.zeros((y_count, x_count), dtype=np.uint16)
    counted = np.zeros((y_count, x_count), dtype=np.uint32)
    discarded = np.zeros((y_count, x_count), dtype=np.uint32)
    discarded_z = {int(value) for value in details.get("discarded_z_slices", [])}
    role_channels = {role: channel for channel, role in manifest["channel_roles"].items()}
    protein_channel = role_channels["protein_clusters"]
    specimen = manifest["specimens"][specimen_index]
    source = channel_source_path(manifest, specimen["channels"][protein_channel])
    with tifffile.TiffFile(source) as tiff:
        series = tiff.series[0]
        for z_index in range(z_count):
            _cancel_if_requested(cancel_event)
            raw = np.squeeze(
                np.asarray(
                    series.asarray()
                    if z_count == 1
                    else series.asarray(key=z_index)
                )
            ).astype(np.uint16, copy=False)
            np.maximum(raw_projection, raw, out=raw_projection)
            mask = np.asarray(clusters[z_index]) == cluster_id
            target = discarded if z_index in discarded_z else counted
            target[mask] = cluster_id
            if progress:
                progress(
                    "Building cluster-end illustration",
                    z_index + 1,
                    z_count,
                    f"Cluster {cluster_id}: Z {z_index + 1}/{z_count}",
                )
    return ClusterTrimPreview(
        raw_projection=raw_projection,
        counted_projection=counted,
        discarded_projection=discarded,
        cluster_id=cluster_id,
        retained_z_slices=tuple(int(value) for value in details["retained_z_slices"]),
        discarded_z_slices=tuple(int(value) for value in details["discarded_z_slices"]),
    )


def load_distribution_preview(
    manifest: dict[str, object],
    specimen_index: int,
    spine_id: int,
    *,
    margin_um: float = 1.0,
) -> DistributionPreview:
    result = load_measurement_result(manifest, specimen_index)
    row = next(
        (item for item in result.get("distribution_rows", []) if int(item["spine_id"]) == spine_id),
        None,
    )
    if row is None:
        raise ValueError("This spine has no cluster-positive distribution result.")
    geometry = result.get("distribution_geometry", {}).get(str(spine_id))
    if geometry is None:
        raise ValueError("Saved distribution geometry is unavailable.")
    xy_size = float(manifest["calibration"]["xy_um_per_pixel"])
    z_step = float(manifest["calibration"]["z_step_um"])
    bounds = geometry["bounds_zyx"]
    _editable, detection, _corrected, _signature = _mask_sources(manifest, specimen_index)
    shape = tuple(int(value) for value in detection["spine_labels"].shape)
    margin_pixels = max(0, int(np.ceil(margin_um / xy_size)))
    y_slice = slice(max(0, int(bounds[1][0]) - margin_pixels), min(shape[1], int(bounds[1][1]) + margin_pixels))
    x_slice = slice(max(0, int(bounds[2][0]) - margin_pixels), min(shape[2], int(bounds[2][1]) + margin_pixels))
    spine_labels = np.asarray(detection["spine_labels"][:, y_slice, x_slice], dtype=np.uint32)
    # Use corrected masks if available, matching the measurement source.
    editable, _detection, _corrected, _signature = _mask_sources(manifest, specimen_index)
    spine_labels = np.asarray(editable["spine_labels"][:, y_slice, x_slice], dtype=np.uint32)
    parent_id = int(row.get("dendrite_id") or 0)
    dendrite_labels = np.asarray(editable["dendrite_labels"][:, y_slice, x_slice], dtype=np.uint32)
    cluster_labels = np.asarray(detection["cluster_labels"][:, y_slice, x_slice], dtype=np.uint32)
    spine = spine_labels == spine_id
    parent = dendrite_labels == parent_id if parent_id else dendrite_labels > 0
    included_ids = {
        int(item["cluster_id"])
        for item in result.get("cluster_rows", [])
        if item.get("row_type") == "individual_cluster"
        and int(item.get("spine_id") or 0) == spine_id
    }
    clusters = np.zeros(spine.shape, dtype=bool)
    details = result.get("cluster_trim_details", {})
    for cluster_id in included_ids:
        discarded = {int(value) for value in details.get(str(cluster_id), {}).get("discarded_z_slices", [])}
        for z_index in range(shape[0]):
            if z_index not in discarded:
                clusters[z_index] |= (cluster_labels[z_index] == cluster_id) & spine[z_index]
    decision = (
        manifest["specimens"][specimen_index]
        .get("distribution_review", {})
        .get("spines", {})
        .get(str(spine_id), {})
    )
    hint_value = decision.get("centerline_endpoint_hint_zyx")
    endpoint_hint = (
        tuple(int(value) for value in hint_value)
        if bool(decision.get("centerline_endpoint_hint_valid", False))
        and isinstance(hint_value, (list, tuple))
        and len(hint_value) == 3
        else None
    )
    calculated = calculate_spine_distribution(
        spine,
        parent,
        clusters,
        sampling_zyx_um=(z_step, xy_size, xy_size),
        global_offset_zyx=(0, y_slice.start, x_slice.start),
        endpoint_hint_zyx=endpoint_hint,
        guidance_image=_distribution_guidance(
            manifest, specimen_index, y_slice, x_slice
        ),
        maximum_gap_um=float(
            manifest["measurements"]["settings"].get(
                "maximum_centerline_gap_um", 1.0
            )
        ),
    )
    if calculated.voxel_bins is None:
        spine_bins = np.zeros(spine.shape[1:], dtype=np.uint8)
        cluster_bins = np.zeros(spine.shape[1:], dtype=np.uint8)
    else:
        spine_bins = np.max(np.where(spine, calculated.voxel_bins + 1, 0), axis=0).astype(np.uint8)
        cluster_bins = np.max(np.where(clusters, calculated.voxel_bins + 1, 0), axis=0).astype(np.uint8)

    role_channels = {role: channel for channel, role in manifest["channel_roles"].items()}
    specimen = manifest["specimens"][specimen_index]
    projections: dict[str, np.ndarray] = {}
    dendrite_stack = np.zeros(spine.shape, dtype=np.uint16)
    for role in ("dendrite_spines", "protein_clusters"):
        channel = role_channels[role]
        source = channel_source_path(manifest, specimen["channels"][channel])
        projection = np.zeros(spine.shape[1:], dtype=np.uint16)
        try:
            stack = np.squeeze(tifffile.memmap(source))
        except ValueError:
            # Compressed TIFFs cannot be mapped directly; tifffile decodes them
            # into a temporary disk-backed array instead of consuming stack RAM.
            stack = np.squeeze(tifffile.imread(source, out="memmap"))
        if stack.ndim == 2:
            cropped_stack = np.asarray(stack[y_slice, x_slice], dtype=np.uint16)[None]
            projection = cropped_stack[0]
        else:
            cropped_stack = np.asarray(stack[:, y_slice, x_slice], dtype=np.uint16)
            projection = np.max(cropped_stack, axis=0).astype(np.uint16, copy=False)
        if role == "dendrite_spines":
            dendrite_stack = cropped_stack
        projections[role] = projection
    axis_xy = tuple(
        (int(point[2]) - x_slice.start, int(point[1]) - y_slice.start)
        for point in calculated.axis_points_zyx
    )
    axis_local = tuple(
        (int(point[0]), int(point[1]) - y_slice.start, int(point[2]) - x_slice.start)
        for point in calculated.axis_points_zyx
    )
    base_local = (
        (
            calculated.base_point_zyx[0],
            calculated.base_point_zyx[1] - y_slice.start,
            calculated.base_point_zyx[2] - x_slice.start,
        )
        if calculated.base_point_zyx
        else None
    )
    endpoint_local = (
        (
            calculated.endpoint_zyx[0],
            calculated.endpoint_zyx[1] - y_slice.start,
            calculated.endpoint_zyx[2] - x_slice.start,
        )
        if calculated.endpoint_zyx
        else None
    )
    bridge_local = tuple(
        (
            int(point[0]),
            int(point[1]) - y_slice.start,
            int(point[2]) - x_slice.start,
        )
        for point in calculated.bridge_points_zyx
    )
    occupied_z = np.flatnonzero(np.any(spine, axis=(1, 2)))
    z_range = (
        (int(occupied_z[0]), int(occupied_z[-1]))
        if len(occupied_z)
        else (0, max(0, spine.shape[0] - 1))
    )
    return DistributionPreview(
        dendrite_projection=projections["dendrite_spines"],
        protein_projection=projections["protein_clusters"],
        spine_bins_projection=spine_bins,
        cluster_bins_projection=cluster_bins,
        axis_xy=axis_xy,
        dendrite_stack=dendrite_stack,
        spine_mask_stack=spine,
        axis_points_local_zyx=axis_local,
        base_point_local_zyx=base_local,
        endpoint_local_zyx=endpoint_local,
        bridge_points_local_zyx=bridge_local,
        crop_origin_yx=(y_slice.start, x_slice.start),
        spine_z_range=z_range,
        row=dict(row),
    )


def load_spine_review_preview(
    manifest: dict[str, object],
    specimen_index: int,
    spine_id: int,
    *,
    margin_um: float = 1.0,
) -> SpineReviewPreview:
    """Load a compact two-channel crop for validity review of any spine."""
    result = load_measurement_result(manifest, specimen_index)
    row = next(
        (item for item in result.get("spine_rows", []) if int(item["spine_id"]) == spine_id),
        None,
    )
    if row is None:
        raise ValueError("This spine is no longer present in the measurement result.")
    editable, detection, _corrected, _signature = _mask_sources(
        manifest, specimen_index
    )
    labels = editable["spine_labels"]
    z_count, y_count, x_count = (int(value) for value in labels.shape)
    z_min, z_max = z_count, -1
    y_min, y_max = y_count, -1
    x_min, x_max = x_count, -1
    for z_index in range(z_count):
        yy, xx = np.nonzero(np.asarray(labels[z_index]) == spine_id)
        if not len(xx):
            continue
        z_min = min(z_min, z_index)
        z_max = max(z_max, z_index)
        y_min = min(y_min, int(yy.min()))
        y_max = max(y_max, int(yy.max()))
        x_min = min(x_min, int(xx.min()))
        x_max = max(x_max, int(xx.max()))
    if z_max < 0:
        raise ValueError("The selected spine is no longer present in the saved masks.")
    xy_size = float(manifest["calibration"]["xy_um_per_pixel"])
    margin = max(0, int(np.ceil(margin_um / xy_size)))
    y_slice = slice(max(0, y_min - margin), min(y_count, y_max + margin + 1))
    x_slice = slice(max(0, x_min - margin), min(x_count, x_max + margin + 1))
    spine_stack = np.asarray(labels[:, y_slice, x_slice], dtype=np.uint32)
    cluster_stack = np.asarray(
        detection["cluster_labels"][:, y_slice, x_slice], dtype=np.uint32
    )

    role_channels = {role: channel for channel, role in manifest["channel_roles"].items()}
    specimen = manifest["specimens"][specimen_index]
    projections: dict[str, np.ndarray] = {}
    for role in ("dendrite_spines", "protein_clusters"):
        channel = role_channels[role]
        source = channel_source_path(manifest, specimen["channels"][channel])
        try:
            stack = np.squeeze(tifffile.memmap(source))
        except ValueError:
            stack = np.squeeze(tifffile.imread(source, out="memmap"))
        cropped = (
            np.asarray(stack[y_slice, x_slice], dtype=np.uint16)[None]
            if stack.ndim == 2
            else np.asarray(stack[:, y_slice, x_slice], dtype=np.uint16)
        )
        projections[role] = np.max(cropped, axis=0).astype(np.uint16, copy=False)
    return SpineReviewPreview(
        dendrite_projection=projections["dendrite_spines"],
        protein_projection=projections["protein_clusters"],
        spine_projection=np.max(
            np.where(spine_stack == spine_id, spine_id, 0), axis=0
        ).astype(np.uint32),
        cluster_projection=np.max(cluster_stack, axis=0).astype(np.uint32),
        crop_origin_yx=(y_slice.start, x_slice.start),
        spine_z_range=(z_min, z_max),
        row=dict(row),
    )


def _binary_surface_area_um2(
    mask: np.ndarray, sampling_zyx_um: tuple[float, float, float]
) -> float:
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    z_step, y_step, x_step = sampling_zyx_um
    return float(
        np.count_nonzero(np.diff(padded, axis=0)) * y_step * x_step
        + np.count_nonzero(np.diff(padded, axis=1)) * z_step * x_step
        + np.count_nonzero(np.diff(padded, axis=2)) * z_step * y_step
    )


def _symmetric_eigenvalues_3x3(matrix: np.ndarray) -> tuple[float, float, float]:
    """Eigenvalues of a real symmetric 3x3 matrix without a LAPACK dependency."""
    value = np.asarray(matrix, dtype=np.float64)
    q = float(np.trace(value) / 3.0)
    centered = value - np.eye(3) * q
    p2 = float(
        centered[0, 0] ** 2
        + centered[1, 1] ** 2
        + centered[2, 2] ** 2
        + 2.0 * (centered[0, 1] ** 2 + centered[0, 2] ** 2 + centered[1, 2] ** 2)
    )
    if p2 <= 0:
        return q, q, q
    p = float(np.sqrt(p2 / 6.0))
    b = centered / p
    determinant = float(
        b[0, 0] * (b[1, 1] * b[2, 2] - b[1, 2] * b[2, 1])
        - b[0, 1] * (b[1, 0] * b[2, 2] - b[1, 2] * b[2, 0])
        + b[0, 2] * (b[1, 0] * b[2, 1] - b[1, 1] * b[2, 0])
    )
    angle = float(np.arccos(np.clip(determinant / 2.0, -1.0, 1.0)) / 3.0)
    largest = q + 2.0 * p * float(np.cos(angle))
    smallest = q + 2.0 * p * float(np.cos(angle + 2.0 * np.pi / 3.0))
    middle = 3.0 * q - largest - smallest
    return tuple(sorted((smallest, middle, largest)))


def _spine_morphology_metrics(
    calculated,
    spine: np.ndarray,
    *,
    sampling_zyx_um: tuple[float, float, float],
    global_offset_zyx: tuple[int, int, int],
    voxel_volume_um3: float,
    decision: dict[str, object],
) -> tuple[dict[str, object], np.ndarray]:
    """Measure reproducible morphology without changing the segmentation mask."""
    mask = np.asarray(spine, dtype=bool)
    empty_head = np.zeros(mask.shape, dtype=bool)
    axis = np.asarray(calculated.axis_points_zyx, dtype=np.int64)
    if len(axis) < 2:
        return (
            {
                "spine_curvilinear_length_um": None,
                "spine_base_to_tip_distance_um": None,
                "centerline_tortuosity": None,
                "maximum_width_um": None,
                "head_maximum_width_um": None,
                "neck_minimum_width_um": None,
                "neck_median_width_um": None,
                "head_to_neck_width_ratio": None,
                "head_volume_um3": None,
                "neck_volume_um3": None,
                "surface_area_um2": _binary_surface_area_um2(mask, sampling_zyx_um),
                "sphericity": None,
                "principal_axis_elongation": None,
                "head_neck_border_path_fraction": None,
                "head_neck_split_status": "no_usable_path",
                "spine_length_status": str(calculated.axis_status),
                "spine_length_uses_virtual_bridge": bool(calculated.bridge_used),
                "centerline_bridge_length_um": float(calculated.bridge_length_um),
                "centerline_base_zyx": None,
                "centerline_tip_zyx": None,
                "centerline_base_source": str(calculated.base_source),
                "centerline_tip_source": str(calculated.endpoint_source),
                "geometry_reviewed": bool(decision.get("reviewed", False)),
                "geometry_review_note": str(decision.get("note", "")),
            },
            empty_head,
        )

    sampling = np.asarray(sampling_zyx_um, dtype=np.float64)
    offset = np.asarray(global_offset_zyx, dtype=np.int64)
    local_axis = axis - offset
    physical_axis = axis.astype(np.float64) * sampling
    steps = np.linalg.norm(np.diff(physical_axis, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(steps)]
    length = float(cumulative[-1])
    straight = float(np.linalg.norm(physical_axis[-1] - physical_axis[0]))
    distance = ndimage.distance_transform_edt(mask, sampling=sampling_zyx_um)
    axis_radii = distance[tuple(local_axis.T)]
    maximum_width = 2.0 * float(np.max(distance[mask])) if np.any(mask) else None

    border_index = max(1, min(len(axis_radii) - 1, int(round(len(axis_radii) * 0.65))))
    split_status = "automatic_low_contrast"
    neck_radius: float | None = None
    head_radius: float | None = None
    if len(axis_radii) >= 4:
        distal_start = min(len(axis_radii) - 2, max(1, int(len(axis_radii) * 0.35)))
        head_peak = distal_start + int(np.argmax(axis_radii[distal_start:]))
        neck_start = max(1, int(len(axis_radii) * 0.05))
        if head_peak > neck_start:
            neck_index = neck_start + int(np.argmin(axis_radii[neck_start:head_peak]))
            neck_radius = float(axis_radii[neck_index])
            head_radius = float(axis_radii[head_peak])
            threshold = neck_radius + (head_radius - neck_radius) * 0.5
            crossings = np.flatnonzero(axis_radii[neck_index : head_peak + 1] >= threshold)
            if len(crossings):
                border_index = neck_index + int(crossings[0])
            split_status = (
                "automatic" if head_radius >= max(1e-12, neck_radius) * 1.10
                else "automatic_no_distinct_neck"
            )

    path_volume = np.zeros(mask.shape, dtype=bool)
    path_volume[tuple(local_axis.T)] = True
    _distance_to_path, nearest = ndimage.distance_transform_edt(
        ~path_volume, sampling=sampling_zyx_um, return_indices=True
    )
    path_indices = np.full(mask.shape, -1, dtype=np.int32)
    path_indices[tuple(local_axis.T)] = np.arange(len(local_axis), dtype=np.int32)
    nearest_index = path_indices[tuple(nearest)]
    head_mask = mask & (nearest_index >= border_index)

    for key, value in (("head_override_voxels_zyx", True), ("neck_override_voxels_zyx", False)):
        for point in decision.get(key, []):
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                continue
            local = tuple(int(point[axis_index]) - int(offset[axis_index]) for axis_index in range(3))
            if all(0 <= local[axis_index] < mask.shape[axis_index] for axis_index in range(3)) and mask[local]:
                head_mask[local] = value
    if decision.get("head_override_voxels_zyx") or decision.get("neck_override_voxels_zyx"):
        split_status = "manual"
    neck_mask = mask & ~head_mask
    surface_area = _binary_surface_area_um2(mask, sampling_zyx_um)
    volume = float(np.count_nonzero(mask)) * voxel_volume_um3
    sphericity = (
        float(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface_area)
        if volume > 0 and surface_area > 0
        else None
    )
    coordinates = np.argwhere(mask).astype(np.float64) * sampling
    elongation = None
    if len(coordinates) >= 3:
        centered_coordinates = coordinates - np.mean(coordinates, axis=0)
        covariance = np.asarray(
            [
                [
                    float(np.sum(centered_coordinates[:, row] * centered_coordinates[:, column]))
                    / max(1, len(coordinates) - 1)
                    for column in range(3)
                ]
                for row in range(3)
            ]
        )
        eigenvalues = _symmetric_eigenvalues_3x3(covariance)
        if eigenvalues[-1] > 0:
            elongation = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-12)))
    head_width = 2.0 * float(np.max(distance[head_mask])) if np.any(head_mask) else None
    neck_axis_radii = axis_radii[: max(1, border_index)]
    neck_minimum = 2.0 * float(np.min(neck_axis_radii)) if len(neck_axis_radii) else None
    neck_median = 2.0 * float(np.median(neck_axis_radii)) if len(neck_axis_radii) else None
    return (
        {
            "spine_curvilinear_length_um": length,
            "spine_base_to_tip_distance_um": straight,
            "centerline_tortuosity": length / straight if straight > 0 else None,
            "maximum_width_um": maximum_width,
            "head_maximum_width_um": head_width,
            "neck_minimum_width_um": neck_minimum,
            "neck_median_width_um": neck_median,
            "head_to_neck_width_ratio": (
                head_width / neck_median
                if head_width is not None and neck_median not in (None, 0.0)
                else None
            ),
            "head_volume_um3": float(np.count_nonzero(head_mask)) * voxel_volume_um3,
            "neck_volume_um3": float(np.count_nonzero(neck_mask)) * voxel_volume_um3,
            "surface_area_um2": surface_area,
            "sphericity": sphericity,
            "principal_axis_elongation": elongation,
            "head_neck_border_path_fraction": float(cumulative[border_index] / length),
            "head_neck_split_status": split_status,
            "spine_length_status": str(calculated.axis_status),
            "spine_length_uses_virtual_bridge": bool(calculated.bridge_used),
            "centerline_bridge_length_um": float(calculated.bridge_length_um),
            "centerline_base_zyx": list(calculated.base_point_zyx) if calculated.base_point_zyx else None,
            "centerline_tip_zyx": list(calculated.endpoint_zyx) if calculated.endpoint_zyx else None,
            "centerline_base_source": str(calculated.base_source),
            "centerline_tip_source": str(calculated.endpoint_source),
            "geometry_reviewed": bool(decision.get("reviewed", False)),
            "geometry_review_note": str(decision.get("note", "")),
        },
        head_mask,
    )


def _calculate_morphology_spine(
    manifest: dict[str, object],
    specimen_index: int,
    spine_id: int,
    result: dict[str, object],
):  # type: ignore[no-untyped-def]
    geometry = result.get("morphology_geometry", result.get("distribution_geometry", {})).get(
        str(spine_id)
    )
    if geometry is None:
        raise ValueError("Saved all-spine morphology geometry is unavailable; recalculate measurements.")
    bounds = geometry["bounds_zyx"]
    y_slice = slice(int(bounds[1][0]), int(bounds[1][1]))
    x_slice = slice(int(bounds[2][0]), int(bounds[2][1]))
    editable, detection, _corrected, _signature = _mask_sources(manifest, specimen_index)
    spine_labels = np.asarray(editable["spine_labels"][:, y_slice, x_slice], dtype=np.uint32)
    spine = spine_labels == spine_id
    if not np.any(spine):
        raise ValueError("The selected spine is no longer present in its saved region.")
    spine_row = next(
        (row for row in result.get("spine_rows", []) if int(row["spine_id"]) == spine_id),
        None,
    )
    if spine_row is None:
        raise ValueError("The selected spine has no saved measurement row.")
    parent_id = int(spine_row.get("dendrite_id") or 0)
    dendrites = np.asarray(editable["dendrite_labels"][:, y_slice, x_slice], dtype=np.uint32)
    parent = dendrites == parent_id if parent_id else dendrites > 0
    cluster_labels = np.asarray(detection["cluster_labels"][:, y_slice, x_slice], dtype=np.uint32)
    clusters = np.zeros(spine.shape, dtype=bool)
    included_ids = {
        int(row["cluster_id"])
        for row in result.get("cluster_rows", [])
        if row.get("row_type") == "individual_cluster"
        and int(row.get("spine_id") or 0) == spine_id
    }
    trim_details = result.get("cluster_trim_details", {})
    for cluster_id in included_ids:
        discarded = {
            int(value)
            for value in trim_details.get(str(cluster_id), {}).get("discarded_z_slices", [])
        }
        for z_index in range(spine.shape[0]):
            if z_index not in discarded:
                clusters[z_index] |= (cluster_labels[z_index] == cluster_id) & spine[z_index]
    specimen = manifest["specimens"][specimen_index]
    decision = specimen.setdefault(
        "morphology_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {}).setdefault(str(spine_id), {})
    base_value = decision.get("centerline_base_hint_zyx")
    tip_value = decision.get("centerline_endpoint_hint_zyx")
    base_hint = tuple(int(value) for value in base_value) if isinstance(base_value, (list, tuple)) and len(base_value) == 3 else None
    tip_hint = tuple(int(value) for value in tip_value) if isinstance(tip_value, (list, tuple)) and len(tip_value) == 3 else None
    xy = float(manifest["calibration"]["xy_um_per_pixel"])
    z_step = float(manifest["calibration"]["z_step_um"])
    calculated = calculate_spine_distribution(
        spine,
        parent,
        clusters,
        sampling_zyx_um=(z_step, xy, xy),
        global_offset_zyx=(0, y_slice.start, x_slice.start),
        endpoint_hint_zyx=tip_hint,
        base_hint_zyx=base_hint,
        guidance_image=_distribution_guidance(manifest, specimen_index, y_slice, x_slice),
        maximum_gap_um=float(
            manifest["measurements"]["settings"].get("maximum_centerline_gap_um", 1.0)
        ),
    )
    morphology, head_mask = _spine_morphology_metrics(
        calculated,
        spine,
        sampling_zyx_um=(z_step, xy, xy),
        global_offset_zyx=(0, y_slice.start, x_slice.start),
        voxel_volume_um3=float(result["voxel_volume_um3"]),
        decision=decision,
    )
    return calculated, morphology, head_mask, spine, clusters, y_slice, x_slice, parent_id


def _checkpoint_morphology_spine(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    spine_id: int,
) -> dict[str, object]:
    result = load_measurement_result(manifest, specimen_index)
    calculated, morphology, _head, _spine, _clusters, y_slice, x_slice, parent_id = (
        _calculate_morphology_spine(manifest, specimen_index, spine_id, result)
    )
    identity = {
        "experimental_group": manifest["specimens"][specimen_index]["experimental_group"],
        "specimen_id": manifest["specimens"][specimen_index]["specimen_id"],
        "roi_id": next(
            int(row.get("roi_id") or 0)
            for row in result.get("spine_rows", [])
            if int(row["spine_id"]) == spine_id
        ),
        "dendrite_id": parent_id,
        "spine_id": spine_id,
    }
    morphology_row = {**identity, **morphology}
    rows = result.setdefault("morphology_rows", [])
    existing = next((row for row in rows if int(row["spine_id"]) == spine_id), None)
    if existing is None:
        rows.append(morphology_row)
    else:
        existing.clear()
        existing.update(morphology_row)
    spine_row = next(row for row in result["spine_rows"] if int(row["spine_id"]) == spine_id)
    for key in (
        "spine_curvilinear_length_um", "spine_base_to_tip_distance_um",
        "spine_length_status", "spine_length_uses_virtual_bridge",
        "centerline_bridge_length_um", "centerline_base_zyx", "centerline_tip_zyx",
        "centerline_base_source", "centerline_tip_source", "geometry_reviewed",
    ):
        spine_row[key] = morphology.get(key)
    validity_decision = manifest["specimens"][specimen_index].setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {}).get(str(spine_id), {})
    spine_row["spine_valid"] = not bool(validity_decision.get("invalid_spine", False))
    spine_row["validity_reviewed"] = bool(
        validity_decision.get("validity_reviewed", validity_decision.get("reviewed", False))
    )
    spine_row["validity_note"] = str(validity_decision.get("note", ""))
    distribution = next(
        (row for row in result.get("distribution_rows", []) if int(row["spine_id"]) == spine_id),
        None,
    )
    if distribution is not None:
        refreshed = distribution_row(
            calculated,
            experimental_group=str(identity["experimental_group"]),
            specimen_id=str(identity["specimen_id"]),
            dendrite_id=parent_id,
            spine_id=spine_id,
            voxel_volume_um3=float(result["voxel_volume_um3"]),
        )
        preserved = {
            key: value
            for key, value in distribution.items()
            if key in {
                "distribution_reviewed", "distribution_included",
                "distribution_review_required", "roi_id", "spine_valid",
                "review_note", "centerline_endpoint_hint_valid",
                "centerline_endpoint_hint_present", "centerline_hint_history",
            }
        }
        distribution.clear()
        distribution.update(refreshed)
        distribution.update(preserved)
        distribution["spine_valid"] = bool(spine_row["spine_valid"])
    geometry = {
        "bounds_zyx": [[0, int(_spine.shape[0])], [y_slice.start, y_slice.stop], [x_slice.start, x_slice.stop]],
        "axis_points_zyx": [list(point) for point in calculated.axis_points_zyx],
        "base_point_zyx": list(calculated.base_point_zyx) if calculated.base_point_zyx else None,
        "endpoint_zyx": list(calculated.endpoint_zyx) if calculated.endpoint_zyx else None,
        "bridge_points_zyx": [list(point) for point in calculated.bridge_points_zyx],
        "bridge_length_um": calculated.bridge_length_um,
    }
    result.setdefault("morphology_geometry", {})[str(spine_id)] = geometry
    result.setdefault("distribution_geometry", {})[str(spine_id)] = geometry
    _refresh_result_summaries(result)
    _write_result(measurement_result_path(manifest, specimen_index), result)
    checkpoint = manifest["specimens"][specimen_index]["checkpoints"].setdefault("measurements", {})
    checkpoint["morphology_review_updated_at"] = time.time()
    for run in manifest.get("morphology_analysis", {}).get("runs", []):
        run["stale"] = True
    save_project(project_path, manifest)
    return result


def apply_morphology_review_edit(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    spine_id: int,
    *,
    operation: str,
    point_zyx: tuple[int, int, int] | None = None,
    strokes_xy: tuple[tuple[tuple[int, int], ...], ...] = (),
    maximum_projection: bool = True,
    z_index: int = 0,
    z_radius: int = 0,
    brush_radius: int = 2,
    reviewed: bool | None = None,
    note: str | None = None,
    invalid_spine: bool | None = None,
) -> dict[str, object]:
    specimen = manifest["specimens"][specimen_index]
    review = specimen.setdefault("morphology_review", {"spines": {}, "updated_at": None})
    decision = review.setdefault("spines", {}).setdefault(str(spine_id), {})
    history = decision.setdefault("history", [])
    history.append({
        "snapshot": {
            key: value
            for key, value in decision.items()
            if key not in {"history", "redo_history"}
        },
        "updated_at": time.time(),
    })
    decision.pop("redo_history", None)
    if operation in {"set_base", "set_tip"}:
        if point_zyx is None:
            raise ValueError("Select a spine voxel for the centerline anchor.")
        key = "centerline_base_hint_zyx" if operation == "set_base" else "centerline_endpoint_hint_zyx"
        decision[key] = [int(value) for value in point_zyx]
    elif operation in {"paint_head", "paint_neck"}:
        result = load_measurement_result(manifest, specimen_index)
        geometry = result.get("morphology_geometry", {}).get(str(spine_id))
        if geometry is None:
            raise ValueError("Recalculate measurements before editing morphology.")
        bounds = geometry["bounds_zyx"]
        y0, x0 = int(bounds[1][0]), int(bounds[2][0])
        y1, x1 = int(bounds[1][1]), int(bounds[2][1])
        editable, _detection, _corrected, _signature = _mask_sources(manifest, specimen_index)
        labels = editable["spine_labels"]
        local_spine = np.asarray(labels[:, y0:y1, x0:x1], dtype=np.uint32) == spine_id
        paint = np.zeros(local_spine.shape[1:], dtype=bool)
        for stroke in strokes_xy:
            for x, y in stroke:
                if 0 <= y < paint.shape[0] and 0 <= x < paint.shape[1]:
                    paint[y, x] = True
        if brush_radius > 0 and np.any(paint):
            coordinates = np.arange(-brush_radius, brush_radius + 1)
            yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
            disk = xx * xx + yy * yy <= brush_radius * brush_radius
            paint = ndimage.binary_dilation(paint, structure=disk)
        selected_mask = local_spine & paint[None, :, :]
        if not maximum_projection:
            allowed_z = np.zeros(local_spine.shape[0], dtype=bool)
            allowed_z[
                max(0, z_index - z_radius) : min(local_spine.shape[0], z_index + z_radius + 1)
            ] = True
            selected_mask &= allowed_z[:, None, None]
        selected = {
            (int(z), int(y0 + y), int(x0 + x))
            for z, y, x in np.argwhere(selected_mask)
        }
        target = "head_override_voxels_zyx" if operation == "paint_head" else "neck_override_voxels_zyx"
        opposite = "neck_override_voxels_zyx" if operation == "paint_head" else "head_override_voxels_zyx"
        target_values = {tuple(int(value) for value in point) for point in decision.get(target, [])}
        opposite_values = {tuple(int(value) for value in point) for point in decision.get(opposite, [])}
        target_values |= selected
        opposite_values -= selected
        decision[target] = [list(point) for point in sorted(target_values)]
        decision[opposite] = [list(point) for point in sorted(opposite_values)]
    elif operation == "reset_border":
        decision.pop("head_override_voxels_zyx", None)
        decision.pop("neck_override_voxels_zyx", None)
    elif operation == "reset_anchors":
        decision.pop("centerline_base_hint_zyx", None)
        decision.pop("centerline_endpoint_hint_zyx", None)
    elif operation == "checkpoint":
        pass
    else:
        raise ValueError("Unknown morphology review operation.")
    if reviewed is not None:
        decision["reviewed"] = bool(reviewed)
    if note is not None:
        decision["note"] = str(note).strip()
    if invalid_spine is not None:
        validity_review = specimen.setdefault(
            "distribution_review", {"spines": {}, "updated_at": None}
        )
        validity = validity_review.setdefault("spines", {}).setdefault(str(spine_id), {})
        validity.update(
            {
                "invalid_spine": bool(invalid_spine),
                "validity_reviewed": True,
                "reviewed": True,
                "note": str(note).strip() if note is not None else str(validity.get("note", "")),
                "review_kind": "morphology_geometry",
                "updated_at": time.time(),
            }
        )
        validity_review["updated_at"] = validity["updated_at"]
    decision["updated_at"] = time.time()
    review["updated_at"] = decision["updated_at"]
    return _checkpoint_morphology_spine(manifest, project_path, specimen_index, spine_id)


def undo_morphology_review(
    manifest: dict[str, object], project_path: str | Path, specimen_index: int, spine_id: int
) -> dict[str, object]:
    decision = manifest["specimens"][specimen_index].setdefault(
        "morphology_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {}).setdefault(str(spine_id), {})
    history = decision.get("history", [])
    if not history:
        raise ValueError("There is no morphology edit to undo.")
    current = {
        key: value
        for key, value in decision.items()
        if key not in {"history", "redo_history"}
    }
    redo_history = list(decision.get("redo_history", []))
    redo_history.append({"snapshot": current, "updated_at": time.time()})
    snapshot = history.pop().get("snapshot", {})
    decision.clear()
    decision.update(snapshot)
    decision["history"] = history
    decision["redo_history"] = redo_history
    return _checkpoint_morphology_spine(manifest, project_path, specimen_index, spine_id)


def redo_morphology_review(
    manifest: dict[str, object], project_path: str | Path, specimen_index: int, spine_id: int
) -> dict[str, object]:
    decision = manifest["specimens"][specimen_index].setdefault(
        "morphology_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {}).setdefault(str(spine_id), {})
    redo_history = decision.get("redo_history", [])
    if not redo_history:
        raise ValueError("There is no morphology edit to redo.")
    current = {
        key: value
        for key, value in decision.items()
        if key not in {"history", "redo_history"}
    }
    history = list(decision.get("history", []))
    history.append({"snapshot": current, "updated_at": time.time()})
    snapshot = redo_history.pop().get("snapshot", {})
    decision.clear()
    decision.update(snapshot)
    decision["history"] = history
    decision["redo_history"] = redo_history
    return _checkpoint_morphology_spine(manifest, project_path, specimen_index, spine_id)


def load_morphology_preview(
    manifest: dict[str, object], specimen_index: int, spine_id: int
) -> MorphologyPreview:
    result = load_measurement_result(manifest, specimen_index)
    calculated, morphology, head, spine, clusters, y_slice, x_slice, _parent = (
        _calculate_morphology_spine(manifest, specimen_index, spine_id, result)
    )
    role_channels = {role: channel for channel, role in manifest["channel_roles"].items()}
    specimen = manifest["specimens"][specimen_index]
    stacks: dict[str, np.ndarray] = {}
    for role in ("dendrite_spines", "protein_clusters"):
        source = channel_source_path(manifest, specimen["channels"][role_channels[role]])
        try:
            stack = np.squeeze(tifffile.memmap(source))
        except ValueError:
            stack = np.squeeze(tifffile.imread(source, out="memmap"))
        stacks[role] = (
            np.array(stack[y_slice, x_slice], dtype=np.uint16, copy=True)[None]
            if stack.ndim == 2
            else np.array(stack[:, y_slice, x_slice], dtype=np.uint16, copy=True)
        )
        del stack
    axis = tuple((int(point[0]), int(point[1]) - y_slice.start, int(point[2]) - x_slice.start) for point in calculated.axis_points_zyx)
    base = axis[0] if axis else None
    tip = axis[-1] if axis else None
    occupied = np.flatnonzero(np.any(spine, axis=(1, 2)))
    row = next(item for item in result.get("morphology_rows", []) if int(item["spine_id"]) == spine_id)
    return MorphologyPreview(
        dendrite_stack=stacks["dendrite_spines"],
        protein_stack=stacks["protein_clusters"],
        spine_mask_stack=spine,
        head_mask_stack=head,
        cluster_mask_stack=clusters,
        axis_points_local_zyx=axis,
        base_point_local_zyx=base,
        tip_point_local_zyx=tip,
        crop_origin_yx=(y_slice.start, x_slice.start),
        spine_z_range=(int(occupied[0]), int(occupied[-1])),
        row={**dict(row), **morphology},
    )


def measure_specimen(
    manifest: dict[str, object],
    specimen_index: int,
    settings: MeasurementSettings,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> tuple[MeasurementSummary, dict[str, object]]:
    started = time.monotonic()
    settings.validate()
    specimen = manifest["specimens"][specimen_index]
    editable, detection, corrected, mask_source_signature = _mask_sources(
        manifest, specimen_index
    )
    signature = measurement_signature(manifest, specimen_index, settings)
    checkpoint = specimen["checkpoints"].setdefault("measurements", {})
    result_path = measurement_result_path(manifest, specimen_index)
    if (
        checkpoint.get("state") == "complete"
        and checkpoint.get("settings_signature") == signature
        and result_path.is_file()
    ):
        saved = load_measurement_result(manifest, specimen_index)
        specimen_row = saved["specimen_rows"][0]
        return (
            MeasurementSummary(
                specimen_index=specimen_index,
                spine_count=int(specimen_row["spine_count"]),
                included_cluster_count=int(specimen_row["included_cluster_count"]),
                dendrite_count=int(specimen_row["dendrite_count"]),
                corrected_masks=bool(saved["corrected_masks"]),
                elapsed_seconds=time.monotonic() - started,
                skipped=True,
            ),
            saved,
        )

    spine_data = editable["spine_labels"]
    dendrite_data = editable["dendrite_labels"]
    cluster_data = detection["cluster_labels"]
    shape = tuple(int(value) for value in spine_data.shape)
    z_count, y_count, x_count = shape
    detection_summary = detection.attrs.get("summary", {})
    maximum_spine = max(
        int(editable.attrs.get("spine_count", detection_summary.get("spine_count", 0))),
        int(editable.attrs.get("next_spine_id", 1)) - 1,
    )
    maximum_dendrite = max(
        int(
            editable.attrs.get(
                "dendrite_count", detection_summary.get("dendrite_count", 0)
            )
        ),
        int(editable.attrs.get("next_dendrite_id", 1)) - 1,
    )
    maximum_cluster = int(detection_summary.get("cluster_count", 0))
    spine_voxels = np.zeros(maximum_spine + 1, dtype=np.int64)
    areas_by_z = np.zeros((z_count, maximum_cluster + 1), dtype=np.int64)
    spine_projection = np.zeros((y_count, x_count), dtype=np.uint32)
    dendrite_projection = np.zeros((y_count, x_count), dtype=np.uint32)
    for z_index in range(z_count):
        _cancel_if_requested(cancel_event)
        spines = np.asarray(spine_data[z_index], dtype=np.uint32)
        dendrites = np.asarray(dendrite_data[z_index], dtype=np.uint32)
        clusters = np.asarray(cluster_data[z_index], dtype=np.uint32)
        spine_voxels += np.bincount(spines.ravel(), minlength=maximum_spine + 1)
        areas_by_z[z_index] = np.bincount(
            clusters.ravel(), minlength=maximum_cluster + 1
        )[: maximum_cluster + 1]
        np.maximum(spine_projection, spines, out=spine_projection)
        np.maximum(dendrite_projection, dendrites, out=dendrite_projection)
        if progress:
            progress(
                "Measuring mask volumes",
                z_index + 1,
                z_count * 2,
                f"{specimen['specimen_id']}: Z {z_index + 1}/{z_count}",
            )

    keep_lookup, trim_details = _cluster_keep_lookup(areas_by_z, settings)
    retained_cluster_voxels = np.zeros(maximum_cluster + 1, dtype=np.int64)
    pair_counts: dict[tuple[int, int], int] = {}
    for z_index in range(z_count):
        _cancel_if_requested(cancel_event)
        spines = np.asarray(spine_data[z_index], dtype=np.uint32)
        clusters = np.asarray(cluster_data[z_index], dtype=np.uint32)
        retained = clusters.copy()
        retained[~keep_lookup[z_index, retained]] = 0
        retained_cluster_voxels += np.bincount(
            retained.ravel(), minlength=maximum_cluster + 1
        )
        overlap = (retained > 0) & (spines > 0)
        if np.any(overlap):
            codes = retained[overlap].astype(np.int64) * (maximum_spine + 1)
            codes += spines[overlap]
            unique, counts = np.unique(codes, return_counts=True)
            for code, count in zip(unique, counts):
                cluster_id, spine_id = divmod(int(code), maximum_spine + 1)
                pair_counts[(cluster_id, spine_id)] = (
                    pair_counts.get((cluster_id, spine_id), 0) + int(count)
                )
        if progress:
            progress(
                "Associating clusters with spines",
                z_count + z_index + 1,
                z_count * 2,
                f"{specimen['specimen_id']}: Z {z_index + 1}/{z_count}",
            )

    xy_size = float(manifest["calibration"]["xy_um_per_pixel"])
    z_step = float(manifest["calibration"]["z_step_um"])
    voxel_volume = xy_size * xy_size * z_step
    spine_parent = _assign_spines_to_dendrites(spine_projection, dendrite_projection)
    dendrite_lengths = _dendrite_lengths(dendrite_projection, xy_size)
    rectangles = normalize_rectangles(
        specimen.get("analysis", {}).get("rois_xy", []), (y_count, x_count)
    ) or [(0, 0, x_count, y_count)]
    spine_roi = {
        spine_id: roi_id_for_mask(spine_projection == spine_id, rectangles)
        for spine_id in range(1, maximum_spine + 1)
        if np.any(spine_projection == spine_id)
    }
    dendrite_roi = {
        dendrite_id: roi_id_for_mask(dendrite_projection == dendrite_id, rectangles)
        for dendrite_id in range(1, maximum_dendrite + 1)
        if np.any(dendrite_projection == dendrite_id)
    }

    cluster_rows: list[dict[str, object]] = []
    included_by_spine: dict[int, list[dict[str, object]]] = {}
    spine_volume_by_id = {
        spine_id: int(spine_voxels[spine_id]) * voxel_volume
        for spine_id in range(1, maximum_spine + 1)
        if int(spine_voxels[spine_id]) > 0
    }
    threshold = settings.minimum_cluster_spine_overlap_percent / 100.0
    for cluster_id in range(1, maximum_cluster + 1):
        retained_count = int(retained_cluster_voxels[cluster_id])
        if not retained_count:
            continue
        candidates = [
            (spine_id, count)
            for (candidate_cluster, spine_id), count in pair_counts.items()
            if candidate_cluster == cluster_id
        ]
        spine_id, inside_count = max(candidates, key=lambda item: item[1]) if candidates else (0, 0)
        overlap_fraction = inside_count / retained_count
        if not spine_id or overlap_fraction < threshold:
            continue
        row = {
            "row_type": "individual_cluster",
            "experimental_group": specimen["experimental_group"],
            "specimen_id": specimen["specimen_id"],
            "roi_id": spine_roi.get(spine_id, 0),
            "dendrite_id": spine_parent.get(spine_id, 0),
            "spine_id": spine_id,
            "cluster_id": cluster_id,
            "retained_cluster_voxels": retained_count,
            "overlap_voxels": inside_count,
            "overlap_percent": overlap_fraction * 100.0,
            "volume_inside_spine_um3": inside_count * voxel_volume,
            "cluster_volume_to_spine_volume_ratio": (
                inside_count * voxel_volume / spine_volume_by_id[spine_id]
                if spine_volume_by_id.get(spine_id)
                else None
            ),
            "distribution_relative_to_spine": None,
            "discarded_z_slices": trim_details.get(cluster_id, {}).get(
                "discarded_z_slices", []
            ),
        }
        cluster_rows.append(row)
        included_by_spine.setdefault(spine_id, []).append(row)

    distribution_rows: list[dict[str, object]] = []
    distribution_geometry: dict[str, dict[str, object]] = {}
    morphology_rows: list[dict[str, object]] = []
    spine_bounds = ndimage.find_objects(spine_projection)
    saved_reviews = specimen.setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {})
    saved_morphology_reviews = specimen.setdefault(
        "morphology_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {})
    for spine_id in sorted(spine_volume_by_id):
        included_clusters = included_by_spine.get(spine_id, [])
        bounds_2d = (
            spine_bounds[spine_id - 1]
            if spine_id - 1 < len(spine_bounds)
            else None
        )
        if bounds_2d is None:
            continue
        y_bounds, x_bounds = bounds_2d
        margin = 2
        y_slice = slice(max(0, y_bounds.start - margin), min(y_count, y_bounds.stop + margin))
        x_slice = slice(max(0, x_bounds.start - margin), min(x_count, x_bounds.stop + margin))
        local_spines = np.asarray(spine_data[:, y_slice, x_slice], dtype=np.uint32)
        local_spine = local_spines == spine_id
        parent_id = spine_parent.get(spine_id, 0)
        local_dendrites = np.asarray(dendrite_data[:, y_slice, x_slice], dtype=np.uint32)
        local_parent = local_dendrites == parent_id if parent_id else local_dendrites > 0
        local_cluster_labels = np.asarray(cluster_data[:, y_slice, x_slice], dtype=np.uint32)
        qualifying = np.zeros(local_spine.shape, dtype=bool)
        included_ids = {int(row["cluster_id"]) for row in included_clusters}
        for z_index in range(z_count):
            retained_ids = [
                cluster_id
                for cluster_id in included_ids
                if keep_lookup[z_index, cluster_id]
            ]
            if retained_ids:
                qualifying[z_index] = (
                    np.isin(local_cluster_labels[z_index], retained_ids)
                    & local_spine[z_index]
                )
        decision = saved_reviews.get(str(spine_id), {})
        morphology_decision = saved_morphology_reviews.get(str(spine_id), {})
        hint_value = morphology_decision.get(
            "centerline_endpoint_hint_zyx",
            decision.get("centerline_endpoint_hint_zyx"),
        )
        hint = (
            tuple(int(value) for value in hint_value)
            if isinstance(hint_value, (list, tuple)) and len(hint_value) == 3
            else None
        )
        hint_valid = False
        if hint is not None:
            local_hint = (hint[0], hint[1] - y_slice.start, hint[2] - x_slice.start)
            hint_valid = (
                0 <= local_hint[0] < local_spine.shape[0]
                and 0 <= local_hint[1] < local_spine.shape[1]
                and 0 <= local_hint[2] < local_spine.shape[2]
                and bool(local_spine[local_hint])
            )
            if not hint_valid:
                history = decision.setdefault("centerline_hint_history", [])
                if not history or history[-1].get("action") != "invalidated_by_resegmentation":
                    history.append(
                        {
                            "action": "invalidated_by_resegmentation",
                            "point_zyx": list(hint),
                            "updated_at": time.time(),
                        }
                    )
                decision["reviewed"] = False
                decision["distribution_reviewed"] = False
        decision["centerline_endpoint_hint_valid"] = hint_valid
        base_hint_value = morphology_decision.get("centerline_base_hint_zyx")
        base_hint = (
            tuple(int(value) for value in base_hint_value)
            if isinstance(base_hint_value, (list, tuple)) and len(base_hint_value) == 3
            else None
        )
        calculated = calculate_spine_distribution(
            local_spine,
            local_parent,
            qualifying,
            sampling_zyx_um=(z_step, xy_size, xy_size),
            global_offset_zyx=(0, y_slice.start, x_slice.start),
            endpoint_hint_zyx=hint if hint_valid else None,
            base_hint_zyx=base_hint,
            guidance_image=_distribution_guidance(
                manifest, specimen_index, y_slice, x_slice
            ),
            maximum_gap_um=settings.maximum_centerline_gap_um,
        )
        row = distribution_row(
            calculated,
            experimental_group=str(specimen["experimental_group"]),
            specimen_id=str(specimen["specimen_id"]),
            dendrite_id=parent_id,
            spine_id=spine_id,
            voxel_volume_um3=voxel_volume,
        )
        default_include = calculated.axis_status in {
            "ok",
            "insufficient_axis_resolution",
        }
        row.update(
            {
                "roi_id": spine_roi.get(spine_id, 0),
                "distribution_reviewed": bool(
                    decision.get("distribution_reviewed", decision.get("reviewed", False))
                ),
                "distribution_included": bool(
                    decision.get("distribution_included", default_include)
                ),
                "spine_valid": not bool(decision.get("invalid_spine", False)),
                "review_note": str(decision.get("note", "")),
                "centerline_endpoint_hint_valid": hint_valid,
                "centerline_endpoint_hint_present": hint is not None,
                "centerline_hint_history": list(
                    decision.get("centerline_hint_history", [])
                ),
            }
        )
        if included_clusters:
            distribution_rows.append(row)
        morphology_values, _head_mask = _spine_morphology_metrics(
            calculated,
            local_spine,
            sampling_zyx_um=(z_step, xy_size, xy_size),
            global_offset_zyx=(0, y_slice.start, x_slice.start),
            voxel_volume_um3=voxel_volume,
            decision=morphology_decision,
        )
        morphology_rows.append(
            {
                "experimental_group": specimen["experimental_group"],
                "specimen_id": specimen["specimen_id"],
                "roi_id": spine_roi.get(spine_id, 0),
                "dendrite_id": parent_id,
                "spine_id": spine_id,
                **morphology_values,
            }
        )
        distribution_geometry[str(spine_id)] = {
            "bounds_zyx": [
                [0, z_count],
                [y_slice.start, y_slice.stop],
                [x_slice.start, x_slice.stop],
            ],
            "axis_points_zyx": [list(point) for point in calculated.axis_points_zyx],
            "base_point_zyx": list(calculated.base_point_zyx) if calculated.base_point_zyx else None,
            "endpoint_zyx": list(calculated.endpoint_zyx) if calculated.endpoint_zyx else None,
            "bridge_points_zyx": [list(point) for point in calculated.bridge_points_zyx],
            "bridge_length_um": calculated.bridge_length_um,
        }

    spine_rows: list[dict[str, object]] = []
    morphology_by_spine = {
        int(row["spine_id"]): row for row in morphology_rows
    }
    for spine_id in range(1, maximum_spine + 1):
        voxel_count = int(spine_voxels[spine_id])
        if not voxel_count:
            continue
        volume = voxel_count * voxel_volume
        included = included_by_spine.get(spine_id, [])
        cluster_sum = sum(float(row["volume_inside_spine_um3"]) for row in included)
        morphology = morphology_by_spine.get(spine_id, {})
        spine_rows.append(
            {
                "experimental_group": specimen["experimental_group"],
                "specimen_id": specimen["specimen_id"],
                "roi_id": spine_roi.get(spine_id, 0),
                "dendrite_id": spine_parent.get(spine_id, 0),
                "spine_id": spine_id,
                "voxel_count": voxel_count,
                "volume_um3": volume,
                "has_protein_cluster": bool(included),
                "included_cluster_count": len(included),
                "inside_cluster_volume_sum_um3": cluster_sum,
                "cluster_to_spine_volume_ratio": cluster_sum / volume if volume else None,
                "spine_curvilinear_length_um": morphology.get(
                    "spine_curvilinear_length_um"
                ),
                "spine_base_to_tip_distance_um": morphology.get(
                    "spine_base_to_tip_distance_um"
                ),
                "spine_length_status": morphology.get("spine_length_status"),
                "spine_length_uses_virtual_bridge": morphology.get(
                    "spine_length_uses_virtual_bridge", False
                ),
                "centerline_bridge_length_um": morphology.get(
                    "centerline_bridge_length_um", 0.0
                ),
                "centerline_base_zyx": morphology.get("centerline_base_zyx"),
                "centerline_tip_zyx": morphology.get("centerline_tip_zyx"),
                "centerline_base_source": morphology.get("centerline_base_source"),
                "centerline_tip_source": morphology.get("centerline_tip_source"),
                "geometry_reviewed": morphology.get("geometry_reviewed", False),
                "protein_distribution_in_spine": next(
                    (
                        [row.get(f"bin_{index:02d}_ratio") for index in range(1, 11)]
                        for row in distribution_rows
                        if int(row["spine_id"]) == spine_id
                    ),
                    None,
                ),
                "spine_valid": not bool(
                    saved_reviews.get(str(spine_id), {}).get("invalid_spine", False)
                ),
                "validity_reviewed": bool(
                    saved_reviews.get(str(spine_id), {}).get(
                        "validity_reviewed",
                        saved_reviews.get(str(spine_id), {}).get("reviewed", False),
                    )
                ),
                "validity_note": str(
                    saved_reviews.get(str(spine_id), {}).get("note", "")
                ),
                "review_kind": str(
                    saved_reviews.get(str(spine_id), {}).get("review_kind", "")
                ),
            }
        )
        if included:
            cluster_rows.append(
                {
                    "row_type": "spine_cluster_sum",
                    "experimental_group": specimen["experimental_group"],
                    "specimen_id": specimen["specimen_id"],
                    "roi_id": spine_roi.get(spine_id, 0),
                    "dendrite_id": spine_parent.get(spine_id, 0),
                    "spine_id": spine_id,
                    "cluster_id": None,
                    "retained_cluster_voxels": sum(
                        int(row["retained_cluster_voxels"]) for row in included
                    ),
                    "overlap_voxels": sum(int(row["overlap_voxels"]) for row in included),
                    "overlap_percent": (
                        100.0
                        * sum(int(row["overlap_voxels"]) for row in included)
                        / sum(int(row["retained_cluster_voxels"]) for row in included)
                    ),
                    "volume_inside_spine_um3": cluster_sum,
                    "cluster_volume_to_spine_volume_ratio": (
                        cluster_sum / volume if volume else None
                    ),
                    "distribution_relative_to_spine": None,
                    "discarded_z_slices": [],
                }
            )

    dendrite_rows: list[dict[str, object]] = []
    present_dendrites = sorted(
        set(int(value) for value in np.unique(dendrite_projection) if value > 0)
        | {int(row["dendrite_id"]) for row in spine_rows if int(row["dendrite_id"]) > 0}
    )
    for dendrite_id in present_dendrites:
        dendrite_spines = [row for row in spine_rows if row["dendrite_id"] == dendrite_id]
        individual_clusters = [
            row
            for row in cluster_rows
            if row["row_type"] == "individual_cluster" and row["dendrite_id"] == dendrite_id
        ]
        length_um = dendrite_lengths.get(dendrite_id, 0.0)
        dendrite_rows.append(
            {
                "experimental_group": specimen["experimental_group"],
                "specimen_id": specimen["specimen_id"],
                "roi_id": dendrite_roi.get(dendrite_id, 0),
                "dendrite_id": dendrite_id,
                "length_um": length_um,
                "spine_count": len(dendrite_spines),
                "spine_density_per_um": len(dendrite_spines) / length_um if length_um else None,
                "average_spine_volume_um3": _mean(
                    [float(row["volume_um3"]) for row in dendrite_spines]
                ),
                "spines_with_clusters_percent": (
                    100.0
                    * sum(bool(row["has_protein_cluster"]) for row in dendrite_spines)
                    / len(dendrite_spines)
                    if dendrite_spines
                    else None
                ),
                "average_cluster_to_spine_volume_ratio": _mean(
                    [
                        float(row["cluster_to_spine_volume_ratio"])
                        for row in dendrite_spines
                        if row["cluster_to_spine_volume_ratio"] is not None
                    ]
                ),
                "average_cluster_volume_um3": _mean(
                    [float(row["volume_inside_spine_um3"]) for row in individual_clusters]
                ),
                "average_protein_distribution": None,
            }
        )

    individual_clusters = [
        row for row in cluster_rows if row["row_type"] == "individual_cluster"
    ]
    total_dendrite_length = sum(float(row.get("length_um") or 0.0) for row in dendrite_rows)
    roi_rows = [
        {
            "experimental_group": specimen["experimental_group"],
            "specimen_id": specimen["specimen_id"],
            "roi_id": roi_id,
            "x0": rectangle[0],
            "y0": rectangle[1],
            "x1": rectangle[2],
            "y1": rectangle[3],
        }
        for roi_id, rectangle in enumerate(rectangles, start=1)
    ]
    specimen_row = {
        "experimental_group": specimen["experimental_group"],
        "specimen_id": specimen["specimen_id"],
        "dendrite_count": len(dendrite_rows),
        "spine_count": len(spine_rows),
        "included_cluster_count": len(individual_clusters),
        "roi_count": len(rectangles),
        "average_spine_density_per_um": (
            len(spine_rows) / total_dendrite_length if total_dendrite_length else None
        ),
        "average_spine_volume_um3": _mean(
            [float(row["volume_um3"]) for row in spine_rows]
        ),
        "spines_with_clusters_percent": (
            100.0 * sum(bool(row["has_protein_cluster"]) for row in spine_rows) / len(spine_rows)
            if spine_rows
            else None
        ),
        "average_cluster_to_spine_volume_ratio": _mean(
            [
                float(row["cluster_to_spine_volume_ratio"])
                for row in spine_rows
                if row["cluster_to_spine_volume_ratio"] is not None
            ]
        ),
        "average_cluster_volume_um3": _mean(
            [float(row["volume_inside_spine_um3"]) for row in individual_clusters]
        ),
        "average_protein_distribution": None,
    }
    result = {
        "algorithm_version": ALGORITHM_VERSION,
        "settings_signature": signature,
        "settings": settings.to_dict(),
        "mask_source_signature": mask_source_signature,
        "corrected_masks": corrected,
        "voxel_volume_um3": voxel_volume,
        "specimen_rows": [specimen_row],
        "roi_rows": roi_rows,
        "dendrite_rows": dendrite_rows,
        "spine_rows": spine_rows,
        "cluster_rows": cluster_rows,
        "distribution_rows": distribution_rows,
        "distribution_geometry": distribution_geometry,
        "morphology_rows": morphology_rows,
        "morphology_geometry": distribution_geometry,
        "cluster_trim_details": {str(key): value for key, value in trim_details.items()},
    }
    _refresh_result_summaries(result)
    _write_result(result_path, result)
    summary = MeasurementSummary(
        specimen_index=specimen_index,
        spine_count=int(result["specimen_rows"][0]["spine_count"]),
        included_cluster_count=int(result["specimen_rows"][0]["included_cluster_count"]),
        dendrite_count=len(dendrite_rows),
        corrected_masks=corrected,
        elapsed_seconds=time.monotonic() - started,
    )
    return summary, result


def measure_project(
    manifest: dict[str, object],
    project_path: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> dict[str, object]:
    settings = MeasurementSettings.from_dict(manifest["measurements"]["settings"])
    eligible = [
        index
        for index, specimen in enumerate(manifest["specimens"])
        if specimen["checkpoints"]["preprocessing"].get("state") == "complete"
        and specimen["checkpoints"]["detection"].get("state") == "complete"
        and specimen["checkpoints"]["review"].get("state") == "complete"
    ]
    if not eligible:
        raise ValueError(
            "No specimen has completed preprocessing, detection, and manual review."
        )
    summaries: list[dict[str, object]] = []
    work_units: dict[int, int] = {}
    for specimen_index in eligible:
        editable, _detection, _corrected, _signature = _mask_sources(
            manifest, specimen_index
        )
        z_count = int(editable["spine_labels"].shape[0])
        work_units[specimen_index] = z_count * 2
    total_work = sum(work_units.values())
    completed_work = 0
    for specimen_index in eligible:
        _cancel_if_requested(cancel_event)
        specimen = manifest["specimens"][specimen_index]
        checkpoint = specimen["checkpoints"].setdefault("measurements", {})
        checkpoint["state"] = "in_progress"

        def specimen_progress(phase: str, current: int, total: int, detail: str) -> None:
            if progress:
                progress(
                    phase,
                    completed_work + min(current, work_units[specimen_index]),
                    total_work,
                    detail,
                )

        summary, _result = measure_specimen(
            manifest,
            specimen_index,
            settings,
            progress=specimen_progress,
            cancel_event=cancel_event,
        )
        signature = measurement_signature(manifest, specimen_index, settings)
        checkpoint.update(
            {
                "state": "complete",
                "updated_at": time.time(),
                "settings_signature": signature,
                "summary": asdict(summary),
            }
        )
        if not summary.skipped:
            for run in manifest.get("morphology_analysis", {}).get("runs", []):
                run["stale"] = True
        save_project(project_path, manifest)
        summaries.append(asdict(summary))
        completed_work += work_units[specimen_index]
        if progress:
            progress(
                "Measurement checkpoint",
                completed_work,
                total_work,
                f"{specimen['specimen_id']} saved",
            )
    if progress:
        progress(
            "Measurements complete",
            total_work,
            total_work,
            f"{len(eligible)} specimen pair(s) checkpointed",
        )
    return {"settings": settings.to_dict(), "summaries": summaries}
