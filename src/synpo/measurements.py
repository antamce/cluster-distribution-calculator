from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
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
from .project import save_project
from .review import review_cache_path


ClusterEndMethod = Literal["untrimmed", "fixed", "adaptive"]
ALGORITHM_VERSION = 2


@dataclass(frozen=True)
class MeasurementSettings:
    minimum_cluster_spine_overlap_percent: float = 80.0
    cluster_end_method: ClusterEndMethod = "adaptive"
    fixed_end_slices: int = 3
    adaptive_area_factor: float = 1.8
    minimum_retained_slices: int = 2

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
        raise ValueError("This specimen has no saved measurement result.")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


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
    specimen.update(
        {
            "spine_count": len(valid_spines),
            "included_cluster_count": len(valid_clusters),
            "average_spine_density_per_um": _mean(
                [float(row["spine_density_per_um"]) for row in dendrite_rows if row.get("spine_density_per_um") is not None]
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
            "invalid_spine_count": len(spine_rows) - len(valid_spines),
            "distribution_included_spine_count": len(included_distribution),
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
    decision = {
        "reviewed": True,
        "distribution_included": bool(distribution_included and not invalid_spine),
        "invalid_spine": bool(invalid_spine),
        "note": str(note).strip(),
        "updated_at": time.time(),
    }
    reviews[str(spine_id)] = decision
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
    if not found:
        raise ValueError("This spine has no cluster-positive distribution row.")
    _refresh_result_summaries(result)
    _write_result(measurement_result_path(manifest, specimen_index), result)
    specimen["checkpoints"].setdefault("measurements", {})["review_updated_at"] = time.time()
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
    source = (
        Path(str(manifest["source_directory"]))
        / specimen["channels"][protein_channel]["filename"]
    )
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
    calculated = calculate_spine_distribution(
        spine,
        parent,
        clusters,
        sampling_zyx_um=(z_step, xy_size, xy_size),
        global_offset_zyx=(0, y_slice.start, x_slice.start),
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
    for role in ("dendrite_spines", "protein_clusters"):
        channel = role_channels[role]
        source = Path(str(manifest["source_directory"])) / specimen["channels"][channel]["filename"]
        projection = np.zeros(spine.shape[1:], dtype=np.uint16)
        try:
            stack = np.squeeze(tifffile.memmap(source))
        except ValueError:
            # Compressed TIFFs cannot be mapped directly; tifffile decodes them
            # into a temporary disk-backed array instead of consuming stack RAM.
            stack = np.squeeze(tifffile.imread(source, out="memmap"))
        if stack.ndim == 2:
            projection = np.asarray(stack[y_slice, x_slice], dtype=np.uint16)
        else:
            projection = np.max(stack[:, y_slice, x_slice], axis=0).astype(np.uint16, copy=False)
        projections[role] = projection
    axis_xy = tuple(
        (int(point[2]) - x_slice.start, int(point[1]) - y_slice.start)
        for point in calculated.axis_points_zyx
    )
    return DistributionPreview(
        dendrite_projection=projections["dendrite_spines"],
        protein_projection=projections["protein_clusters"],
        spine_bins_projection=spine_bins,
        cluster_bins_projection=cluster_bins,
        axis_xy=axis_xy,
        row=dict(row),
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
    spine_bounds = ndimage.find_objects(spine_projection)
    saved_reviews = specimen.setdefault(
        "distribution_review", {"spines": {}, "updated_at": None}
    ).setdefault("spines", {})
    for spine_id, included_clusters in sorted(included_by_spine.items()):
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
        calculated = calculate_spine_distribution(
            local_spine,
            local_parent,
            qualifying,
            sampling_zyx_um=(z_step, xy_size, xy_size),
            global_offset_zyx=(0, y_slice.start, x_slice.start),
        )
        row = distribution_row(
            calculated,
            experimental_group=str(specimen["experimental_group"]),
            specimen_id=str(specimen["specimen_id"]),
            dendrite_id=parent_id,
            spine_id=spine_id,
            voxel_volume_um3=voxel_volume,
        )
        decision = saved_reviews.get(str(spine_id), {})
        default_include = calculated.axis_status in {
            "ok",
            "insufficient_axis_resolution",
        }
        row.update(
            {
                "distribution_reviewed": bool(decision.get("reviewed", False)),
                "distribution_included": bool(
                    decision.get("distribution_included", default_include)
                ),
                "spine_valid": not bool(decision.get("invalid_spine", False)),
                "review_note": str(decision.get("note", "")),
            }
        )
        distribution_rows.append(row)
        distribution_geometry[str(spine_id)] = {
            "bounds_zyx": [
                [0, z_count],
                [y_slice.start, y_slice.stop],
                [x_slice.start, x_slice.stop],
            ],
            "axis_points_zyx": [list(point) for point in calculated.axis_points_zyx],
        }

    spine_rows: list[dict[str, object]] = []
    for spine_id in range(1, maximum_spine + 1):
        voxel_count = int(spine_voxels[spine_id])
        if not voxel_count:
            continue
        volume = voxel_count * voxel_volume
        included = included_by_spine.get(spine_id, [])
        cluster_sum = sum(float(row["volume_inside_spine_um3"]) for row in included)
        spine_rows.append(
            {
                "experimental_group": specimen["experimental_group"],
                "specimen_id": specimen["specimen_id"],
                "dendrite_id": spine_parent.get(spine_id, 0),
                "spine_id": spine_id,
                "voxel_count": voxel_count,
                "volume_um3": volume,
                "has_protein_cluster": bool(included),
                "included_cluster_count": len(included),
                "inside_cluster_volume_sum_um3": cluster_sum,
                "cluster_to_spine_volume_ratio": cluster_sum / volume if volume else None,
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
            }
        )
        if included:
            cluster_rows.append(
                {
                    "row_type": "spine_cluster_sum",
                    "experimental_group": specimen["experimental_group"],
                    "specimen_id": specimen["specimen_id"],
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
    specimen_row = {
        "experimental_group": specimen["experimental_group"],
        "specimen_id": specimen["specimen_id"],
        "dendrite_count": len(dendrite_rows),
        "spine_count": len(spine_rows),
        "included_cluster_count": len(individual_clusters),
        "average_spine_density_per_um": _mean(
            [float(row["spine_density_per_um"]) for row in dendrite_rows if row["spine_density_per_um"] is not None]
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
        "dendrite_rows": dendrite_rows,
        "spine_rows": spine_rows,
        "cluster_rows": cluster_rows,
        "distribution_rows": distribution_rows,
        "distribution_geometry": distribution_geometry,
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
        if specimen["checkpoints"]["detection"].get("state") == "complete"
    ]
    if not eligible:
        raise ValueError("No detected specimens are ready for measurement.")
    summaries: list[dict[str, object]] = []
    for position, specimen_index in enumerate(eligible):
        _cancel_if_requested(cancel_event)
        specimen = manifest["specimens"][specimen_index]
        checkpoint = specimen["checkpoints"].setdefault("measurements", {})
        checkpoint["state"] = "in_progress"

        def specimen_progress(phase: str, current: int, total: int, detail: str) -> None:
            if progress:
                progress(
                    phase,
                    position * total + current,
                    len(eligible) * total,
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
        save_project(project_path, manifest)
        summaries.append(asdict(summary))
    return {"settings": settings.to_dict(), "summaries": summaries}
