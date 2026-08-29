from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Literal

import numpy as np
import psutil
import tifffile
import zarr
from numcodecs import Blosc
from scipy import ndimage

from .detection import detection_cache_path
from .models import ProgressCallback
from .preprocessing import ProcessingCancelled, project_cache_path
from .project import save_project


ObjectType = Literal["dendrite", "spine"]
Operation = Literal[
    "add",
    "exclude",
    "filopodium",
    "split",
    "merge",
    "expand",
    "trim",
    "accept",
    "needs_attention",
]


@dataclass(frozen=True)
class ReviewAction:
    object_type: ObjectType
    operation: Operation
    z_index: int
    points: tuple[tuple[int, int], ...]
    brush_radius_pixels: int = 4
    projection_hint: bool = False

    def validate(self) -> None:
        if self.object_type not in {"dendrite", "spine"}:
            raise ValueError("Review object type must be dendrite or spine.")
        if self.operation not in {
            "add",
            "exclude",
            "filopodium",
            "split",
            "merge",
            "expand",
            "trim",
            "accept",
            "needs_attention",
        }:
            raise ValueError("Unknown review operation.")
        if not self.points:
            raise ValueError("Draw or click a hint before applying this action.")
        if self.brush_radius_pixels < 1:
            raise ValueError("Hint brush radius must be positive.")
        if self.operation == "filopodium" and self.object_type != "spine":
            raise ValueError("Filopodium exclusion is available only for spines.")


@dataclass(frozen=True)
class ReviewResult:
    specimen_index: int
    action_id: str
    operation: str
    object_type: str
    affected_ids: tuple[int, ...]
    new_ids: tuple[int, ...]
    edit_count: int
    dendrite_count: int
    spine_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewSlice:
    raw: np.ndarray
    dendrites: np.ndarray
    spines: np.ndarray
    clusters: np.ndarray
    z_index: int
    z_count: int
    corrected: bool


def review_cache_path(manifest: dict[str, object]) -> Path:
    return project_cache_path(manifest).parent / "review.zarr"


def _compressor() -> Blosc:
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _cancel_if_requested(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled(
            "Review operation was cancelled before its checkpoint was committed."
        )


def _review_group_key(specimen_index: int) -> str:
    return f"specimens/{specimen_index:04d}"


def _ensure_review_group(
    manifest: dict[str, object],
    specimen_index: int,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> zarr.Group:
    detection_root = zarr.open_group(str(detection_cache_path(manifest)), mode="r")
    key = _review_group_key(specimen_index)
    if key not in detection_root or not bool(
        detection_root[key].attrs.get("complete", False)
    ):
        raise ValueError("Automatic detection must finish before this specimen can be reviewed.")
    detection_group = detection_root[key]
    source_signature = str(detection_group.attrs["settings_signature"])
    review_root = zarr.open_group(str(review_cache_path(manifest)), mode="a")
    if key in review_root:
        existing = review_root[key]
        if (
            bool(existing.attrs.get("initialized", False))
            and existing.attrs.get("detection_signature") == source_signature
        ):
            return existing
        del review_root[key]
    group = review_root.require_group(key)
    group.attrs.update(
        {
            "initialized": False,
            "detection_signature": source_signature,
            "created_at": time.time(),
        }
    )
    # Protein clusters are not manually corrected, so keep using the immutable
    # detection labels rather than duplicating a potentially large third volume.
    names = ("dendrite_labels", "spine_labels")
    total = sum(int(detection_group[name].shape[0]) for name in names)
    current = 0
    for name in names:
        source = detection_group[name]
        output = group.create_dataset(
            name,
            shape=source.shape,
            chunks=source.chunks,
            dtype="uint32",
            compressor=_compressor(),
            overwrite=True,
        )
        for z_index in range(source.shape[0]):
            _cancel_if_requested(cancel_event)
            output[z_index] = source[z_index]
            current += 1
            if progress:
                progress(
                    "Preparing editable masks",
                    current,
                    total,
                    f"{name}: Z {z_index + 1}/{source.shape[0]}",
                )
    summary = detection_group.attrs.get("summary", {})
    group.attrs.update(
        {
            "initialized": True,
            "dendrite_count": int(summary.get("dendrite_count", 0)),
            "spine_count": int(summary.get("spine_count", 0)),
            "cluster_count": int(summary.get("cluster_count", 0)),
            "next_dendrite_id": int(summary.get("dendrite_count", 0)) + 1,
            "next_spine_id": int(summary.get("spine_count", 0)) + 1,
        }
    )
    return group


def _disk_mask(
    shape: tuple[int, int], points: tuple[tuple[int, int], ...], radius: int
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for x, y in points:
        if (
            x + radius < 0
            or y + radius < 0
            or x - radius >= shape[1]
            or y - radius >= shape[0]
        ):
            continue
        mask |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return mask


def _sample_ids(
    plane: np.ndarray, points: tuple[tuple[int, int], ...], radius: int
) -> tuple[int, ...]:
    mask = _disk_mask(tuple(plane.shape), points, radius)
    values = np.unique(plane[mask])
    return tuple(int(value) for value in values if value > 0)


def _hint_crop(
    shape: tuple[int, int], points: tuple[tuple[int, int], ...], radius: int
) -> tuple[slice, slice, np.ndarray]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = max(0, min(xs) - radius)
    x1 = min(shape[1], max(xs) + radius + 1)
    y0 = max(0, min(ys) - radius)
    y1 = min(shape[0], max(ys) + radius + 1)
    local_points = tuple((x - x0, y - y0) for x, y in points)
    return (
        slice(y0, y1),
        slice(x0, x1),
        _disk_mask((y1 - y0, x1 - x0), local_points, radius),
    )


def _sample_ids_projection(
    array: zarr.Array, points: tuple[tuple[int, int], ...], radius: int
) -> tuple[int, ...]:
    y_slice, x_slice, hint = _hint_crop(tuple(array.shape[1:]), points, radius)
    selected: set[int] = set()
    for z_index in range(array.shape[0]):
        values = np.unique(np.asarray(array[z_index, y_slice, x_slice])[hint])
        selected.update(int(value) for value in values if value > 0)
    return tuple(sorted(selected))


def _best_projection_z(
    array: zarr.Array,
    ids: tuple[int, ...],
    points: tuple[tuple[int, int], ...],
    radius: int,
) -> int:
    y_slice, x_slice, hint = _hint_crop(tuple(array.shape[1:]), points, radius)
    best_z = 0
    best_score = -1
    wanted = np.asarray(ids, dtype=np.uint32)
    for z_index in range(array.shape[0]):
        patch = np.asarray(array[z_index, y_slice, x_slice])
        score = int(np.count_nonzero(np.isin(patch[hint], wanted)))
        if score > best_score:
            best_z, best_score = z_index, score
    return best_z


def _brightest_projection_z(
    processed: zarr.Array,
    points: tuple[tuple[int, int], ...],
    radius: int,
) -> int:
    y_slice, x_slice, hint = _hint_crop(tuple(processed.shape[1:]), points, radius)
    best_z = 0
    best_score = float("-inf")
    for z_index in range(processed.shape[0]):
        patch = np.asarray(processed[z_index, y_slice, x_slice], dtype=np.float32)
        score = float(np.max(patch[hint])) if np.any(hint) else float("-inf")
        if score > best_score:
            best_z, best_score = z_index, score
    return best_z


def _bbox_for_ids(
    array: zarr.Array, ids: tuple[int, ...]
) -> tuple[int, int, int, int, int, int]:
    wanted = np.asarray(ids, dtype=np.uint32)
    z_min, y_min, x_min = array.shape[0], array.shape[1], array.shape[2]
    z_max = y_max = x_max = -1
    for z_index in range(array.shape[0]):
        plane = np.asarray(array[z_index])
        coordinates = np.argwhere(np.isin(plane, wanted))
        if not len(coordinates):
            continue
        z_min = min(z_min, z_index)
        z_max = max(z_max, z_index)
        y_min = min(y_min, int(coordinates[:, 0].min()))
        y_max = max(y_max, int(coordinates[:, 0].max()))
        x_min = min(x_min, int(coordinates[:, 1].min()))
        x_max = max(x_max, int(coordinates[:, 1].max()))
    if z_max < 0:
        raise ValueError("The selected object is no longer present in the corrected mask.")
    return z_min, z_max + 1, y_min, y_max + 1, x_min, x_max + 1


def _local_bbox(
    shape: tuple[int, int, int],
    z_index: int,
    points: tuple[tuple[int, int], ...],
    xy_margin: int,
    z_radius: int,
) -> tuple[int, int, int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        max(0, z_index - z_radius),
        min(shape[0], z_index + z_radius + 1),
        max(0, min(ys) - xy_margin),
        min(shape[1], max(ys) + xy_margin + 1),
        max(0, min(xs) - xy_margin),
        min(shape[2], max(xs) + xy_margin + 1),
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int, int, int],
    shape: tuple[int, int, int],
    xy: int,
    z: int = 0,
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    return (
        max(0, z0 - z),
        min(shape[0], z1 + z),
        max(0, y0 - xy),
        min(shape[1], y1 + xy),
        max(0, x0 - xy),
        min(shape[2], x1 + xy),
    )


def _bbox_slices(
    bbox: tuple[int, int, int, int, int, int]
) -> tuple[slice, slice, slice]:
    z0, z1, y0, y1, x0, x1 = bbox
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def _enforce_review_ram_policy(
    manifest: dict[str, object],
    bbox: tuple[int, int, int, int, int, int],
) -> None:
    z0, z1, y0, y1, x0, x1 = bbox
    voxels = (z1 - z0) * (y1 - y0) * (x1 - x0)
    # Covers label copies, boolean candidates, component labels, and working space.
    estimated_extra = voxels * 32 + 256 * 1024 * 1024
    fraction = min(
        0.8, float(manifest["resource_policy"].get("maximum_ram_fraction", 0.8))
    )
    if not 0 < fraction <= 0.8:
        raise ValueError("Maximum RAM fraction must be above zero and at most 0.8.")
    limit = int(psutil.virtual_memory().total * fraction)
    if psutil.Process().memory_info().rss + estimated_extra > limit:
        raise MemoryError(
            "This correction would exceed Synpo's RAM safety limit. Narrow the hint "
            "or split/exclude the object in smaller local steps."
        )


def _local_points(
    points: tuple[tuple[int, int], ...], bbox: tuple[int, int, int, int, int, int]
) -> tuple[tuple[int, int], ...]:
    _, _, y0, _, x0, _ = bbox
    return tuple((x - x0, y - y0) for x, y in points)


def _store_undo_patch(
    group: zarr.Group,
    action_id: str,
    dataset_name: str,
    bbox: tuple[int, int, int, int, int, int],
    patch: np.ndarray,
) -> None:
    undo = group.require_group("undo").require_group(action_id)
    undo.create_dataset(
        "before",
        data=np.asarray(patch, dtype=np.uint32),
        chunks=(1, min(256, patch.shape[1]), min(256, patch.shape[2])),
        compressor=_compressor(),
        overwrite=True,
    )
    undo.attrs.update({"dataset_name": dataset_name, "bbox": list(bbox)})


def _processed_dendrite_data(
    manifest: dict[str, object], specimen_index: int
) -> zarr.Array:
    dendrite_channel = next(
        channel
        for channel, role in manifest["channel_roles"].items()
        if role == "dendrite_spines"
    )
    checkpoint = manifest["specimens"][specimen_index]["checkpoints"]["preprocessing"]
    key = checkpoint["channels"][dendrite_channel]["dataset_key"]
    return zarr.open_group(str(project_cache_path(manifest)), mode="r")[key]


def _status_changes(
    specimen: dict[str, object],
    object_type: str,
    ids: tuple[int, ...],
    status: str,
) -> dict[str, str | None]:
    statuses = specimen["review"]["object_status"].setdefault(object_type, {})
    previous: dict[str, str | None] = {}
    for object_id in ids:
        key = str(object_id)
        previous[key] = statuses.get(key)
        statuses[key] = status
    return previous


def apply_review_action(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    action: ReviewAction,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> ReviewResult:
    action.validate()
    specimen = manifest["specimens"][specimen_index]
    if specimen["checkpoints"]["detection"].get("state") != "complete":
        raise ValueError("Automatic detection must finish before review.")
    group = _ensure_review_group(
        manifest,
        specimen_index,
        progress=progress,
        cancel_event=cancel_event,
    )
    dataset_name = f"{action.object_type}_labels"
    target = group[dataset_name]
    if not 0 <= action.z_index < target.shape[0]:
        raise ValueError("The selected Z slice is outside the stack.")
    if action.projection_hint:
        selected_ids = _sample_ids_projection(
            target, action.points, action.brush_radius_pixels
        )
        if selected_ids:
            effective_z = _best_projection_z(
                target, selected_ids, action.points, action.brush_radius_pixels
            )
        elif action.operation == "add":
            effective_z = _brightest_projection_z(
                _processed_dendrite_data(manifest, specimen_index),
                action.points,
                action.brush_radius_pixels,
            )
        else:
            effective_z = action.z_index
    else:
        selected_ids = _sample_ids(
            np.asarray(target[action.z_index]),
            action.points,
            action.brush_radius_pixels,
        )
        effective_z = action.z_index
    action_id = uuid.uuid4().hex
    bbox: tuple[int, int, int, int, int, int] | None = None
    before: np.ndarray | None = None
    affected_ids = selected_ids
    new_ids: tuple[int, ...] = ()
    count_delta = 0
    previous_status: dict[str, str | None] = {}
    pending_status: str | None = None

    if action.operation in {"accept", "needs_attention"}:
        if not selected_ids:
            raise ValueError("The hint did not touch a selected object.")
        pending_status = (
            "accepted" if action.operation == "accept" else "needs_attention"
        )
    elif action.operation in {"exclude", "filopodium", "merge", "split"}:
        required = 2 if action.operation == "merge" else 1
        if len(selected_ids) < required:
            raise ValueError(
                "This action needs the hint to touch "
                + ("at least two objects." if required == 2 else "an existing object.")
            )
        bbox = _expand_bbox(_bbox_for_ids(target, selected_ids), tuple(target.shape), 2)
        _enforce_review_ram_policy(manifest, bbox)
        slices = _bbox_slices(bbox)
        before = np.asarray(target[slices])
        patch = before.copy()
        if action.operation in {"exclude", "filopodium"}:
            patch[np.isin(patch, selected_ids)] = 0
            count_delta = -len(selected_ids)
            pending_status = (
                "excluded_filopodium"
                if action.operation == "filopodium"
                else "excluded"
            )
        elif action.operation == "merge":
            keep = min(selected_ids)
            patch[np.isin(patch, selected_ids)] = keep
            count_delta = -(len(selected_ids) - 1)
            pending_status = "merged"
            new_ids = (keep,)
        else:
            if len(selected_ids) != 1:
                raise ValueError("Split one object at a time.")
            object_id = selected_ids[0]
            object_mask = patch == object_id
            local_hint = _disk_mask(
                tuple(patch.shape[1:]),
                _local_points(action.points, bbox),
                action.brush_radius_pixels,
            )
            object_mask &= ~local_hint[None, :, :]
            components, component_count = ndimage.label(
                object_mask, structure=np.ones((3, 3, 3), dtype=bool)
            )
            sizes = np.bincount(components.ravel())
            component_ids = [
                value for value in range(1, component_count + 1) if sizes[value] >= 5
            ]
            if len(component_ids) < 2:
                raise ValueError(
                    "The hint did not separate this object. Draw across the full neck/contact."
                )
            patch[patch == object_id] = 0
            assigned = [object_id]
            next_key = f"next_{action.object_type}_id"
            next_id = int(group.attrs[next_key])
            for component_id in component_ids[1:]:
                assigned.append(next_id)
                next_id += 1
            for component_id, assigned_id in zip(component_ids, assigned):
                patch[components == component_id] = assigned_id
            group.attrs[next_key] = next_id
            new_ids = tuple(assigned)
            count_delta = len(assigned) - 1
        _store_undo_patch(group, action_id, dataset_name, bbox, before)
        target[slices] = patch
    else:
        xy = float(manifest["calibration"]["xy_um_per_pixel"])
        margin = max(
            action.brush_radius_pixels + 4,
            int(round(float(manifest["review_settings"]["local_margin_um"]) / xy)),
        )
        z_radius = int(manifest["review_settings"]["z_radius_slices"])
        if action.operation in {"expand", "trim"}:
            if len(selected_ids) != 1:
                raise ValueError("This boundary action must touch exactly one object.")
            bbox = _local_bbox(
                tuple(target.shape), effective_z, action.points, margin, z_radius
            )
        else:
            bbox = _local_bbox(
                tuple(target.shape), effective_z, action.points, margin, z_radius
            )
        _enforce_review_ram_policy(manifest, bbox)
        slices = _bbox_slices(bbox)
        before = np.asarray(target[slices])
        patch = before.copy()
        local_points = _local_points(action.points, bbox)
        hint_2d = _disk_mask(
            tuple(patch.shape[1:]), local_points, action.brush_radius_pixels
        )
        hint_3d = np.zeros(patch.shape, dtype=bool)
        local_z = effective_z - bbox[0]
        hint_3d[local_z, hint_2d] = True
        hint_3d = ndimage.binary_dilation(
            hint_3d, structure=np.ones((3, 3, 3), dtype=bool), iterations=1
        )
        if action.operation == "trim":
            object_id = selected_ids[0]
            candidate = (patch == object_id) & ~hint_3d
            components, count = ndimage.label(
                candidate, structure=np.ones((3, 3, 3), dtype=bool)
            )
            if not count:
                raise ValueError("This hint would remove the complete object.")
            sizes = np.bincount(components.ravel())
            keep = int(np.argmax(sizes[1:]) + 1)
            patch[patch == object_id] = 0
            patch[components == keep] = object_id
        else:
            processed = _processed_dendrite_data(manifest, specimen_index)
            signal = np.asarray(processed[slices])
            sensitivity = float(
                manifest["detection"]["settings"]["dendrite_sensitivity"]
            )
            threshold = float(processed.attrs["statistics"]["applied_threshold"])
            candidate = signal >= (threshold / sensitivity) * 0.8
            candidate = ndimage.binary_closing(
                candidate, structure=np.ones((1, 3, 3), dtype=bool)
            )
            if action.object_type == "spine":
                dendrite_patch = np.asarray(group["dendrite_labels"][slices])
                candidate &= dendrite_patch == 0
            candidate |= hint_3d
            if action.operation == "expand":
                object_id = selected_ids[0]
                seed = (patch == object_id) | hint_3d
                candidate |= patch == object_id
                region = ndimage.binary_propagation(
                    seed, structure=np.ones((3, 3, 3), dtype=bool), mask=candidate
                )
                region &= (patch == 0) | (patch == object_id)
                patch[region] = object_id
            else:
                region = ndimage.binary_propagation(
                    hint_3d,
                    structure=np.ones((3, 3, 3), dtype=bool),
                    mask=candidate,
                )
                region &= patch == 0
                if int(region.sum()) < 5:
                    raise ValueError(
                        "No image-supported object was found around the hint. "
                        "Increase sensitivity or draw farther inside the signal."
                    )
                next_key = f"next_{action.object_type}_id"
                object_id = int(group.attrs[next_key])
                group.attrs[next_key] = object_id + 1
                patch[region] = object_id
                new_ids = (object_id,)
                affected_ids = (object_id,)
                count_delta = 1
        _store_undo_patch(group, action_id, dataset_name, bbox, before)
        target[slices] = patch

    count_key = f"{action.object_type}_count"
    group.attrs[count_key] = max(0, int(group.attrs[count_key]) + count_delta)
    if pending_status is not None:
        previous_status = _status_changes(
            specimen, action.object_type, selected_ids, pending_status
        )
    history_entry = {
        "action_id": action_id,
        "timestamp": time.time(),
        "operation": action.operation,
        "object_type": action.object_type,
        "z_index": effective_z,
        "projection_hint": action.projection_hint,
        "hint_point_count": len(action.points),
        "brush_radius_pixels": action.brush_radius_pixels,
        "affected_ids": list(affected_ids),
        "new_ids": list(new_ids),
        "bbox": list(bbox) if bbox else None,
        "count_delta": count_delta,
        "previous_status": previous_status,
        "undone": False,
        "undo_available": True,
    }
    specimen["review"]["history"].append(history_entry)
    maximum_undo = int(manifest["review_settings"].get("maximum_undo_actions", 100))
    mask_undo_entries = [
        item
        for item in specimen["review"]["history"]
        if item.get("bbox") is not None
        and not item.get("undone")
        and item.get("undo_available", True)
    ]
    while len(mask_undo_entries) > maximum_undo:
        expired = mask_undo_entries.pop(0)
        undo_key = f"undo/{expired['action_id']}"
        if undo_key in group:
            del group[undo_key]
        expired["undo_available"] = False
    specimen["review"]["state"] = "in_progress"
    checkpoint = specimen["checkpoints"]["review"]
    checkpoint.update(
        {
            "state": "in_progress",
            "updated_at": time.time(),
            "edit_count": int(checkpoint.get("edit_count", 0)) + 1,
            "cache_path": str(review_cache_path(manifest)),
            "detection_signature": group.attrs["detection_signature"],
        }
    )
    save_project(project_path, manifest)
    return ReviewResult(
        specimen_index=specimen_index,
        action_id=action_id,
        operation=action.operation,
        object_type=action.object_type,
        affected_ids=affected_ids,
        new_ids=new_ids,
        edit_count=int(checkpoint["edit_count"]),
        dendrite_count=int(group.attrs["dendrite_count"]),
        spine_count=int(group.attrs["spine_count"]),
    )


def undo_last_review_action(
    manifest: dict[str, object], project_path: str | Path, specimen_index: int
) -> ReviewResult:
    specimen = manifest["specimens"][specimen_index]
    history = specimen["review"]["history"]
    entry = next(
        (
            item
            for item in reversed(history)
            if not item.get("undone") and item.get("undo_available", True)
        ),
        None,
    )
    if entry is None:
        raise ValueError("There is no review action to undo for this specimen.")
    root = zarr.open_group(str(review_cache_path(manifest)), mode="a")
    group = root[_review_group_key(specimen_index)]
    action_id = str(entry["action_id"])
    if entry.get("bbox") is not None:
        undo = group[f"undo/{action_id}"]
        bbox = tuple(int(value) for value in undo.attrs["bbox"])
        group[str(undo.attrs["dataset_name"])][_bbox_slices(bbox)] = np.asarray(
            undo["before"]
        )
    statuses = specimen["review"]["object_status"].setdefault(
        str(entry["object_type"]), {}
    )
    for object_id, previous in entry.get("previous_status", {}).items():
        if previous is None:
            statuses.pop(object_id, None)
        else:
            statuses[object_id] = previous
    count_key = f"{entry['object_type']}_count"
    group.attrs[count_key] = max(
        0, int(group.attrs[count_key]) - int(entry.get("count_delta", 0))
    )
    entry["undone"] = True
    checkpoint = specimen["checkpoints"]["review"]
    checkpoint["updated_at"] = time.time()
    checkpoint["edit_count"] = max(0, int(checkpoint.get("edit_count", 1)) - 1)
    specimen["review"]["state"] = "in_progress"
    save_project(project_path, manifest)
    return ReviewResult(
        specimen_index=specimen_index,
        action_id=action_id,
        operation="undo",
        object_type=str(entry["object_type"]),
        affected_ids=tuple(int(value) for value in entry.get("affected_ids", [])),
        new_ids=(),
        edit_count=int(checkpoint["edit_count"]),
        dendrite_count=int(group.attrs["dendrite_count"]),
        spine_count=int(group.attrs["spine_count"]),
    )


def set_specimen_review_state(
    manifest: dict[str, object],
    project_path: str | Path,
    specimen_index: int,
    *,
    complete: bool,
    comment: str,
) -> None:
    specimen = manifest["specimens"][specimen_index]
    state = "complete" if complete else "in_progress"
    specimen["review"]["state"] = state
    specimen["review"]["comment"] = comment.strip()
    checkpoint = specimen["checkpoints"]["review"]
    checkpoint["state"] = state
    checkpoint["updated_at"] = time.time()
    checkpoint["detection_signature"] = specimen["checkpoints"]["detection"].get(
        "settings_signature"
    )
    save_project(project_path, manifest)


def load_review_slice(
    manifest: dict[str, object], specimen_index: int, z_index: int, background_channel: str
) -> ReviewSlice:
    specimen = manifest["specimens"][specimen_index]
    shape = tuple(
        int(value)
        for value in specimen["channels"][background_channel]["metadata"]["shape"]
    )
    z_count = 1 if len(shape) == 2 else shape[0]
    source = (
        Path(str(manifest["source_directory"]))
        / specimen["channels"][background_channel]["filename"]
    )
    with tifffile.TiffFile(source) as tiff:
        series = tiff.series[0]
        raw = np.squeeze(
            np.asarray(series.asarray() if z_count == 1 else series.asarray(key=z_index))
        )
    key = _review_group_key(specimen_index)
    review_path = review_cache_path(manifest)
    detection_root = zarr.open_group(str(detection_cache_path(manifest)), mode="r")
    detection_group = detection_root[key]
    corrected = False
    if review_path.exists():
        review_root = zarr.open_group(str(review_path), mode="r")
        if key in review_root and bool(review_root[key].attrs.get("initialized", False)):
            candidate = review_root[key]
            current_signature = detection_group.attrs.get("settings_signature")
            if candidate.attrs.get("detection_signature") == current_signature:
                group = candidate
                corrected = True
            else:
                group = None
        else:
            group = None
    else:
        group = None
    if group is None:
        group = detection_group
    return ReviewSlice(
        raw=raw,
        dendrites=np.asarray(group["dendrite_labels"][z_index]),
        spines=np.asarray(group["spine_labels"][z_index]),
        clusters=np.asarray(detection_group["cluster_labels"][z_index]),
        z_index=z_index,
        z_count=z_count,
        corrected=corrected,
    )
