from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np


Rectangle = tuple[int, int, int, int]


def normalize_rectangles(
    rectangles: Iterable[object], shape_yx: tuple[int, int]
) -> list[Rectangle]:
    """Clamp, discard empty entries, and merge touching analysis rectangles."""
    height, width = (int(value) for value in shape_yx)
    pending: list[Rectangle] = []
    for value in rectangles:
        if isinstance(value, dict):
            raw = (value.get("x0"), value.get("y0"), value.get("x1"), value.get("y1"))
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            raw = tuple(value)
        else:
            continue
        try:
            x0, y0, x1, y1 = (int(item) for item in raw)
        except (TypeError, ValueError):
            continue
        x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
        y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
        if x1 > x0 and y1 > y0:
            pending.append((x0, y0, x1, y1))

    changed = True
    while changed:
        changed = False
        merged: list[Rectangle] = []
        while pending:
            current = pending.pop(0)
            index = 0
            while index < len(pending):
                other = pending[index]
                # Half-open rectangles touch when one boundary equals the other.
                separated = (
                    current[2] < other[0]
                    or other[2] < current[0]
                    or current[3] < other[1]
                    or other[3] < current[1]
                )
                if separated:
                    index += 1
                    continue
                current = (
                    min(current[0], other[0]),
                    min(current[1], other[1]),
                    max(current[2], other[2]),
                    max(current[3], other[3]),
                )
                pending.pop(index)
                changed = True
                index = 0
            merged.append(current)
        pending = merged
    return sorted(pending, key=lambda item: (item[1], item[0], item[3], item[2]))


def specimen_shape_yx(specimen: dict[str, object]) -> tuple[int, int]:
    channel = next(iter(specimen["channels"].values()))
    shape = tuple(int(value) for value in channel["metadata"]["shape"])
    return shape[-2], shape[-1]


def specimen_rectangles(
    manifest: dict[str, object], specimen_index: int, *, full_if_empty: bool = True
) -> list[Rectangle]:
    specimen = manifest["specimens"][specimen_index]
    shape_yx = specimen_shape_yx(specimen)
    saved = specimen.get("analysis", {}).get("rois_xy", [])
    rectangles = normalize_rectangles(saved, shape_yx)
    if not rectangles and full_if_empty:
        height, width = shape_yx
        return [(0, 0, width, height)]
    return rectangles


def rectangle_records(rectangles: Iterable[Rectangle]) -> list[dict[str, int]]:
    return [
        {"id": index, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        for index, (x0, y0, x1, y1) in enumerate(rectangles, start=1)
    ]


def roi_mask(shape_yx: tuple[int, int], rectangles: Iterable[Rectangle]) -> np.ndarray:
    mask = np.zeros(shape_yx, dtype=bool)
    for x0, y0, x1, y1 in rectangles:
        mask[y0:y1, x0:x1] = True
    return mask


def roi_signature(rectangles: Iterable[Rectangle]) -> str:
    payload = [list(rectangle) for rectangle in rectangles]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def roi_id_for_mask(mask_yx: np.ndarray, rectangles: Iterable[Rectangle]) -> int:
    mask = np.asarray(mask_yx, dtype=bool)
    best_id = 0
    best_overlap = 0
    for roi_id, (x0, y0, x1, y1) in enumerate(rectangles, start=1):
        overlap = int(np.count_nonzero(mask[y0:y1, x0:x1]))
        if overlap > best_overlap:
            best_id, best_overlap = roi_id, overlap
    return best_id
