from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


BIN_COUNT = 10


@dataclass(frozen=True)
class SpineDistribution:
    axis_status: str
    axis_note: str
    axis_points_zyx: tuple[tuple[int, int, int], ...]
    spine_voxels_by_bin: tuple[int, ...]
    cluster_voxels_by_bin: tuple[int, ...]
    voxel_bins: np.ndarray | None
    base_point_zyx: tuple[int, int, int] | None
    endpoint_zyx: tuple[int, int, int] | None
    endpoint_source: str


_NEIGHBOURS = tuple(
    offset
    for offset in product((-1, 0, 1), repeat=3)
    if offset != (0, 0, 0)
)


def _nearest_index(
    points: np.ndarray, targets: np.ndarray, sampling: tuple[float, float, float]
) -> int:
    scaled_points = points.astype(np.float64) * np.asarray(sampling)
    scaled_targets = targets.astype(np.float64) * np.asarray(sampling)
    best_index = 0
    best_distance = np.inf
    # Contact patches and spine skeletons are normally small. Chunking avoids an
    # unexpectedly large pairwise matrix for unusually broad shaft contacts.
    for start in range(0, len(points), 1024):
        distances = np.sum(
            (scaled_points[start : start + 1024, None] - scaled_targets[None]) ** 2,
            axis=2,
        )
        flat = int(np.argmin(distances))
        distance = float(distances.ravel()[flat])
        if distance < best_distance:
            local_point, _target = np.unravel_index(flat, distances.shape)
            best_index = start + int(local_point)
            best_distance = distance
    return best_index


def _skeleton_graph(
    coordinates: np.ndarray, sampling: tuple[float, float, float]
) -> tuple[list[list[tuple[int, float]]], dict[tuple[int, int, int], int]]:
    lookup = {tuple(int(value) for value in point): index for index, point in enumerate(coordinates)}
    graph: list[list[tuple[int, float]]] = [[] for _ in range(len(coordinates))]
    sampling_array = np.asarray(sampling, dtype=np.float64)
    for index, point in enumerate(coordinates):
        point_tuple = tuple(int(value) for value in point)
        for offset in _NEIGHBOURS:
            neighbour = tuple(point_tuple[axis] + offset[axis] for axis in range(3))
            neighbour_index = lookup.get(neighbour)
            if neighbour_index is None or neighbour_index <= index:
                continue
            weight = float(np.linalg.norm(np.asarray(offset) * sampling_array))
            graph[index].append((neighbour_index, weight))
            graph[neighbour_index].append((index, weight))
    return graph, lookup


def _dijkstra(
    graph: list[list[tuple[int, float]]], start: int
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(len(graph), np.inf, dtype=np.float64)
    predecessors = np.full(len(graph), -1, dtype=np.int64)
    distances[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                predecessors[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return distances, predecessors


def _reconstruct_path(predecessors: np.ndarray, start: int, end: int) -> list[int]:
    path = [end]
    while path[-1] != start:
        parent = int(predecessors[path[-1]])
        if parent < 0:
            return []
        path.append(parent)
    path.reverse()
    return path


def _longest_path(
    graph: list[list[tuple[int, float]]], start: int
) -> tuple[list[int], bool]:
    distances, predecessors = _dijkstra(graph, start)
    endpoints = [index for index, edges in enumerate(graph) if len(edges) <= 1 and index != start]
    if not endpoints:
        endpoints = [index for index in range(len(graph)) if index != start]
    reachable = [index for index in endpoints if np.isfinite(distances[index])]
    if not reachable:
        return [], False
    reachable.sort(key=lambda index: float(distances[index]), reverse=True)
    end = reachable[0]
    competing = (
        len(reachable) > 1
        and distances[reachable[1]] >= distances[end] * 0.90
    )
    return _reconstruct_path(predecessors, start, end), competing


def _path_toward_hint(
    graph: list[list[tuple[int, float]]],
    coordinates: np.ndarray,
    start: int,
    hint: tuple[int, int, int],
    sampling: tuple[float, float, float],
) -> list[int]:
    distances, predecessors = _dijkstra(graph, start)
    reachable = np.flatnonzero(np.isfinite(distances))
    if not len(reachable):
        return []
    scaled = (coordinates[reachable] - np.asarray(hint)) * np.asarray(sampling)
    end = int(reachable[int(np.argmin(np.sum(scaled * scaled, axis=1)))])
    return _reconstruct_path(predecessors, start, end)


def _inside_spine_path(
    spine: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    sampling: tuple[float, float, float],
) -> list[tuple[int, int, int]]:
    """A* path constrained to spine voxels, used for the final hinted segment."""
    if start == end:
        return [start]
    sampling_array = np.asarray(sampling, dtype=np.float64)

    def heuristic(point: tuple[int, int, int]) -> float:
        return float(np.linalg.norm((np.asarray(point) - np.asarray(end)) * sampling_array))

    queue: list[tuple[float, float, tuple[int, int, int]]] = [(heuristic(start), 0.0, start)]
    distances = {start: 0.0}
    predecessors: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    while queue:
        _score, distance, point = heapq.heappop(queue)
        if distance != distances.get(point):
            continue
        if point == end:
            path = [end]
            while path[-1] != start:
                path.append(predecessors[path[-1]])
            path.reverse()
            return path
        for offset in _NEIGHBOURS:
            neighbour = tuple(point[axis] + offset[axis] for axis in range(3))
            if any(value < 0 or value >= spine.shape[axis] for axis, value in enumerate(neighbour)):
                continue
            if not spine[neighbour]:
                continue
            weight = float(np.linalg.norm(np.asarray(offset) * sampling_array))
            candidate = distance + weight
            if candidate < distances.get(neighbour, np.inf):
                distances[neighbour] = candidate
                predecessors[neighbour] = point
                heapq.heappush(queue, (candidate + heuristic(neighbour), candidate, neighbour))
    return []


def calculate_spine_distribution(
    spine_mask: np.ndarray,
    dendrite_mask: np.ndarray,
    qualifying_cluster_mask: np.ndarray,
    *,
    sampling_zyx_um: tuple[float, float, float],
    global_offset_zyx: tuple[int, int, int] = (0, 0, 0),
    endpoint_hint_zyx: tuple[int, int, int] | None = None,
) -> SpineDistribution:
    """Split a 3-D spine into ten calibrated geodesic shaft-to-tip bins."""
    spine = np.asarray(spine_mask, dtype=bool)
    dendrite = np.asarray(dendrite_mask, dtype=bool)
    clusters = np.asarray(qualifying_cluster_mask, dtype=bool) & spine
    empty = (0,) * BIN_COUNT
    if not np.any(spine):
        return SpineDistribution("no_usable_path", "Spine mask is empty.", (), empty, empty, None, None, None, "automatic")

    skeleton = np.asarray(skeletonize(spine), dtype=bool)
    coordinates = np.argwhere(skeleton)
    if len(coordinates) < 2:
        return SpineDistribution(
            "no_usable_path",
            "The 3-D spine skeleton has fewer than two voxels.",
            (),
            empty,
            empty,
            None,
            None,
            None,
            "automatic",
        )

    graph, _lookup = _skeleton_graph(coordinates, sampling_zyx_um)
    contact_voxels = spine & ndimage.binary_dilation(dendrite, structure=np.ones((3, 3, 3), dtype=bool))
    contact_note = ""
    contact_ambiguous = False
    if np.any(contact_voxels):
        contact_labels, count = ndimage.label(contact_voxels)
        sizes = np.bincount(contact_labels.ravel())[1:]
        order = np.argsort(sizes)[::-1]
        chosen_label = int(order[0]) + 1
        if count > 1 and sizes[order[1]] >= sizes[order[0]] * 0.80:
            contact_ambiguous = True
            contact_note = "Multiple similarly sized spine/dendrite contact regions."
        targets = np.argwhere(contact_labels == chosen_label)
    else:
        # Segmentation may leave a one-voxel gap. Use the nearest spine voxel to
        # the parent dendrite and flag the missing direct contact for review.
        contact_ambiguous = True
        contact_note = "No direct spine/dendrite contact; nearest region was used."
        if not np.any(dendrite):
            return SpineDistribution(
                "no_usable_path",
                "No parent-dendrite voxels were available near this spine.",
                (),
                empty,
                empty,
                None,
                None,
                None,
                "automatic",
            )
        distances = ndimage.distance_transform_edt(~dendrite, sampling=sampling_zyx_um)
        candidate_distances = np.where(spine, distances, np.inf)
        targets = np.asarray(
            [np.unravel_index(int(np.argmin(candidate_distances)), spine.shape)],
            dtype=np.int64,
        )

    start = _nearest_index(coordinates, targets, sampling_zyx_um)
    local_hint: tuple[int, int, int] | None = None
    if endpoint_hint_zyx is not None:
        local_hint = tuple(
            int(endpoint_hint_zyx[axis]) - int(global_offset_zyx[axis])
            for axis in range(3)
        )
        if any(value < 0 or value >= spine.shape[axis] for axis, value in enumerate(local_hint)) or not spine[local_hint]:
            local_hint = None
    if local_hint is None:
        path_indices, endpoint_ambiguous = _longest_path(graph, start)
    else:
        path_indices = _path_toward_hint(
            graph, coordinates, start, local_hint, sampling_zyx_um
        )
        endpoint_ambiguous = False
    if not path_indices or (local_hint is None and len(path_indices) < 2):
        return SpineDistribution(
            "no_usable_path",
            "No connected shaft-to-tip skeleton path could be constructed.",
            (),
            empty,
            empty,
            None,
            None,
            None,
            "manual" if local_hint is not None else "automatic",
        )

    path_points = coordinates[path_indices]
    if local_hint is not None:
        final_segment = _inside_spine_path(
            spine,
            tuple(int(value) for value in path_points[-1]),
            local_hint,
            sampling_zyx_um,
        )
        if final_segment:
            appended = np.asarray(final_segment[1:], dtype=np.int64)
            if len(appended):
                path_points = np.vstack((path_points, appended))
    if len(path_points) < 2:
        return SpineDistribution(
            "no_usable_path",
            "The selected endpoint does not produce a non-zero centerline.",
            (),
            empty,
            empty,
            None,
            None,
            None,
            "manual" if local_hint is not None else "automatic",
        )
    cumulative = np.zeros(len(path_points), dtype=np.float64)
    sampling = np.asarray(sampling_zyx_um, dtype=np.float64)
    cumulative[1:] = np.cumsum(
        np.linalg.norm(np.diff(path_points, axis=0) * sampling, axis=1)
    )
    if cumulative[-1] <= 0:
        return SpineDistribution(
            "no_usable_path", "The centerline has zero calibrated length.", (), empty, empty, None, None, None, "manual" if local_hint is not None else "automatic"
        )

    path_volume = np.zeros(spine.shape, dtype=bool)
    path_volume[tuple(path_points.T)] = True
    _distance, nearest = ndimage.distance_transform_edt(
        ~path_volume,
        sampling=sampling_zyx_um,
        return_indices=True,
    )
    path_progress = np.full(spine.shape, np.nan, dtype=np.float64)
    path_progress[tuple(path_points.T)] = cumulative
    nearest_progress = path_progress[tuple(nearest)]
    voxel_bins = np.full(spine.shape, -1, dtype=np.int8)
    assigned = np.floor(nearest_progress[spine] / cumulative[-1] * BIN_COUNT).astype(np.int16)
    voxel_bins[spine] = np.clip(assigned, 0, BIN_COUNT - 1).astype(np.int8)
    spine_counts = np.bincount(voxel_bins[spine], minlength=BIN_COUNT)[:BIN_COUNT]
    cluster_counts = np.bincount(voxel_bins[clusters], minlength=BIN_COUNT)[:BIN_COUNT]

    notes = [note for note in (contact_note,) if note]
    if endpoint_ambiguous:
        notes.append("Multiple similarly long distal skeleton paths were found.")
    if local_hint is not None:
        notes.append("Manual distal endpoint hint was used.")
    zero_bins = np.flatnonzero(spine_counts == 0)
    if len(zero_bins):
        notes.append(
            "No spine voxel centres fell in bin(s) "
            + ", ".join(str(int(index) + 1) for index in zero_bins)
            + "."
        )
    if contact_ambiguous or endpoint_ambiguous:
        status = "ambiguous_axis"
    elif len(zero_bins):
        status = "insufficient_axis_resolution"
    else:
        status = "ok"

    offset = np.asarray(global_offset_zyx, dtype=np.int64)
    global_points = tuple(
        tuple(int(value) for value in point + offset) for point in path_points
    )
    return SpineDistribution(
        axis_status=status,
        axis_note=" ".join(notes),
        axis_points_zyx=global_points,
        spine_voxels_by_bin=tuple(int(value) for value in spine_counts),
        cluster_voxels_by_bin=tuple(int(value) for value in cluster_counts),
        voxel_bins=voxel_bins,
        base_point_zyx=global_points[0],
        endpoint_zyx=global_points[-1],
        endpoint_source="manual" if local_hint is not None else "automatic",
    )


def distribution_row(
    distribution: SpineDistribution,
    *,
    experimental_group: str,
    specimen_id: str,
    dendrite_id: int,
    spine_id: int,
    voxel_volume_um3: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "experimental_group": experimental_group,
        "specimen_id": specimen_id,
        "dendrite_id": dendrite_id,
        "spine_id": spine_id,
        "distribution_axis_status": distribution.axis_status,
        "distribution_axis_note": distribution.axis_note,
        "centerline_base_zyx": list(distribution.base_point_zyx) if distribution.base_point_zyx else None,
        "centerline_endpoint_zyx": list(distribution.endpoint_zyx) if distribution.endpoint_zyx else None,
        "centerline_endpoint_source": distribution.endpoint_source,
    }
    for index, (spine_count, cluster_count) in enumerate(
        zip(distribution.spine_voxels_by_bin, distribution.cluster_voxels_by_bin),
        start=1,
    ):
        row[f"bin_{index:02d}_spine_volume_um3"] = (
            spine_count * voxel_volume_um3 if spine_count else None
        )
        row[f"bin_{index:02d}_cluster_volume_um3"] = cluster_count * voxel_volume_um3
        row[f"bin_{index:02d}_ratio"] = cluster_count / spine_count if spine_count else None
    return row
