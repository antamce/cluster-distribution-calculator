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
    bridge_used: bool = False
    bridge_length_um: float = 0.0
    bridge_points_zyx: tuple[tuple[int, int, int], ...] = ()


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
    guidance_image: np.ndarray | None = None,
    maximum_gap_um: float = 1.0,
) -> SpineDistribution:
    """Split a 3-D spine into ten calibrated geodesic shaft-to-tip bins."""
    spine = np.asarray(spine_mask, dtype=bool)
    dendrite = np.asarray(dendrite_mask, dtype=bool)
    clusters = np.asarray(qualifying_cluster_mask, dtype=bool) & spine
    empty = (0,) * BIN_COUNT
    if not np.any(spine):
        return SpineDistribution("no_usable_path", "Spine mask is empty.", (), empty, empty, None, None, None, "automatic")

    bridge_points: tuple[tuple[int, int, int], ...] = ()
    bridge_length = 0.0
    topology_status: str | None = None
    topology_notes: list[str] = []
    sampling_array = np.asarray(sampling_zyx_um, dtype=np.float64)
    components, component_count = ndimage.label(
        spine, structure=np.ones((3, 3, 3), dtype=bool)
    )
    sizes = np.bincount(components.ravel())
    component_ids = list(range(1, component_count + 1))
    largest_size = max((int(sizes[value]) for value in component_ids), default=0)
    minimum_substantial_size = max(5, int(np.ceil(largest_size * 0.02)))

    requested_hint: tuple[int, int, int] | None = None
    hinted_component = 0
    if endpoint_hint_zyx is not None:
        candidate = tuple(
            int(endpoint_hint_zyx[axis]) - int(global_offset_zyx[axis])
            for axis in range(3)
        )
        if all(0 <= value < spine.shape[axis] for axis, value in enumerate(candidate)):
            hinted_component = int(components[candidate])
            if hinted_component > 0:
                requested_hint = candidate

    # The base is selected before any bridging. Direct shaft contact is strongest;
    # otherwise use the nearest substantial component rather than a tiny satellite.
    contact_counts: dict[int, int] = {}
    if np.any(dendrite):
        near_dendrite = ndimage.binary_dilation(
            dendrite, structure=np.ones((3, 3, 3), dtype=bool)
        )
        for component_id in component_ids:
            contact_counts[component_id] = int(
                np.count_nonzero((components == component_id) & near_dendrite)
            )
    contacting = [
        value
        for value in component_ids
        if contact_counts.get(value, 0) > 0
        and int(sizes[value]) >= minimum_substantial_size
    ]
    if contacting:
        base_component = max(
            contacting,
            key=lambda value: (contact_counts[value], int(sizes[value])),
        )
    elif np.any(dendrite):
        distance_to_dendrite = ndimage.distance_transform_edt(
            ~dendrite, sampling=sampling_zyx_um
        )
        candidates = [
            value
            for value in component_ids
            if int(sizes[value]) >= minimum_substantial_size
        ] or component_ids
        base_component = min(
            candidates,
            key=lambda value: (
                float(np.min(distance_to_dendrite[components == value])),
                -int(sizes[value]),
            ),
        )
    else:
        base_component = max(component_ids, key=lambda value: int(sizes[value]))

    substantial = {
        value
        for value in component_ids
        if int(sizes[value]) >= minimum_substantial_size
    }
    substantial.add(base_component)
    if hinted_component:
        substantial.add(hinted_component)

    distal_component: int | None = None
    if hinted_component:
        if hinted_component != base_component:
            distal_component = hinted_component
        elif len(substantial) > 1:
            topology_status = "disconnected_components_ignored"
            topology_notes.append(
                "The manual endpoint is in the component nearest the dendrite; "
                "other disconnected components were not used for the centerline."
            )
    elif len(substantial) == 2:
        distal_component = next(value for value in substantial if value != base_component)
    elif len(substantial) > 2:
        topology_status = "multiple_disconnected_components"
        topology_notes.append(
            f"{len(substantial)} substantial disconnected components were found; "
            "the component nearest the dendrite was retained until a distal endpoint is selected."
        )

    axis_components = {base_component}
    bridge_failure_note = ""
    if distal_component is not None:
        from scipy.spatial import cKDTree
        from skimage.graph import route_through_array

        if maximum_gap_um <= 0:
            bridge_failure_note = "Virtual bridging is disabled."
        coordinates_by_component = [
            np.argwhere(components == value)
            for value in (base_component, distal_component)
        ]
        first_scaled = coordinates_by_component[0].astype(np.float64) * sampling_array
        second_scaled = coordinates_by_component[1].astype(np.float64) * sampling_array
        tree = cKDTree(second_scaled)
        distances, neighbours = tree.query(first_scaled, k=1)
        first_index = int(np.argmin(distances))
        gap_distance = float(distances[first_index])
        second_index = int(neighbours[first_index])
        if not bridge_failure_note and gap_distance > float(maximum_gap_um):
            bridge_failure_note = (
                f"Disconnected spine gap {gap_distance:.3f} µm exceeds the "
                f"{maximum_gap_um:.3f} µm limit."
            )
        if not bridge_failure_note:
            start_point = tuple(int(value) for value in coordinates_by_component[0][first_index])
            end_point = tuple(int(value) for value in coordinates_by_component[1][second_index])
            lower = np.maximum(0, np.minimum(start_point, end_point) - 2)
            upper = np.minimum(np.asarray(spine.shape), np.maximum(start_point, end_point) + 3)
            slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
            if guidance_image is not None and np.asarray(guidance_image).shape == spine.shape:
                local_signal = np.asarray(guidance_image[slices], dtype=np.float32)
                low, high = np.percentile(local_signal, (5.0, 99.0))
                normalized = np.clip((local_signal - low) / max(1e-6, high - low), 0.0, 1.0)
                costs = 1.0 + (1.0 - normalized) * 4.0
            else:
                costs = np.ones(
                    tuple(int(upper[axis] - lower[axis]) for axis in range(3)),
                    dtype=np.float32,
                )
            selected_components = np.isin(
                components[slices], [base_component, distal_component]
            )
            costs[selected_components] = 0.1
            local_start = tuple(int(start_point[axis] - lower[axis]) for axis in range(3))
            local_end = tuple(int(end_point[axis] - lower[axis]) for axis in range(3))
            route, _cost = route_through_array(
                costs, local_start, local_end, fully_connected=True, geometric=True
            )
            global_route = [
                tuple(int(point[axis] + lower[axis]) for axis in range(3))
                for point in route
            ]
            if len(global_route) >= 2:
                route_array = np.asarray(global_route, dtype=np.int64)
                candidate_length = float(
                    np.sum(
                        np.linalg.norm(
                            np.diff(route_array, axis=0) * sampling_array, axis=1
                        )
                    )
                )
                if candidate_length <= float(maximum_gap_um):
                    bridge_length = candidate_length
                    bridge_points = tuple(global_route)
                    axis_components.add(distal_component)
                else:
                    bridge_failure_note = (
                        f"The signal-guided bridge path is {candidate_length:.3f} µm, "
                        "beyond the permitted gap corridor."
                    )
            else:
                bridge_failure_note = "No signal-guided bridge path could be constructed."
        if bridge_failure_note:
            topology_status = "unbridged_disconnected_part"
            topology_notes.append(
                bridge_failure_note
                + " A centerline was retained in the component nearest the dendrite."
            )

    ignored_components = set(component_ids) - axis_components
    if ignored_components:
        ignored_voxels = sum(int(sizes[value]) for value in ignored_components)
        ignored_substantial = ignored_components & substantial
        topology_notes.append(
            f"{len(ignored_components)} disconnected component(s), {ignored_voxels} voxel(s), "
            "were ignored for centerline topology but retained in mask-volume measurements."
        )
        if topology_status is None:
            topology_status = (
                "multiple_disconnected_components"
                if ignored_substantial
                else "disconnected_fragments_ignored"
            )

    augmented_spine = np.isin(components, list(axis_components))
    if bridge_points:
        augmented_spine[tuple(np.asarray(bridge_points, dtype=np.int64).T)] = True

    skeleton = np.asarray(skeletonize(augmented_spine), dtype=bool)
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
    base_mask = components == base_component
    contact_voxels = base_mask & ndimage.binary_dilation(
        dendrite, structure=np.ones((3, 3, 3), dtype=bool)
    )
    contact_note = ""
    contact_ambiguous = False
    if np.any(contact_voxels):
        contact_labels, count = ndimage.label(contact_voxels)
        contact_sizes = np.bincount(contact_labels.ravel())[1:]
        order = np.argsort(contact_sizes)[::-1]
        chosen_label = int(order[0]) + 1
        if count > 1 and contact_sizes[order[1]] >= contact_sizes[order[0]] * 0.80:
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
        candidate_distances = np.where(base_mask, distances, np.inf)
        targets = np.asarray(
            [np.unravel_index(int(np.argmin(candidate_distances)), spine.shape)],
            dtype=np.int64,
        )

    start = _nearest_index(coordinates, targets, sampling_zyx_um)
    local_hint = (
        requested_hint
        if requested_hint is not None and hinted_component in axis_components
        else None
    )
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

    notes = [note for note in (contact_note, *topology_notes) if note]
    if bridge_points:
        notes.append(f"A virtual signal-guided bridge of {bridge_length:.3f} µm was used.")
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
    if bridge_points:
        status = "bridged_gap"
    elif topology_status is not None:
        status = topology_status
    elif contact_ambiguous or endpoint_ambiguous:
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
        bridge_used=bool(bridge_points),
        bridge_length_um=bridge_length,
        bridge_points_zyx=tuple(
            tuple(int(value) for value in np.asarray(point) + offset)
            for point in bridge_points
        ),
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
        "centerline_bridge_used": distribution.bridge_used,
        "centerline_bridge_length_um": distribution.bridge_length_um,
        "centerline_bridge_points_zyx": [list(point) for point in distribution.bridge_points_zyx],
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
