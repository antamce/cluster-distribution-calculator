from __future__ import annotations

import hashlib
import json
import math
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

from .models import ProgressCallback
from .project import save_project


ALGORITHM_VERSION = 1
ProgressDetailCallback = Callable[[str, int, int, str], None]


class ProcessingCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class PreprocessingSettings:
    background_percentile: float = 20.0
    gaussian_sigma_xy_um: float = 0.07
    gaussian_sigma_z_um: float = 0.0
    threshold_sensitivity: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.background_percentile <= 99.9:
            raise ValueError("Background percentile must be between 0 and 99.9.")
        if self.gaussian_sigma_xy_um < 0 or self.gaussian_sigma_z_um < 0:
            raise ValueError("Gaussian smoothing values cannot be negative.")
        if not 0.1 <= self.threshold_sensitivity <= 3.0:
            raise ValueError("Threshold sensitivity must be between 0.1 and 3.0.")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PreprocessingSettings":
        result = cls(
            background_percentile=float(value.get("background_percentile", 20.0)),
            gaussian_sigma_xy_um=float(value.get("gaussian_sigma_xy_um", 0.07)),
            gaussian_sigma_z_um=float(value.get("gaussian_sigma_z_um", 0.0)),
            threshold_sensitivity=float(value.get("threshold_sensitivity", 1.0)),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class StackStatistics:
    background: float
    otsu_threshold: float
    applied_threshold: float
    raw_low: float
    raw_high: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PreviewResult:
    z_index: int
    z_count: int
    raw: np.ndarray
    processed: np.ndarray
    threshold: float
    background: float
    raw_low: float
    raw_high: float


@dataclass(frozen=True)
class CachedStackResult:
    dataset_key: str
    slices_written: int
    resumed_from: int
    elapsed_seconds: float
    statistics: StackStatistics


def _cancel_if_requested(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled("Preprocessing was cancelled. The partial cache can be resumed.")


def _z_count_and_shape(channel_data: dict[str, object]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in channel_data["metadata"]["shape"])
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) != 3:
        raise ValueError(f"Only 2D or 3D grayscale TIFF data are supported; found shape {shape}.")
    return shape


def _read_series_slice(series: tifffile.TiffPageSeries, z_index: int) -> np.ndarray:
    array = np.asarray(series.asarray(key=z_index))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale XY slice, but read shape {array.shape}.")
    return array


def load_raw_slice(path: str | Path, z_index: int) -> tuple[np.ndarray, int]:
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = tuple(int(value) for value in series.shape)
        z_count = 1 if len(shape) == 2 else shape[0]
        if not 0 <= z_index < z_count:
            raise IndexError(f"Z slice {z_index} is outside 0..{z_count - 1}.")
        if z_count == 1:
            array = np.squeeze(np.asarray(series.asarray()))
        else:
            array = _read_series_slice(series, z_index)
    if array.ndim != 2:
        raise ValueError(f"Expected a grayscale XY slice, but read shape {array.shape}.")
    return array, z_count


def _sample_indices(z_count: int, count: int = 12) -> np.ndarray:
    return np.unique(np.linspace(0, z_count - 1, min(z_count, count), dtype=int))


def _sample_stride(y: int, x: int, target_pixels: int = 200_000) -> int:
    return max(1, int(math.ceil(math.sqrt((y * x) / target_pixels))))


def _otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    maximum = float(values.max())
    if maximum <= 0:
        return 0.0
    hist, edges = np.histogram(values, bins=1024, range=(0.0, maximum))
    probabilities = hist.astype(np.float64)
    total = probabilities.sum()
    if total == 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) * 0.5
    cumulative_weight = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * centers)
    denominator = cumulative_weight * (total - cumulative_weight)
    between = np.zeros_like(denominator)
    valid = denominator > 0
    between[valid] = (
        (cumulative_mean[-1] * cumulative_weight[valid] - cumulative_mean[valid] * total)
        ** 2
        / denominator[valid]
    )
    return float(centers[int(np.argmax(between))])


def _physical_sigmas(
    settings: PreprocessingSettings, xy_um_per_pixel: float, z_step_um: float
) -> tuple[float, float]:
    if xy_um_per_pixel <= 0 or z_step_um <= 0:
        raise ValueError("Voxel calibration values must be greater than zero.")
    return (
        settings.gaussian_sigma_z_um / z_step_um,
        settings.gaussian_sigma_xy_um / xy_um_per_pixel,
    )


def _process_plane_or_slab(
    slab: np.ndarray,
    *,
    background: float,
    sigma_z: float,
    sigma_xy: float,
    center: int,
) -> np.ndarray:
    work = np.asarray(slab, dtype=np.float32)
    np.subtract(work, np.float32(background), out=work)
    np.maximum(work, 0.0, out=work)
    if work.ndim == 2:
        if sigma_xy > 0:
            work = ndimage.gaussian_filter(work, sigma=sigma_xy, mode="nearest")
        return work
    if sigma_z > 0 or sigma_xy > 0:
        work = ndimage.gaussian_filter(
            work, sigma=(sigma_z, sigma_xy, sigma_xy), mode="nearest"
        )
    return work[center]


def estimate_stack_statistics(
    path: str | Path,
    settings: PreprocessingSettings,
    *,
    xy_um_per_pixel: float,
    z_step_um: float,
    cancel_event: Event | None = None,
) -> StackStatistics:
    settings.validate()
    sigma_z, sigma_xy = _physical_sigmas(settings, xy_um_per_pixel, z_step_um)
    raw_samples: list[np.ndarray] = []
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = tuple(int(value) for value in series.shape)
        z_count, y, x = (1, *shape) if len(shape) == 2 else shape
        stride = _sample_stride(y, x)
        for z_index in _sample_indices(z_count):
            _cancel_if_requested(cancel_event)
            raw = (
                np.squeeze(np.asarray(series.asarray()))
                if z_count == 1
                else _read_series_slice(series, int(z_index))
            )
            raw_samples.append(np.asarray(raw[::stride, ::stride]))
    sampled_raw = np.concatenate([sample.reshape(-1) for sample in raw_samples])
    background = float(np.percentile(sampled_raw, settings.background_percentile))
    raw_low, raw_high = (
        float(value) for value in np.percentile(sampled_raw, (0.5, 99.8))
    )

    processed_samples: list[np.ndarray] = []
    for raw in raw_samples:
        # Statistics use XY smoothing. Z smoothing is applied during preview/cache writing.
        plane = np.asarray(raw, dtype=np.float32)
        plane = np.maximum(plane - background, 0.0)
        if sigma_xy / max(1, stride) > 0:
            plane = ndimage.gaussian_filter(
                plane, sigma=sigma_xy / max(1, stride), mode="nearest"
            )
        processed_samples.append(plane.reshape(-1))
    otsu = _otsu_threshold(np.concatenate(processed_samples))
    return StackStatistics(
        background=background,
        otsu_threshold=otsu,
        # Higher sensitivity deliberately lowers the cutoff and retains more candidates.
        applied_threshold=otsu / settings.threshold_sensitivity,
        raw_low=raw_low,
        raw_high=max(raw_low + 1.0, raw_high),
    )


def make_preview(
    path: str | Path,
    z_index: int,
    settings: PreprocessingSettings,
    *,
    xy_um_per_pixel: float,
    z_step_um: float,
    statistics: StackStatistics | None = None,
    cancel_event: Event | None = None,
) -> PreviewResult:
    stats = statistics or estimate_stack_statistics(
        path,
        settings,
        xy_um_per_pixel=xy_um_per_pixel,
        z_step_um=z_step_um,
        cancel_event=cancel_event,
    )
    sigma_z, sigma_xy = _physical_sigmas(settings, xy_um_per_pixel, z_step_um)
    radius = int(math.ceil(3 * sigma_z)) if sigma_z > 0 else 0
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = tuple(int(value) for value in series.shape)
        z_count = 1 if len(shape) == 2 else shape[0]
        if not 0 <= z_index < z_count:
            raise IndexError(f"Z slice {z_index} is outside 0..{z_count - 1}.")
        raw = (
            np.squeeze(np.asarray(series.asarray()))
            if z_count == 1
            else _read_series_slice(series, z_index)
        )
        if radius:
            indices = [min(max(index, 0), z_count - 1) for index in range(z_index - radius, z_index + radius + 1)]
            slab = np.stack([_read_series_slice(series, index) for index in indices])
            processed = _process_plane_or_slab(
                slab,
                background=stats.background,
                sigma_z=sigma_z,
                sigma_xy=sigma_xy,
                center=radius,
            )
        else:
            processed = _process_plane_or_slab(
                raw,
                background=stats.background,
                sigma_z=0.0,
                sigma_xy=sigma_xy,
                center=0,
            )
    return PreviewResult(
        z_index=z_index,
        z_count=z_count,
        raw=np.asarray(raw),
        processed=processed,
        threshold=stats.applied_threshold,
        background=stats.background,
        raw_low=stats.raw_low,
        raw_high=stats.raw_high,
    )


def settings_signature(
    settings: PreprocessingSettings,
    *,
    source_sha256: str | None,
    xy_um_per_pixel: float,
    z_step_um: float,
) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "settings": settings.to_dict(),
        "source_sha256": source_sha256,
        "xy_um_per_pixel": xy_um_per_pixel,
        "z_step_um": z_step_um,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _enforce_ram_policy(shape: tuple[int, int], radius: int, fraction: float) -> None:
    if not 0 < fraction <= 0.8:
        raise ValueError("Maximum RAM fraction must be greater than zero and no more than 0.8.")
    y, x = shape
    planes = max(1, 2 * radius + 1)
    estimated_extra = y * x * (2 * planes + 4 * planes + 4) + 64 * 1024 * 1024
    process_rss = psutil.Process().memory_info().rss
    limit = int(psutil.virtual_memory().total * fraction)
    if process_rss + estimated_extra > limit:
        raise MemoryError(
            "This operation would exceed Synpo's RAM safety limit. Close other Synpo "
            "views or reduce Z smoothing before trying again."
        )


def process_stack_to_cache(
    path: str | Path,
    cache_path: str | Path,
    dataset_key: str,
    settings: PreprocessingSettings,
    *,
    xy_um_per_pixel: float,
    z_step_um: float,
    source_sha256: str | None,
    maximum_ram_fraction: float = 0.8,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> CachedStackResult:
    started = time.monotonic()
    settings.validate()
    sigma_z, sigma_xy = _physical_sigmas(settings, xy_um_per_pixel, z_step_um)
    radius = int(math.ceil(3 * sigma_z)) if sigma_z > 0 else 0
    signature = settings_signature(
        settings,
        source_sha256=source_sha256,
        xy_um_per_pixel=xy_um_per_pixel,
        z_step_um=z_step_um,
    )
    statistics = estimate_stack_statistics(
        path,
        settings,
        xy_um_per_pixel=xy_um_per_pixel,
        z_step_um=z_step_um,
        cancel_event=cancel_event,
    )

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(cache_path), mode="a")
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        shape = tuple(int(value) for value in series.shape)
        z_count, y, x = (1, *shape) if len(shape) == 2 else shape
        _enforce_ram_policy((y, x), radius, maximum_ram_fraction)

        if dataset_key in root:
            existing = root[dataset_key]
            valid_existing = (
                tuple(existing.shape) == (z_count, y, x)
                and existing.attrs.get("settings_signature") == signature
            )
            if not valid_existing:
                del root[dataset_key]

        if dataset_key not in root:
            dataset = root.create_dataset(
                dataset_key,
                shape=(z_count, y, x),
                chunks=(1, min(512, y), min(512, x)),
                dtype="uint16",
                compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
                overwrite=False,
            )
            dataset.attrs.update(
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "settings_signature": signature,
                    "settings": settings.to_dict(),
                    "statistics": statistics.to_dict(),
                    "source_sha256": source_sha256 or "",
                    "slices_completed": 0,
                    "complete": False,
                }
            )
        else:
            dataset = root[dataset_key]

        start_z = int(dataset.attrs.get("slices_completed", 0))
        if bool(dataset.attrs.get("complete", False)) and start_z >= z_count:
            return CachedStackResult(dataset_key, 0, z_count, time.monotonic() - started, statistics)

        slice_cache: dict[int, np.ndarray] = {}

        def read(index: int) -> np.ndarray:
            bounded = min(max(index, 0), z_count - 1)
            if bounded not in slice_cache:
                slice_cache[bounded] = (
                    np.squeeze(np.asarray(series.asarray()))
                    if z_count == 1
                    else _read_series_slice(series, bounded)
                )
            return slice_cache[bounded]

        for z_index in range(start_z, z_count):
            _cancel_if_requested(cancel_event)
            if radius:
                indices = range(z_index - radius, z_index + radius + 1)
                slab = np.stack([read(index) for index in indices])
                processed = _process_plane_or_slab(
                    slab,
                    background=statistics.background,
                    sigma_z=sigma_z,
                    sigma_xy=sigma_xy,
                    center=radius,
                )
                keep = {min(max(index, 0), z_count - 1) for index in range(z_index - radius + 1, z_index + radius + 2)}
                slice_cache = {key: value for key, value in slice_cache.items() if key in keep}
            else:
                processed = _process_plane_or_slab(
                    read(z_index),
                    background=statistics.background,
                    sigma_z=0.0,
                    sigma_xy=sigma_xy,
                    center=0,
                )
                slice_cache.clear()
            dataset[z_index, :, :] = np.clip(np.rint(processed), 0, 65535).astype(np.uint16)
            dataset.attrs["slices_completed"] = z_index + 1
            if progress:
                elapsed = max(0.001, time.monotonic() - started)
                completed = z_index - start_z + 1
                eta = elapsed / completed * (z_count - z_index - 1)
                progress(
                    "Preprocessing",
                    z_index + 1,
                    z_count,
                    f"slice {z_index + 1}/{z_count}; about {eta:.0f} s remaining",
                )
        dataset.attrs["complete"] = True
        dataset.attrs["completed_at"] = time.time()

    return CachedStackResult(
        dataset_key=dataset_key,
        slices_written=z_count - start_z,
        resumed_from=start_z,
        elapsed_seconds=time.monotonic() - started,
        statistics=statistics,
    )


def project_cache_path(manifest: dict[str, object]) -> Path:
    saved = manifest.get("cache", {}).get("path")
    if saved:
        return Path(str(saved))
    return (
        Path(str(manifest["output_directory"]))
        / ".synpo-cache"
        / str(manifest["project_id"])
        / "preprocessed.zarr"
    )


def process_project_cache(
    manifest: dict[str, object],
    project_path: str | Path,
    *,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> dict[str, object]:
    cache_path = project_cache_path(manifest)
    manifest["cache"].update(
        {"format": "zarr-v2-blosc-zstd", "path": str(cache_path), "deletion_eligible": False}
    )
    calibration = manifest["calibration"]
    xy = float(calibration["xy_um_per_pixel"])
    z_step = float(calibration["z_step_um"])
    ram_fraction = min(0.8, float(manifest["resource_policy"].get("maximum_ram_fraction", 0.8)))
    specimens = manifest["specimens"]
    total_slices = sum(
        _z_count_and_shape(channel_data)[0]
        for specimen in specimens
        for channel_data in specimen["channels"].values()
    )
    completed_before = 0
    started = time.monotonic()
    channel_results: list[dict[str, object]] = []

    for specimen_index, specimen in enumerate(specimens):
        checkpoint = specimen["checkpoints"]["preprocessing"]
        checkpoint["state"] = "in_progress"
        for channel in ("ChanA", "ChanB"):
            _cancel_if_requested(cancel_event)
            channel_data = specimen["channels"][channel]
            z_count, _, _ = _z_count_and_shape(channel_data)
            settings = PreprocessingSettings.from_dict(
                manifest["preprocessing"]["settings_by_channel"][channel]
            )
            filename = str(channel_data["filename"])
            source = Path(str(manifest["source_directory"])) / filename
            dataset_key = f"specimens/{specimen_index:04d}/{channel}/data"

            def channel_progress(phase: str, current: int, total: int, detail: str) -> None:
                if progress:
                    elapsed = max(0.001, time.monotonic() - started)
                    overall = completed_before + current
                    eta = elapsed / max(1, overall) * max(0, total_slices - overall)
                    progress(
                        phase,
                        overall,
                        total_slices,
                        f"{specimen['specimen_id']} {channel}: {detail} | batch ETA {eta / 60:.1f} min",
                    )

            result = process_stack_to_cache(
                source,
                cache_path,
                dataset_key,
                settings,
                xy_um_per_pixel=xy,
                z_step_um=z_step,
                source_sha256=channel_data["fingerprint"].get("sha256"),
                maximum_ram_fraction=ram_fraction,
                progress=channel_progress,
                cancel_event=cancel_event,
            )
            completed_before += z_count
            checkpoint["channels"][channel] = {
                "state": "complete",
                "dataset_key": dataset_key,
                "settings_signature": settings_signature(
                    settings,
                    source_sha256=channel_data["fingerprint"].get("sha256"),
                    xy_um_per_pixel=xy,
                    z_step_um=z_step,
                ),
                "statistics": result.statistics.to_dict(),
                "elapsed_seconds": result.elapsed_seconds,
            }
            checkpoint["state"] = (
                "complete"
                if set(checkpoint["channels"]) == {"ChanA", "ChanB"}
                else "in_progress"
            )
            checkpoint["updated_at"] = time.time()
            save_project(project_path, manifest)
            channel_results.append(
                {
                    "specimen_index": specimen_index,
                    "channel": channel,
                    "slices_written": result.slices_written,
                    "elapsed_seconds": result.elapsed_seconds,
                }
            )
    return {
        "cache_path": str(cache_path),
        "total_slices": total_slices,
        "elapsed_seconds": time.monotonic() - started,
        "channels": channel_results,
    }


def load_cached_slice(
    manifest: dict[str, object], specimen_index: int, channel: str, z_index: int
) -> np.ndarray:
    root = zarr.open_group(str(project_cache_path(manifest)), mode="r")
    dataset_key = f"specimens/{specimen_index:04d}/{channel}/data"
    if dataset_key not in root:
        raise KeyError("This channel has not been preprocessed yet.")
    dataset = root[dataset_key]
    return np.asarray(dataset[z_index, :, :])
