from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from . import __version__
from .importer import fingerprint_file
from .models import Calibration, ProgressCallback, ScanReport


SCHEMA_VERSION = 1
_PROJECT_SAVE_LOCK = RLock()


def default_preprocessing_manifest() -> dict[str, object]:
    settings = {
        "background_percentile": 20.0,
        "gaussian_sigma_xy_um": 0.07,
        "gaussian_sigma_z_um": 0.0,
        "threshold_sensitivity": 1.0,
    }
    return {
        "algorithm_version": 2,
        "parameter_model": "per_specimen_v1",
        "settings_by_channel": {
            "ChanA": dict(settings),
            "ChanB": dict(settings),
        },
        "representative_specimens": [],
        "special_specimens": [],
        "settings_by_specimen": {},
        "parameters_set_by_specimen": {},
    }


def default_detection_manifest() -> dict[str, object]:
    return {
        "algorithm_version": 2,
        "memory_mode": "automatic",
        "settings_by_specimen": {},
        "settings": {
            "dendrite_sensitivity": 1.25,
            "cluster_sensitivity": 0.65,
            "spine_branch_length_um": 3.0,
            "minimum_dendrite_length_um": 4.0,
            "minimum_spine_projection_pixels": 6,
            "minimum_cluster_voxels": 25,
        },
    }


def default_review_manifest() -> dict[str, object]:
    return {
        "algorithm_version": 2,
        "memory_mode": "automatic",
        "correction_sensitivity": 1.0,
        "local_margin_um": 1.0,
        "z_radius_slices": 2,
        "add_z_radius_slices": 6,
        "maximum_undo_actions": 100,
    }


def default_measurements_manifest() -> dict[str, object]:
    return {
        "algorithm_version": 4,
        "settings": {
            "minimum_cluster_spine_overlap_percent": 80.0,
            "cluster_end_method": "adaptive",
            "fixed_end_slices": 3,
            "adaptive_area_factor": 1.8,
            "minimum_retained_slices": 2,
            "maximum_centerline_gap_um": 1.0,
            "spine_volume_filter_enabled": False,
            "spine_volume_filter_cutoff_um3": 0.0,
        },
    }


def migrate_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Add newly introduced fields without invalidating Stage 1 projects."""
    application = manifest.setdefault("application", {"name": "Synpo"})
    application.setdefault(
        "created_with_version", application.get("version", __version__)
    )
    application["name"] = "Synpo"
    application["version"] = __version__
    manifest.setdefault("resource_policy", {"maximum_ram_fraction": 0.8})
    preprocessing = manifest.setdefault(
        "preprocessing", default_preprocessing_manifest()
    )
    legacy_preprocessing = preprocessing.get("parameter_model") != "per_specimen_v1"
    preprocessing.setdefault("representative_specimens", [])
    preprocessing.setdefault("special_specimens", [])
    preprocessing.setdefault("settings_by_specimen", {})
    preprocessing.setdefault("parameters_set_by_specimen", {})
    manifest.setdefault("detection", default_detection_manifest())
    manifest["detection"].setdefault("memory_mode", "automatic")
    manifest["detection"].setdefault("settings_by_specimen", {})
    review_settings = manifest.setdefault("review_settings", default_review_manifest())
    review_settings["algorithm_version"] = 2
    review_settings.setdefault("memory_mode", "automatic")
    review_settings.setdefault("correction_sensitivity", 1.0)
    review_settings.setdefault("add_z_radius_slices", 6)
    existing_measurements = manifest.get("measurements")
    previous_measurement_algorithm = (
        int(existing_measurements.get("algorithm_version", 0))
        if isinstance(existing_measurements, dict)
        else 0
    )
    measurements = manifest.setdefault("measurements", default_measurements_manifest())
    measurements["algorithm_version"] = 4
    measurements.setdefault("settings", {})
    measurements["settings"].setdefault("maximum_centerline_gap_um", 1.0)
    measurements["settings"].setdefault("spine_volume_filter_enabled", False)
    measurements["settings"].setdefault("spine_volume_filter_cutoff_um3", 0.0)
    morphology_analysis = manifest.setdefault(
        "morphology_analysis", {"runs": [], "active_run_id": None}
    )
    morphology_analysis.setdefault("runs", [])
    morphology_analysis.setdefault("active_run_id", None)
    import_settings = manifest.setdefault(
        "import_settings",
        {
            "mode": "strict",
            "channel_markers": {"ChanA": "ChanA", "ChanB": "ChanB"},
            "default_experimental_group": "Experiment",
        },
    )
    import_settings.setdefault("mode", "strict")
    import_settings.setdefault(
        "channel_markers", {"ChanA": "ChanA", "ChanB": "ChanB"}
    )
    import_settings.setdefault("default_experimental_group", "Experiment")
    cache = manifest.setdefault("cache", {})
    if cache.get("format") == "pending_stage_2":
        cache["format"] = "zarr-v2-blosc-zstd"
    cache.setdefault("format", "zarr-v2-blosc-zstd")
    cache.setdefault("path", None)
    cache.setdefault("deletion_eligible", False)
    legacy_special = {
        int(value) for value in preprocessing.get("special_specimens", [])
    }
    legacy_saved = preprocessing.get("settings_by_specimen", {})
    legacy_defaults = preprocessing.get(
        "settings_by_channel", default_preprocessing_manifest()["settings_by_channel"]
    )
    for specimen_index, specimen in enumerate(manifest.get("specimens", [])):
        for channel_data in specimen.get("channels", {}).values():
            channel_data.setdefault(
                "source_path",
                str(
                    Path(str(manifest.get("source_directory", "")))
                    / str(channel_data.get("filename", ""))
                ),
            )
        checkpoints = specimen.setdefault("checkpoints", {})
        value = checkpoints.get("preprocessing", "not_started")
        if isinstance(value, str):
            checkpoints["preprocessing"] = {
                "state": value,
                "channels": {},
                "updated_at": None,
            }
        checkpoints.setdefault("detection", "not_started")
        detection_value = checkpoints.get("detection", "not_started")
        if isinstance(detection_value, str):
            checkpoints["detection"] = {
                "state": detection_value,
                "updated_at": None,
            }
        checkpoints.setdefault("review", "not_started")
        review_checkpoint = checkpoints.get("review", "not_started")
        if isinstance(review_checkpoint, str):
            checkpoints["review"] = {
                "state": review_checkpoint,
                "updated_at": None,
                "edit_count": 0,
            }
        else:
            review_checkpoint.setdefault("state", "not_started")
            review_checkpoint.setdefault("updated_at", None)
            review_checkpoint.setdefault("edit_count", 0)
        measurement_checkpoint = checkpoints.setdefault("measurements", {})
        if isinstance(measurement_checkpoint, str):
            checkpoints["measurements"] = {
                "state": measurement_checkpoint,
                "updated_at": None,
            }
        else:
            measurement_checkpoint.setdefault("state", "not_started")
            measurement_checkpoint.setdefault("updated_at", None)
        if (
            previous_measurement_algorithm < 4
            and checkpoints["measurements"].get("state") == "complete"
        ):
            checkpoints["measurements"].update(
                {
                    "state": "not_started",
                    "updated_at": None,
                    "reason": "All-spine morphology and calibrated length measurements were added; recalculate measurements.",
                }
            )
        review = specimen.setdefault(
            "review", {"state": "needs_attention", "comment": "", "history": []}
        )
        review.setdefault("state", "needs_attention")
        review.setdefault("comment", "")
        review.setdefault("history", [])
        object_status = review.setdefault(
            "object_status", {"dendrite": {}, "spine": {}}
        )
        object_status.setdefault("dendrite", {})
        object_status.setdefault("spine", {})
        distribution_review = specimen.setdefault(
            "distribution_review", {"spines": {}, "updated_at": None}
        )
        distribution_review.setdefault("spines", {})
        distribution_review.setdefault("updated_at", None)
        morphology_review = specimen.setdefault(
            "morphology_review", {"spines": {}, "updated_at": None}
        )
        morphology_review.setdefault("spines", {})
        morphology_review.setdefault("updated_at", None)
        analysis = specimen.setdefault(
            "analysis", {"excluded": False, "exclusion_reason": "", "rois_xy": []}
        )
        analysis.setdefault("excluded", False)
        analysis.setdefault("exclusion_reason", "")
        analysis.setdefault("rois_xy", [])
        if legacy_preprocessing:
            specimen_settings = legacy_saved.setdefault(str(specimen_index), {})
            for channel in ("ChanA", "ChanB"):
                if not (specimen_index in legacy_special and channel in specimen_settings):
                    specimen_settings[channel] = dict(legacy_defaults[channel])
            completed_channels = checkpoints["preprocessing"].get("channels", {})
            if checkpoints["preprocessing"].get("state") == "complete":
                preprocessing["parameters_set_by_specimen"][str(specimen_index)] = [
                    channel
                    for channel in ("ChanA", "ChanB")
                    if channel in completed_channels
                ] or ["ChanA", "ChanB"]
    preprocessing["parameter_model"] = "per_specimen_v1"
    preprocessing["algorithm_version"] = 2
    return manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_project_manifest(
    report: ScanReport,
    *,
    output_directory: str | Path,
    channel_roles: dict[str, str],
    calibration: Calibration,
) -> dict[str, object]:
    if not report.valid:
        raise ValueError("The import contains errors and cannot be saved as a project.")
    if set(channel_roles) != {"ChanA", "ChanB"} or len(set(channel_roles.values())) != 2:
        raise ValueError("ChanA and ChanB must have different assigned roles.")
    calibration.validate()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    specimens: list[dict[str, object]] = []
    seen_labels: set[tuple[str, str]] = set()
    for pair in report.pairs:
        label_key = (pair.experimental_group.casefold(), pair.specimen_id.casefold())
        if label_key in seen_labels:
            raise ValueError(
                f"Duplicate edited group/specimen label: {pair.experimental_group} / {pair.specimen_id}"
            )
        seen_labels.add(label_key)
        channels: dict[str, object] = {}
        for channel, channel_file in pair.channels.items():
            if channel_file.metadata is None or channel_file.fingerprint is None:
                raise ValueError(f"Missing metadata or fingerprint for {channel_file.filename}")
            channels[channel] = {
                "filename": channel_file.filename,
                "source_path": str(channel_file.path.resolve()),
                "metadata": channel_file.metadata.to_dict(),
                "fingerprint": channel_file.fingerprint.to_dict(),
            }
        specimens.append(
            {
                "batch_prefix": pair.batch_prefix,
                "experimental_group": pair.experimental_group,
                "specimen_id": pair.specimen_id,
                "channels": channels,
                "checkpoints": {
                    "preprocessing": "not_started",
                    "detection": "not_started",
                    "review": "not_started",
                },
                "review": {"state": "needs_attention", "comment": "", "history": []},
            }
        )

    timestamp = _utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "application": {"name": "Synpo", "version": __version__},
        "project_id": str(uuid.uuid4()),
        "created_at": timestamp,
        "updated_at": timestamp,
        "source_directory": str(report.source_directory),
        "output_directory": str(output),
        "batch_prefix": report.pairs[0].batch_prefix,
        "channel_roles": dict(channel_roles),
        "import_settings": {
            "mode": report.import_mode,
            "channel_markers": dict(report.channel_markers),
            "default_experimental_group": report.default_experimental_group,
        },
        "calibration": calibration.to_dict(),
        "resource_policy": {"maximum_ram_fraction": 0.8},
        "preprocessing": default_preprocessing_manifest(),
        "detection": default_detection_manifest(),
        "review_settings": default_review_manifest(),
        "measurements": default_measurements_manifest(),
        "morphology_analysis": {"runs": [], "active_run_id": None},
        "cache": {"format": "zarr-v2-blosc-zstd", "path": None, "deletion_eligible": False},
        "specimens": specimens,
    }
    return migrate_manifest(manifest)


def save_project(path: str | Path, manifest: dict[str, object]) -> Path:
    with _PROJECT_SAVE_LOCK:
        migrate_manifest(manifest)
        destination = Path(path).expanduser().resolve()
        if destination.suffix.lower() != ".json" or not destination.name.lower().endswith(
            ".synpo.json"
        ):
            destination = destination.with_name(destination.stem + ".synpo.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest["updated_at"] = _utc_now()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        return destination


def load_project(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot open project: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    required = {"project_id", "source_directory", "output_directory", "specimens"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Project is missing required fields: {', '.join(sorted(missing))}")
    return migrate_manifest(manifest)


def channel_source_path(
    manifest: dict[str, object], channel_data: dict[str, object]
) -> Path:
    """Resolve a channel source while remaining compatible with older projects."""
    saved = str(channel_data.get("source_path", "")).strip()
    if saved:
        path = Path(saved).expanduser()
        if not path.is_absolute():
            path = Path(str(manifest["source_directory"])) / path
        return path.resolve()
    return (
        Path(str(manifest["source_directory"])) / str(channel_data["filename"])
    ).expanduser().resolve()


def verify_project_sources(
    manifest: dict[str, object],
    *,
    source_directory: str | Path | None = None,
    full_checksums: bool = True,
    allow_renamed: bool = False,
    progress: ProgressCallback | None = None,
) -> list[dict[str, str]]:
    directory = (
        Path(source_directory).expanduser().resolve()
        if source_directory is not None
        else None
    )
    expected: list[tuple[str, dict[str, object]]] = []
    for specimen in manifest["specimens"]:
        for channel, channel_data in specimen["channels"].items():
            expected.append((channel, channel_data))

    results: list[dict[str, str]] = []
    total = len(expected)
    renamed_candidates_by_size: dict[int, list[Path]] = {}
    fingerprint_cache: dict[Path, object] = {}
    if directory is not None and directory.is_dir() and allow_renamed:
        for path in sorted(directory.rglob("*"), key=lambda value: str(value).casefold()):
            if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}:
                renamed_candidates_by_size.setdefault(path.stat().st_size, []).append(path)
    for index, (channel, channel_data) in enumerate(expected, start=1):
        filename = str(channel_data["filename"])
        if directory is None:
            saved_path = channel_source_path(manifest, channel_data)
            candidates = [saved_path] if saved_path.is_file() else []
        else:
            direct = directory / filename
            candidates = [direct] if direct.is_file() else []
            if not candidates and directory.is_dir():
                candidates = sorted(
                    (path for path in directory.rglob(filename) if path.is_file()),
                    key=lambda path: str(path).casefold(),
                )
        if progress:
            progress("Verifying source files", index - 1, total, filename)
        status, detail = "missing", "File not found"
        matched_path: Path | None = None
        saved = channel_data["fingerprint"]
        exact_candidates = list(candidates)
        if allow_renamed and saved.get("sha256"):
            for candidate in renamed_candidates_by_size.get(
                int(saved["size_bytes"]), []
            ):
                if candidate not in candidates:
                    candidates.append(candidate)
        for path in candidates:
            stat = path.stat()
            if stat.st_size != int(saved["size_bytes"]):
                status, detail = "modified", "File size differs"
                continue
            if not full_checksums:
                status, detail = "ok", "File size matches; checksum not recalculated"
                matched_path = path
                break
            current = fingerprint_cache.get(path)
            if current is None:
                current = fingerprint_file(path, include_checksum=True)
                fingerprint_cache[path] = current
            saved_hash = saved.get("sha256")
            if not saved_hash:
                status, detail = "unverified", "Project has no saved checksum"
                matched_path = path
                break
            if current.sha256 == saved_hash:
                status, detail = "ok", "Checksum matches"
                matched_path = path
                break
            status, detail = "modified", "SHA-256 checksum differs"
        duplicate_matches = 0
        if matched_path is not None and allow_renamed and saved.get("sha256"):
            matching_paths: list[Path] = []
            for candidate in candidates:
                if candidate.stat().st_size != int(saved["size_bytes"]):
                    continue
                fingerprint = fingerprint_cache.get(candidate)
                if fingerprint is None:
                    fingerprint = fingerprint_file(candidate, include_checksum=True)
                    fingerprint_cache[candidate] = fingerprint
                if fingerprint.sha256 == saved["sha256"]:
                    matching_paths.append(candidate)
            duplicate_matches = max(0, len(matching_paths) - 1)
            if duplicate_matches:
                detail = (
                    f"Checksum matches; {duplicate_matches + 1} identical copies found, "
                    f"using {matched_path}"
                )
        results.append(
            {
                "filename": filename,
                "channel": channel,
                "status": status,
                "detail": detail,
                "path": str(matched_path or (candidates[0] if candidates else "")),
                "matched_by": (
                    "filename"
                    if matched_path is not None and matched_path in exact_candidates
                    else "size_and_sha256"
                    if matched_path is not None
                    else "none"
                ),
                "duplicate_matches": duplicate_matches,
            }
        )
        if progress:
            progress("Verifying source files", index, total, filename)
    return results


def relink_project_sources(
    manifest: dict[str, object],
    new_source_directory: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> list[dict[str, str]]:
    directory = Path(new_source_directory).expanduser().resolve()
    results = verify_project_sources(
        manifest,
        source_directory=directory,
        full_checksums=True,
        allow_renamed=True,
        progress=progress,
    )
    if all(item["status"] == "ok" for item in results):
        result_index = 0
        for specimen in manifest["specimens"]:
            for channel_data in specimen["channels"].values():
                channel_data["source_path"] = results[result_index]["path"]
                result_index += 1
        manifest["source_directory"] = str(directory)
        manifest["updated_at"] = _utc_now()
    return results
