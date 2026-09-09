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
        "algorithm_version": 1,
        "settings_by_channel": {
            "ChanA": dict(settings),
            "ChanB": dict(settings),
        },
        "representative_specimens": [],
        "special_specimens": [],
        "settings_by_specimen": {},
    }


def default_detection_manifest() -> dict[str, object]:
    return {
        "algorithm_version": 2,
        "memory_mode": "automatic",
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
        "algorithm_version": 2,
        "settings": {
            "minimum_cluster_spine_overlap_percent": 80.0,
            "cluster_end_method": "adaptive",
            "fixed_end_slices": 3,
            "adaptive_area_factor": 1.8,
            "minimum_retained_slices": 2,
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
    preprocessing.setdefault("representative_specimens", [])
    preprocessing.setdefault("special_specimens", [])
    preprocessing.setdefault("settings_by_specimen", {})
    manifest.setdefault("detection", default_detection_manifest())
    manifest["detection"].setdefault("memory_mode", "automatic")
    review_settings = manifest.setdefault("review_settings", default_review_manifest())
    review_settings["algorithm_version"] = 2
    review_settings.setdefault("memory_mode", "automatic")
    review_settings.setdefault("correction_sensitivity", 1.0)
    review_settings.setdefault("add_z_radius_slices", 6)
    manifest.setdefault("measurements", default_measurements_manifest())
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
    for specimen in manifest.get("specimens", []):
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
        for path in candidates:
            stat = path.stat()
            if stat.st_size != int(saved["size_bytes"]):
                status, detail = "modified", "File size differs"
                continue
            if not full_checksums:
                status, detail = "ok", "File size matches; checksum not recalculated"
                matched_path = path
                break
            current = fingerprint_file(path, include_checksum=True)
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
        results.append(
            {
                "filename": filename,
                "channel": channel,
                "status": status,
                "detail": detail,
                "path": str(matched_path or (candidates[0] if candidates else "")),
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
