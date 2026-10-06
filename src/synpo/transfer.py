from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Event
from typing import BinaryIO

from . import __version__
from .models import ProgressCallback
from .preprocessing import project_cache_path
from .project import (
    SCHEMA_VERSION,
    default_detection_manifest,
    default_measurements_manifest,
    default_preprocessing_manifest,
    default_review_manifest,
    migrate_manifest,
    relink_project_sources,
)


TRANSFER_FORMAT = "synpo-transfer"
TRANSFER_VERSION = 1
TRANSFER_SUFFIX = ".synpo-transfer.zip"
METADATA_MEMBER = "synpo-transfer.json"


def _current_algorithm_versions() -> dict[str, int]:
    return {
        "preprocessing": int(default_preprocessing_manifest()["algorithm_version"]),
        "detection": int(default_detection_manifest()["algorithm_version"]),
        "review": int(default_review_manifest()["algorithm_version"]),
        "measurements": int(default_measurements_manifest()["algorithm_version"]),
    }


class TransferError(ValueError):
    """Base class for safe, user-facing transfer failures."""


class TransferValidationError(TransferError):
    """The archive is malformed or failed an integrity check."""


class TransferCacheError(TransferValidationError):
    """The full-state cache is missing or damaged and can be discarded safely."""


class TransferCancelled(TransferError):
    """The user cancelled at a safe file boundary."""


@dataclass(frozen=True)
class TransferInfo:
    mode: str
    include_raw: bool
    project_name: str
    project_id: str
    created_at: str
    uncompressed_size: int


@dataclass(frozen=True)
class TransferImportResult:
    project_path: Path
    manifest: dict[str, object]
    mode: str
    included_raw: bool
    recovered_as_settings_only: bool
    renamed_source_matches: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_project_filename(path: str | Path) -> str:
    name = Path(path).name
    if not name.casefold().endswith(".synpo.json"):
        name = f"{Path(name).stem or 'project'}.synpo.json"
    return (
        re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
        or "project.synpo.json"
    )


def _safe_folder_name(project_name: str) -> str:
    stem = project_name[: -len(".synpo.json")]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-") or "Synpo-project"


def _validate_member_name(name: str) -> None:
    if not name or "\\" in name:
        raise TransferValidationError(f"Unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TransferValidationError(f"Unsafe archive path: {name!r}")


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TransferCancelled("Transfer cancelled; no partial project was kept.")


def _copy_stream(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    hasher: object,
    transferred: list[int],
    total: int,
    phase: str,
    detail: str,
    progress: ProgressCallback | None,
    cancel_event: Event | None,
) -> None:
    while True:
        _check_cancel(cancel_event)
        block = source.read(1024 * 1024)
        if not block:
            break
        destination.write(block)
        hasher.update(block)  # type: ignore[attr-defined]
        transferred[0] += len(block)
        if progress:
            progress(phase, transferred[0], max(1, total), detail)


def settings_only_manifest(manifest: dict[str, object]) -> dict[str, object]:
    """Copy project setup while removing all derived and correction state."""
    result = migrate_manifest(copy.deepcopy(manifest))
    result["cache"] = {
        "format": str(result.get("cache", {}).get("format", "zarr-v2-blosc-zstd")),
        "path": None,
        "deletion_eligible": False,
    }
    for specimen in result.get("specimens", []):
        excluded = bool(specimen.get("analysis", {}).get("excluded", False))
        specimen["checkpoints"] = {
            "preprocessing": {
                "state": "excluded" if excluded else "not_started",
                "channels": {},
                "updated_at": None,
            },
            "detection": {"state": "not_started", "updated_at": None},
            "review": {
                "state": "not_started",
                "updated_at": None,
                "edit_count": 0,
            },
            "measurements": {"state": "not_started", "updated_at": None},
        }
        review = specimen.setdefault("review", {})
        comment = str(review.get("comment", ""))
        specimen["review"] = {
            "state": "needs_attention",
            "comment": comment,
            "history": [],
            "object_status": {"dendrite": {}, "spine": {}},
        }
        specimen["distribution_review"] = {"spines": {}, "updated_at": None}
        specimen["morphology_review"] = {"spines": {}, "updated_at": None}
    result["morphology_analysis"] = {"runs": [], "active_run_id": None}
    return result


def _raw_sources(
    manifest: dict[str, object],
) -> list[tuple[int, str, dict[str, object], Path]]:
    values: list[tuple[int, str, dict[str, object], Path]] = []
    for specimen_index, specimen in enumerate(manifest.get("specimens", [])):
        for channel, channel_data in specimen.get("channels", {}).items():
            saved = str(channel_data.get("source_path", "")).strip()
            path = Path(saved) if saved else Path(str(channel_data["filename"]))
            if not path.is_absolute():
                path = Path(str(manifest["source_directory"])) / path
            values.append(
                (
                    specimen_index,
                    str(channel),
                    channel_data,
                    path.expanduser().resolve(),
                )
            )
    return values


def _validate_full_cache(manifest: dict[str, object], root: Path) -> None:
    def require_tree(relative: str, description: str) -> None:
        path = root / relative
        if not path.is_dir() or not any(item.is_file() for item in path.rglob("*")):
            raise TransferCacheError(
                f"The {description} cache is missing or incomplete ({relative})."
            )

    specimens = manifest.get("specimens", [])
    if any(s["checkpoints"]["preprocessing"].get("state") == "complete" for s in specimens):
        require_tree("preprocessed.zarr", "preprocessing")
    if any(s["checkpoints"]["detection"].get("state") == "complete" for s in specimens):
        require_tree("detection.zarr", "detection")
    if any(int(s["checkpoints"]["review"].get("edit_count", 0)) > 0 for s in specimens):
        require_tree("review.zarr", "manual-correction")
    for index, specimen in enumerate(specimens):
        if specimen["checkpoints"].get("measurements", {}).get("state") == "complete":
            path = root / "measurements" / f"specimen-{index:04d}.json.gz"
            if not path.is_file():
                raise TransferCacheError(
                    f"The measurement cache is missing for specimen {index + 1}."
                )


def create_transfer_archive(
    manifest: dict[str, object],
    project_path: str | Path,
    archive_path: str | Path,
    *,
    mode: str = "full",
    include_raw: bool = False,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> Path:
    if mode not in {"full", "settings_only"}:
        raise TransferError("Transfer mode must be 'full' or 'settings_only'.")
    source_manifest = migrate_manifest(copy.deepcopy(manifest))
    archive_manifest = (
        source_manifest if mode == "full" else settings_only_manifest(source_manifest)
    )
    project_name = _safe_project_filename(project_path)
    project_member = f"project/{project_name}"

    sources: list[tuple[str, Path, str]] = []
    if mode == "full":
        cache_root = project_cache_path(source_manifest).expanduser().resolve().parent
        _validate_full_cache(source_manifest, cache_root)
        if cache_root.is_dir():
            for path in sorted(cache_root.rglob("*"), key=lambda value: str(value).casefold()):
                if path.is_symlink():
                    raise TransferError(f"Cache contains an unsupported symbolic link: {path}")
                if path.is_file():
                    relative = path.relative_to(cache_root).as_posix()
                    sources.append((f"cache/{relative}", path, "cache"))

    raw_records: list[dict[str, object]] = []
    if include_raw:
        for specimen_index, channel, channel_data, path in _raw_sources(source_manifest):
            if not path.is_file():
                raise TransferError(f"Raw TIFF is missing: {path}")
            fingerprint = channel_data.get("fingerprint", {})
            if path.stat().st_size != int(fingerprint.get("size_bytes", -1)):
                raise TransferError(f"Raw TIFF size differs from the project fingerprint: {path}")
            saved_hash = str(fingerprint.get("sha256", ""))
            if not saved_hash:
                raise TransferError(f"Raw TIFF has no saved SHA-256 fingerprint: {path}")
            safe_raw_name = re.sub(r"[^A-Za-z0-9._-]+", "-", path.name).strip(".-")
            member = (
                f"raw/specimen-{specimen_index:04d}/{channel}/"
                f"{safe_raw_name or 'source.tif'}"
            )
            sources.append((member, path, "raw"))
            raw_records.append(
                {
                    "specimen_index": specimen_index,
                    "channel": channel,
                    "member": member,
                    "filename": str(channel_data.get("filename", path.name)),
                    "size": path.stat().st_size,
                    "sha256": saved_hash,
                }
            )

    project_bytes = json.dumps(archive_manifest, indent=2).encode("utf-8")
    total = len(project_bytes) + sum(path.stat().st_size for _, path, _ in sources)
    records: list[dict[str, object]] = []
    destination = Path(archive_path).expanduser().resolve()
    if not destination.name.casefold().endswith(TRANSFER_SUFFIX):
        destination = destination.with_name(destination.stem + TRANSFER_SUFFIX)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    transferred = [0]
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr(project_member, project_bytes)
            transferred[0] += len(project_bytes)
            records.append(
                {
                    "path": project_member,
                    "kind": "project",
                    "size": len(project_bytes),
                    "sha256": _sha256_bytes(project_bytes),
                }
            )
            if progress:
                progress("Creating transfer ZIP", transferred[0], max(1, total), project_name)
            for member, path, kind in sources:
                _check_cancel(cancel_event)
                digest = hashlib.sha256()
                with path.open("rb") as source, archive.open(
                    member, "w", force_zip64=True
                ) as target:
                    _copy_stream(
                        source,
                        target,
                        hasher=digest,
                        transferred=transferred,
                        total=total,
                        phase="Creating transfer ZIP",
                        detail=path.name,
                        progress=progress,
                        cancel_event=cancel_event,
                    )
                calculated = digest.hexdigest()
                if kind == "raw":
                    expected = next(
                        record["sha256"]
                        for record in raw_records
                        if record["member"] == member
                    )
                    if calculated != expected:
                        raise TransferError(
                            "Raw TIFF checksum differs from the project fingerprint: "
                            f"{path}"
                        )
                records.append(
                    {
                        "path": member,
                        "kind": kind,
                        "size": path.stat().st_size,
                        "sha256": calculated,
                    }
                )
            metadata = {
                "format": TRANSFER_FORMAT,
                "format_version": TRANSFER_VERSION,
                "created_at": _utc_now(),
                "created_with_version": __version__,
                "mode": mode,
                "include_raw": include_raw,
                "project_id": str(source_manifest["project_id"]),
                "project_member": project_member,
                "project_filename": project_name,
                "algorithm_versions": _current_algorithm_versions(),
                "raw_files": raw_records,
                "files": records,
            }
            archive.writestr(
                METADATA_MEMBER, json.dumps(metadata, indent=2).encode("utf-8")
            )
        _check_cancel(cancel_event)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _read_metadata(
    archive: zipfile.ZipFile, *, allow_damaged_cache: bool = False
) -> dict[str, object]:
    names = archive.namelist()
    for name in names:
        _validate_member_name(name)
    if len(names) != len(set(names)):
        raise TransferValidationError("The transfer ZIP contains duplicate archive paths.")
    try:
        metadata = json.loads(archive.read(METADATA_MEMBER).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransferValidationError(f"Transfer metadata cannot be read: {exc}") from exc
    if metadata.get("format") != TRANSFER_FORMAT:
        raise TransferValidationError("This ZIP is not a Synpo transfer archive.")
    if metadata.get("format_version") != TRANSFER_VERSION:
        raise TransferValidationError(
            f"Unsupported transfer format {metadata.get('format_version')!r}; expected {TRANSFER_VERSION}."
        )
    if metadata.get("mode") not in {"full", "settings_only"}:
        raise TransferValidationError("The transfer archive has an invalid mode.")
    if metadata.get("mode") == "full":
        algorithms = metadata.get("algorithm_versions")
        if algorithms != _current_algorithm_versions() and not allow_damaged_cache:
            raise TransferCacheError(
                "The cached results were created with incompatible processing algorithms."
            )
    members = set(names)
    records = metadata.get("files")
    if not isinstance(records, list):
        raise TransferValidationError("Transfer metadata has no file inventory.")
    recorded: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TransferValidationError("Transfer file inventory is malformed.")
        member = str(record.get("path", ""))
        _validate_member_name(member)
        if member in recorded or member not in members:
            raise TransferValidationError(f"Transfer file inventory is inconsistent: {member}")
        kind = str(record.get("kind", ""))
        if kind not in {"project", "cache", "raw"}:
            raise TransferValidationError(f"Transfer file has an invalid kind: {kind!r}")
        expected_prefix = {"project": "project/", "cache": "cache/", "raw": "raw/"}[kind]
        if not member.startswith(expected_prefix):
            raise TransferValidationError(f"Transfer file is stored in the wrong location: {member}")
        try:
            size = int(record.get("size", -1))
        except (TypeError, ValueError) as exc:
            raise TransferValidationError(f"Transfer file size is invalid: {member}") from exc
        if size < 0 or archive.getinfo(member).file_size != size:
            if kind == "cache" and allow_damaged_cache:
                pass
            elif kind == "cache":
                raise TransferCacheError(
                    f"Cached file size inventory is inconsistent: {member}"
                )
            else:
                raise TransferValidationError(
                    f"Transfer file size inventory is inconsistent: {member}"
                )
        checksum = str(record.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise TransferValidationError(f"Transfer checksum is invalid: {member}")
        recorded.add(member)
    project_member = str(metadata.get("project_member", ""))
    if project_member not in recorded:
        raise TransferValidationError("The archived project is missing from the inventory.")
    project_records = [record for record in records if record.get("kind") == "project"]
    if len(project_records) != 1:
        raise TransferValidationError("The transfer must contain exactly one project JSON.")
    raw_records = metadata.get("raw_files", [])
    if not isinstance(raw_records, list):
        raise TransferValidationError("The raw TIFF inventory is malformed.")
    raw_members = {
        str(record["path"]) for record in records if record.get("kind") == "raw"
    }
    mapped_raw_members = {
        str(record.get("member", "")) for record in raw_records if isinstance(record, dict)
    }
    if raw_members != mapped_raw_members:
        raise TransferValidationError("The raw TIFF mapping is incomplete or inconsistent.")
    if bool(metadata.get("include_raw", False)) != bool(raw_members):
        raise TransferValidationError("The raw TIFF inclusion flag is inconsistent.")
    return metadata


def inspect_transfer_archive(path: str | Path) -> TransferInfo:
    source = Path(path).expanduser().resolve()
    try:
        with zipfile.ZipFile(source, "r") as archive:
            metadata = _read_metadata(archive, allow_damaged_cache=True)
            total = sum(int(record.get("size", 0)) for record in metadata["files"])
    except (OSError, zipfile.BadZipFile) as exc:
        raise TransferValidationError(f"Cannot open transfer ZIP: {exc}") from exc
    return TransferInfo(
        mode=str(metadata["mode"]),
        include_raw=bool(metadata.get("include_raw", False)),
        project_name=str(metadata.get("project_filename", "project.synpo.json")),
        project_id=str(metadata.get("project_id", "")),
        created_at=str(metadata.get("created_at", "")),
        uncompressed_size=total,
    )


def _unique_target(parent: Path, folder_name: str) -> Path:
    target = parent / folder_name
    if not target.exists():
        return target
    number = 2
    while (parent / f"{folder_name}-{number}").exists():
        number += 1
    return parent / f"{folder_name}-{number}"


def _rewrite_cache_paths(manifest: dict[str, object], target: Path, full: bool) -> None:
    cache_root = target / ".synpo-cache" / str(manifest["project_id"])
    preprocessed = cache_root / "preprocessed.zarr"
    manifest.setdefault("cache", {})["path"] = str(preprocessed) if full else None
    manifest["cache"]["deletion_eligible"] = False
    if not full:
        return
    replacements = {
        "preprocessing": preprocessed,
        "detection": cache_root / "detection.zarr",
        "review": cache_root / "review.zarr",
        "measurements": cache_root / "measurements",
    }
    for specimen in manifest.get("specimens", []):
        for stage, new_path in replacements.items():
            checkpoint = specimen.get("checkpoints", {}).get(stage)
            if not isinstance(checkpoint, dict):
                continue
            for key in tuple(checkpoint):
                if key == "cache_path" or key.endswith("_path") and "cache" in key:
                    checkpoint[key] = str(new_path)


def _write_project(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    manifest["updated_at"] = _utc_now()
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def import_transfer_archive(
    archive_path: str | Path,
    destination_parent: str | Path,
    *,
    external_raw_directory: str | Path | None = None,
    conflict_policy: str = "copy",
    recover_as_settings_only: bool = False,
    progress: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> TransferImportResult:
    if conflict_policy not in {"copy", "replace"}:
        raise TransferError("Conflict policy must be 'copy' or 'replace'.")
    source = Path(archive_path).expanduser().resolve()
    parent = Path(destination_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            metadata = _read_metadata(
                archive, allow_damaged_cache=recover_as_settings_only
            )
            project_name = _safe_project_filename(
                str(metadata.get("project_filename", "project.synpo.json"))
            )
            folder_name = _safe_folder_name(project_name)
            nominal_target = (parent / folder_name).resolve()
            if nominal_target.parent != parent:
                raise TransferValidationError("The destination project path is unsafe.")
            target = (
                nominal_target
                if conflict_policy == "replace" or not nominal_target.exists()
                else _unique_target(parent, folder_name).resolve()
            )
            usage = shutil.disk_usage(parent)
            effective_mode = (
                "settings_only" if recover_as_settings_only else str(metadata["mode"])
            )
            required_records = [
                record
                for record in metadata["files"]
                if record["kind"] in {"project", "raw"}
                or (record["kind"] == "cache" and effective_mode == "full")
            ]
            total = sum(int(record.get("size", 0)) for record in required_records)
            if usage.free < total + 16 * 1024 * 1024:
                raise TransferError("There is not enough free disk space to import this transfer.")
            staging: Path | None = (
                parent / f".{folder_name}-import-{uuid.uuid4().hex}"
            )
            staging.mkdir()
            transferred = [0]
            try:
                project_record = next(
                    record
                    for record in metadata["files"]
                    if record["path"] == metadata["project_member"]
                )
                project_bytes = archive.read(str(metadata["project_member"]))
                if len(project_bytes) != int(project_record["size"]) or _sha256_bytes(
                    project_bytes
                ) != str(project_record["sha256"]):
                    raise TransferValidationError(
                        "The archived project JSON failed its integrity check."
                    )
                transferred[0] = len(project_bytes)
                if progress:
                    progress(
                        "Importing transfer ZIP",
                        transferred[0],
                        max(1, total),
                        project_name,
                    )
                try:
                    manifest = json.loads(project_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                    raise TransferValidationError(
                        f"The archived project JSON is invalid: {exc}"
                    ) from exc
                if manifest.get("schema_version") != SCHEMA_VERSION:
                    raise TransferValidationError(
                        f"Unsupported project schema {manifest.get('schema_version')!r}; "
                        f"expected {SCHEMA_VERSION}."
                    )
                required = {
                    "project_id",
                    "source_directory",
                    "output_directory",
                    "specimens",
                }
                missing = required.difference(manifest)
                if missing:
                    raise TransferValidationError(
                        "The archived project is missing required fields: "
                        + ", ".join(sorted(missing))
                    )
                manifest = migrate_manifest(manifest)
                if recover_as_settings_only:
                    manifest = settings_only_manifest(manifest)

                raw_mapping: dict[tuple[int, str], dict[str, object]] = {}
                for raw in metadata.get("raw_files", []):
                    try:
                        key = (int(raw["specimen_index"]), str(raw["channel"]))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise TransferValidationError(
                            "The embedded raw TIFF mapping is malformed."
                        ) from exc
                    if key in raw_mapping:
                        raise TransferValidationError(
                            "The embedded raw TIFF mapping contains duplicates."
                        )
                    raw_mapping[key] = raw
                if bool(metadata.get("include_raw", False)):
                    expected_raw = {
                        (index, str(channel))
                        for index, specimen in enumerate(manifest["specimens"])
                        for channel in specimen.get("channels", {})
                    }
                    if set(raw_mapping) != expected_raw:
                        raise TransferValidationError(
                            "The embedded raw TIFF mapping does not match the project."
                        )
                    inventory_by_member = {
                        str(record["path"]): record
                        for record in metadata["files"]
                        if record["kind"] == "raw"
                    }
                    for (index, channel), raw in raw_mapping.items():
                        channel_data = manifest["specimens"][index]["channels"][channel]
                        fingerprint = channel_data.get("fingerprint", {})
                        inventory = inventory_by_member[str(raw["member"])]
                        if str(raw.get("sha256", "")) != str(
                            fingerprint.get("sha256", "")
                        ) or int(raw.get("size", -1)) != int(
                            fingerprint.get("size_bytes", -2)
                        ) or str(raw.get("sha256", "")) != str(
                            inventory.get("sha256", "")
                        ) or int(raw.get("size", -1)) != int(
                            inventory.get("size", -2)
                        ):
                            raise TransferValidationError(
                                "An embedded raw TIFF does not match its project fingerprint."
                            )

                selected_records = [
                    record
                    for record in metadata["files"]
                    if record["kind"] == "raw"
                    or (record["kind"] == "cache" and effective_mode == "full")
                ]
                for record in selected_records:
                    _check_cancel(cancel_event)
                    member = str(record["path"])
                    if record["kind"] == "cache":
                        relative = PurePosixPath(member).relative_to("cache")
                        output = (
                            staging
                            / ".synpo-cache"
                            / str(manifest["project_id"])
                            / Path(*relative.parts)
                        )
                    else:
                        relative = PurePosixPath(member).relative_to("raw")
                        output = staging / "raw" / Path(*relative.parts)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    try:
                        with archive.open(member, "r") as input_file, output.open(
                            "wb"
                        ) as output_file:
                            _copy_stream(
                                input_file,
                                output_file,
                                hasher=digest,
                                transferred=transferred,
                                total=total,
                                phase="Importing transfer ZIP",
                                detail=output.name,
                                progress=progress,
                                cancel_event=cancel_event,
                            )
                    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                        if record["kind"] == "cache":
                            raise TransferCacheError(
                                f"A cached result could not be extracted: {exc}"
                            ) from exc
                        raise TransferValidationError(
                            f"An archived raw TIFF could not be extracted: {exc}"
                        ) from exc
                    if output.stat().st_size != int(
                        record["size"]
                    ) or digest.hexdigest() != str(record["sha256"]):
                        if record["kind"] == "cache":
                            raise TransferCacheError(
                                f"Cached file failed its integrity check: {member}"
                            )
                        raise TransferValidationError(
                            f"Raw TIFF failed its integrity check: {member}"
                        )

                manifest["output_directory"] = str(target)
                _rewrite_cache_paths(manifest, target, effective_mode == "full")
                if effective_mode == "full":
                    _validate_full_cache(
                        manifest,
                        staging / ".synpo-cache" / str(manifest["project_id"]),
                    )

                included_raw = bool(metadata.get("include_raw", False))
                renamed = 0
                if included_raw:
                    for (index, channel), raw in raw_mapping.items():
                        relative = PurePosixPath(str(raw["member"])).relative_to("raw")
                        final_path = target / "raw" / Path(*relative.parts)
                        manifest["specimens"][index]["channels"][channel]["source_path"] = str(final_path)
                    manifest["source_directory"] = str(target / "raw")
                else:
                    if external_raw_directory is None:
                        raise TransferError(
                            "Select the folder containing this project's raw TIFF files."
                        )
                    results = relink_project_sources(manifest, external_raw_directory)
                    failures = [item for item in results if item["status"] != "ok"]
                    if failures:
                        examples = "; ".join(
                            f"{item['filename']}: {item['detail']}"
                            for item in failures[:4]
                        )
                        raise TransferError(
                            f"Raw TIFF relinking failed for {len(failures)} file(s): {examples}"
                        )
                    renamed = sum(
                        item.get("matched_by") == "size_and_sha256"
                        for item in results
                    )

                project_output = staging / project_name
                _write_project(project_output, manifest)
                _check_cancel(cancel_event)

                backup: Path | None = None
                if target.exists():
                    if conflict_policy != "replace":
                        raise TransferError(f"Destination already exists: {target}")
                    backup = parent / f".{folder_name}-backup-{uuid.uuid4().hex}"
                    target.rename(backup)
                try:
                    staging.rename(target)
                except Exception:
                    if backup is not None and backup.exists() and not target.exists():
                        backup.rename(target)
                    raise
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
                staging = None
                return TransferImportResult(
                    project_path=target / project_name,
                    manifest=manifest,
                    mode=effective_mode,
                    included_raw=included_raw,
                    recovered_as_settings_only=recover_as_settings_only,
                    renamed_source_matches=renamed,
                )
            finally:
                if staging is not None and staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
    except TransferError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError, TypeError, ValueError) as exc:
        raise TransferValidationError(f"Cannot import transfer ZIP: {exc}") from exc
