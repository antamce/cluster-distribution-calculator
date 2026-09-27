from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import threading
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import psutil

from . import __version__


DIAGNOSTIC_LEVEL_CORRECTION = "correction"
DIAGNOSTIC_LEVEL_FULL = "full"
VALID_DIAGNOSTIC_LEVELS = {
    DIAGNOSTIC_LEVEL_CORRECTION,
    DIAGNOSTIC_LEVEL_FULL,
}

DiagnosticCallback = Callable[[str, str, dict[str, object]], None]

_PACKAGE_NAMES = (
    "synpo-microscopy",
    "PySide6",
    "numpy",
    "scipy",
    "scikit-image",
    "tifffile",
    "zarr",
    "numcodecs",
    "psutil",
    "openpyxl",
    "matplotlib",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def diagnostic_protocol_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "Synpo_Diagnostic_Protocol.pdf"


def diagnostic_session_parent(output_directory: str | Path) -> Path:
    return Path(output_directory).expanduser().resolve() / "Synpo diagnostics"


def _path_kind(path: Path) -> dict[str, object]:
    lowered = str(path).casefold()
    markers = {
        "onedrive": "onedrive" in lowered,
        "dropbox": "dropbox" in lowered,
        "sharepoint": "sharepoint" in lowered,
        "network_unc": str(path).startswith("\\\\"),
    }
    return {
        "drive": path.anchor,
        "synchronization_markers": [name for name, found in markers.items() if found],
    }


class DiagnosticSession:
    """Append-only, privacy-conscious diagnostic evidence for one app session."""

    SAMPLE_FIELDS = (
        "utc_time",
        "elapsed_seconds",
        "process_cpu_percent",
        "system_cpu_percent",
        "process_rss_bytes",
        "process_vms_bytes",
        "available_memory_bytes",
        "memory_percent",
        "system_read_bytes",
        "system_write_bytes",
        "process_read_bytes",
        "process_write_bytes",
        "disk_free_bytes",
        "thread_count",
        "active_operation",
        "active_phase",
    )

    def __init__(self, directory: str | Path, *, level: str) -> None:
        if level not in VALID_DIAGNOSTIC_LEVELS:
            raise ValueError(f"Unknown diagnostic level: {level!r}.")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.level = level
        self.session_id = uuid.uuid4().hex
        self.started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        self._lock = threading.RLock()
        self._finished = False
        self._write_error: str | None = None
        self._event_counts: Counter[str] = Counter()
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._sample_count = 0
        self._active_operation = "idle"
        self._active_phase = ""
        self._process = psutil.Process()
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._events_path = self.directory / "events.jsonl"
        self._samples_path = self.directory / "system-samples.csv"
        self._events_handle = self._events_path.open("a", encoding="utf-8", newline="")
        self._samples_handle = self._samples_path.open("a", encoding="utf-8", newline="")
        self._sample_writer = csv.DictWriter(
            self._samples_handle, fieldnames=self.SAMPLE_FIELDS
        )
        self._sample_writer.writeheader()
        self._samples_handle.flush()
        self._write_environment()
        self._write_readme()
        self.record(
            "diagnostic_session_started",
            scope="diagnostic",
            level=level,
            session_id=self.session_id,
        )

    @classmethod
    def create(cls, parent: str | Path, *, level: str) -> "DiagnosticSession":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parent_path = Path(parent).expanduser().resolve()
        parent_path.mkdir(parents=True, exist_ok=True)
        for suffix in range(1000):
            label = timestamp if suffix == 0 else f"{timestamp}-{suffix:03d}"
            destination = parent_path / label
            if not destination.exists():
                return cls(destination, level=level)
        raise OSError("Could not allocate a unique diagnostic session folder.")

    @property
    def active(self) -> bool:
        return not self._finished

    @property
    def write_error(self) -> str | None:
        return self._write_error

    def _sanitize_path_text(self, value: str) -> str:
        result = value
        candidates = [Path.home()]
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            candidates.append(Path(user_profile))
        for candidate in candidates:
            text = str(candidate)
            if text and result.casefold().startswith(text.casefold()):
                result = "%USERPROFILE%" + result[len(text) :]
                break
        return result

    def _sanitize(self, value: object) -> object:
        if isinstance(value, Path):
            return self._sanitize_path_text(str(value))
        if isinstance(value, str):
            if "\\" in value or "/" in value:
                return self._sanitize_path_text(value)
            return value
        if isinstance(value, dict):
            return {
                str(key): self._sanitize(item)
                for key, item in value.items()
                if str(key).casefold() not in {"comment", "note", "notes"}
            }
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(item) for item in value]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return repr(value)

    def _package_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for name in _PACKAGE_NAMES:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        return versions

    @staticmethod
    def _process_names() -> list[str]:
        names: set[str] = set()
        for process in psutil.process_iter(["name"]):
            try:
                name = str(process.info.get("name") or "").strip()
                if name:
                    names.add(name)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return sorted(names, key=str.casefold)

    def _write_environment(self) -> None:
        memory = psutil.virtual_memory()
        environment = {
            "schema_version": 1,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "diagnostic_level": self.level,
            "synpo_version": __version__,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": self._sanitize_path_text(sys.executable),
                "prefix": self._sanitize_path_text(sys.prefix),
            },
            "operating_system": {
                "platform": platform.platform(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
            "hardware": {
                "processor": platform.processor(),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "total_memory_bytes": int(memory.total),
            },
            "packages": self._package_versions(),
            "running_process_names": self._process_names(),
            "privacy": {
                "image_pixels_collected": False,
                "command_lines_collected": False,
                "window_titles_collected": False,
                "comments_collected": False,
                "home_paths_replaced_with": "%USERPROFILE%",
            },
        }
        (self.directory / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_readme(self) -> None:
        protocol = diagnostic_protocol_path()
        text = (
            "Synpo diagnostic session\n"
            "========================\n\n"
            "This folder contains timings and system-usage metadata. It does not "
            "contain microscopy pixels, project comments, process command lines, "
            "or window titles. Review environment.json and session-summary.json "
            "before sharing the folder.\n\n"
            f"Standalone protocol: {self._sanitize_path_text(str(protocol))}\n"
        )
        (self.directory / "README.txt").write_text(text, encoding="utf-8")

    def record(self, event: str, scope: str = "full", **details: object) -> None:
        if (
            self.level == DIAGNOSTIC_LEVEL_CORRECTION
            and scope not in {"correction", "diagnostic"}
        ):
            return
        with self._lock:
            if self._finished or self._write_error is not None:
                return
            try:
                payload = {
                    "utc_time": _utc_now(),
                    "elapsed_seconds": round(
                        time.monotonic() - self._started_monotonic, 6
                    ),
                    "event": str(event),
                    "scope": str(scope),
                    "thread": threading.current_thread().name,
                    "details": self._sanitize(details),
                }
                self._events_handle.write(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
                )
                self._events_handle.flush()
                self._event_counts[str(event)] += 1
                if event == "correction_started":
                    self._active_operation = str(details.get("operation", "correction"))
                    self._active_phase = "starting"
                elif event == "operation_started":
                    self._active_operation = str(details.get("kind", "operation"))
                    self._active_phase = "starting"
                elif event == "context_generation_started":
                    self._active_operation = "context_generation"
                    self._active_phase = str(details.get("request_view", "starting"))
                elif event in {
                    "correction_progress",
                    "operation_progress",
                    "context_generation_progress",
                }:
                    self._active_phase = str(details.get("phase", ""))
                elif event in {
                    "correction_finished",
                    "operation_finished",
                    "context_generation_completed",
                    "context_generation_failed",
                }:
                    self._active_operation = "idle"
                    self._active_phase = ""
                duration = details.get("duration_seconds")
                if isinstance(duration, (int, float)):
                    self._durations[str(event)].append(float(duration))
            except Exception as exc:  # Diagnostics must never fail scientific work.
                self._write_error = f"{type(exc).__name__}: {exc}"

    def callback(self, event: str, scope: str, details: dict[str, object]) -> None:
        self.record(event, scope=scope, **details)

    def sample_system(self) -> None:
        with self._lock:
            if self._finished or self._write_error is not None:
                return
            try:
                memory = psutil.virtual_memory()
                process_memory = self._process.memory_info()
                system_io = psutil.disk_io_counters()
                try:
                    process_io = self._process.io_counters()
                except (AttributeError, psutil.AccessDenied, NotImplementedError):
                    process_io = None
                usage = shutil.disk_usage(self.directory)
                row = {
                    "utc_time": _utc_now(),
                    "elapsed_seconds": round(
                        time.monotonic() - self._started_monotonic, 3
                    ),
                    "process_cpu_percent": self._process.cpu_percent(None),
                    "system_cpu_percent": psutil.cpu_percent(None),
                    "process_rss_bytes": int(process_memory.rss),
                    "process_vms_bytes": int(process_memory.vms),
                    "available_memory_bytes": int(memory.available),
                    "memory_percent": float(memory.percent),
                    "system_read_bytes": int(system_io.read_bytes) if system_io else "",
                    "system_write_bytes": int(system_io.write_bytes) if system_io else "",
                    "process_read_bytes": int(process_io.read_bytes) if process_io else "",
                    "process_write_bytes": int(process_io.write_bytes) if process_io else "",
                    "disk_free_bytes": int(usage.free),
                    "thread_count": self._process.num_threads(),
                    "active_operation": self._active_operation,
                    "active_phase": self._active_phase,
                }
                self._sample_writer.writerow(row)
                self._samples_handle.flush()
                self._sample_count += 1
            except Exception as exc:  # Diagnostics must never fail scientific work.
                self._write_error = f"{type(exc).__name__}: {exc}"

    def record_project_context(
        self,
        *,
        output_directory: str | Path | None,
        cache_path: str | Path | None,
        specimen_count: int | None,
    ) -> None:
        details: dict[str, object] = {
            "specimen_count": specimen_count,
        }
        if output_directory is not None:
            output = Path(output_directory).expanduser().resolve()
            details["output_directory"] = output
            details["output_path_characteristics"] = _path_kind(output)
        if cache_path is not None:
            cache = Path(cache_path).expanduser().resolve()
            details["cache_path"] = cache
            details["cache_path_characteristics"] = _path_kind(cache)
            try:
                usage = shutil.disk_usage(cache if cache.exists() else cache.parent)
                details["cache_disk_free_bytes"] = int(usage.free)
            except OSError:
                pass
        self.record("project_context", scope="full", **details)

    def finish(self, *, reason: str = "user_finished") -> Path:
        with self._lock:
            if self._finished:
                return self.directory
            self.record(
                "diagnostic_session_finished",
                scope="diagnostic",
                reason=reason,
            )
            finished_at = _utc_now()
            summary = {
                "schema_version": 1,
                "session_id": self.session_id,
                "started_at": self.started_at,
                "finished_at": finished_at,
                "elapsed_seconds": round(
                    time.monotonic() - self._started_monotonic, 3
                ),
                "diagnostic_level": self.level,
                "event_counts": dict(sorted(self._event_counts.items())),
                "duration_summary_seconds": {
                    name: {
                        "count": len(values),
                        "total": round(sum(values), 6),
                        "minimum": round(min(values), 6),
                        "maximum": round(max(values), 6),
                        "average": round(sum(values) / len(values), 6),
                    }
                    for name, values in sorted(self._durations.items())
                    if values
                },
                "system_sample_count": self._sample_count,
                "diagnostic_write_error": self._write_error,
            }
            (self.directory / "session-summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            self._events_handle.close()
            self._samples_handle.close()
            self._finished = True
            return self.directory

    def package(self, *, reason: str = "user_packaged") -> Path:
        directory = self.finish(reason=reason)
        archive = directory.with_suffix(".zip")
        with zipfile.ZipFile(
            archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            for source in sorted(directory.rglob("*")):
                if source.is_file():
                    bundle.write(source, source.relative_to(directory.parent))
        return archive


def emit_diagnostic(
    callback: DiagnosticCallback | None,
    event: str,
    *,
    scope: str = "correction",
    **details: object,
) -> None:
    if callback is None:
        return
    try:
        callback(event, scope, dict(details))
    except Exception:
        pass


class diagnostic_span:
    """Emit start/finish timing records without allowing logging to affect work."""

    def __init__(
        self,
        callback: DiagnosticCallback | None,
        event: str,
        *,
        scope: str = "correction",
        **details: object,
    ) -> None:
        self.callback = callback
        self.event = event
        self.scope = scope
        self.details = details
        self.started = 0.0

    def __enter__(self) -> "diagnostic_span":
        self.started = time.monotonic()
        emit_diagnostic(
            self.callback,
            f"{self.event}_started",
            scope=self.scope,
            **self.details,
        )
        return self

    def __exit__(self, exc_type, exc, _traceback) -> bool:
        details = dict(self.details)
        details.update(
            {
                "duration_seconds": round(time.monotonic() - self.started, 6),
                "status": "failed" if exc is not None else "completed",
            }
        )
        if exc is not None:
            details["error_type"] = type(exc).__name__
            details["error_message"] = str(exc)
        emit_diagnostic(
            self.callback,
            f"{self.event}_finished",
            scope=self.scope,
            **details,
        )
        return False
