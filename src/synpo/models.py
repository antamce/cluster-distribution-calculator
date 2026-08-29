from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


Severity = Literal["error", "warning"]
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class ParsedFilename:
    batch_prefix: str
    experimental_group: str
    specimen_id: str
    channel: Literal["ChanA", "ChanB"]


@dataclass(frozen=True)
class TiffMetadata:
    shape: tuple[int, ...]
    dtype: str
    axes: str
    pages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "axes": self.axes,
            "pages": self.pages,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TiffMetadata":
        return cls(
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
            axes=str(value["axes"]),
            pages=int(value["pages"]),
        )


@dataclass(frozen=True)
class FileFingerprint:
    size_bytes: int
    mtime_ns: int
    sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FileFingerprint":
        sha256 = value.get("sha256")
        return cls(
            size_bytes=int(value["size_bytes"]),
            mtime_ns=int(value["mtime_ns"]),
            sha256=None if sha256 is None else str(sha256),
        )


@dataclass(frozen=True)
class ImportIssue:
    severity: Severity
    message: str
    filename: str | None = None


@dataclass
class ChannelFile:
    path: Path
    parsed: ParsedFilename
    metadata: TiffMetadata | None = None
    fingerprint: FileFingerprint | None = None
    issues: list[ImportIssue] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return self.path.name


@dataclass
class SpecimenPair:
    batch_prefix: str
    experimental_group: str
    specimen_id: str
    channels: dict[str, ChannelFile] = field(default_factory=dict)
    issues: list[ImportIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        all_issues = self.issues + [
            issue
            for channel_file in self.channels.values()
            for issue in channel_file.issues
        ]
        return set(self.channels) == {"ChanA", "ChanB"} and not any(
            issue.severity == "error" for issue in all_issues
        )

    @property
    def shape_text(self) -> str:
        metadata = [item.metadata for item in self.channels.values() if item.metadata]
        if not metadata:
            return ""
        shapes = {item.shape for item in metadata}
        return " × ".join(str(value) for value in metadata[0].shape) if len(shapes) == 1 else "mismatch"

    @property
    def dtype_text(self) -> str:
        metadata = [item.metadata for item in self.channels.values() if item.metadata]
        if not metadata:
            return ""
        dtypes = {item.dtype for item in metadata}
        return metadata[0].dtype if len(dtypes) == 1 else "mismatch"


@dataclass
class ScanReport:
    source_directory: Path
    pairs: list[SpecimenPair]
    issues: list[ImportIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return bool(self.pairs) and not any(
            issue.severity == "error"
            for issue in self.issues
        ) and all(pair.valid for pair in self.pairs)

    @property
    def error_count(self) -> int:
        pair_issues = [issue for pair in self.pairs for issue in pair.issues]
        file_issues = [
            issue
            for pair in self.pairs
            for channel_file in pair.channels.values()
            for issue in channel_file.issues
        ]
        return sum(
            issue.severity == "error"
            for issue in self.issues + pair_issues + file_issues
        )


@dataclass(frozen=True)
class Calibration:
    preset_name: str
    xy_um_per_pixel: float
    z_step_um: float

    def validate(self) -> None:
        if not self.preset_name.strip():
            raise ValueError("Calibration preset name is required.")
        if self.xy_um_per_pixel <= 0 or self.z_step_um <= 0:
            raise ValueError("Calibration values must be greater than zero.")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "preset_name": self.preset_name.strip(),
            "xy_um_per_pixel": self.xy_um_per_pixel,
            "z_step_um": self.z_step_um,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Calibration":
        calibration = cls(
            preset_name=str(value["preset_name"]),
            xy_um_per_pixel=float(value["xy_um_per_pixel"]),
            z_step_um=float(value["z_step_um"]),
        )
        calibration.validate()
        return calibration
