from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import tifffile

from .models import (
    ChannelFile,
    FileFingerprint,
    ImportIssue,
    ParsedFilename,
    ProgressCallback,
    ScanReport,
    SpecimenPair,
    TiffMetadata,
)


FILENAME_PATTERN = re.compile(
    r"^(?P<batch_prefix>[^_]+)_(?P<experimental_group>[^_]+)_"
    r"(?P<specimen_id>[^_]+)_(?P<channel>ChanA|ChanB)_registered\."
    r"(?P<extension>tif|tiff)$",
    re.IGNORECASE,
)

DEFAULT_CHANNEL_MARKERS = {"ChanA": "ChanA", "ChanB": "ChanB"}
DEFAULT_EXPERIMENTAL_GROUP = "Experiment"


def parse_filename(filename: str | Path) -> ParsedFilename:
    name = Path(filename).name
    match = FILENAME_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(
            "Expected <batch_prefix>_<experimental_group>_<specimen_id>_"
            "<ChanA|ChanB>_registered.tif"
        )
    raw_channel = match.group("channel").lower()
    channel = "ChanA" if raw_channel == "chana" else "ChanB"
    return ParsedFilename(
        batch_prefix=match.group("batch_prefix").strip(),
        experimental_group=match.group("experimental_group").strip(),
        specimen_id=match.group("specimen_id").strip(),
        channel=channel,
    )


def validate_channel_markers(channel_markers: dict[str, str]) -> dict[str, str]:
    markers = {
        "ChanA": str(channel_markers.get("ChanA", "")).strip(),
        "ChanB": str(channel_markers.get("ChanB", "")).strip(),
    }
    if not markers["ChanA"] or not markers["ChanB"]:
        raise ValueError("Both channel filename markers are required.")
    if markers["ChanA"].casefold() == markers["ChanB"].casefold():
        raise ValueError("Channel filename markers must be different.")
    return markers


def _remove_marker(stem: str, marker: str) -> str | None:
    lowered = stem.casefold()
    marker_lower = marker.casefold()
    if lowered.endswith(marker_lower):
        return stem[: -len(marker)].rstrip(" _-.")
    token = re.compile(
        rf"(?i)(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])"
    )
    matches = list(token.finditer(stem))
    if len(matches) != 1:
        return None
    match = matches[0]
    value = stem[: match.start()] + stem[match.end() :]
    value = re.sub(r"([ _.-])\1+", r"\1", value).strip(" _-.")
    return value


def _fallback_specimen_name(pair_stem: str) -> str:
    value = re.sub(r"(?i)(?:[_ .-]+registered)$", "", pair_stem)
    return value.strip(" _-.") or "Specimen"


def _identify_flexible_channel(
    path: Path, markers: dict[str, str]
) -> tuple[str, str]:
    matches = [
        (channel, base)
        for channel, marker in markers.items()
        if (base := _remove_marker(path.stem, marker)) is not None
    ]
    if not matches:
        raise ValueError(
            f"Filename does not contain either configured channel marker "
            f"({markers['ChanA']!r} or {markers['ChanB']!r})"
        )
    if len(matches) > 1:
        raise ValueError("Filename matches both configured channel markers")
    return matches[0]


def read_tiff_metadata(path: Path) -> TiffMetadata:
    with tifffile.TiffFile(path) as image:
        if not image.series:
            raise ValueError("TIFF has no readable image series")
        series = image.series[0]
        return TiffMetadata(
            shape=tuple(int(item) for item in series.shape),
            dtype=str(series.dtype),
            axes=str(series.axes),
            pages=len(image.pages),
        )


def fingerprint_file(path: Path, include_checksum: bool = True) -> FileFingerprint:
    before = path.stat()
    digest: str | None = None
    if include_checksum:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise OSError(f"File changed while it was being fingerprinted: {path.name}")
    return FileFingerprint(
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest,
    )


def _inspect_file(channel_file: ChannelFile, include_checksum: bool) -> None:
    try:
        channel_file.metadata = read_tiff_metadata(channel_file.path)
    except Exception as exc:
        channel_file.issues.append(
            ImportIssue("error", f"Cannot read TIFF metadata: {exc}", channel_file.filename)
        )
        return

    metadata = channel_file.metadata
    if len(metadata.shape) != 3:
        channel_file.issues.append(
            ImportIssue("error", f"Expected a 3D stack, found shape {metadata.shape}", channel_file.filename)
        )
    if metadata.dtype != "uint16":
        channel_file.issues.append(
            ImportIssue("error", f"Expected uint16 data, found {metadata.dtype}", channel_file.filename)
        )
    if metadata.axes.upper() != "ZYX":
        channel_file.issues.append(
            ImportIssue(
                "warning",
                f"Axis metadata is {metadata.axes!r}, not 'ZYX'; confirm slice order",
                channel_file.filename,
            )
        )

    try:
        channel_file.fingerprint = fingerprint_file(channel_file.path, include_checksum)
    except Exception as exc:
        channel_file.issues.append(
            ImportIssue("error", f"Cannot fingerprint file: {exc}", channel_file.filename)
        )


def scan_batch(
    source_directory: str | Path,
    *,
    channel_markers: dict[str, str] | None = None,
    default_experimental_group: str = DEFAULT_EXPERIMENTAL_GROUP,
    include_checksums: bool = True,
    progress: ProgressCallback | None = None,
) -> ScanReport:
    directory = Path(source_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Source folder does not exist: {directory}")
    markers = validate_channel_markers(
        channel_markers or DEFAULT_CHANNEL_MARKERS
    )
    default_group = str(default_experimental_group).strip()
    if not default_group:
        raise ValueError("The fallback experimental-group name cannot be empty.")

    paths = sorted(
        (item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}),
        key=lambda item: item.name.casefold(),
    )
    report_issues: list[ImportIssue] = []
    grouped: dict[tuple[str, str, str], dict[str, list[ChannelFile]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_modes: dict[tuple[str, str, str], str] = {}

    for path in paths:
        try:
            channel, pair_stem = _identify_flexible_channel(path, markers)
        except ValueError as exc:
            report_issues.append(ImportIssue("error", str(exc), path.name))
            continue
        try:
            strict = parse_filename(path.name)
        except ValueError:
            strict = None
        if strict is not None and strict.channel == channel:
            parsed = strict
            key = (parsed.batch_prefix, parsed.experimental_group, parsed.specimen_id)
            mode = "strict"
        else:
            parsed = ParsedFilename(
                batch_prefix=directory.name or "batch",
                experimental_group=default_group,
                specimen_id=_fallback_specimen_name(pair_stem),
                channel=channel,  # type: ignore[arg-type]
            )
            key = (
                parsed.batch_prefix,
                parsed.experimental_group,
                pair_stem.casefold(),
            )
            mode = "flexible"
        grouped[key][channel].append(ChannelFile(path=path, parsed=parsed))
        pair_modes[key] = (
            "flexible" if pair_modes.get(key) == "flexible" or mode == "flexible" else "strict"
        )

    unique_files = [
        files[0]
        for channels in grouped.values()
        for files in channels.values()
        if len(files) == 1
    ]
    total = len(unique_files)
    for index, channel_file in enumerate(unique_files, start=1):
        if progress:
            progress("Inspecting TIFF files", index - 1, total, channel_file.filename)
        _inspect_file(channel_file, include_checksums)
        if progress:
            progress("Inspecting TIFF files", index, total, channel_file.filename)

    pairs: list[SpecimenPair] = []
    for key, channel_lists in grouped.items():
        pair = SpecimenPair(
            batch_prefix=key[0],
            experimental_group=key[1],
            specimen_id=(
                next(iter(channel_lists.values()))[0].parsed.specimen_id
                if channel_lists
                else key[2]
            ),
            import_mode=pair_modes.get(key, "strict"),
        )
        if pair.import_mode == "flexible":
            pair.issues.append(
                ImportIssue(
                    "warning",
                    "Metadata-free filename mode: using one experimental group; "
                    "confirm or edit the group and specimen labels.",
                )
            )
        for channel in ("ChanA", "ChanB"):
            files = channel_lists.get(channel, [])
            if not files:
                pair.issues.append(ImportIssue("error", f"Missing {channel} file"))
            elif len(files) > 1:
                names = ", ".join(item.filename for item in files)
                pair.issues.append(ImportIssue("error", f"Duplicate {channel} files: {names}"))
            else:
                pair.channels[channel] = files[0]

        if set(pair.channels) == {"ChanA", "ChanB"}:
            metadata_a = pair.channels["ChanA"].metadata
            metadata_b = pair.channels["ChanB"].metadata
            if metadata_a and metadata_b:
                if metadata_a.shape != metadata_b.shape:
                    pair.issues.append(
                        ImportIssue(
                            "error",
                            f"Channel dimensions differ: {metadata_a.shape} vs {metadata_b.shape}",
                        )
                    )
                if metadata_a.dtype != metadata_b.dtype:
                    pair.issues.append(
                        ImportIssue(
                            "error",
                            f"Channel data types differ: {metadata_a.dtype} vs {metadata_b.dtype}",
                        )
                    )
        pairs.append(pair)

    batch_prefixes = {
        pair.batch_prefix for pair in pairs if pair.import_mode == "strict"
    }
    if len(batch_prefixes) > 1:
        report_issues.append(
            ImportIssue("error", "More than one batch prefix was found in the selected folder")
        )
    if not paths:
        report_issues.append(ImportIssue("error", "No TIFF files were found directly in the folder"))
    elif not pairs:
        report_issues.append(ImportIssue("error", "No valid filename pairs could be parsed"))

    pairs.sort(key=lambda item: (item.experimental_group.casefold(), item.specimen_id.casefold()))
    import_mode = (
        "flexible" if any(pair.import_mode == "flexible" for pair in pairs) else "strict"
    )
    return ScanReport(
        source_directory=directory,
        pairs=pairs,
        issues=report_issues,
        import_mode=import_mode,
        channel_markers=markers,
        default_experimental_group=default_group,
    )


def inspect_manual_pair(
    channel_a_path: str | Path,
    channel_b_path: str | Path,
    *,
    default_experimental_group: str = DEFAULT_EXPERIMENTAL_GROUP,
    specimen_id: str | None = None,
    include_checksums: bool = True,
    progress: ProgressCallback | None = None,
) -> ScanReport:
    """Inspect two explicitly selected TIFFs as one specimen pair."""
    paths = {
        "ChanA": Path(channel_a_path).expanduser().resolve(),
        "ChanB": Path(channel_b_path).expanduser().resolve(),
    }
    for channel, path in paths.items():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"{channel} must be an existing TIFF file: {path}")
    group = str(default_experimental_group).strip()
    if not group:
        raise ValueError("The experimental-group name cannot be empty.")
    shared = os.path.commonprefix((paths["ChanA"].stem, paths["ChanB"].stem))
    default_specimen = shared.rstrip(" _-.") or paths["ChanA"].stem
    specimen = str(specimen_id or default_specimen).strip()
    if not specimen:
        raise ValueError("The specimen name cannot be empty.")
    try:
        common = Path(
            os.path.commonpath((paths["ChanA"].parent, paths["ChanB"].parent))
        ).resolve()
    except ValueError:
        common = paths["ChanA"].parent
    pair = SpecimenPair(
        batch_prefix="manual",
        experimental_group=group,
        specimen_id=specimen,
        import_mode="manual",
        issues=[
            ImportIssue(
                "warning",
                "Manually paired files; confirm or edit the group and specimen labels.",
            )
        ],
    )
    total = 2
    for index, (channel, path) in enumerate(paths.items(), start=1):
        parsed = ParsedFilename(
            batch_prefix="manual",
            experimental_group=group,
            specimen_id=specimen,
            channel=channel,  # type: ignore[arg-type]
        )
        channel_file = ChannelFile(path=path, parsed=parsed)
        if progress:
            progress("Inspecting TIFF files", index - 1, total, path.name)
        _inspect_file(channel_file, include_checksums)
        pair.channels[channel] = channel_file
        if progress:
            progress("Inspecting TIFF files", index, total, path.name)
    metadata_a = pair.channels["ChanA"].metadata
    metadata_b = pair.channels["ChanB"].metadata
    if metadata_a and metadata_b:
        if metadata_a.shape != metadata_b.shape:
            pair.issues.append(
                ImportIssue(
                    "error",
                    f"Channel dimensions differ: {metadata_a.shape} vs {metadata_b.shape}",
                )
            )
        if metadata_a.dtype != metadata_b.dtype:
            pair.issues.append(
                ImportIssue(
                    "error",
                    f"Channel data types differ: {metadata_a.dtype} vs {metadata_b.dtype}",
                )
            )
    return ScanReport(
        source_directory=common,
        pairs=[pair],
        import_mode="manual",
        channel_markers=dict(DEFAULT_CHANNEL_MARKERS),
        default_experimental_group=group,
    )
