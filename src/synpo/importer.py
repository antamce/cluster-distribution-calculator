from __future__ import annotations

import hashlib
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
    include_checksums: bool = True,
    progress: ProgressCallback | None = None,
) -> ScanReport:
    directory = Path(source_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Source folder does not exist: {directory}")

    paths = sorted(
        (item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}),
        key=lambda item: item.name.casefold(),
    )
    report_issues: list[ImportIssue] = []
    grouped: dict[tuple[str, str, str], dict[str, list[ChannelFile]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in paths:
        try:
            parsed = parse_filename(path.name)
        except ValueError as exc:
            report_issues.append(ImportIssue("error", str(exc), path.name))
            continue
        grouped[(parsed.batch_prefix, parsed.experimental_group, parsed.specimen_id)][
            parsed.channel
        ].append(ChannelFile(path=path, parsed=parsed))

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
            specimen_id=key[2],
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

    batch_prefixes = {pair.batch_prefix for pair in pairs}
    if len(batch_prefixes) > 1:
        report_issues.append(
            ImportIssue("error", "More than one batch prefix was found in the selected folder")
        )
    if not paths:
        report_issues.append(ImportIssue("error", "No TIFF files were found directly in the folder"))
    elif not pairs:
        report_issues.append(ImportIssue("error", "No valid filename pairs could be parsed"))

    pairs.sort(key=lambda item: (item.experimental_group.casefold(), item.specimen_id.casefold()))
    return ScanReport(source_directory=directory, pairs=pairs, issues=report_issues)
