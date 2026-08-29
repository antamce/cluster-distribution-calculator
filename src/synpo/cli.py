from __future__ import annotations

import argparse
import json
from pathlib import Path

from .importer import scan_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Synpo microscopy batches")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="Validate a folder of paired TIFF files")
    scan.add_argument("folder", type=Path)
    scan.add_argument("--skip-checksums", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "scan":
        report = scan_batch(arguments.folder, include_checksums=not arguments.skip_checksums)
        payload = {
            "source_directory": str(report.source_directory),
            "valid": report.valid,
            "errors": report.error_count,
            "issues": [issue.__dict__ for issue in report.issues],
            "pairs": [
                {
                    "batch_prefix": pair.batch_prefix,
                    "experimental_group": pair.experimental_group,
                    "specimen_id": pair.specimen_id,
                    "valid": pair.valid,
                    "shape": pair.shape_text,
                    "dtype": pair.dtype_text,
                    "channels": {
                        name: {
                            "filename": item.filename,
                            "sha256": item.fingerprint.sha256 if item.fingerprint else None,
                            "issues": [issue.__dict__ for issue in item.issues],
                        }
                        for name, item in pair.channels.items()
                    },
                    "issues": [issue.__dict__ for issue in pair.issues],
                }
                for pair in report.pairs
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0 if report.valid else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
