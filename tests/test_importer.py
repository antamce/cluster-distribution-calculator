from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile

from synpo.importer import parse_filename, scan_batch


@contextlib.contextmanager
def workspace_directory():
    root = Path.cwd() / ".test-work" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class ImporterTests(unittest.TestCase):
    def write_stack(self, path: Path, shape: tuple[int, int, int] = (3, 12, 10)) -> None:
        tifffile.imwrite(
            path,
            np.zeros(shape, dtype=np.uint16),
            metadata={"axes": "ZYX"},
            photometric="minisblack",
        )

    def test_parse_expected_filename(self) -> None:
        parsed = parse_filename(
            "exp 1606 - tomato_Gliu 120_Untitled001_ChanB_registered.tif"
        )
        self.assertEqual(parsed.batch_prefix, "exp 1606 - tomato")
        self.assertEqual(parsed.experimental_group, "Gliu 120")
        self.assertEqual(parsed.specimen_id, "Untitled001")
        self.assertEqual(parsed.channel, "ChanB")

    def test_scan_valid_pair(self) -> None:
        with workspace_directory() as root:
            stem = "experiment_group one_specimen01"
            self.write_stack(root / f"{stem}_ChanA_registered.tif")
            self.write_stack(root / f"{stem}_ChanB_registered.tif")
            report = scan_batch(root)
            self.assertTrue(report.valid)
            self.assertEqual(len(report.pairs), 1)
            self.assertEqual(report.pairs[0].shape_text, "3 × 12 × 10")
            self.assertEqual(report.pairs[0].dtype_text, "uint16")
            self.assertTrue(report.pairs[0].channels["ChanA"].fingerprint.sha256)

    def test_missing_channel_is_an_error(self) -> None:
        with workspace_directory() as root:
            self.write_stack(root / "experiment_group_specimen_ChanA_registered.tif")
            report = scan_batch(root)
            self.assertFalse(report.valid)
            self.assertEqual(report.error_count, 1)

    def test_mismatched_shapes_are_an_error(self) -> None:
        with workspace_directory() as root:
            self.write_stack(root / "experiment_group_specimen_ChanA_registered.tif", (3, 12, 10))
            self.write_stack(root / "experiment_group_specimen_ChanB_registered.tif", (4, 12, 10))
            report = scan_batch(root)
            self.assertFalse(report.valid)
            self.assertIn("dimensions differ", report.pairs[0].issues[0].message)


if __name__ == "__main__":
    unittest.main()
