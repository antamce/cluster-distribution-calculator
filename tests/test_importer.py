from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile

from synpo.importer import inspect_manual_pair, parse_filename, scan_batch


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

    def test_custom_suffix_markers_use_one_fallback_group(self) -> None:
        with workspace_directory() as root:
            self.write_stack(root / "Untitled001cy.tif")
            self.write_stack(root / "Untitled001cl.tif")
            self.write_stack(root / "Untitled002cy.tif")
            self.write_stack(root / "Untitled002cl.tif")
            report = scan_batch(
                root,
                channel_markers={"ChanA": "cy", "ChanB": "cl"},
                default_experimental_group="single experiment",
            )
            self.assertTrue(report.valid)
            self.assertEqual(report.import_mode, "flexible")
            self.assertEqual(
                [pair.specimen_id for pair in report.pairs],
                ["Untitled001", "Untitled002"],
            )
            self.assertEqual(
                {pair.experimental_group for pair in report.pairs},
                {"single experiment"},
            )

    def test_default_markers_allow_names_without_registered_flag(self) -> None:
        with workspace_directory() as root:
            self.write_stack(root / "plain specimen_ChanA.tif")
            self.write_stack(root / "plain specimen_ChanB.tif")
            report = scan_batch(root, default_experimental_group="all files")
            self.assertTrue(report.valid)
            self.assertEqual(report.pairs[0].specimen_id, "plain specimen")
            self.assertEqual(report.pairs[0].experimental_group, "all files")

    def test_manual_pair_can_use_different_folders(self) -> None:
        with workspace_directory() as root:
            first = root / "protein"
            second = root / "morphology"
            first.mkdir()
            second.mkdir()
            channel_a = first / "protein-view.tif"
            channel_b = second / "dendrite-view.tif"
            self.write_stack(channel_a)
            self.write_stack(channel_b)
            report = inspect_manual_pair(
                channel_a,
                channel_b,
                default_experimental_group="manual group",
                specimen_id="chosen specimen",
            )
            self.assertTrue(report.valid)
            self.assertEqual(report.import_mode, "manual")
            self.assertEqual(report.pairs[0].channels["ChanA"].path, channel_a.resolve())
            self.assertEqual(report.pairs[0].channels["ChanB"].path, channel_b.resolve())


if __name__ == "__main__":
    unittest.main()
