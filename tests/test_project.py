from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile

from synpo.calibration import CalibrationStore
from synpo.importer import inspect_manual_pair, scan_batch
from synpo.models import Calibration
from synpo.project import (
    channel_source_path,
    create_project_manifest,
    load_project,
    relink_project_sources,
    save_project,
    verify_project_sources,
)


@contextlib.contextmanager
def workspace_directory():
    root = Path.cwd() / ".test-work" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class ProjectTests(unittest.TestCase):
    def make_pair(self, root: Path) -> None:
        data = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
        for channel in ("ChanA", "ChanB"):
            tifffile.imwrite(
                root / f"batch_group_specimen_{channel}_registered.tif",
                data,
                metadata={"axes": "ZYX"},
                photometric="minisblack",
            )

    def test_project_round_trip_verify_and_relink(self) -> None:
        with workspace_directory() as root:
            source = root / "source"
            output = root / "output"
            moved = root / "moved"
            source.mkdir()
            moved.mkdir()
            self.make_pair(source)
            report = scan_batch(source)
            manifest = create_project_manifest(
                report,
                output_directory=output,
                channel_roles={"ChanA": "protein_clusters", "ChanB": "dendrite_spines"},
                calibration=Calibration("example", 0.0462584, 0.5),
            )
            project_path = save_project(root / "example.synpo.json", manifest)
            loaded = load_project(project_path)
            results = verify_project_sources(loaded)
            self.assertTrue(all(item["status"] == "ok" for item in results))

            for path in source.iterdir():
                shutil.copy2(path, moved / path.name)
            relinked = relink_project_sources(loaded, moved)
            self.assertTrue(all(item["status"] == "ok" for item in relinked))
            self.assertEqual(Path(loaded["source_directory"]), moved.resolve())

    def test_calibration_store_round_trip(self) -> None:
        with workspace_directory() as root:
            path = root / "calibrations.json"
            store = CalibrationStore(path)
            expected = Calibration("63x objective", 0.0462584, 0.5)
            store.save_preset(expected)
            self.assertEqual(store.load()[expected.preset_name], expected)

    def test_stage_1_project_is_migrated_for_preprocessing(self) -> None:
        with workspace_directory() as root:
            source = root / "source"
            source.mkdir()
            self.make_pair(source)
            manifest = create_project_manifest(
                scan_batch(source),
                output_directory=root / "output",
                channel_roles={"ChanA": "protein_clusters", "ChanB": "dendrite_spines"},
                calibration=Calibration("example", 0.05, 0.5),
            )
            manifest.pop("preprocessing")
            manifest["cache"] = {
                "format": "pending_stage_2",
                "path": None,
                "deletion_eligible": False,
            }
            manifest["specimens"][0]["checkpoints"]["preprocessing"] = "not_started"
            loaded = load_project(save_project(root / "old.synpo.json", manifest))
            self.assertIn("preprocessing", loaded)
            self.assertEqual(loaded["detection"]["memory_mode"], "automatic")
            self.assertIsInstance(
                loaded["specimens"][0]["checkpoints"]["preprocessing"], dict
            )

    def test_manual_sources_keep_individual_paths_and_relink_recursively(self) -> None:
        with workspace_directory() as root:
            source_a = root / "original" / "protein"
            source_b = root / "original" / "dendrite"
            source_a.mkdir(parents=True)
            source_b.mkdir(parents=True)
            data = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
            path_a = source_a / "view-a.tif"
            path_b = source_b / "view-b.tif"
            for path in (path_a, path_b):
                tifffile.imwrite(
                    path,
                    data,
                    metadata={"axes": "ZYX"},
                    photometric="minisblack",
                )
            report = inspect_manual_pair(
                path_a, path_b, specimen_id="manual specimen"
            )
            manifest = create_project_manifest(
                report,
                output_directory=root / "output",
                channel_roles={
                    "ChanA": "protein_clusters",
                    "ChanB": "dendrite_spines",
                },
                calibration=Calibration("example", 0.05, 0.5),
            )
            self.assertEqual(
                channel_source_path(manifest, manifest["specimens"][0]["channels"]["ChanA"]),
                path_a.resolve(),
            )
            moved = root / "relocated"
            moved_a = moved / "one" / path_a.name
            moved_b = moved / "two" / path_b.name
            moved_a.parent.mkdir(parents=True)
            moved_b.parent.mkdir(parents=True)
            shutil.copy2(path_a, moved_a)
            shutil.copy2(path_b, moved_b)
            path_a.unlink()
            path_b.unlink()
            self.assertTrue(
                all(
                    item["status"] == "missing"
                    for item in verify_project_sources(manifest)
                )
            )
            results = relink_project_sources(manifest, moved)
            self.assertTrue(all(item["status"] == "ok" for item in results))
            self.assertEqual(
                channel_source_path(manifest, manifest["specimens"][0]["channels"]["ChanA"]),
                moved_a.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
