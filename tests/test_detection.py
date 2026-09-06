from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
import zarr

import synpo.detection as detection_module
from synpo.detection import (
    ALWAYS_LOW_MEMORY_MODE,
    DetectionSettings,
    InsufficientDetectionDiskSpace,
    detect_project,
    detection_cache_path,
    load_detection_slice,
)
from synpo.importer import scan_batch
from synpo.models import Calibration
from synpo.preprocessing import process_project_cache
from synpo.project import create_project_manifest, load_project, save_project


@contextlib.contextmanager
def workspace_directory():
    root = Path.cwd() / ".test-work" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class DetectionTests(unittest.TestCase):
    def make_pair(self, source: Path, specimen: str = "specimen") -> None:
        shape = (8, 64, 80)
        dendrites = np.full(shape, 100, dtype=np.uint16)
        dendrites[2:7, 29:36, 6:74] = 2600
        dendrites[2:7, 18:30, 22:29] = 2600
        dendrites[2:7, 35:49, 47:55] = 2800
        proteins = np.full(shape, 100, dtype=np.uint16)
        proteins[2:6, 20:25, 23:28] = 3600
        proteins[3:7, 38:44, 48:54] = 4200
        for channel, data in (("ChanA", proteins), ("ChanB", dendrites)):
            tifffile.imwrite(
                source / f"batch_group_{specimen}_{channel}_registered.tif",
                data,
                metadata={"axes": "ZYX"},
                photometric="minisblack",
            )

    def test_detection_masks_checkpoint_and_resume(self) -> None:
        with workspace_directory() as root:
            source = root / "source"
            source.mkdir()
            self.make_pair(source)
            manifest = create_project_manifest(
                scan_batch(source),
                output_directory=root / "output",
                channel_roles={
                    "ChanA": "protein_clusters",
                    "ChanB": "dendrite_spines",
                },
                calibration=Calibration("test", 0.1, 0.5),
            )
            project_path = save_project(root / "batch.synpo.json", manifest)
            process_project_cache(manifest, project_path)
            manifest["detection"]["settings"] = DetectionSettings(
                dendrite_sensitivity=1.25,
                cluster_sensitivity=0.8,
                spine_branch_length_um=2.0,
                minimum_dendrite_length_um=1.0,
                minimum_spine_projection_pixels=2,
                minimum_cluster_voxels=4,
            ).to_dict()
            first = detect_project(manifest, project_path)
            second = detect_project(manifest, project_path)
            reopened = load_project(project_path)
            checkpoint = reopened["specimens"][0]["checkpoints"]["detection"]
            self.assertEqual(checkpoint["state"], "complete")
            self.assertGreater(checkpoint["summary"]["dendrite_count"], 0)
            self.assertGreater(checkpoint["summary"]["cluster_count"], 0)
            self.assertTrue(second["summaries"][0]["skipped"])

            detection_root = zarr.open_group(
                checkpoint["cache_path"], mode="r"
            )["specimens/0000"]
            self.assertTrue(detection_root.attrs["complete"])
            self.assertEqual(
                tuple(detection_root["cluster_labels"].shape), (8, 64, 80)
            )
            view = load_detection_slice(reopened, 0, 3, "ChanB")
            self.assertEqual(view.raw.dtype, np.dtype("uint16"))
            self.assertGreater(int(view.clusters.max()), 0)

    def _prepared_manifest(self, root: Path, *specimens: str):
        source = root / "source"
        source.mkdir()
        for specimen in specimens:
            self.make_pair(source, specimen)
        manifest = create_project_manifest(
            scan_batch(source),
            output_directory=root / "output",
            channel_roles={
                "ChanA": "protein_clusters",
                "ChanB": "dendrite_spines",
            },
            calibration=Calibration("test", 0.1, 0.5),
        )
        project_path = save_project(root / "batch.synpo.json", manifest)
        process_project_cache(manifest, project_path)
        manifest["detection"]["settings"] = DetectionSettings(
            dendrite_sensitivity=1.25,
            cluster_sensitivity=0.8,
            spine_branch_length_um=2.0,
            minimum_dendrite_length_um=1.0,
            minimum_spine_projection_pixels=2,
            minimum_cluster_voxels=4,
        ).to_dict()
        return manifest, project_path

    def test_low_memory_detection_matches_masks_and_merges_slab_objects(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self._prepared_manifest(root, "specimen")
            detect_project(manifest, project_path)
            detection_root = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )
            standard = {
                name: np.asarray(detection_root["specimens/0000"][name])
                for name in (
                    "dendrite_labels",
                    "spine_labels",
                    "cluster_labels",
                )
            }
            del detection_root["specimens/0000"]
            manifest["detection"]["memory_mode"] = ALWAYS_LOW_MEMORY_MODE

            with patch("synpo.detection._low_memory_slab_depth", return_value=2):
                result = detect_project(manifest, project_path)

            low_group = zarr.open_group(
                str(detection_cache_path(manifest)), mode="r"
            )["specimens/0000"]
            self.assertEqual(result["low_memory_pairs"], 1)
            self.assertEqual(result["summaries"][0]["processing_mode"], "low_memory")
            self.assertNotIn("_cluster_work", low_group)
            for name, expected in standard.items():
                actual = np.asarray(low_group[name])
                self.assertTrue(np.array_equal(expected > 0, actual > 0), name)
                for label_id in np.unique(expected):
                    if label_id:
                        mapped = np.unique(actual[expected == label_id])
                        self.assertEqual(len(mapped), 1, name)
                for label_id in np.unique(actual):
                    if label_id:
                        mapped = np.unique(expected[actual == label_id])
                        self.assertEqual(len(mapped), 1, name)
            # Both synthetic protein objects cross forced Z-slab seams. Each must
            # remain one 3-D object after seam reconciliation.
            self.assertEqual(int(low_group["cluster_labels"][:].max()), 2)

    def test_insufficient_disk_skips_specimen_then_retries_it(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self._prepared_manifest(root, "a", "b")
            with (
                patch("synpo.detection._standard_detection_fits", return_value=False),
                patch(
                    "synpo.detection._preflight_low_memory_disk",
                    side_effect=[
                        InsufficientDetectionDiskSpace(10_000, 1_000),
                        (10_000, 20_000),
                    ],
                ),
                patch("synpo.detection._low_memory_slab_depth", return_value=2),
            ):
                first = detect_project(manifest, project_path)

            self.assertEqual(first["skipped_pairs"], 1)
            self.assertEqual(first["completed_pairs"], 1)
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["detection"]["state"],
                "skipped",
            )
            self.assertEqual(
                manifest["specimens"][1]["checkpoints"]["detection"]["state"],
                "complete",
            )

            with (
                patch("synpo.detection._standard_detection_fits", return_value=False),
                patch("synpo.detection._low_memory_slab_depth", return_value=2),
            ):
                retried = detect_project(manifest, project_path)
            self.assertEqual(retried["skipped_pairs"], 0)
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["detection"]["state"],
                "complete",
            )
            self.assertNotIn(
                "reason", manifest["specimens"][0]["checkpoints"]["detection"]
            )

    def test_specimen_failure_is_recorded_without_stopping_batch(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self._prepared_manifest(root, "a", "b")
            original = detection_module.detect_specimen

            def fail_first(current_manifest, specimen_index, settings, **kwargs):
                if specimen_index == 0:
                    raise RuntimeError("synthetic specimen failure")
                return original(
                    current_manifest, specimen_index, settings, **kwargs
                )

            with patch("synpo.detection.detect_specimen", side_effect=fail_first):
                result = detect_project(manifest, project_path)

            self.assertEqual(result["failed_pairs"], 1)
            self.assertEqual(result["completed_pairs"], 1)
            first_checkpoint = manifest["specimens"][0]["checkpoints"]["detection"]
            self.assertEqual(first_checkpoint["state"], "failed")
            self.assertIn("synthetic specimen failure", first_checkpoint["reason"])
            self.assertEqual(
                manifest["specimens"][1]["checkpoints"]["detection"]["state"],
                "complete",
            )

    def test_memory_strategy_change_reuses_completed_detection(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self._prepared_manifest(root, "specimen")
            first = detect_project(manifest, project_path)
            manifest["detection"]["memory_mode"] = ALWAYS_LOW_MEMORY_MODE
            second = detect_project(manifest, project_path)
            self.assertEqual(first["completed_pairs"], 1)
            self.assertEqual(second["reused_pairs"], 1)
            self.assertEqual(second["low_memory_pairs"], 0)
            self.assertEqual(second["summaries"][0]["processing_mode"], "standard")


if __name__ == "__main__":
    unittest.main()
