from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile
import zarr

from synpo.detection import DetectionSettings, detect_project, load_detection_slice
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
    def make_pair(self, source: Path) -> None:
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
                source / f"batch_group_specimen_{channel}_registered.tif",
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


if __name__ == "__main__":
    unittest.main()
