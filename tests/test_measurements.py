from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import zarr

from synpo.detection import detection_cache_path
from synpo.measurements import (
    MeasurementSettings,
    _cluster_keep_lookup,
    cluster_end_comparison_rows,
    load_measurement_result,
    measure_project,
)
from synpo.project import migrate_manifest, save_project


@contextlib.contextmanager
def workspace_directory():
    root = Path.cwd() / ".test-work" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class MeasurementTests(unittest.TestCase):
    def make_project(self, root: Path) -> tuple[dict[str, object], Path]:
        manifest: dict[str, object] = {
            "schema_version": 1,
            "application": {"name": "Synpo", "version": "0.5.0"},
            "project_id": uuid.uuid4().hex,
            "source_directory": str(root / "source"),
            "output_directory": str(root / "output"),
            "batch_prefix": "batch",
            "channel_roles": {
                "ChanA": "protein_clusters",
                "ChanB": "dendrite_spines",
            },
            "calibration": {
                "preset_name": "test",
                "xy_um_per_pixel": 0.1,
                "z_step_um": 0.5,
            },
            "resource_policy": {"maximum_ram_fraction": 0.8},
            "cache": {"format": "zarr-v2-blosc-zstd", "path": None},
            "specimens": [
                {
                    "batch_prefix": "batch",
                    "experimental_group": "treated",
                    "specimen_id": "cell-1",
                    "channels": {},
                    "checkpoints": {
                        "preprocessing": "complete",
                        "detection": {"state": "complete"},
                        "review": "not_started",
                    },
                    "review": {
                        "state": "needs_attention",
                        "comment": "",
                        "history": [],
                    },
                }
            ],
        }
        migrate_manifest(manifest)
        manifest["measurements"]["settings"] = MeasurementSettings(
            cluster_end_method="untrimmed"
        ).to_dict()
        output = Path(str(manifest["output_directory"]))
        output.mkdir(parents=True)
        project_path = save_project(root / "batch.synpo.json", manifest)

        shape = (6, 24, 24)
        dendrites = np.zeros(shape, dtype=np.uint32)
        dendrites[:, 11:14, 2:22] = 1
        spines = np.zeros(shape, dtype=np.uint32)
        spines[1:5, 7:11, 6:10] = 1
        clusters = np.zeros(shape, dtype=np.uint32)
        clusters[0:5, 8:10, 7:9] = 1
        clusters[2:5, 2:4, 17:19] = 2
        root_group = zarr.open_group(str(detection_cache_path(manifest)), mode="a")
        group = root_group.require_group("specimens/0000")
        for name, data in (
            ("dendrite_labels", dendrites),
            ("spine_labels", spines),
            ("cluster_labels", clusters),
        ):
            group.create_dataset(name, data=data, chunks=(1, 24, 24), overwrite=True)
        group.attrs.update(
            {
                "complete": True,
                "settings_signature": "detection-test",
                "summary": {
                    "dendrite_count": 1,
                    "spine_count": 1,
                    "cluster_count": 2,
                },
            }
        )
        return manifest, project_path

    def test_overlap_metrics_and_sum_row(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_project(root)
            output = measure_project(manifest, project_path)
            self.assertEqual(len(output["summaries"]), 1)
            result = load_measurement_result(manifest, 0)
            self.assertEqual(len(result["spine_rows"]), 1)
            self.assertEqual(result["spine_rows"][0]["included_cluster_count"], 1)
            self.assertAlmostEqual(result["spine_rows"][0]["volume_um3"], 0.32)
            self.assertAlmostEqual(
                result["spine_rows"][0]["cluster_to_spine_volume_ratio"], 0.25
            )
            individual = [
                row
                for row in result["cluster_rows"]
                if row["row_type"] == "individual_cluster"
            ]
            sums = [
                row
                for row in result["cluster_rows"]
                if row["row_type"] == "spine_cluster_sum"
            ]
            self.assertEqual(len(individual), 1)
            self.assertEqual(len(sums), 1)
            self.assertAlmostEqual(individual[0]["overlap_percent"], 80.0)
            self.assertAlmostEqual(
                individual[0]["cluster_volume_to_spine_volume_ratio"], 0.25
            )
            self.assertAlmostEqual(
                sums[0]["cluster_volume_to_spine_volume_ratio"], 0.25
            )
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["measurements"]["state"],
                "complete",
            )
            comparison = cluster_end_comparison_rows(result)
            self.assertEqual(len(comparison), 2)
            self.assertIn("fixed_candidate_volume_um3", comparison[0])
            self.assertIn("adaptive_candidate_volume_um3", comparison[0])

    def test_fixed_and_adaptive_end_trimming(self) -> None:
        areas = np.zeros((8, 2), dtype=np.int64)
        areas[1:7, 1] = [80, 70, 20, 22, 19, 20]
        fixed, fixed_details = _cluster_keep_lookup(
            areas,
            MeasurementSettings(
                cluster_end_method="fixed",
                fixed_end_slices=2,
                minimum_retained_slices=2,
            ),
        )
        adaptive, adaptive_details = _cluster_keep_lookup(
            areas,
            MeasurementSettings(
                cluster_end_method="adaptive",
                adaptive_area_factor=1.8,
                minimum_retained_slices=2,
            ),
        )
        self.assertFalse(fixed[1, 1])
        self.assertFalse(fixed[2, 1])
        self.assertEqual(fixed_details[1]["discarded_z_slices"], [1, 2])
        self.assertFalse(adaptive[1, 1])
        self.assertFalse(adaptive[2, 1])
        self.assertEqual(adaptive_details[1]["discarded_z_slices"], [1, 2])


if __name__ == "__main__":
    unittest.main()
