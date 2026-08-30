from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile
import zarr

from synpo.detection import detection_cache_path
from synpo.measurements import (
    MeasurementSettings,
    _cluster_keep_lookup,
    clear_centerline_endpoint_hint,
    cluster_end_comparison_rows,
    load_measurement_result,
    measure_project,
    set_centerline_endpoint_hint,
    set_distribution_review,
)
from synpo.distribution import calculate_spine_distribution
from synpo.exporting import export_measurements
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
        spines[1:4, 7:11, 6:18] = 1
        clusters = np.zeros(shape, dtype=np.uint32)
        clusters[1:4, 8:10, 7:9] = 1
        clusters[2:5, 2:4, 17:19] = 2
        source = Path(str(manifest["source_directory"]))
        source.mkdir(parents=True)
        channel_a = "batch_treated_cell-1_ChanA_registered.tif"
        channel_b = "batch_treated_cell-1_ChanB_registered.tif"
        tifffile.imwrite(source / channel_a, (clusters > 0).astype(np.uint16) * 1200)
        tifffile.imwrite(source / channel_b, ((dendrites > 0) | (spines > 0)).astype(np.uint16) * 1800)
        manifest["specimens"][0]["channels"] = {
            "ChanA": {"filename": channel_a},
            "ChanB": {"filename": channel_b},
        }
        save_project(project_path, manifest)
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
            self.assertEqual(len(result["distribution_rows"]), 1)
            self.assertEqual(result["distribution_rows"][0]["spine_id"], 1)
            self.assertEqual(result["spine_rows"][0]["included_cluster_count"], 1)
            self.assertAlmostEqual(result["spine_rows"][0]["volume_um3"], 0.72)
            self.assertAlmostEqual(
                result["spine_rows"][0]["cluster_to_spine_volume_ratio"], 1 / 12
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
            self.assertAlmostEqual(individual[0]["overlap_percent"], 100.0)
            self.assertAlmostEqual(
                individual[0]["cluster_volume_to_spine_volume_ratio"], 1 / 12
            )
            self.assertAlmostEqual(
                sums[0]["cluster_volume_to_spine_volume_ratio"], 1 / 12
            )
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["measurements"]["state"],
                "complete",
            )
            comparison = cluster_end_comparison_rows(result)
            self.assertEqual(len(comparison), 2)
            self.assertIn("fixed_candidate_volume_um3", comparison[0])
            self.assertIn("adaptive_candidate_volume_um3", comparison[0])

            hinted_point = (3, 8, 17)
            set_centerline_endpoint_hint(
                manifest, project_path, 0, 1, hinted_point
            )
            hinted = load_measurement_result(manifest, 0)
            hinted_row = hinted["distribution_rows"][0]
            self.assertEqual(hinted_row["centerline_endpoint_source"], "manual")
            self.assertEqual(hinted_row["centerline_endpoint_zyx"], list(hinted_point))
            self.assertTrue(hinted_row["centerline_endpoint_hint_valid"])
            self.assertEqual(hinted_row["centerline_hint_history"][-1]["action"], "placed")
            set_distribution_review(
                manifest,
                project_path,
                0,
                1,
                distribution_included=True,
                invalid_spine=False,
                note="endpoint checked",
            )
            reviewed_hint = load_measurement_result(manifest, 0)["distribution_rows"][0]
            self.assertEqual(reviewed_hint["centerline_endpoint_source"], "manual")
            self.assertEqual(reviewed_hint["centerline_hint_history"][-1]["action"], "placed")

            replacement = (2, 9, 17)
            set_centerline_endpoint_hint(
                manifest, project_path, 0, 1, replacement
            )
            replaced = load_measurement_result(manifest, 0)["distribution_rows"][0]
            self.assertEqual(replaced["centerline_endpoint_zyx"], list(replacement))
            self.assertEqual(replaced["centerline_hint_history"][-1]["action"], "replaced")

            detection_root = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )
            detection_root["specimens/0000/spine_labels"][replacement] = 0
            manifest["specimens"][0]["checkpoints"]["measurements"]["state"] = "not_started"
            measure_project(manifest, project_path)
            invalid_hint = load_measurement_result(manifest, 0)["distribution_rows"][0]
            self.assertEqual(invalid_hint["centerline_endpoint_source"], "automatic")
            self.assertTrue(invalid_hint["centerline_endpoint_hint_present"])
            self.assertFalse(invalid_hint["centerline_endpoint_hint_valid"])
            self.assertFalse(invalid_hint["distribution_reviewed"])
            self.assertEqual(
                invalid_hint["centerline_hint_history"][-1]["action"],
                "invalidated_by_resegmentation",
            )

            export = export_measurements(
                manifest,
                root / "measurements.xlsx",
                validation_pdf=True,
            )
            self.assertTrue(Path(str(export["workbook"])).is_file())
            self.assertTrue((root / "measurements_csv" / "Distribution_Individual.csv").is_file())
            self.assertEqual(len(export["pdfs"]), 1)

            clear_centerline_endpoint_hint(manifest, project_path, 0, 1)
            cleared = load_measurement_result(manifest, 0)["distribution_rows"][0]
            self.assertEqual(cleared["centerline_endpoint_source"], "automatic")
            self.assertFalse(cleared["centerline_endpoint_hint_present"])
            self.assertEqual(cleared["centerline_hint_history"][-1]["action"], "cleared")

            set_distribution_review(
                manifest,
                project_path,
                0,
                1,
                distribution_included=False,
                invalid_spine=True,
                note="broken",
            )
            invalidated = load_measurement_result(manifest, 0)
            self.assertEqual(invalidated["specimen_rows"][0]["spine_count"], 0)
            self.assertFalse(invalidated["spine_rows"][0]["spine_valid"])

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

    def test_calibrated_curved_axis_assigns_every_voxel_once(self) -> None:
        spine = np.zeros((7, 18, 28), dtype=bool)
        centers = []
        for x in range(3, 24):
            y = 7 + int(round(3 * np.sin((x - 3) / 20 * np.pi)))
            centers.append((3, y, x))
            spine[2:5, y - 1 : y + 2, x] = True
        dendrite = np.zeros_like(spine)
        dendrite[:, :, :3] = True
        clusters = np.zeros_like(spine)
        clusters[:, :, 15:23] = spine[:, :, 15:23]
        result = calculate_spine_distribution(
            spine,
            dendrite,
            clusters,
            sampling_zyx_um=(0.5, 0.1, 0.1),
        )
        self.assertNotEqual(result.axis_status, "no_usable_path")
        self.assertEqual(sum(result.spine_voxels_by_bin), int(spine.sum()))
        self.assertEqual(sum(result.cluster_voxels_by_bin), int(clusters.sum()))
        self.assertGreater(len(result.axis_points_zyx), 10)

        hinted_endpoint = centers[13]
        hinted = calculate_spine_distribution(
            spine,
            dendrite,
            clusters,
            sampling_zyx_um=(0.5, 0.1, 0.1),
            endpoint_hint_zyx=hinted_endpoint,
        )
        self.assertEqual(hinted.endpoint_source, "manual")
        self.assertEqual(hinted.endpoint_zyx, hinted_endpoint)
        self.assertEqual(sum(hinted.spine_voxels_by_bin), int(spine.sum()))


if __name__ == "__main__":
    unittest.main()
