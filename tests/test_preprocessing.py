from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile
import zarr

from synpo.preprocessing import (
    PreprocessingSettings,
    effective_preprocessing_settings,
    estimate_stack_statistics,
    make_preview,
    process_stack_to_cache,
    process_project_cache,
)
from synpo.importer import scan_batch
from synpo.models import Calibration
from synpo.project import create_project_manifest, load_project, save_project


@contextlib.contextmanager
def workspace_directory():
    root = Path.cwd() / ".test-work" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class PreprocessingTests(unittest.TestCase):
    def make_stack(self, path: Path) -> np.ndarray:
        data = np.full((5, 32, 36), 100, dtype=np.uint16)
        data[:, 11:21, 14:24] = 1100
        data[2, 15:18, 18:21] = 5000
        tifffile.imwrite(
            path, data, metadata={"axes": "ZYX"}, photometric="minisblack"
        )
        return data

    def test_adaptive_preview_preserves_raw_and_removes_background(self) -> None:
        with workspace_directory() as root:
            source = root / "stack.tif"
            original = self.make_stack(source)
            settings = PreprocessingSettings(
                background_percentile=20.0,
                gaussian_sigma_xy_um=0.05,
                gaussian_sigma_z_um=0.0,
            )
            statistics = estimate_stack_statistics(
                source,
                settings,
                xy_um_per_pixel=0.05,
                z_step_um=0.5,
            )
            preview = make_preview(
                source,
                2,
                settings,
                xy_um_per_pixel=0.05,
                z_step_um=0.5,
                statistics=statistics,
            )
            np.testing.assert_array_equal(preview.raw, original[2])
            self.assertAlmostEqual(statistics.background, 100.0)
            self.assertEqual(float(preview.processed[0, 0]), 0.0)
            self.assertGreater(preview.processed[16, 19], preview.threshold)

    def test_compressed_cache_is_idempotent_and_resumable(self) -> None:
        with workspace_directory() as root:
            source = root / "stack.tif"
            original = self.make_stack(source)
            cache = root / "cache.zarr"
            settings = PreprocessingSettings(gaussian_sigma_xy_um=0.0)
            first = process_stack_to_cache(
                source,
                cache,
                "specimens/0000/ChanA/data",
                settings,
                xy_um_per_pixel=0.05,
                z_step_um=0.5,
                source_sha256="example",
            )
            second = process_stack_to_cache(
                source,
                cache,
                "specimens/0000/ChanA/data",
                settings,
                xy_um_per_pixel=0.05,
                z_step_um=0.5,
                source_sha256="example",
            )
            dataset = zarr.open_group(str(cache), mode="r")[
                "specimens/0000/ChanA/data"
            ]
            self.assertEqual(first.slices_written, original.shape[0])
            self.assertEqual(second.slices_written, 0)
            self.assertTrue(dataset.attrs["complete"])
            self.assertEqual(dataset.attrs["slices_completed"], original.shape[0])
            self.assertEqual(dataset.dtype, np.dtype("uint16"))
            self.assertEqual(int(dataset[2, 0, 0]), 0)

    def test_project_batch_writes_pair_checkpoint(self) -> None:
        with workspace_directory() as root:
            source = root / "source"
            source.mkdir()
            for channel in ("ChanA", "ChanB"):
                self.make_stack(
                    source / f"batch_group_specimen_{channel}_registered.tif"
                )
            manifest = create_project_manifest(
                scan_batch(source),
                output_directory=root / "output",
                channel_roles={
                    "ChanA": "protein_clusters",
                    "ChanB": "dendrite_spines",
                },
                calibration=Calibration("test", 0.05, 0.5),
            )
            project_path = save_project(root / "batch.synpo.json", manifest)
            result = process_project_cache(manifest, project_path)
            reopened = load_project(project_path)
            checkpoint = reopened["specimens"][0]["checkpoints"]["preprocessing"]
            self.assertEqual(checkpoint["state"], "complete")
            self.assertEqual(set(checkpoint["channels"]), {"ChanA", "ChanB"})
            self.assertEqual(result["total_slices"], 10)

    def test_special_pair_settings_override_defaults_and_remain_dormant(self) -> None:
        manifest = {
            "preprocessing": {
                "settings_by_channel": {
                    "ChanA": PreprocessingSettings(
                        threshold_sensitivity=1.0
                    ).to_dict(),
                    "ChanB": PreprocessingSettings().to_dict(),
                },
                "special_specimens": [0],
                "settings_by_specimen": {
                    "0": {
                        "ChanA": PreprocessingSettings(
                            threshold_sensitivity=2.5
                        ).to_dict()
                    }
                },
            }
        }
        self.assertEqual(
            effective_preprocessing_settings(
                manifest, 0, "ChanA"
            ).threshold_sensitivity,
            2.5,
        )
        manifest["preprocessing"]["special_specimens"] = []
        self.assertEqual(
            effective_preprocessing_settings(
                manifest, 0, "ChanA"
            ).threshold_sensitivity,
            1.0,
        )
        self.assertEqual(
            manifest["preprocessing"]["settings_by_specimen"]["0"]["ChanA"][
                "threshold_sensitivity"
            ],
            2.5,
        )


if __name__ == "__main__":
    unittest.main()
