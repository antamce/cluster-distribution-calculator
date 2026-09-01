from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile
import zarr

from synpo.detection import DetectionSettings, detect_project
from synpo.importer import scan_batch
from synpo.models import Calibration
from synpo.preprocessing import process_project_cache, project_cache_path
from synpo.project import create_project_manifest, load_project, save_project
from synpo.review import (
    ReviewAction,
    apply_review_action,
    load_review_slice,
    set_specimen_review_state,
    undo_last_review_action,
)
from synpo.visualization import generate_context_volume


@contextlib.contextmanager
def workspace_directory():
    root = Path.cwd() / ".test-work" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class ReviewTests(unittest.TestCase):
    @staticmethod
    def add_preprocessed_signal(
        manifest: dict[str, object], mask: np.ndarray, value: int = 5000
    ) -> None:
        key = manifest["specimens"][0]["checkpoints"]["preprocessing"]["channels"][
            "ChanB"
        ]["dataset_key"]
        dataset = zarr.open_group(str(project_cache_path(manifest)), mode="a")[key]
        data = np.asarray(dataset)
        data[mask] = value
        dataset[:] = data

    def make_detected_project(self, root: Path) -> tuple[dict[str, object], Path]:
        source = root / "source"
        source.mkdir()
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
        detect_project(manifest, project_path)
        return manifest, project_path

    def test_exclude_checkpoint_undo_and_reopen(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            initial = load_review_slice(manifest, 0, 3, "ChanB")
            self.assertFalse(initial.corrected)
            automatic_context = generate_context_volume(
                manifest,
                0,
                "ChanB",
                corrected=False,
                include_3d=False,
                maximum_3d_points=500,
            )
            self.assertEqual(automatic_context.xy.raw.shape, (64, 80))
            self.assertEqual(automatic_context.xz.raw.shape, (8, 80))
            self.assertEqual(automatic_context.yz.raw.shape, (8, 64))
            self.assertEqual(len(automatic_context.points_um), 0)
            cropped_context = generate_context_volume(
                manifest,
                0,
                "ChanB",
                corrected=False,
                include_3d=True,
                roi_xy=(0, 0, 40, 40),
                maximum_3d_points=500,
            )
            self.assertEqual(cropped_context.roi_xy, (0, 0, 40, 40))
            self.assertLessEqual(len(cropped_context.points_um), 500)
            self.assertGreater(len(cropped_context.mesh_faces), 0)
            mesh_z_layers = cropped_context.mesh_vertices_um[:, 2] / 0.5
            self.assertTrue(np.any(np.abs(mesh_z_layers - np.rint(mesh_z_layers)) > 0.1))
            self.assertTrue(
                np.all(cropped_context.points_um[:, 0] < 4.0)
                if len(cropped_context.points_um)
                else True
            )
            coordinate = np.argwhere(initial.dendrites > 0)[0]
            y, x = (int(value) for value in coordinate)
            object_id = int(initial.dendrites[y, x])

            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="dendrite",
                    operation="exclude",
                    z_index=3,
                    points=((x, y),),
                    brush_radius_pixels=1,
                    projection_hint=True,
                ),
            )
            corrected = load_review_slice(manifest, 0, 3, "ChanB")
            self.assertTrue(corrected.corrected)
            self.assertFalse(np.any(corrected.dendrites == object_id))
            self.assertEqual(result.edit_count, 1)
            corrected_context = generate_context_volume(
                manifest,
                0,
                "ChanB",
                corrected=True,
                maximum_3d_points=500,
            )
            self.assertTrue(corrected_context.corrected)
            self.assertFalse(np.any(corrected_context.xy.dendrites == object_id))

            reopened = load_project(project_path)
            checkpoint = reopened["specimens"][0]["checkpoints"]["review"]
            self.assertEqual(checkpoint["state"], "in_progress")
            self.assertEqual(checkpoint["edit_count"], 1)
            self.assertEqual(
                reopened["specimens"][0]["review"]["object_status"]["dendrite"][
                    str(object_id)
                ],
                "excluded",
            )

            undo = undo_last_review_action(reopened, project_path, 0)
            restored = load_review_slice(reopened, 0, 3, "ChanB")
            self.assertEqual(int(restored.dendrites[y, x]), object_id)
            self.assertEqual(undo.edit_count, 0)

            set_specimen_review_state(
                reopened,
                project_path,
                0,
                complete=True,
                comment="Checked dendrite boundary",
            )
            final = load_project(project_path)
            self.assertEqual(final["specimens"][0]["review"]["state"], "complete")
            self.assertEqual(
                final["specimens"][0]["review"]["comment"],
                "Checked dendrite boundary",
            )
            self.assertEqual(
                final["specimens"][0]["checkpoints"]["review"]["state"],
                "complete",
            )

            final["detection"]["settings"]["cluster_sensitivity"] = 0.9
            detect_project(final, project_path)
            invalidated = load_project(project_path)
            self.assertEqual(
                invalidated["specimens"][0]["review"]["state"],
                "needs_attention",
            )
            self.assertEqual(
                invalidated["specimens"][0]["checkpoints"]["review"]["state"],
                "not_started",
            )
            refreshed_view = load_review_slice(invalidated, 0, 3, "ChanB")
            self.assertFalse(refreshed_view.corrected)

    def test_each_add_stroke_creates_one_preprocessed_object_atomically(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            source = root / "source" / "batch_group_specimen_ChanB_registered.tif"
            raw = tifffile.imread(source)
            zz, yy, xx = np.ogrid[: raw.shape[0], : raw.shape[1], : raw.shape[2]]
            first_signal = (
                ((zz - 4) / 2.0) ** 2
                + ((yy - 9) / 3.0) ** 2
                + ((xx - 10) / 3.0) ** 2
                <= 1.0
            )
            second_signal = (
                ((zz - 4) / 2.0) ** 2
                + ((yy - 9) / 3.0) ** 2
                + ((xx - 67) / 3.0) ** 2
                <= 1.0
            )
            raw[first_signal | second_signal] = 5000
            tifffile.imwrite(
                source,
                raw,
                metadata={"axes": "ZYX"},
                photometric="minisblack",
            )
            self.add_preprocessed_signal(manifest, first_signal | second_signal)

            before = load_review_slice(manifest, 0, 4, "ChanB")
            self.assertEqual(int(before.spines[9, 10]), 0)
            self.assertEqual(int(before.spines[9, 67]), 0)
            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="spine",
                    operation="add",
                    z_index=4,
                    points=((10, 9), (16, 9), (67, 9), (40, 6)),
                    strokes=(((10, 9), (16, 9)), ((67, 9),), ((40, 6),)),
                    brush_radius_pixels=1,
                ),
            )
            self.assertTrue(result.checkpoint_written)
            self.assertEqual(len(result.new_ids), 2)
            self.assertEqual(
                [item["status"] for item in result.hint_results],
                ["created", "created", "skipped"],
            )
            corrected = load_review_slice(manifest, 0, 4, "ChanB")
            self.assertEqual(int(corrected.spines[9, 10]), result.new_ids[0])
            self.assertEqual(int(corrected.spines[9, 67]), result.new_ids[1])
            self.assertEqual(int(corrected.spines[9, 16]), 0)
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["review"]["edit_count"],
                1,
            )
            history = manifest["specimens"][0]["review"]["history"][-1]
            self.assertEqual(history["hint_count"], 3)
            self.assertEqual(len(history["bboxes"]), 2)

            undo_last_review_action(manifest, project_path, 0)
            restored = load_review_slice(manifest, 0, 4, "ChanB")
            self.assertEqual(int(restored.spines[9, 10]), 0)
            self.assertEqual(int(restored.spines[9, 67]), 0)

    def test_raw_only_signal_writes_no_mask_or_checkpoint(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            source = root / "source" / "batch_group_specimen_ChanB_registered.tif"
            raw = tifffile.imread(source)
            raw[0:2, 4:9, 38:43] = 6000
            tifffile.imwrite(
                source,
                raw,
                metadata={"axes": "ZYX"},
                photometric="minisblack",
            )
            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="spine",
                    operation="add",
                    z_index=0,
                    points=((40, 6),),
                    strokes=(((40, 6),),),
                    brush_radius_pixels=1,
                ),
            )
            self.assertFalse(result.checkpoint_written)
            self.assertEqual(result.new_ids, ())
            self.assertEqual(result.hint_results[0]["status"], "skipped")
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["review"]["edit_count"],
                0,
            )
            view = load_review_slice(manifest, 0, 0, "ChanB")
            self.assertEqual(int(view.spines[6, 40]), 0)

    def test_two_hints_split_one_connected_preprocessed_region(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            source = root / "source" / "batch_group_specimen_ChanB_registered.tif"
            raw = tifffile.imread(source)
            zz, yy, xx = np.ogrid[: raw.shape[0], : raw.shape[1], : raw.shape[2]]
            connected_signal = (
                ((zz - 4) / 2.0) ** 2
                + ((yy - 8) / 3.0) ** 2
                + ((xx - 40) / 10.0) ** 2
                <= 1.0
            )
            raw[connected_signal] = 5000
            tifffile.imwrite(
                source,
                raw,
                metadata={"axes": "ZYX"},
                photometric="minisblack",
            )
            self.add_preprocessed_signal(manifest, connected_signal)
            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="spine",
                    operation="add",
                    z_index=4,
                    points=((36, 8), (44, 8)),
                    strokes=(((36, 8),), ((44, 8),)),
                    brush_radius_pixels=1,
                ),
            )
            self.assertEqual(len(result.new_ids), 2)
            corrected = load_review_slice(manifest, 0, 4, "ChanB")
            self.assertEqual(int(corrected.spines[8, 36]), result.new_ids[0])
            self.assertEqual(int(corrected.spines[8, 44]), result.new_ids[1])
            self.assertNotEqual(
                int(corrected.spines[8, 36]), int(corrected.spines[8, 44])
            )


if __name__ == "__main__":
    unittest.main()
