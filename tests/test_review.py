from __future__ import annotations

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import tifffile
import zarr

from synpo.detection import DetectionSettings, detect_project, detection_cache_path
from synpo.importer import scan_batch
from synpo.models import Calibration
from synpo.preprocessing import process_project_cache, project_cache_path
from synpo.project import create_project_manifest, load_project, save_project
from synpo.review import (
    ALWAYS_LOW_MEMORY_REVIEW_MODE,
    ReviewAction,
    _segment_add_hint_group,
    apply_review_action,
    load_review_slice,
    review_cache_path,
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

    def test_two_touching_add_hints_become_one_new_object(self) -> None:
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
            self.assertEqual(len(result.new_ids), 1)
            corrected = load_review_slice(manifest, 0, 4, "ChanB")
            self.assertEqual(int(corrected.spines[8, 36]), result.new_ids[0])
            self.assertEqual(int(corrected.spines[8, 44]), result.new_ids[0])
            self.assertEqual(
                int(corrected.spines[8, 36]), int(corrected.spines[8, 44])
            )
            self.assertEqual(
                [item["status"] for item in result.hint_results],
                ["created", "joined"],
            )

    def test_add_region_joins_one_existing_object_and_rejects_ambiguous_contact(self) -> None:
        shape = (3, 15, 15)
        processed = np.zeros(shape, dtype=np.float32)
        processed[1, 5:10, 6:10] = 1000.0
        target = np.zeros(shape, dtype=np.uint32)
        target[1, 7, 10] = 7
        hints = [
            {
                "index": 1,
                "points": ((7, 7),),
                "z_index": 1,
                "bbox": (0, 3, 0, 15, 0, 15),
            }
        ]
        output, next_id, results = _segment_add_hint_group(
            processed,
            target,
            None,
            (0, 3, 0, 15, 0, 15),
            hints,
            object_type="dendrite",
            brush_radius=1,
            sensitivity=1.0,
            preprocessing_threshold=10.0,
            next_id=20,
        )
        self.assertEqual(next_id, 20)
        self.assertEqual(results[0]["status"], "joined")
        self.assertEqual(results[0]["object_id"], 7)
        self.assertGreater(np.count_nonzero(output == 7), 1)

        ambiguous_target = target.copy()
        ambiguous_target[1, 7, 5] = 8
        rejected, next_id, results = _segment_add_hint_group(
            processed,
            ambiguous_target,
            None,
            (0, 3, 0, 15, 0, 15),
            hints,
            object_type="dendrite",
            brush_radius=1,
            sensitivity=1.0,
            preprocessing_threshold=10.0,
            next_id=20,
        )
        self.assertEqual(next_id, 20)
        self.assertEqual(results[0]["status"], "skipped")
        self.assertIn("multiple existing objects", results[0]["message"])
        np.testing.assert_array_equal(rejected, ambiguous_target)

    def test_add_action_joining_existing_object_writes_one_undoable_checkpoint(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            detection = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )["specimens/0000"]
            dendrites = np.zeros(detection["dendrite_labels"].shape, dtype=np.uint32)
            spines = np.zeros_like(dendrites)
            spines[2:6, 8:12, 20:24] = 1
            detection["dendrite_labels"][:] = dendrites
            detection["spine_labels"][:] = spines
            summary = dict(detection.attrs["summary"])
            summary.update({"dendrite_count": 0, "spine_count": 1})
            detection.attrs["summary"] = summary

            key = manifest["specimens"][0]["checkpoints"]["preprocessing"][
                "channels"
            ]["ChanB"]["dataset_key"]
            processed = zarr.open_group(
                str(project_cache_path(manifest)), mode="a"
            )[key]
            signal = np.zeros(processed.shape, dtype=np.uint16)
            signal[2:6, 8:12, 20:28] = 5000
            processed[:] = signal

            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="spine",
                    operation="add",
                    z_index=4,
                    points=((26, 10),),
                    strokes=(((26, 10),),),
                    brush_radius_pixels=1,
                    sensitivity=1.0,
                ),
            )
            self.assertTrue(result.checkpoint_written)
            self.assertEqual(result.new_ids, ())
            self.assertEqual(result.affected_ids, (1,))
            self.assertEqual(result.hint_results[0]["status"], "joined")
            corrected = load_review_slice(manifest, 0, 4, "ChanB")
            self.assertEqual(int(corrected.spines[10, 26]), 1)

            undo_last_review_action(manifest, project_path, 0)
            restored = load_review_slice(manifest, 0, 4, "ChanB")
            self.assertEqual(int(restored.spines[10, 22]), 1)
            self.assertEqual(int(restored.spines[10, 26]), 0)

    def test_projection_brush_transfers_only_dendrite_voxels_to_one_spine_and_undoes(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            detection = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )["specimens/0000"]
            dendrites = np.zeros(detection["dendrite_labels"].shape, dtype=np.uint32)
            spines = np.zeros_like(dendrites)
            dendrites[:, 30:36, 10:50] = 1
            spines[:, 25:30, 20:30] = 1
            detection["dendrite_labels"][:] = dendrites
            detection["spine_labels"][:] = spines
            summary = dict(detection.attrs["summary"])
            summary.update({"dendrite_count": 1, "spine_count": 1})
            detection.attrs["summary"] = summary

            points = tuple((25, y) for y in range(27, 35)) + ((5, 45),)
            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="spine",
                    operation="dendrite_to_spine",
                    z_index=3,
                    points=points,
                    brush_radius_pixels=1,
                    projection_hint=True,
                ),
            )
            self.assertGreater(result.transferred_voxel_count, 0)
            for z_index in range(dendrites.shape[0]):
                corrected = load_review_slice(
                    manifest, 0, z_index, "ChanB"
                )
                self.assertEqual(int(corrected.dendrites[32, 25]), 0)
                self.assertEqual(int(corrected.spines[32, 25]), 1)
                self.assertEqual(int(corrected.spines[45, 5]), 0)
                self.assertEqual(int(corrected.dendrites[32, 15]), 1)
                self.assertEqual(int(corrected.dendrites[32, 45]), 1)
            history = manifest["specimens"][0]["review"]["history"][-1]
            self.assertEqual(history["transferred_voxel_count"], result.transferred_voxel_count)
            self.assertEqual(history["source_dendrite_ids"], [1])

            undo_last_review_action(manifest, project_path, 0)
            restored = load_review_slice(manifest, 0, 3, "ChanB")
            self.assertEqual(int(restored.dendrites[32, 25]), 1)
            self.assertEqual(int(restored.spines[32, 25]), 0)

    def test_projection_transfer_rejects_multiple_spines_without_writing(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            detection = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )["specimens/0000"]
            dendrites = np.zeros(detection["dendrite_labels"].shape, dtype=np.uint32)
            spines = np.zeros_like(dendrites)
            dendrites[:, 30:36, 10:50] = 1
            spines[:, 25:30, 20:30] = 1
            spines[:, 36:41, 20:30] = 2
            detection["dendrite_labels"][:] = dendrites
            detection["spine_labels"][:] = spines
            summary = dict(detection.attrs["summary"])
            summary.update({"dendrite_count": 1, "spine_count": 2})
            detection.attrs["summary"] = summary
            points = tuple((25, y) for y in range(27, 40))
            with self.assertRaisesRegex(ValueError, "touched multiple spines"):
                apply_review_action(
                    manifest,
                    project_path,
                    0,
                    ReviewAction(
                        object_type="spine",
                        operation="dendrite_to_spine",
                        z_index=3,
                        points=points,
                        brush_radius_pixels=1,
                        projection_hint=True,
                    ),
                )
            corrected = load_review_slice(manifest, 0, 3, "ChanB")
            np.testing.assert_array_equal(corrected.dendrites, dendrites[3])
            np.testing.assert_array_equal(corrected.spines, spines[3])
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["review"].get(
                    "edit_count", 0
                ),
                0,
            )

    def test_projection_brush_transfers_all_covered_spines_to_one_dendrite_and_undoes(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            detection = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )["specimens/0000"]
            dendrites = np.zeros(detection["dendrite_labels"].shape, dtype=np.uint32)
            spines = np.zeros_like(dendrites)
            dendrites[:, 30:36, 10:50] = 1
            spines[:, 25:30, 20:30] = 1
            spines[:, 10, 40] = 2
            detection["dendrite_labels"][:] = dendrites
            detection["spine_labels"][:] = spines
            summary = dict(detection.attrs["summary"])
            summary.update({"dendrite_count": 1, "spine_count": 2})
            detection.attrs["summary"] = summary

            points = tuple((25, y) for y in range(27, 35)) + ((40, 10),)
            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="dendrite",
                    operation="spine_to_dendrite",
                    z_index=3,
                    points=points,
                    brush_radius_pixels=1,
                    projection_hint=True,
                ),
            )
            self.assertGreater(result.transferred_voxel_count, 0)
            self.assertEqual(result.dendrite_count, 1)
            self.assertEqual(result.spine_count, 1)
            for z_index in range(dendrites.shape[0]):
                corrected = load_review_slice(manifest, 0, z_index, "ChanB")
                self.assertEqual(int(corrected.spines[28, 25]), 0)
                self.assertEqual(int(corrected.dendrites[28, 25]), 1)
                self.assertEqual(int(corrected.spines[10, 40]), 0)
                self.assertEqual(int(corrected.dendrites[10, 40]), 1)
                self.assertEqual(int(corrected.spines[26, 21]), 1)
            history = manifest["specimens"][0]["review"]["history"][-1]
            self.assertEqual(history["source_spine_ids"], [1, 2])
            self.assertEqual(history["removed_spine_ids"], [2])

            undo = undo_last_review_action(manifest, project_path, 0)
            self.assertEqual(undo.spine_count, 2)
            restored = load_review_slice(manifest, 0, 3, "ChanB")
            np.testing.assert_array_equal(restored.dendrites, dendrites[3])
            np.testing.assert_array_equal(restored.spines, spines[3])

    def test_spine_transfer_rejects_multiple_dendrites_without_writing(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            detection = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )["specimens/0000"]
            dendrites = np.zeros(detection["dendrite_labels"].shape, dtype=np.uint32)
            spines = np.zeros_like(dendrites)
            dendrites[:, 30:35, 20:30] = 1
            dendrites[:, 40:45, 20:30] = 2
            spines[:, 35:40, 20:30] = 1
            detection["dendrite_labels"][:] = dendrites
            detection["spine_labels"][:] = spines
            summary = dict(detection.attrs["summary"])
            summary.update({"dendrite_count": 2, "spine_count": 1})
            detection.attrs["summary"] = summary

            with self.assertRaisesRegex(ValueError, "touched multiple dendrites"):
                apply_review_action(
                    manifest,
                    project_path,
                    0,
                    ReviewAction(
                        object_type="dendrite",
                        operation="spine_to_dendrite",
                        z_index=3,
                        points=tuple((25, y) for y in range(32, 43)),
                        brush_radius_pixels=1,
                        projection_hint=True,
                    ),
                )
            corrected = load_review_slice(manifest, 0, 3, "ChanB")
            np.testing.assert_array_equal(corrected.dendrites, dendrites[3])
            np.testing.assert_array_equal(corrected.spines, spines[3])
            self.assertEqual(
                manifest["specimens"][0]["checkpoints"]["review"].get(
                    "edit_count", 0
                ),
                0,
            )

    def test_trim_sensitivity_is_captured_per_action_and_higher_removes_more(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            detection = zarr.open_group(
                str(detection_cache_path(manifest)), mode="a"
            )["specimens/0000"]
            dendrites = np.zeros(detection["dendrite_labels"].shape, dtype=np.uint32)
            dendrites[:, 20:45, 20:60] = 1
            detection["dendrite_labels"][:] = dendrites
            detection["spine_labels"][:] = 0
            summary = dict(detection.attrs["summary"])
            summary.update({"dendrite_count": 1, "spine_count": 0})
            detection.attrs["summary"] = summary

            key = manifest["specimens"][0]["checkpoints"]["preprocessing"][
                "channels"
            ]["ChanB"]["dataset_key"]
            processed = zarr.open_group(
                str(project_cache_path(manifest)), mode="a"
            )[key]
            signal = np.zeros(processed.shape, dtype=np.uint16)
            signal[:, 20:45, 20:40] = 3000
            signal[:, 20:45, 40:60] = 800
            processed[:] = signal
            statistics = dict(processed.attrs["statistics"])
            statistics["applied_threshold"] = 1000.0
            processed.attrs["statistics"] = statistics

            def trim(sensitivity: float) -> int:
                apply_review_action(
                    manifest,
                    project_path,
                    0,
                    ReviewAction(
                        object_type="dendrite",
                        operation="trim",
                        z_index=3,
                        points=((42, 32),),
                        brush_radius_pixels=1,
                        sensitivity=sensitivity,
                    ),
                )
                group = zarr.open_group(
                    str(review_cache_path(manifest)), mode="r"
                )["specimens/0000"]
                return int(np.count_nonzero(np.asarray(group["dendrite_labels"])))

            low_count = trim(0.5)
            self.assertEqual(
                manifest["specimens"][0]["review"]["history"][-1]["sensitivity"],
                0.5,
            )
            undo_last_review_action(manifest, project_path, 0)
            high_count = trim(2.0)
            self.assertEqual(
                manifest["specimens"][0]["review"]["history"][-1]["sensitivity"],
                2.0,
            )
            self.assertLess(high_count, low_count)

    def test_forced_low_memory_correction_keeps_undo_available(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            manifest["review_settings"][
                "memory_mode"
            ] = ALWAYS_LOW_MEMORY_REVIEW_MODE
            initial = load_review_slice(manifest, 0, 3, "ChanB")
            y, x = (int(value) for value in np.argwhere(initial.dendrites > 0)[0])
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
                ),
            )
            self.assertEqual(result.processing_mode, "low_memory")
            self.assertFalse(
                np.any(load_review_slice(manifest, 0, 3, "ChanB").dendrites == object_id)
            )
            undo_last_review_action(manifest, project_path, 0)
            self.assertEqual(
                int(load_review_slice(manifest, 0, 3, "ChanB").dendrites[y, x]),
                object_id,
            )

    def test_forced_low_memory_split_labels_both_connected_sides(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            manifest["review_settings"][
                "memory_mode"
            ] = ALWAYS_LOW_MEMORY_REVIEW_MODE
            before = load_review_slice(manifest, 0, 3, "ChanB")
            object_id = int(before.dendrites[32, 40])
            self.assertGreater(object_id, 0)
            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="dendrite",
                    operation="split",
                    z_index=3,
                    points=((40, 32),),
                    brush_radius_pixels=4,
                ),
            )
            self.assertEqual(result.processing_mode, "low_memory")
            self.assertEqual(len(result.new_ids), 2)
            corrected = load_review_slice(manifest, 0, 3, "ChanB")
            self.assertNotEqual(
                int(corrected.dendrites[32, 30]),
                int(corrected.dendrites[32, 50]),
            )

    def test_forced_low_memory_expand_runs_disk_backed_propagation(self) -> None:
        with workspace_directory() as root:
            manifest, project_path = self.make_detected_project(root)
            manifest["review_settings"][
                "memory_mode"
            ] = ALWAYS_LOW_MEMORY_REVIEW_MODE
            before = load_review_slice(manifest, 0, 3, "ChanB")
            self.assertGreater(int(before.dendrites[32, 40]), 0)
            result = apply_review_action(
                manifest,
                project_path,
                0,
                ReviewAction(
                    object_type="dendrite",
                    operation="expand",
                    z_index=3,
                    points=((40, 32),),
                    brush_radius_pixels=1,
                ),
            )
            self.assertEqual(result.processing_mode, "low_memory")
            self.assertTrue(result.checkpoint_written)


if __name__ == "__main__":
    unittest.main()
