from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from synpo.app import MainWindow, REVIEW_BRUSH_COLORS, ReviewCanvas, SliceView, SpineMapView
from synpo.detection import ALWAYS_LOW_MEMORY_MODE, DetectionSettings
from synpo.preprocessing import PreprocessingSettings
from synpo.project import default_preprocessing_manifest, migrate_manifest


class Stage7AUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_zoom_mapping_fit_and_native(self) -> None:
        view = SliceView("test")
        view.resize(400, 300)
        view.show_rgb(np.zeros((100, 200, 3), dtype=np.uint8))
        view.show()
        self.app.processEvents()

        self.assertEqual(view.zoom_percent(), 200)
        self.assertEqual(view.image_coordinate(QPointF(200, 150)), (100, 50))
        self.assertIsNone(view.image_coordinate(QPointF(200, 20)))
        view.set_native_zoom()
        self.assertEqual(view.zoom_percent(), 100)
        self.assertEqual(view.image_coordinate(QPointF(200, 150)), (100, 50))
        view.reset_view()
        self.assertEqual(view.zoom_percent(), 200)
        view.close()

    def test_detection_overlay_remains_available_on_zoomable_slice_view(self) -> None:
        view = SliceView("detection")
        raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
        dendrites = np.zeros((8, 8), dtype=np.uint32)
        spines = np.zeros((8, 8), dtype=np.uint32)
        clusters = np.zeros((8, 8), dtype=np.uint32)
        dendrites[1:3, 1:3] = 1
        spines[3:5, 3:5] = 2
        clusters[5:7, 5:7] = 3
        view.show_detection(
            raw,
            0,
            63,
            dendrites=dendrites,
            spines=spines,
            clusters=clusters,
        )
        self.assertIsNotNone(view._image)
        self.assertFalse(view._image.isNull())

    def test_sensitivity_limits_and_nonblocking_warnings(self) -> None:
        PreprocessingSettings(threshold_sensitivity=10.0).validate()
        DetectionSettings(
            dendrite_sensitivity=10.0, cluster_sensitivity=10.0
        ).validate()
        with self.assertRaises(ValueError):
            PreprocessingSettings(threshold_sensitivity=10.01).validate()
        with self.assertRaises(ValueError):
            DetectionSettings(dendrite_sensitivity=10.01).validate()

        window = MainWindow()
        self.assertEqual(window.sensitivity_spin.maximum(), 10.0)
        self.assertEqual(window.dendrite_detection_sensitivity.maximum(), 10.0)
        self.assertEqual(window.cluster_detection_sensitivity.maximum(), 10.0)
        self.assertEqual(
            window.detection_memory_mode.itemData(1), ALWAYS_LOW_MEMORY_MODE
        )
        window.sensitivity_spin.setValue(3.5)
        window.cluster_detection_sensitivity.setValue(4.0)
        self.assertIn("High-sensitivity", window.preprocessing_sensitivity_warning.text())
        self.assertIn("protein-cluster", window.detection_sensitivity_warning.text())
        window.deleteLater()
        self.app.processEvents()

    def test_stage7c_review_controls_and_numbered_map_render(self) -> None:
        window = MainWindow()
        self.assertEqual(window.distribution_review_mode.count(), 2)
        self.assertEqual(
            window.distribution_review_mode.itemData(1), "cluster_less"
        )
        self.assertIn("numbered spine map", window.open_spine_map_button.text().lower())
        view = SpineMapView()
        view.resize(500, 400)
        raw = np.arange(2400, dtype=np.uint16).reshape(40, 60)
        spines = np.zeros((40, 60), dtype=np.uint32)
        spines[5:15, 8:20] = 3
        spines[20:32, 35:50] = 9
        clusters = np.zeros_like(spines)
        clusters[8:11, 12:15] = 1
        view.set_scene(raw, spines, clusters, {3, 9}, 9, focus_only=False)
        self.assertIsNotNone(view._image)
        self.assertEqual(view._visible_ids, {3, 9})
        self.assertEqual(view._image.pixelColor(6, 10).name(), "#00ffff")
        self.assertEqual(view._image.pixelColor(33, 25).name(), "#00ff00")
        self.assertEqual(window.review_memory_mode.count(), 2)
        self.assertFalse(window.context_status_widget.isVisible())
        view.close()
        window.deleteLater()
        self.app.processEvents()

    def test_review_brush_palette_changes_with_action(self) -> None:
        window = MainWindow()
        canvas = window.review_view
        self.assertIsInstance(canvas, ReviewCanvas)
        for operation, expected in REVIEW_BRUSH_COLORS.items():
            index = window.review_operation.findData(operation)
            window.review_operation.setCurrentIndex(index)
            self.assertEqual(canvas._hint_color.name(), expected.name())
        window.deleteLater()
        self.app.processEvents()

    def test_review_correction_controls_cover_transfer_sensitivity_and_large_diameter(self) -> None:
        window = MainWindow()
        self.assertEqual(window.review_brush_diameter.maximum(), 1000)
        self.assertEqual(window.review_brush_diameter.value(), 9)
        self.assertTrue(window.centralWidget().isAncestorOf(window.save_button))
        window.tabs.setTabEnabled(3, True)
        window.review_specimen.addItem("test specimen", 0)
        window._set_review_busy(False)

        add_index = window.review_operation.findData("add")
        window.review_operation.setCurrentIndex(add_index)
        self.assertTrue(window.review_sensitivity_widget.isEnabled())
        window.review_sensitivity.setValue(235)
        self.assertEqual(window.review_sensitivity_value.text(), "2.35")

        exclude_index = window.review_operation.findData("exclude")
        window.review_operation.setCurrentIndex(exclude_index)
        self.assertFalse(window.review_sensitivity_widget.isEnabled())

        transfer_index = window.review_operation.findData("dendrite_to_spine")
        window.review_operation.setCurrentIndex(transfer_index)
        self.assertEqual(window.review_view_mode.currentData(), "xy_max")
        self.assertEqual(window.review_object_type.currentData(), "spine")
        self.assertFalse(window.review_object_type.isEnabled())
        with patch("synpo.app.QMessageBox.warning") as warning:
            window._review_action_failed("Dendrite transfer touched multiple spines 1, 2.")
            warning.assert_not_called()
        self.assertIn("multiple spines", window.review_status.text())

        reverse_index = window.review_operation.findData("spine_to_dendrite")
        window.review_operation.setCurrentIndex(reverse_index)
        self.assertEqual(window.review_view_mode.currentData(), "xy_max")
        self.assertEqual(window.review_object_type.currentData(), "dendrite")
        self.assertFalse(window.review_object_type.isEnabled())
        self.assertFalse(window.review_sensitivity_widget.isEnabled())
        with patch("synpo.app.QMessageBox.warning") as warning:
            window._review_action_failed(
                "Spine transfer touched multiple dendrites 1, 2."
            )
            warning.assert_not_called()
        self.assertIn("multiple dendrites", window.review_status.text())
        window.deleteLater()
        self.app.processEvents()

    def test_distinct_review_colors_separate_dendrite_and_spine_ids_only(self) -> None:
        view = SliceView("distinct labels")
        raw = np.zeros((8, 8), dtype=np.uint16)
        dendrites = np.zeros((8, 8), dtype=np.uint32)
        spines = np.zeros((8, 8), dtype=np.uint32)
        clusters = np.zeros((8, 8), dtype=np.uint32)
        dendrites[1, 1] = 1
        dendrites[1, 2] = 2
        spines[3, 1] = 1
        spines[3, 2] = 2
        clusters[5, 1] = 1
        clusters[5, 2] = 2
        view.show_detection(
            raw,
            0,
            1,
            dendrites=dendrites,
            spines=spines,
            clusters=clusters,
            distinct_dendrites_spines=True,
        )
        self.assertNotEqual(
            view._image.pixelColor(1, 1), view._image.pixelColor(2, 1)
        )
        self.assertNotEqual(
            view._image.pixelColor(1, 3), view._image.pixelColor(2, 3)
        )
        self.assertEqual(
            view._image.pixelColor(1, 5), view._image.pixelColor(2, 5)
        )

    def test_special_preprocessing_invalidates_only_effectively_changed_pairs(self) -> None:
        window = MainWindow()
        manifest = {
            "application": {},
            "preprocessing": default_preprocessing_manifest(),
            "specimens": [
                {
                    "channels": {},
                    "checkpoints": {
                        "preprocessing": {
                            "state": "complete",
                            "channels": {"ChanA": {}, "ChanB": {}},
                        },
                        "detection": {"state": "complete"},
                        "review": {"state": "complete"},
                        "measurements": {"state": "complete"},
                    },
                    "review": {"state": "complete"},
                }
                for _ in range(2)
            ],
        }
        migrate_manifest(manifest)
        window.manifest = manifest
        window.preprocess_specimen.blockSignals(True)
        window.preprocess_specimen.addItem("first", 0)
        window.preprocess_specimen.addItem("second", 1)
        window.preprocess_specimen.setCurrentIndex(0)
        window.preprocess_specimen.blockSignals(False)
        window.special_preprocessing_check.blockSignals(True)
        window.special_preprocessing_check.setChecked(True)
        window.special_preprocessing_check.blockSignals(False)
        window.sensitivity_spin.setValue(2.0)
        window._store_preprocessing_selection()
        self.assertEqual(manifest["preprocessing"]["special_specimens"], [0])
        self.assertEqual(
            manifest["specimens"][0]["checkpoints"]["preprocessing"]["state"],
            "not_started",
        )
        self.assertEqual(
            manifest["specimens"][1]["checkpoints"]["preprocessing"]["state"],
            "complete",
        )
        self.assertEqual(
            manifest["preprocessing"]["settings_by_specimen"]["0"]["ChanA"][
                "threshold_sensitivity"
            ],
            2.0,
        )
        window.deleteLater()
        self.app.processEvents()

    def test_stage7d_flexible_import_controls_are_available(self) -> None:
        window = MainWindow()
        self.assertEqual(window.channel_a_marker.text(), "ChanA")
        self.assertEqual(window.channel_b_marker.text(), "ChanB")
        window.channel_a_marker.setText("cy")
        window.channel_b_marker.setText("cl")
        self.assertEqual(
            window._current_channel_markers(), {"ChanA": "cy", "ChanB": "cl"}
        )
        self.assertIn("manually choose", window.manual_pair_button.text().lower())
        window.deleteLater()
        self.app.processEvents()

    def test_batch_progress_widgets_show_counts_elapsed_and_eta(self) -> None:
        window = MainWindow()
        window._job_kind = "detection"
        window._progress_started_at = time.monotonic() - 10.0
        window._progress_last_at = window._progress_started_at
        window._progress_last_current = 0
        window._progress_last_total = 0
        window._progress_rate_ema = None
        window._update_progress(
            "Detecting protein clusters", 5, 10, "specimen 1: writing Z 5/10"
        )
        self.assertEqual(window.detection_progress_bar.value(), 5)
        self.assertEqual(window.detection_progress_bar.maximum(), 10)
        label = window.detection_progress_label.text()
        self.assertIn("5/10 (50%)", label)
        self.assertIn("elapsed", label)
        self.assertIn("remaining", label)
        window.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
