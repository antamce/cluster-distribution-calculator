from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from synpo.app import MainWindow, SliceView, SpineMapView
from synpo.detection import DetectionSettings
from synpo.preprocessing import PreprocessingSettings


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
        view.close()
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
