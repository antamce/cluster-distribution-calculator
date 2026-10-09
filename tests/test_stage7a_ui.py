from __future__ import annotations

import os
import shutil
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QAbstractItemView

from synpo.app import (
    BatchPreprocessWorker,
    ClusterCountScorePanel,
    CorrelationMatrixPanel,
    MainWindow,
    MorphologyPlotCanvas,
    PreviewWorker,
    REVIEW_BRUSH_COLORS,
    ReviewCanvas,
    ReviewWorker,
    SliceView,
    SpineMapView,
    SpineVolumeFilterDialog,
)
from synpo.detection import ALWAYS_LOW_MEMORY_MODE, DetectionSettings
from synpo.morphology import DEFAULT_FEATURES
from synpo.preprocessing import PreprocessingSettings
from synpo.project import default_preprocessing_manifest, migrate_manifest
from synpo.review import ReviewAction, ReviewResult


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
        menu_titles = [action.text().replace("&", "") for action in window.menuBar().actions()]
        self.assertIn("Advanced", menu_titles)
        self.assertFalse(window.diagnostic_mode_action.isChecked())
        self.assertTrue(window.diagnostic_full_action.isChecked())
        self.assertFalse(window.finish_diagnostic_action.isEnabled())
        self.assertTrue(window.open_diagnostic_protocol_action.isEnabled())
        self.assertEqual(window.distribution_review_mode.count(), 2)
        self.assertEqual(
            window.distribution_review_mode.itemData(1), "cluster_less"
        )
        self.assertIn(
            "all eligible spines",
            window.accept_all_distribution_spines.text().lower(),
        )
        window.manifest = {
            "specimens": [
                {"checkpoints": {"measurements": {"state": "complete"}}}
            ]
        }
        window.tabs.setTabEnabled(4, True)
        with patch(
            "synpo.app.filtered_measurement_result",
            return_value=(
                {
                    "distribution_rows": [
                        {
                            "spine_id": 1,
                            "spine_valid": True,
                            "volume_filter_excluded": False,
                            "distribution_axis_status": "bridged_gap",
                            "distribution_reviewed": False,
                            "distribution_included": False,
                            "bin_01_ratio": 0.25,
                        },
                        {
                            "spine_id": 2,
                            "spine_valid": True,
                            "volume_filter_excluded": False,
                            "distribution_axis_status": "ok",
                            "distribution_reviewed": True,
                            "distribution_included": True,
                            "bin_01_ratio": 0.5,
                        }
                    ]
                },
                [],
            ),
        ):
            window._sync_distribution_bulk_acceptance()
        self.assertTrue(window.accept_all_distribution_spines.isEnabled())
        self.assertFalse(window.accept_all_distribution_spines.isChecked())
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

    def test_spine_volume_filter_controls_and_preview_dialog(self) -> None:
        window = MainWindow()
        self.assertIn("Filter exported measurement workbook", window.filter_export_action.text())
        self.assertEqual(window.volume_filter_cutoff.suffix(), " µm³")
        rows = [
            {
                "experimental_group": "group-a",
                "specimen_id": "cell-1",
                "roi_id": 1,
                "dendrite_id": 2,
                "spine_id": 3,
                "volume_um3": 0.1,
                "spine_valid": True,
                "has_protein_cluster": True,
                "included_cluster_count": 1,
            }
        ]
        dialog = SpineVolumeFilterDialog(
            rows, cutoff_um3=0.2, enabled=True, parent=window
        )
        self.assertEqual(dialog.table.rowCount(), 1)
        self.assertIn("Would exclude 1", dialog.summary.text())
        dialog.table.item(0, 7).setCheckState(Qt.CheckState.Checked)
        self.app.processEvents()
        self.assertIn("Would exclude 0", dialog.summary.text())
        dialog.close()
        window.deleteLater()
        self.app.processEvents()

    def test_diagnostic_menu_starts_and_finalizes_full_session(self) -> None:
        root = Path.cwd() / ".test-work" / uuid.uuid4().hex
        root.mkdir(parents=True)
        try:
            window = MainWindow()
            window.output_edit.setText(str(root))
            window.diagnostic_mode_action.setChecked(True)
            self.assertIsNotNone(window._diagnostics)
            self.assertEqual(window._diagnostics.level, "full")
            directory = window._diagnostics.directory
            self.assertTrue(window.finish_diagnostic_action.isEnabled())
            self.assertFalse(window.diagnostic_level_menu.isEnabled())

            window.diagnostic_mode_action.setChecked(False)
            self.assertIsNone(window._diagnostics)
            self.assertTrue((directory / "session-summary.json").is_file())
            self.assertTrue(window.diagnostic_level_menu.isEnabled())
            window.deleteLater()
            self.app.processEvents()
        finally:
            shutil.rmtree(root)

    def test_worker_diagnostics_are_attached_only_to_review_worker(self) -> None:
        preview_results: list[object] = []
        preview = PreviewWorker(
            Path("unused.tif"),
            0,
            PreprocessingSettings(),
            0.1,
            0.5,
            None,
            ("preview",),
        )
        preview.completed.connect(preview_results.append)
        with patch("synpo.app.make_preview", return_value="preview-result"):
            preview.run()
        self.assertEqual(preview_results, [(('preview',), "preview-result")])

        batch_results: list[object] = []
        batch = BatchPreprocessWorker({}, Path("unused.synpo.json"))
        batch.completed.connect(batch_results.append)
        with patch("synpo.app.process_project_cache", return_value="batch-result"):
            batch.run()
        self.assertEqual(batch_results, ["batch-result"])

        diagnostic_events: list[tuple[str, str, dict[str, object]]] = []
        action = ReviewAction(
            object_type="spine",
            operation="accept",
            z_index=0,
            points=((1, 1),),
        )
        review_result = ReviewResult(
            specimen_index=0,
            action_id="test",
            operation="accept",
            object_type="spine",
            affected_ids=(1,),
            new_ids=(),
            edit_count=1,
            dendrite_count=1,
            spine_count=1,
        )
        worker = ReviewWorker(
            {},
            Path("unused.synpo.json"),
            0,
            action,
            diagnostic=lambda event, scope, details: diagnostic_events.append(
                (event, scope, details)
            ),
        )
        with patch("synpo.app.apply_review_action", return_value=review_result):
            worker.run()
        self.assertEqual(
            [event for event, _scope, _details in diagnostic_events],
            ["correction_started", "correction_finished"],
        )
        self.assertEqual(diagnostic_events[-1][2]["status"], "completed")

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
        for dendrite_x in (1, 2):
            for spine_x in (1, 2):
                self.assertNotEqual(
                    view._image.pixelColor(dendrite_x, 1),
                    view._image.pixelColor(spine_x, 3),
                )
        self.assertEqual(
            view._image.pixelColor(1, 5), view._image.pixelColor(2, 5)
        )

    def test_independent_preprocessing_invalidates_only_changed_specimen(self) -> None:
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
        window.sensitivity_spin.setValue(2.0)
        window._store_preprocessing_selection()
        self.assertEqual(manifest["preprocessing"]["special_specimens"], [])
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
        self.assertIn(
            "ChanA",
            manifest["preprocessing"]["parameters_set_by_specimen"]["0"],
        )
        window.deleteLater()
        self.app.processEvents()

    def test_review_shortcuts_match_the_approved_tool_order(self) -> None:
        window = MainWindow()
        shortcuts = {
            action.shortcut().toString(): action.text()
            for action in window.review_shortcut_actions
        }
        expected = {
            "Ctrl+1": "add",
            "Ctrl+2": "exclude",
            "Ctrl+3": "split",
            "Ctrl+4": "merge",
            "Ctrl+5": "spine_to_dendrite",
            "Ctrl+6": "dendrite_to_spine",
            "Ctrl+7": "trim",
            "Ctrl+8": "expand",
            "Ctrl+9": "erase",
            "Ctrl+0": "filopodium",
        }
        for shortcut, operation in expected.items():
            self.assertEqual(shortcuts[shortcut], f"Select {operation}")
        self.assertEqual(shortcuts["Return"], "Apply correction")
        window.deleteLater()
        self.app.processEvents()

    def test_morphology_panel_keeps_protein_descriptive_only(self) -> None:
        window = MainWindow()
        self.assertEqual(window.tabs.tabText(5), "6. Morphology clustering")
        self.assertEqual(
            {key for key, checkbox in window.morphology_features.items() if checkbox.isChecked()},
            set(DEFAULT_FEATURES),
        )
        self.assertFalse(
            any("protein" in key.casefold() for key in window.morphology_features)
        )
        self.assertEqual(
            {window.morphology_algorithm.itemData(index) for index in range(window.morphology_algorithm.count())},
            {"gaussian_mixture", "ward", "kmeans"},
        )
        self.assertEqual(
            window.morphology_groups.selectionMode(),
            QAbstractItemView.SelectionMode.MultiSelection,
        )
        self.assertEqual(window.morphology_axes_alpha.value(), 1.0)
        self.assertEqual(window.morphology_background_alpha.value(), 1.0)
        self.assertFalse(window.morphology_reviewed_only.isChecked())
        self.assertFalse(window._current_morphology_settings().reviewed_only)
        self.assertTrue(window.morphology_show_legend.isChecked())
        self.assertEqual(
            window.morphology_legend_position.currentData(), "outside_right"
        )
        self.assertGreaterEqual(
            window.morphology_plot_mode.findData("custom_features"), 0
        )
        self.assertGreaterEqual(
            window.morphology_plot_mode.findData("pca_interpretation"), 0
        )
        self.assertGreaterEqual(
            window.morphology_plot_mode.findData("pca_3d_features"), 0
        )
        self.assertTrue(window.morphology_pca_show_points.isChecked())
        self.assertAlmostEqual(window.morphology_pca_point_alpha.value(), 0.68)
        self.assertEqual(len(window.morphology_correlation_panel.feature_checks), 15)
        self.assertEqual(window.morphology_display_tabs.count(), 4)
        self.assertEqual(window.advanced_display_tabs.count(), 3)
        window.morphology_plot_mode.setCurrentIndex(
            window.morphology_plot_mode.findData("pca_interpretation")
        )
        self.assertFalse(window.morphology_pca_point_row.isHidden())
        window.deleteLater()
        self.app.processEvents()

    def test_correlation_panel_has_independent_features_groups_and_custom_colors(self) -> None:
        assignments = []
        for index in range(8):
            volume = 0.02 + index * 0.01
            assignments.append(
                {
                    "experimental_group": "control" if index < 4 else "treated",
                    "volume_um3": volume,
                    "spine_curvilinear_length_um": volume * 4.0,
                    "spine_base_to_tip_distance_um": 0.5 + index * 0.02,
                    "centerline_tortuosity": 1.0 + index * 0.01,
                    "maximum_width_um": 0.2 + index * 0.01,
                    "principal_axis_elongation": 1.2 + index * 0.03,
                }
            )
        result = {
            "settings": {"features": ["linear_volume", "curvilinear_length"]},
            "assignments": assignments,
            "plot_style": {
                "correlation_features": ["linear_volume", "curvilinear_length"],
                "correlation_scope": "control",
                "correlation_colormap": "custom",
                "correlation_negative_color": "#0011aa",
                "correlation_zero_color": "#eeeeee",
                "correlation_positive_color": "#aa1100",
                "correlation_alpha": 0.75,
            },
        }
        panel = CorrelationMatrixPanel()
        panel.show_run(result)
        self.assertEqual(panel.scope.currentData(), "control")
        self.assertEqual(panel.colormap.currentData(), "custom")
        self.assertAlmostEqual(panel.alpha.value(), 0.75)
        self.assertEqual(panel._data["included_spine_count"], 4)
        self.assertIn("fewer than 30", panel.summary.text())
        settings = panel.settings()
        self.assertEqual(
            settings["correlation_features"],
            ["linear_volume", "curvilinear_length"],
        )
        self.assertEqual(settings["correlation_positive_color"], "#aa1100")
        panel.close()

    def test_morphology_puncta_and_custom_feature_plots_render(self) -> None:
        result = {
            "settings": {
                "features": ["log_volume", "curvilinear_length"],
                "pca_dimensions": 2,
            },
            "cluster_definitions": [
                {"morphology_cluster_id": 1, "color": "#e63946"},
                {"morphology_cluster_id": 2, "color": "#457b9d"},
            ],
            "pca_loadings": [
                {
                    "feature": "log_volume",
                    "pca_1_loading": 0.8,
                    "pca_2_loading": -0.6,
                    "pca_1_explained_fraction": 0.7,
                    "pca_2_explained_fraction": 0.3,
                },
                {
                    "feature": "curvilinear_length",
                    "pca_1_loading": 0.6,
                    "pca_2_loading": 0.8,
                    "pca_1_explained_fraction": 0.7,
                    "pca_2_explained_fraction": 0.3,
                },
            ],
            "feature_centers": [-1.2, 1.0],
            "feature_scales": [0.5, 0.8],
            "pca_feature_means": [0.0, 0.0],
            "assignments": [
                {
                    "morphology_cluster_id": 1,
                    "experimental_group": "control",
                    "volume_um3": 0.02,
                    "spine_curvilinear_length_um": 0.7,
                    "has_protein_cluster": True,
                    "protein_puncta_total_volume_um3": 0.003,
                    "pca_1": -0.8,
                    "pca_2": 0.2,
                },
                {
                    "morphology_cluster_id": 2,
                    "experimental_group": "treated",
                    "volume_um3": 0.12,
                    "spine_curvilinear_length_um": 1.8,
                    "has_protein_cluster": True,
                    "protein_puncta_total_volume_um3": 0.011,
                    "pca_1": 0.9,
                    "pca_2": -0.1,
                },
            ],
            "plot_style": {
                "show_legend": True,
                "legend_position": "outside_right",
                "custom_x_feature": "log_volume",
                "custom_y_feature": "curvilinear_length",
            },
        }
        canvas = MorphologyPlotCanvas()
        canvas.show_run(result, "protein_volume")
        puncta_axis = canvas.figure.axes[0]
        self.assertGreater(len(puncta_axis.patches), 0)
        self.assertIn("Protein puncta volume", puncta_axis.get_ylabel())
        self.assertIsNotNone(puncta_axis.get_legend())
        puncta_labels = {
            text.get_text() for text in puncta_axis.get_legend().get_texts()
        }
        self.assertTrue(
            {
                "Cluster 1 box",
                "Cluster 2 box",
                "Colored points: individual protein-positive spines",
                "Hollow black points: box-plot outliers (same spines)",
                "Experimental group: control",
                "Experimental group: treated",
            }.issubset(puncta_labels)
        )

        canvas.show_run(result, "custom_features")
        axis = canvas.figure.axes[0]
        self.assertEqual(axis.get_xlabel(), "Log spine volume")
        self.assertEqual(axis.get_ylabel(), "Curvilinear length")
        self.assertIsNotNone(axis.get_legend())

        result["plot_style"]["show_legend"] = False
        canvas.show_run(result, "custom_features")
        self.assertIsNone(canvas.figure.axes[0].get_legend())

        canvas.show_run(result, "pca_interpretation")
        self.assertGreaterEqual(len(canvas.figure.axes), 4)
        self.assertTrue(
            any(
                axis.get_title().startswith("PCA scores")
                for axis in canvas.figure.axes
            )
        )
        result["plot_style"].update(
            {"pca_show_points": True, "pca_point_alpha": 0.12}
        )
        canvas.show_run(result, "pca_interpretation")
        score_axis = canvas.figure.axes[0]
        self.assertTrue(
            any(collection.get_alpha() == 0.12 for collection in score_axis.collections)
        )
        result["plot_style"]["pca_show_points"] = False
        canvas.show_run(result, "pca_interpretation")
        self.assertEqual(len(canvas.figure.axes[0].collections), 0)
        result["settings"].update(
            {
                "reduction_method": "pcumap",
                "embedding_dimensions": 5,
                "embedding_plot_dimensions": 3,
            }
        )
        for index, assignment in enumerate(result["assignments"]):
            assignment.update(
                {
                    "embedding_1": float(index),
                    "embedding_2": float(index) / 2.0,
                    "embedding_3": -float(index),
                }
            )
        canvas.show_run(result, "embedding")
        embedding_axis = canvas.figure.axes[0]
        self.assertEqual(embedding_axis.name, "3d")
        self.assertEqual(embedding_axis.get_xlabel(), "PCC/PCUMAP 1")
        self.assertIn("5D PCC/PCUMAP clustering space", embedding_axis.get_title())
        canvas.close()

    def test_cluster_count_score_panel_plots_and_tabulates_each_method(self) -> None:
        result = {
            "settings": {"cluster_count_selection": "silhouette"},
            "selected_cluster_count": 3,
            "candidate_diagnostics": [
                {
                    "cluster_count": 1,
                    "criterion": 140.0,
                    "criterion_name": "SSE-BIC",
                    "silhouette": None,
                    "within_cluster_sse": 120.0,
                    "elbow_distance": 0.0,
                    "cluster_sizes": [60],
                    "accepted": True,
                    "selected": False,
                },
                {
                    "cluster_count": 2,
                    "criterion": 105.0,
                    "criterion_name": "SSE-BIC",
                    "silhouette": 0.51,
                    "within_cluster_sse": 70.0,
                    "elbow_distance": 0.31,
                    "cluster_sizes": [31, 29],
                    "accepted": True,
                    "selected": False,
                },
                {
                    "cluster_count": 3,
                    "criterion": 101.0,
                    "criterion_name": "SSE-BIC",
                    "silhouette": 0.62,
                    "within_cluster_sse": 51.0,
                    "elbow_distance": 0.42,
                    "cluster_sizes": [20, 19, 21],
                    "accepted": True,
                    "selected": True,
                },
                {
                    "cluster_count": 4,
                    "criterion": 110.0,
                    "criterion_name": "SSE-BIC",
                    "silhouette": 0.48,
                    "within_cluster_sse": 45.0,
                    "elbow_distance": 0.0,
                    "cluster_sizes": [3, 17, 19, 21],
                    "accepted": False,
                    "selected": False,
                },
            ],
        }
        panel = ClusterCountScorePanel()
        panel.show_run(result)
        self.assertEqual(panel.score_method.currentData(), "silhouette")
        self.assertEqual(panel.table.rowCount(), 4)
        axis = panel.figure.axes[0]
        self.assertEqual(axis.get_xlabel(), "Number of clusters")
        self.assertEqual(axis.get_ylabel(), "Mean silhouette score")
        self.assertIn("selected 3 clusters", panel.summary.text())
        panel.score_method.setCurrentIndex(panel.score_method.findData("elbow"))
        self.app.processEvents()
        self.assertEqual(panel.figure.axes[0].get_ylabel(), "Within-cluster SSE")
        self.assertEqual(panel.table.item(2, 1).text(), "51")
        self.assertEqual(panel.table.item(3, 3).text(), "No")
        panel.close()

    def test_morphology_checkpoint_advances_from_the_edited_spine(self) -> None:
        window = MainWindow()
        window.morphology_spine.clear()
        for spine_id in (11, 12, 13):
            window.morphology_spine.addItem(f"Spine {spine_id}", spine_id)
        window.morphology_spine.setCurrentIndex(1)
        with patch.object(window, "_prepare_measurements_tab"), patch.object(
            window, "_prepare_morphology_tab"
        ):
            window._morphology_edit_completed(
                {"spine_id": 12, "advance": True, "result": {}}
            )
        self.assertEqual(window.morphology_spine.currentData(), 13)
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

    def test_advanced_clustering_tab_exposes_real_reducer_controls(self) -> None:
        window = MainWindow()
        self.assertEqual(window.tabs.tabText(6), "7. Advanced clustering")
        self.assertEqual(window.advanced_reduction.currentData(), "umap")
        self.assertEqual(window.advanced_embedding_dimensions.value(), 5)
        self.assertEqual(window.advanced_plot_dimensions.currentData(), 2)
        self.assertEqual(window.morphology_cluster_count_selection.count(), 3)
        self.assertEqual(
            window.morphology_cluster_count_selection.currentData(),
            "information_criterion",
        )
        self.assertEqual(window.advanced_cluster_count_selection.count(), 3)
        window.advanced_cluster_count_selection.setCurrentIndex(
            window.advanced_cluster_count_selection.findData("silhouette")
        )
        self.assertFalse(window.advanced_pcumap_group.isVisible())
        window.advanced_reduction.setCurrentIndex(
            window.advanced_reduction.findData("pcumap")
        )
        self.assertEqual(window.advanced_reduction.currentData(), "pcumap")
        settings = window._current_advanced_settings()
        self.assertEqual(settings.reduction_method, "pcumap")
        self.assertEqual(settings.embedding_dimensions, 5)
        self.assertEqual(settings.cluster_count_selection, "silhouette")
        window.advanced_fixed_clusters.setValue(3)
        self.assertFalse(window.advanced_cluster_count_selection.isEnabled())
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
