from __future__ import annotations

import re
import sys
import time
from threading import Event
from pathlib import Path

import numpy as np
from scipy import ndimage
from PySide6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QImage, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .calibration import CalibrationStore
from .importer import inspect_manual_pair, scan_batch
from .models import Calibration, ScanReport, SpecimenPair
from .project import (
    channel_source_path,
    create_project_manifest,
    load_project,
    relink_project_sources,
    save_project,
    verify_project_sources,
)
from .preprocessing import (
    PreprocessingSettings,
    PreviewResult,
    ProcessingCancelled,
    StackStatistics,
    effective_preprocessing_settings,
    make_preview,
    process_project_cache,
)
from .detection import (
    ALWAYS_LOW_MEMORY_MODE,
    AUTOMATIC_MEMORY_MODE,
    DetectionSettings,
    DetectionSlice,
    detect_project,
    load_detection_slice,
)
from .review import (
    ALWAYS_LOW_MEMORY_REVIEW_MODE,
    AUTOMATIC_REVIEW_MEMORY_MODE,
    ReviewAction,
    ReviewSlice,
    apply_review_action,
    load_review_slice,
    set_specimen_review_state,
    undo_last_review_action,
)
from .visualization import (
    ContextVolume,
    ProjectionData,
    context_signature,
    generate_context_volume,
)
from .measurements import (
    ClusterTrimPreview,
    DistributionPreview,
    SpineReviewPreview,
    MeasurementSettings,
    cluster_end_comparison_rows,
    clear_centerline_endpoint_hint,
    distribution_summary_rows,
    load_cluster_trim_preview,
    load_distribution_preview,
    load_spine_review_preview,
    load_measurement_result,
    measure_project,
    set_centerline_endpoint_hint,
    set_distribution_review,
    set_spine_quality_review,
)
from .exporting import export_measurements


ROLE_LABELS = {
    "protein_clusters": "Protein clusters",
    "dendrite_spines": "Dendrites and spines",
}

VOLUME_KIND_LABELS = ("Dendrites", "Spines", "Protein clusters")
VOLUME_DEFAULT_COLORS = (QColor(55, 220, 85), QColor(35, 195, 245), QColor(245, 55, 200))
VOLUME_DEFAULT_OPACITIES = (0.75, 0.60, 1.0)
DISTRIBUTION_COLORS = (
    (35, 0, 75), (75, 3, 110), (112, 14, 117), (147, 37, 103), (177, 63, 82),
    (204, 93, 58), (224, 127, 36), (239, 167, 25), (246, 210, 42), (240, 249, 33),
)
REVIEW_BRUSH_COLORS = {
    "exclude": QColor("#ff3030"),
    "add": QColor("#2ecc71"),
    "trim": QColor("#ff00ff"),
    "expand": QColor("#2389ff"),
    "split": QColor("#ffe119"),
    "filopodium": QColor("#9b59b6"),
    "merge": QColor("#ED6291"),
    "accept": QColor("#22d3ee"),
    "needs_attention": QColor("#ff8c1a"),
}


def _label_colors(labels: np.ndarray, kind: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.uint64)
    hashed = values * np.uint64(2654435761 + kind * 7919)
    variation = ((hashed >> np.uint64(16)) & np.uint64(63)).astype(np.uint8)
    colors = np.zeros((*values.shape, 3), dtype=np.uint8)
    if kind == 0:
        colors[..., 0] = 25 + variation // 2
        colors[..., 1] = 170 + variation
        colors[..., 2] = 45 + variation // 3
    elif kind == 1:
        colors[..., 0] = variation
        colors[..., 1] = 170 + variation
        colors[..., 2] = 210 + variation // 2
    else:
        colors[..., 0] = 205 + variation // 2
        colors[..., 1] = 30 + variation
        colors[..., 2] = 165 + variation
    return colors


def _screen_limited_size(widget: QWidget, width: int, height: int) -> tuple[int, int]:
    screen = widget.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return width, height
    available = screen.availableGeometry()
    return (
        max(320, min(width, round(available.width() * 0.9))),
        max(240, min(height, round(available.height() * 0.9))),
    )


class SliceView(QLabel):
    zoom_changed = Signal(int)

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 200)
        self.setStyleSheet("background: #171717; color: #bdbdbd; border: 1px solid #444;")
        self._image: QImage | None = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._last_pan_position: QPointF | None = None
        self._cursor_before_pan = self.cursor()

    def _display_aspect_ratio(self) -> float:
        if self._image is None:
            return 1.0
        return self._image.width() / max(1, self._image.height())

    def _native_display_size(self) -> tuple[float, float]:
        if self._image is None:
            return 1.0, 1.0
        width = float(self._image.width())
        return width, width / max(1e-9, self._display_aspect_ratio())

    def _fit_scale(self) -> float:
        native_width, native_height = self._native_display_size()
        return max(
            1e-9,
            min(
                max(1, self.width()) / native_width,
                max(1, self.height()) / native_height,
            ),
        )

    def _display_scale(self) -> float:
        return self._fit_scale() * self._zoom

    def zoom_percent(self) -> int:
        return max(1, round(self._display_scale() * 100.0))

    def _target_rect(self, *, clamp_pan: bool = True) -> QRectF:
        native_width, native_height = self._native_display_size()
        scale = self._display_scale()
        target_width = native_width * scale
        target_height = native_height * scale
        if clamp_pan:
            maximum_x = max(0.0, (target_width - self.width()) / 2.0)
            maximum_y = max(0.0, (target_height - self.height()) / 2.0)
            self._pan.setX(max(-maximum_x, min(maximum_x, self._pan.x())))
            self._pan.setY(max(-maximum_y, min(maximum_y, self._pan.y())))
        return QRectF(
            (self.width() - target_width) / 2.0 + self._pan.x(),
            (self.height() - target_height) / 2.0 + self._pan.y(),
            target_width,
            target_height,
        )

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._render()
        self.zoom_changed.emit(self.zoom_percent())

    def set_native_zoom(self) -> None:
        self._zoom = max(0.05, min(40.0, 1.0 / self._fit_scale()))
        self._pan = QPointF(0.0, 0.0)
        self._render()
        self.zoom_changed.emit(self.zoom_percent())

    def image_coordinate(self, position: QPointF) -> tuple[int, int] | None:
        if self._image is None:
            return None
        target = self._target_rect()
        if not target.contains(position) or target.width() <= 0 or target.height() <= 0:
            return None
        column = int((position.x() - target.left()) * self._image.width() / target.width())
        row = int((position.y() - target.top()) * self._image.height() / target.height())
        return (
            min(self._image.width() - 1, max(0, column)),
            min(self._image.height() - 1, max(0, row)),
        )

    def show_array(
        self,
        array: np.ndarray,
        low: float,
        high: float,
        mask: np.ndarray | None = None,
    ) -> None:
        scale = max(1.0, float(high) - float(low))
        gray = np.clip((np.asarray(array, dtype=np.float32) - low) * 255.0 / scale, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        if mask is not None:
            selected = np.asarray(mask, dtype=bool)
            rgb[selected, 0] = 255
            rgb[selected, 1] = (rgb[selected, 1].astype(np.uint16) * 35 // 100).astype(np.uint8)
            rgb[selected, 2] = 210
        height, width = gray.shape
        self._image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self._render()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._render()
        self.zoom_changed.emit(self.zoom_percent())

    def show_rgb(self, rgb: np.ndarray) -> None:
        image = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width = image.shape[:2]
        self._image = QImage(
            image.data, width, height, image.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self._render()

    def show_detection(
        self,
        array: np.ndarray,
        low: float,
        high: float,
        *,
        dendrites: np.ndarray | None = None,
        spines: np.ndarray | None = None,
        clusters: np.ndarray | None = None,
    ) -> None:
        scale = max(1.0, float(high) - float(low))
        gray = np.clip(
            (np.asarray(array, dtype=np.float32) - low) * 255.0 / scale, 0, 255
        ).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        for labels, color in (
            (dendrites, np.array([35, 220, 70], dtype=np.float32)),
            (spines, np.array([0, 205, 255], dtype=np.float32)),
            (clusters, np.array([255, 40, 205], dtype=np.float32)),
        ):
            if labels is None:
                continue
            mask = np.asarray(labels) > 0
            rgb[mask] = np.clip(
                rgb[mask].astype(np.float32) * 0.3 + color * 0.7, 0, 255
            ).astype(np.uint8)
        height, width = gray.shape
        self._image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self._render()

    def _render(self) -> None:
        if self._image is None:
            return
        canvas = QPixmap(max(1, self.width()), max(1, self.height()))
        canvas.fill(QColor("#171717"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(
            self._target_rect(),
            self._image,
            QRectF(0, 0, self._image.width(), self._image.height()),
        )
        painter.end()
        self.setPixmap(canvas)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._image is None or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        position = event.position()
        old_target = self._target_rect()
        if old_target.width() <= 0 or old_target.height() <= 0:
            return
        anchor_x = (position.x() - old_target.left()) / old_target.width()
        anchor_y = (position.y() - old_target.top()) / old_target.height()
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        self._zoom = max(0.05, min(40.0, self._zoom * factor))
        new_target = self._target_rect(clamp_pan=False)
        self._pan.setX(
            position.x()
            - anchor_x * new_target.width()
            - (self.width() - new_target.width()) / 2.0
        )
        self._pan.setY(
            position.y()
            - anchor_y * new_target.height()
            - (self.height() - new_target.height()) / 2.0
        )
        self._render()
        self.zoom_changed.emit(self.zoom_percent())
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan_position = event.position()
            self._cursor_before_pan = self.cursor()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning and self._last_pan_position is not None:
            delta = event.position() - self._last_pan_position
            self._pan += delta
            self._last_pan_position = event.position()
            self._render()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning and event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._last_pan_position = None
            self.setCursor(self._cursor_before_pan)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ZoomControls(QWidget):
    def __init__(self, view: SliceView) -> None:
        super().__init__()
        self.view = view
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setMinimumWidth(82)
        layout.addWidget(self.zoom_label)
        fit_button = QPushButton("Fit image")
        fit_button.setToolTip("Fit the complete image in the available panel.")
        fit_button.clicked.connect(view.reset_view)
        layout.addWidget(fit_button)
        native_button = QPushButton("100%")
        native_button.setToolTip("Show one source image pixel per display pixel.")
        native_button.clicked.connect(view.set_native_zoom)
        layout.addWidget(native_button)
        view.zoom_changed.connect(self._set_zoom)
        self.setToolTip("Mouse wheel: zoom around pointer. Middle-button drag: pan.")
        QTimer.singleShot(0, lambda: self._set_zoom(view.zoom_percent()))

    @Slot(int)
    def _set_zoom(self, percent: int) -> None:
        self.zoom_label.setText(f"Zoom: {percent}%")


class EndpointHintView(SliceView):
    point_clicked = Signal(int, int)
    context_requested = Signal()

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self.point_mode = False

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton or self._image is None:
            super().mousePressEvent(event)
            return
        if not self.point_mode:
            self.context_requested.emit()
            event.accept()
            return
        coordinate = self.image_coordinate(event.position())
        if coordinate is None:
            return
        self.point_clicked.emit(*coordinate)
        event.accept()


class ClickableSliceView(SliceView):
    context_requested = Signal()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self._image is not None:
            self.context_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class SpineMapView(SliceView):
    spine_selected = Signal(int)

    def __init__(self) -> None:
        super().__init__("Numbered spine map")
        self._labels: np.ndarray | None = None
        self._visible_ids: set[int] = set()

    @staticmethod
    def _boundaries(labels: np.ndarray) -> np.ndarray:
        present = labels > 0
        interior = present.copy()
        interior[1:, :] &= labels[1:, :] == labels[:-1, :]
        interior[:-1, :] &= labels[:-1, :] == labels[1:, :]
        interior[:, 1:] &= labels[:, 1:] == labels[:, :-1]
        interior[:, :-1] &= labels[:, :-1] == labels[:, 1:]
        return present & ~interior

    def set_scene(
        self,
        raw: np.ndarray,
        spines: np.ndarray,
        clusters: np.ndarray,
        visible_ids: set[int],
        current_id: int | None,
        *,
        focus_only: bool,
    ) -> None:
        labels = np.asarray(spines, dtype=np.uint32)
        self._labels = labels
        self._visible_ids = set(visible_ids)
        low, high = np.percentile(raw, (0.5, 99.8))
        scale = max(1.0, float(high) - float(low))
        gray = np.clip(
            (np.asarray(raw, dtype=np.float32) - float(low)) * 255.0 / scale,
            0,
            255,
        ).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        cluster_mask = np.asarray(clusters) > 0
        rgb[cluster_mask] = np.clip(
            rgb[cluster_mask].astype(np.float32) * 0.35
            + np.asarray((245, 45, 205), dtype=np.float32) * 0.65,
            0,
            255,
        ).astype(np.uint8)
        filtered = np.where(np.isin(labels, list(visible_ids)), labels, 0).astype(
            np.uint32
        )
        boundary_pixels = self._boundaries(filtered)
        other = ndimage.binary_dilation(
            boundary_pixels & (filtered != int(current_id or 0)),
            structure=np.ones((3, 3), dtype=bool),
            iterations=2,
        )
        rgb[other] = (0, 255, 255)
        if current_id is not None:
            current = ndimage.binary_dilation(
                boundary_pixels & (filtered == current_id),
                structure=np.ones((3, 3), dtype=bool),
                iterations=2,
            )
            rgb[current] = (0, 255, 0)
        image = QImage(
            np.ascontiguousarray(rgb).data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        painter = QPainter(image)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        for spine_id in sorted(visible_ids):
            yy, xx = np.nonzero(labels == spine_id)
            if not len(xx):
                continue
            x, y = int(np.median(xx)), int(np.median(yy))
            color = QColor(255, 235, 25) if spine_id == current_id else QColor(230, 250, 255)
            painter.setPen(QPen(QColor(20, 20, 20), 3))
            painter.drawText(x + 3, y - 3, str(spine_id))
            painter.setPen(QPen(color, 1))
            painter.drawText(x + 3, y - 3, str(spine_id))
        painter.end()
        self._image = image
        self._render()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton or self._labels is None:
            super().mousePressEvent(event)
            return
        coordinate = self.image_coordinate(event.position())
        if coordinate is None:
            return
        x, y = coordinate
        spine_id = int(self._labels[y, x])
        if spine_id not in self._visible_ids:
            y0, y1 = max(0, y - 6), min(self._labels.shape[0], y + 7)
            x0, x1 = max(0, x - 6), min(self._labels.shape[1], x + 7)
            nearby = self._labels[y0:y1, x0:x1]
            candidates = nearby[np.isin(nearby, list(self._visible_ids))]
            if not len(candidates):
                return
            spine_id = int(np.bincount(candidates.astype(np.int64)).argmax())
        self.spine_selected.emit(spine_id)
        event.accept()


class SpineMapDialog(QDialog):
    spine_selected = Signal(int)

    def __init__(
        self,
        title: str,
        volume: ContextVolume,
        result: dict[str, object],
        slice_loader,
        current_id: int | None,
        *,
        focus_only: bool,
        parent=None,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*_screen_limited_size(self, 1100, 820))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.volume = volume
        self.result = result
        self.slice_loader = slice_loader
        self.current_id = current_id
        self.focus_only = focus_only
        self.rows = {
            int(row["spine_id"]): row for row in result.get("spine_rows", [])
        }
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.filter_combo = QComboBox()
        for label, value in (
            ("All spines", "all"),
            ("Cluster-positive", "positive"),
            ("Cluster-less", "cluster_less"),
            ("Valid", "valid"),
            ("Invalid", "invalid"),
        ):
            self.filter_combo.addItem(label, value)
        self.filter_combo.currentIndexChanged.connect(self._render_scene)
        controls.addWidget(QLabel("Show:"))
        controls.addWidget(self.filter_combo)
        self.plane_combo = QComboBox()
        self.plane_combo.addItem("XY maximum projection", "maximum")
        self.plane_combo.addItem("Single Z slice", "slice")
        self.plane_combo.currentIndexChanged.connect(self._plane_changed)
        controls.addWidget(QLabel("View:"))
        controls.addWidget(self.plane_combo)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setRange(0, max(0, volume.z_count - 1))
        self.z_slider.setEnabled(False)
        self.z_slider.valueChanged.connect(self._render_scene)
        controls.addWidget(self.z_slider, 1)
        self.z_label = QLabel("XY maximum")
        controls.addWidget(self.z_label)
        layout.addLayout(controls)
        self.view = SpineMapView()
        self.view.spine_selected.connect(self._spine_clicked)
        layout.addWidget(self.view, 1)
        layout.addWidget(ZoomControls(self.view))
        self.status = QLabel(
            "Click a numbered spine to open it in the appropriate review queue. "
            "Mouse wheel zooms; middle-button drag pans."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        if focus_only:
            self.filter_combo.setVisible(False)
        self._render_scene()

    def _visible_ids(self) -> set[int]:
        if self.focus_only:
            return set(self.rows)
        mode = str(self.filter_combo.currentData() or "all")
        return {
            spine_id
            for spine_id, row in self.rows.items()
            if mode == "all"
            or (mode == "positive" and bool(row.get("has_protein_cluster", False)))
            or (mode == "cluster_less" and not bool(row.get("has_protein_cluster", False)))
            or (mode == "valid" and bool(row.get("spine_valid", True)))
            or (mode == "invalid" and not bool(row.get("spine_valid", True)))
        }

    @Slot()
    def _plane_changed(self) -> None:
        single_slice = self.plane_combo.currentData() == "slice"
        self.z_slider.setEnabled(single_slice)
        self._render_scene()

    @Slot()
    def _render_scene(self) -> None:
        if self.plane_combo.currentData() == "slice":
            z_index = self.z_slider.value()
            loaded = self.slice_loader(z_index)
            raw, spines, clusters = loaded.raw, loaded.spines, loaded.clusters
            self.z_label.setText(f"Z {z_index + 1}/{self.volume.z_count}")
        else:
            raw = self.volume.xy.raw
            spines = self.volume.xy.spines
            clusters = self.volume.xy.clusters
            self.z_label.setText("XY maximum")
        self.view.set_scene(
            raw,
            spines,
            clusters,
            self._visible_ids(),
            self.current_id,
            focus_only=self.focus_only,
        )

    @Slot(int)
    def _spine_clicked(self, spine_id: int) -> None:
        self.current_id = spine_id
        self._render_scene()
        self.status.setText(
            f"Spine {spine_id} selected in the main review window."
        )
        self.spine_selected.emit(spine_id)


class DistributionChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(230)
        self._row: dict[str, object] | None = None
        self._mode = "line"
        self._fixed_scale = False

    def set_profile(self, row: dict[str, object] | None) -> None:
        self._row = row
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def set_fixed_scale(self, fixed: bool) -> None:
        self._fixed_scale = fixed
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        plot = QRectF(58, 24, max(40, self.width() - 82), max(40, self.height() - 76))
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        if not self._row:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No group distribution profile yet")
            return
        means = [self._row.get(f"bin_{index:02d}_mean") for index in range(1, 11)]
        sems = [self._row.get(f"bin_{index:02d}_sem") for index in range(1, 11)]
        finite = [float(value) for value in means if value is not None]
        if not finite:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No included distribution values")
            return
        maximum = 1.0 if self._fixed_scale else max(0.01, max(finite) * 1.15)
        points: list[QPoint] = []
        for index, value in enumerate(means):
            if value is None:
                continue
            x = plot.left() + (index + 0.5) * plot.width() / 10
            y = plot.bottom() - min(maximum, float(value)) / maximum * plot.height()
            point = QPoint(int(x), int(y))
            points.append(point)
            color = QColor(*DISTRIBUTION_COLORS[index])
            painter.setPen(QPen(color, 2))
            sem = float(sems[index] or 0.0)
            error = sem / maximum * plot.height()
            painter.drawLine(QPoint(int(x), int(y - error)), QPoint(int(x), int(y + error)))
            painter.drawLine(QPoint(int(x - 4), int(y - error)), QPoint(int(x + 4), int(y - error)))
            painter.drawLine(QPoint(int(x - 4), int(y + error)), QPoint(int(x + 4), int(y + error)))
            if self._mode == "bar":
                painter.fillRect(QRectF(x - plot.width() / 28, y, plot.width() / 14, plot.bottom() - y), color)
            else:
                painter.setBrush(color)
                painter.drawEllipse(point, 4, 4)
            painter.setPen(QColor(60, 60, 60))
            painter.drawText(QRectF(x - 15, plot.bottom() + 4, 30, 18), Qt.AlignmentFlag.AlignCenter, str(index + 1))
        if self._mode == "line" and len(points) > 1:
            painter.setPen(QPen(QColor(55, 90, 170), 2))
            painter.drawPolyline(QPolygon(points))
        painter.setPen(QColor(40, 40, 40))
        painter.drawText(QRectF(0, 0, self.width(), 20), Qt.AlignmentFlag.AlignCenter, f"{self._row.get('experimental_group', '')}: mean ± SEM across specimen means")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 24, plot.width(), 20), Qt.AlignmentFlag.AlignCenter, "Spine part: shaft → tip")
        painter.drawText(QRectF(4, plot.top(), 48, 20), Qt.AlignmentFlag.AlignRight, f"{maximum:.3g}")
        painter.drawText(QRectF(4, plot.bottom() - 10, 48, 20), Qt.AlignmentFlag.AlignRight, "0")

class ReviewCanvas(SliceView):
    hint_changed = Signal(int)

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self._base_image: QImage | None = None
        self._strokes: list[list[tuple[int, int]]] = []
        self._drawing = False
        self._brush_radius = 4
        self._hint_color = QColor(REVIEW_BRUSH_COLORS["add"])
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))
        self._draw_hints()

    def set_hint_color(self, color: QColor) -> None:
        self._hint_color = QColor(color)
        self._draw_hints()

    def clear_hint(self) -> None:
        self._strokes.clear()
        self._drawing = False
        self._draw_hints()
        self.hint_changed.emit(0)

    def undo_stroke(self) -> None:
        if self._strokes:
            self._strokes.pop()
            self._draw_hints()
            self.hint_changed.emit(len(self.hint_points()))

    def hint_points(self) -> tuple[tuple[int, int], ...]:
        return tuple(point for stroke in self._strokes for point in stroke)

    def hint_strokes(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(tuple(stroke) for stroke in self._strokes if stroke)

    def show_detection(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().show_detection(*args, **kwargs)
        self._base_image = self._image.copy() if self._image is not None else None
        self._draw_hints()

    def _image_position(self, position) -> tuple[int, int] | None:  # type: ignore[no-untyped-def]
        if self._base_image is None:
            return None
        return self.image_coordinate(position)

    def _append_to_stroke(self, point: tuple[int, int]) -> None:
        stroke = self._strokes[-1]
        if not stroke:
            stroke.append(point)
            return
        x0, y0 = stroke[-1]
        x1, y1 = point
        distance = max(abs(x1 - x0), abs(y1 - y0))
        steps = max(1, int(distance / max(1, self._brush_radius)))
        for step in range(1, steps + 1):
            sample = (
                round(x0 + (x1 - x0) * step / steps),
                round(y0 + (y1 - y0) * step / steps),
            )
            if sample != stroke[-1]:
                stroke.append(sample)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._image_position(event.position())
            if point is not None:
                self._strokes.append([])
                self._append_to_stroke(point)
                self._drawing = True
                self._draw_hints()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drawing:
            point = self._image_position(event.position())
            if point is not None:
                self._append_to_stroke(point)
                self._draw_hints()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            point = self._image_position(event.position())
            if point is not None:
                self._append_to_stroke(point)
            self._drawing = False
            self._draw_hints()
            self.hint_changed.emit(len(self.hint_points()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _draw_hints(self) -> None:
        if self._base_image is None:
            return
        image = self._base_image.copy()
        painter = QPainter(image)
        pen = QPen(
            self._hint_color,
            self._brush_radius * 2 + 1,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        for stroke in self._strokes:
            if len(stroke) == 1:
                painter.drawPoint(QPoint(*stroke[0]))
            for start, end in zip(stroke, stroke[1:]):
                painter.drawLine(QPoint(*start), QPoint(*end))
        painter.end()
        self._image = image
        self._render()


class ProjectionView(SliceView):
    coordinate_selected = Signal(str, int, int)
    selection_changed = Signal(object)

    def __init__(self, axis: str) -> None:
        super().__init__(f"{axis} maximum projection")
        self.axis = axis
        self._projection: ProjectionData | None = None
        self._crosshair = (0, 0, 0)
        self._display_aspect = 1.0
        self._visible_kinds = (True, True, True)
        self._selection_enabled = False
        self._selection_start: tuple[int, int] | None = None
        self._selection_end: tuple[int, int] | None = None
        self._selecting = False
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_projection(
        self,
        projection: ProjectionData,
        crosshair: tuple[int, int, int],
        *,
        xy_um_per_pixel: float,
        z_step_um: float,
    ) -> None:
        self._projection = projection
        self._crosshair = crosshair
        height, width = projection.raw.shape
        if self.axis == "XY":
            self._display_aspect = width / max(1, height)
        else:
            self._display_aspect = (width * xy_um_per_pixel) / max(
                xy_um_per_pixel, height * z_step_um
            )
        self._render_projection()

    def _display_aspect_ratio(self) -> float:
        return self._display_aspect

    def set_crosshair(self, crosshair: tuple[int, int, int]) -> None:
        self._crosshair = crosshair
        self._render_projection()

    def set_visible_kinds(self, visible: tuple[bool, bool, bool]) -> None:
        self._visible_kinds = visible
        self._render_projection()

    def enable_rectangle_selection(self, enabled: bool = True) -> None:
        self._selection_enabled = enabled
        self._selection_start = None
        self._selection_end = None
        self._selecting = False
        self._render_projection()

    def selected_rectangle(self) -> tuple[int, int, int, int] | None:
        if self._selection_start is None or self._selection_end is None:
            return None
        x0, y0 = self._selection_start
        x1, y1 = self._selection_end
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        return left, top, right + 1, bottom + 1

    def _render_projection(self) -> None:
        if self._projection is None:
            return
        raw = self._projection.raw
        low, high = np.percentile(raw, (0.5, 99.8))
        scale = max(1.0, float(high) - float(low))
        gray = np.clip(
            (raw.astype(np.float32) - float(low)) * 255.0 / scale, 0, 255
        ).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        for kind, labels in enumerate(
            (
                self._projection.dendrites,
                self._projection.spines,
                self._projection.clusters,
            )
        ):
            if not self._visible_kinds[kind]:
                continue
            mask = labels > 0
            if np.any(mask):
                colors = _label_colors(labels, kind)
                rgb[mask] = np.clip(
                    rgb[mask].astype(np.float32) * 0.25
                    + colors[mask].astype(np.float32) * 0.75,
                    0,
                    255,
                ).astype(np.uint8)
        height, width = gray.shape
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        painter = QPainter(image)
        painter.setPen(QPen(QColor(255, 230, 30, 220), 1))
        x, y, z = self._crosshair
        if self.axis == "XY":
            horizontal, vertical = y, x
        elif self.axis == "XZ":
            horizontal, vertical = z, x
        else:
            horizontal, vertical = z, y
        painter.drawLine(0, horizontal, width - 1, horizontal)
        painter.drawLine(vertical, 0, vertical, height - 1)
        if self._selection_start is not None and self._selection_end is not None:
            x0, y0 = self._selection_start
            x1, y1 = self._selection_end
            painter.setPen(QPen(QColor(255, 145, 20, 255), 3))
            painter.drawRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        painter.end()
        self._image = image
        self._render()

    def _event_image_coordinate(self, event) -> tuple[int, int] | None:  # type: ignore[no-untyped-def]
        return self.image_coordinate(event.position())

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        coordinate = self._event_image_coordinate(event)
        if coordinate is None:
            return
        if self._selection_enabled:
            self._selection_start = coordinate
            self._selection_end = coordinate
            self._selecting = True
            self._render_projection()
            event.accept()
            return
        column, row = coordinate
        self.coordinate_selected.emit(self.axis, column, row)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._selection_enabled and self._selecting:
            coordinate = self._event_image_coordinate(event)
            if coordinate is not None:
                self._selection_end = coordinate
                self._render_projection()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._selection_enabled and self._selecting:
            coordinate = self._event_image_coordinate(event)
            if coordinate is not None:
                self._selection_end = coordinate
            self._selecting = False
            self._render_projection()
            self.selection_changed.emit(self.selected_rectangle())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class Volume3DView(QWidget):
    rotation_changed = Signal(float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background: #111; color: white;")
        self._volume: ContextVolume | None = None
        self._rotation_x = 25.0
        self._rotation_y = 0.0
        self._rotation_z = -35.0
        self._zoom = 1.0
        self._last_mouse = None
        self._visible_kinds = (True, True, True)
        self._kind_colors = tuple(QColor(color) for color in VOLUME_DEFAULT_COLORS)
        self._kind_opacities = list(VOLUME_DEFAULT_OPACITIES)
        self._z_spacing_factor = 1.0

    def set_volume(self, volume: ContextVolume) -> None:
        self._volume = volume
        self._rotation_x = 25.0
        self._rotation_y = 0.0
        self._rotation_z = -35.0
        self._zoom = 1.0
        self.update()
        self.rotation_changed.emit(*self.rotation())

    @staticmethod
    def _bounded_rotation(value: float) -> float:
        return max(-180.0, min(180.0, float(value)))

    def rotation(self) -> tuple[float, float, float]:
        return self._rotation_x, self._rotation_y, self._rotation_z

    def set_rotation(self, x: float, y: float, z: float) -> None:
        values = tuple(self._bounded_rotation(value) for value in (x, y, z))
        if values == self.rotation():
            return
        self._rotation_x, self._rotation_y, self._rotation_z = values
        self.update()
        self.rotation_changed.emit(*values)

    def reset_rotation(self) -> None:
        self.set_rotation(25.0, 0.0, -35.0)

    def set_visible_kinds(self, visible: tuple[bool, bool, bool]) -> None:
        self._visible_kinds = visible
        self.update()

    def kind_color(self, kind: int) -> QColor:
        return QColor(self._kind_colors[kind])

    def set_kind_color(self, kind: int, color: QColor) -> None:
        colors = list(self._kind_colors)
        colors[kind] = QColor(color)
        self._kind_colors = tuple(colors)
        self.update()

    def reset_kind_colors(self) -> None:
        self._kind_colors = tuple(QColor(color) for color in VOLUME_DEFAULT_COLORS)
        self.update()

    def set_kind_opacity(self, kind: int, opacity: float) -> None:
        self._kind_opacities[kind] = max(0.05, min(1.0, float(opacity)))
        self.update()

    def kind_opacity(self, kind: int) -> float:
        return self._kind_opacities[kind]

    def set_z_spacing_factor(self, factor: float) -> None:
        self._z_spacing_factor = max(0.2, min(5.0, float(factor)))
        self.update()

    def z_spacing_factor(self) -> float:
        return self._z_spacing_factor

    def _rotate_coordinates(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        points = points.astype(np.float32, copy=False)
        centered = points - (points.min(axis=0) + points.max(axis=0)) / 2.0
        centered[:, 2] *= self._z_spacing_factor
        angle_x, angle_y, angle_z = np.deg2rad(self.rotation())
        cos_x, sin_x = np.float32(np.cos(angle_x)), np.float32(np.sin(angle_x))
        cos_y, sin_y = np.float32(np.cos(angle_y)), np.float32(np.sin(angle_y))
        cos_z, sin_z = np.float32(np.cos(angle_z)), np.float32(np.sin(angle_z))
        z_x = centered[:, 0] * cos_z - centered[:, 1] * sin_z
        z_y = centered[:, 0] * sin_z + centered[:, 1] * cos_z
        x_y = z_y * cos_x - centered[:, 2] * sin_x
        x_z = z_y * sin_x + centered[:, 2] * cos_x
        rotated = np.empty_like(centered)
        rotated[:, 0] = z_x * cos_y + x_z * sin_y
        rotated[:, 1] = x_y
        rotated[:, 2] = -z_x * sin_y + x_z * cos_y
        return rotated, centered, cos_z, sin_z, cos_x, sin_x

    def _render_mesh_image(self, render_width: int, render_height: int) -> QImage:
        assert self._volume is not None
        vertices = self._volume.mesh_vertices_um
        faces = self._volume.mesh_faces
        rotated, centered, _cy, _sy, _cp, _sp = self._rotate_coordinates(vertices)
        span = max(1e-6, float(np.ptp(centered, axis=0).max()))
        scale = min(render_width, render_height) * 0.78 * self._zoom / span
        screen_x = np.rint(rotated[:, 0] * scale + render_width / 2).astype(np.int32)
        screen_y = np.rint(-rotated[:, 1] * scale + render_height / 2).astype(np.int32)

        first = rotated[faces[:, 0]]
        edge_a = rotated[faces[:, 1]] - first
        edge_b = rotated[faces[:, 2]] - first
        normal_x = edge_a[:, 1] * edge_b[:, 2] - edge_a[:, 2] * edge_b[:, 1]
        normal_y = edge_a[:, 2] * edge_b[:, 0] - edge_a[:, 0] * edge_b[:, 2]
        normal_z = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
        normal_length = np.sqrt(
            normal_x * normal_x + normal_y * normal_y + normal_z * normal_z
        )
        lighting = 0.42 + 0.58 * np.abs(normal_z) / np.maximum(normal_length, 1e-6)
        face_depth = (
            rotated[faces[:, 0], 2]
            + rotated[faces[:, 1], 2]
            + rotated[faces[:, 2], 2]
        ) / 3.0

        image = QImage(
            render_width,
            render_height,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor("#111111"))
        compositor = QPainter(image)
        compositor.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for kind in range(3):
            if not self._visible_kinds[kind]:
                continue
            selected = np.flatnonzero(self._volume.mesh_face_kinds == kind)
            if not len(selected):
                continue
            selected = selected[np.argsort(face_depth[selected])]
            layer = QImage(
                render_width,
                render_height,
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            layer.fill(Qt.GlobalColor.transparent)
            layer_painter = QPainter(layer)
            layer_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            layer_painter.setPen(Qt.PenStyle.NoPen)
            base = self._kind_colors[kind]
            shade_cache: dict[int, QColor] = {}
            for face_index in selected:
                triangle = faces[face_index]
                xs = screen_x[triangle]
                ys = screen_y[triangle]
                if (
                    xs.max() < 0
                    or xs.min() >= render_width
                    or ys.max() < 0
                    or ys.min() >= render_height
                ):
                    continue
                shade = int(np.clip(round(float(lighting[face_index]) * 31), 0, 31))
                color = shade_cache.get(shade)
                if color is None:
                    factor = shade / 31.0
                    color = QColor(
                        round(base.red() * factor),
                        round(base.green() * factor),
                        round(base.blue() * factor),
                    )
                    shade_cache[shade] = color
                layer_painter.setBrush(color)
                layer_painter.drawPolygon(
                    QPolygon(
                        [
                            QPoint(int(xs[0]), int(ys[0])),
                            QPoint(int(xs[1]), int(ys[1])),
                            QPoint(int(xs[2]), int(ys[2])),
                        ]
                    )
                )
            layer_painter.end()
            compositor.setOpacity(self._kind_opacities[kind])
            compositor.drawImage(0, 0, layer)
            compositor.setOpacity(1.0)
        compositor.end()
        return image

    def _point_radius(
        self,
        scale: float,
        cos_yaw: float,
        sin_yaw: float,
        cos_pitch: float,
        sin_pitch: float,
    ) -> int:
        if self._volume is None:
            return 2
        xy = self._volume.xy_um_per_pixel
        z = self._volume.z_step_um * self._z_spacing_factor
        horizontal_extent = (
            0.5 * xy * scale * (abs(cos_yaw) + abs(sin_yaw))
        )
        vertical_extent = 0.5 * scale * (
            xy * (abs(sin_yaw) + abs(cos_yaw)) * abs(cos_pitch)
            + z * abs(sin_pitch)
        )
        return max(2, min(12, int(np.ceil(max(horizontal_extent, vertical_extent)))))

    def _rasterize_solid_objects(
        self,
        screen_x: np.ndarray,
        screen_y: np.ndarray,
        depth: np.ndarray,
        visible: np.ndarray,
        render_width: int,
        render_height: int,
        point_radius: int,
    ) -> np.ndarray:
        canvas = np.full(
            (render_height, render_width, 3), 17.0, dtype=np.float32
        )
        if self._volume is None or not np.any(visible):
            return canvas.astype(np.uint8)
        visible_depth = depth[visible]
        depth_low = float(visible_depth.min())
        depth_span = max(1e-6, float(visible_depth.max()) - depth_low)
        kinds = self._volume.point_kinds
        offsets = [
            (dy, dx)
            for dy in range(-point_radius, point_radius + 1)
            for dx in range(-point_radius, point_radius + 1)
            if dx * dx + dy * dy <= point_radius * point_radius
        ]
        canvas_pixels = canvas.reshape(-1, 3)
        for kind in range(3):
            selected = visible & (kinds == kind)
            if not np.any(selected):
                continue
            xs = screen_x[selected]
            ys = screen_y[selected]
            zs = depth[selected]
            depth_buffer = np.full(
                render_width * render_height, -np.inf, dtype=np.float32
            )
            for dy, dx in offsets:
                shifted_x = xs + dx
                shifted_y = ys + dy
                inside = (
                    (shifted_x >= 0)
                    & (shifted_x < render_width)
                    & (shifted_y >= 0)
                    & (shifted_y < render_height)
                )
                if np.any(inside):
                    flat = shifted_y[inside] * render_width + shifted_x[inside]
                    np.maximum.at(depth_buffer, flat, zs[inside])
            covered = np.isfinite(depth_buffer)
            if not np.any(covered):
                continue
            lighting = 0.58 + 0.42 * (
                (depth_buffer[covered] - depth_low) / depth_span
            )
            chosen = self._kind_colors[kind]
            base = np.asarray(
                (chosen.red(), chosen.green(), chosen.blue()), dtype=np.float32
            )
            surface = np.clip(lighting[:, None] * base[None, :], 0, 255)
            opacity = self._kind_opacities[kind]
            canvas_pixels[covered] = (
                canvas_pixels[covered] * (1.0 - opacity) + surface * opacity
            )
        return np.rint(canvas).astype(np.uint8)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._last_mouse is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._last_mouse
            self._last_mouse = event.position()
            self.set_rotation(
                self._rotation_x + delta.y() * 0.6,
                self._rotation_y,
                self._rotation_z + delta.x() * 0.6,
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._last_mouse = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._zoom = max(
            0.35, min(4.0, self._zoom * (1.12 ** (event.angleDelta().y() / 120.0)))
        )
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._zoom = 1.0
        self.reset_rotation()
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self._volume is None or not (
            len(self._volume.mesh_faces) or len(self._volume.points_um)
        ):
            painter.setPen(QColor("#dddddd"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No 3D mask surface"
            )
            painter.end()
            return
        render_width = max(64, self.width())
        render_height = max(64, self.height())
        if len(self._volume.mesh_faces):
            image = self._render_mesh_image(render_width, render_height)
        else:
            points = self._volume.points_um.astype(np.float32, copy=False)
            (
                rotated,
                centered,
                cos_yaw,
                sin_yaw,
                cos_pitch,
                sin_pitch,
            ) = self._rotate_coordinates(points)
            span = max(1e-6, float(np.ptp(centered, axis=0).max()))
            scale = min(render_width, render_height) * 0.78 * self._zoom / span
            screen_x = np.rint(
                rotated[:, 0] * scale + render_width / 2
            ).astype(np.int32)
            screen_y = np.rint(
                -rotated[:, 1] * scale + render_height / 2
            ).astype(np.int32)
            visible = (
                (screen_x >= 1)
                & (screen_x < render_width - 1)
                & (screen_y >= 1)
                & (screen_y < render_height - 1)
            )
            visible &= np.isin(
                self._volume.point_kinds,
                np.flatnonzero(self._visible_kinds).astype(np.uint8),
            )
            point_radius = self._point_radius(
                scale, cos_yaw, sin_yaw, cos_pitch, sin_pitch
            )
            canvas = self._rasterize_solid_objects(
                screen_x,
                screen_y,
                rotated[:, 2],
                visible,
                render_width,
                render_height,
                point_radius,
            )
            image = QImage(
                canvas.data,
                render_width,
                render_height,
                canvas.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
        painter.drawImage(self.rect(), image)
        painter.setPen(QColor("#eeeeee"))
        painter.drawText(16, 24, "Drag: rotate   Wheel: zoom   Double-click: reset")
        legend_x = 16
        for kind, label in enumerate(("Dendrites", "Spines", "Clusters")):
            painter.setPen(self._kind_colors[kind])
            painter.drawText(legend_x, 46, label)
            legend_x += (84, 60, 72)[kind]
        painter.end()


class ContextViewerDialog(QDialog):
    z_selected = Signal(int)

    def __init__(
        self, title: str, volume: ContextVolume, initial_z: int, parent=None
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*_screen_limited_size(self, 1220, 860))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.volume = volume
        y_count, x_count = volume.xy.raw.shape
        self._crosshair = (
            x_count // 2,
            y_count // 2,
            max(0, min(volume.z_count - 1, initial_z)),
        )
        outer = QVBoxLayout(self)
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Visible objects:"))
        self.context_overlay_checks: list[QCheckBox] = []
        for label in ("Dendrites", "Spines", "Protein clusters"):
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._visibility_changed)
            self.context_overlay_checks.append(checkbox)
            overlay_row.addWidget(checkbox)
        overlay_row.addStretch(1)
        outer.addLayout(overlay_row)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        projections_tab = QWidget()
        projection_layout = QGridLayout(projections_tab)
        self.projection_views: dict[str, ProjectionView] = {}
        for column, axis in enumerate(("XY", "XZ", "YZ")):
            label = QLabel(f"{axis} maximum projection")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            projection_layout.addWidget(label, 0, column)
            view = ProjectionView(axis)
            view.coordinate_selected.connect(self._projection_clicked)
            self.projection_views[axis] = view
            projection_layout.addWidget(view, 1, column)
            projection_layout.addWidget(ZoomControls(view), 2, column)
        help_label = QLabel(
            "Click any projection to link the yellow crosshairs and update the main Z slice. "
            "Orthogonal views use the confirmed physical voxel calibration."
        )
        help_label.setWordWrap(True)
        projection_layout.addWidget(help_label, 3, 0, 1, 3)
        self.tabs.addTab(projections_tab, "XY / XZ / YZ maxima")

        volume_tab = QWidget()
        volume_layout = QVBoxLayout(volume_tab)
        material_note = QLabel(
            "Dendrites and spines are translucent; protein clusters are opaque. "
            "Triangular surfaces are interpolated continuously between Z layers."
        )
        material_note.setWordWrap(True)
        volume_layout.addWidget(material_note)
        self.volume_view = Volume3DView()
        self.volume_view.set_volume(volume)
        rotation_group = QGroupBox("Rotation")
        rotation_layout = QGridLayout(rotation_group)
        self.volume_rotation_sliders: list[QSlider] = []
        self.volume_rotation_spins: list[QSpinBox] = []
        for row, (axis, initial) in enumerate(
            zip(("X", "Y", "Z"), self.volume_view.rotation())
        ):
            rotation_layout.addWidget(QLabel(f"{axis} axis:"), row, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-180, 180)
            slider.setSingleStep(1)
            slider.setPageStep(10)
            slider.setValue(round(initial))
            slider.valueChanged.connect(self._rotation_control_changed)
            self.volume_rotation_sliders.append(slider)
            rotation_layout.addWidget(slider, row, 1)
            spin = QSpinBox()
            spin.setRange(-180, 180)
            spin.setSingleStep(1)
            spin.setSuffix("°")
            spin.setValue(round(initial))
            spin.valueChanged.connect(self._rotation_control_changed)
            self.volume_rotation_spins.append(spin)
            rotation_layout.addWidget(spin, row, 2)
        reset_rotation = QPushButton("Reset rotation")
        reset_rotation.clicked.connect(self.volume_view.reset_rotation)
        rotation_layout.addWidget(reset_rotation, 0, 3, 3, 1)
        self.volume_view.rotation_changed.connect(self._rotation_view_changed)
        volume_layout.addWidget(rotation_group)
        render_row = QHBoxLayout()
        render_row.addWidget(QLabel("Z-layer spacing:"))
        self.volume_z_spacing = QDoubleSpinBox()
        self.volume_z_spacing.setRange(0.2, 5.0)
        self.volume_z_spacing.setDecimals(2)
        self.volume_z_spacing.setSingleStep(0.1)
        self.volume_z_spacing.setValue(1.0)
        self.volume_z_spacing.setSuffix("×")
        self.volume_z_spacing.setToolTip(
            "Display only. 1.00× uses the confirmed physical Z calibration; "
            "measurements and masks are never changed."
        )
        render_row.addWidget(self.volume_z_spacing)
        reset_spacing = QPushButton("Use calibrated spacing")
        reset_spacing.clicked.connect(lambda: self.volume_z_spacing.setValue(1.0))
        render_row.addWidget(reset_spacing)
        render_row.addStretch(1)
        volume_layout.addLayout(render_row)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Surface opacity:"))
        self.volume_opacity_spins: list[QSpinBox] = []
        for kind, label in enumerate(("Dendrites", "Spines")):
            opacity_row.addWidget(QLabel(f"{label}:"))
            control = QSpinBox()
            control.setRange(5, 100)
            control.setSingleStep(5)
            control.setSuffix("%")
            control.setValue(round(VOLUME_DEFAULT_OPACITIES[kind] * 100))
            control.setToolTip(
                "Display and snapshot only; segmentation and measurements are unchanged."
            )
            control.valueChanged.connect(
                lambda value, index=kind: self.volume_view.set_kind_opacity(
                    index, value / 100.0
                )
            )
            self.volume_opacity_spins.append(control)
            opacity_row.addWidget(control)
        opacity_row.addWidget(QLabel("Protein clusters: 100% (opaque)"))
        reset_opacity = QPushButton("Reset opacity")
        reset_opacity.clicked.connect(self._reset_volume_opacity)
        opacity_row.addWidget(reset_opacity)
        opacity_row.addStretch(1)
        volume_layout.addLayout(opacity_row)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("3D colors:"))
        self.volume_color_buttons: list[QPushButton] = []
        for kind, label in enumerate(VOLUME_KIND_LABELS):
            button = QPushButton(f"{label} color…")
            button.clicked.connect(
                lambda _checked=False, index=kind: self._choose_volume_color(index)
            )
            self.volume_color_buttons.append(button)
            color_row.addWidget(button)
        reset_colors = QPushButton("Reset colors")
        reset_colors.clicked.connect(self._reset_volume_colors)
        color_row.addWidget(reset_colors)
        color_row.addStretch(1)
        volume_layout.addLayout(color_row)
        self.volume_z_spacing.valueChanged.connect(
            self.volume_view.set_z_spacing_factor
        )
        self._refresh_volume_color_buttons()
        volume_layout.addWidget(self.volume_view, 1)
        save_snapshot = QPushButton("Save current 3D snapshot…")
        save_snapshot.clicked.connect(self._save_snapshot)
        volume_layout.addWidget(save_snapshot)
        self.tabs.addTab(volume_tab, "Rotatable 3D objects")
        self.tabs.setTabEnabled(
            1, bool(len(volume.mesh_faces) or len(volume.points_um))
        )
        self._refresh_projections()

    @Slot()
    def _rotation_control_changed(self) -> None:
        sender = self.sender()
        for slider, spin in zip(
            self.volume_rotation_sliders, self.volume_rotation_spins
        ):
            if sender is slider:
                spin.blockSignals(True)
                spin.setValue(slider.value())
                spin.blockSignals(False)
            elif sender is spin:
                slider.blockSignals(True)
                slider.setValue(spin.value())
                slider.blockSignals(False)
        self.volume_view.set_rotation(
            *(control.value() for control in self.volume_rotation_spins)
        )

    @Slot(float, float, float)
    def _rotation_view_changed(self, x: float, y: float, z: float) -> None:
        for slider, spin, value in zip(
            self.volume_rotation_sliders,
            self.volume_rotation_spins,
            (x, y, z),
        ):
            rounded = round(value)
            slider.blockSignals(True)
            spin.blockSignals(True)
            slider.setValue(rounded)
            spin.setValue(rounded)
            slider.blockSignals(False)
            spin.blockSignals(False)

    def _choose_volume_color(self, kind: int) -> None:
        color = QColorDialog.getColor(
            self.volume_view.kind_color(kind),
            self,
            f"Choose {VOLUME_KIND_LABELS[kind].lower()} color",
        )
        if color.isValid():
            self.volume_view.set_kind_color(kind, color)
            self._refresh_volume_color_buttons()

    def _reset_volume_colors(self) -> None:
        self.volume_view.reset_kind_colors()
        self._refresh_volume_color_buttons()

    def _reset_volume_opacity(self) -> None:
        for kind, control in enumerate(self.volume_opacity_spins):
            control.setValue(round(VOLUME_DEFAULT_OPACITIES[kind] * 100))

    def _refresh_volume_color_buttons(self) -> None:
        for kind, button in enumerate(self.volume_color_buttons):
            color = self.volume_view.kind_color(kind)
            text = "#111111" if color.lightness() > 145 else "#ffffff"
            button.setStyleSheet(
                f"background-color: {color.name()}; color: {text};"
            )

    def select_view(self, view: str) -> None:
        self.tabs.setCurrentIndex(1 if view == "3d" else 0)

    def _refresh_projections(self) -> None:
        for axis, projection in (
            ("XY", self.volume.xy),
            ("XZ", self.volume.xz),
            ("YZ", self.volume.yz),
        ):
            self.projection_views[axis].set_projection(
                projection,
                self._crosshair,
                xy_um_per_pixel=self.volume.xy_um_per_pixel,
                z_step_um=self.volume.z_step_um,
            )

    def _visibility_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        visible = tuple(
            checkbox.isChecked() for checkbox in self.context_overlay_checks
        )
        for view in self.projection_views.values():
            view.set_visible_kinds(visible)
        self.volume_view.set_visible_kinds(visible)

    @Slot(str, int, int)
    def _projection_clicked(self, axis: str, column: int, row: int) -> None:
        x, y, z = self._crosshair
        if axis == "XY":
            x, y = column, row
        elif axis == "XZ":
            x, z = column, row
        else:
            y, z = column, row
        self._crosshair = (x, y, max(0, min(self.volume.z_count - 1, z)))
        for view in self.projection_views.values():
            view.set_crosshair(self._crosshair)
        self.z_selected.emit(self._crosshair[2])

    def _save_snapshot(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self, "Save 3D snapshot", "synpo-3d-view.png", "PNG image (*.png)"
        )
        if selected:
            destination = Path(selected)
            if destination.suffix.lower() != ".png":
                destination = destination.with_suffix(".png")
            if not self.volume_view.grab().save(str(destination), "PNG"):
                QMessageBox.warning(self, "Cannot save snapshot", str(destination))


class AreaSelectionDialog(QDialog):
    area_selected = Signal(object)

    def __init__(self, title: str, volume: ContextVolume, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*_screen_limited_size(self, 900, 820))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        outer = QVBoxLayout(self)
        instructions = QLabel(
            "Drag an orange rectangle around only the structures needed in the 3D view. "
            "A smaller area renders faster and uses much less memory."
        )
        instructions.setWordWrap(True)
        outer.addWidget(instructions)
        self.selection_view = ProjectionView("XY")
        self.selection_view.set_projection(
            volume.xy,
            (volume.xy.raw.shape[1] // 2, volume.xy.raw.shape[0] // 2, 0),
            xy_um_per_pixel=volume.xy_um_per_pixel,
            z_step_um=volume.z_step_um,
        )
        self.selection_view.enable_rectangle_selection(True)
        self.selection_view.selection_changed.connect(self._selection_changed)
        outer.addWidget(self.selection_view, 1)
        outer.addWidget(ZoomControls(self.selection_view))
        self.selection_label = QLabel("No area selected.")
        outer.addWidget(self.selection_label)
        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.close)
        buttons.addWidget(cancel)
        buttons.addStretch(1)
        self.generate_button = QPushButton("Generate 3D view from selected area")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self._accept_area)
        buttons.addWidget(self.generate_button)
        outer.addLayout(buttons)

    @Slot(object)
    def _selection_changed(self, rectangle) -> None:  # type: ignore[no-untyped-def]
        if rectangle is None:
            self.generate_button.setEnabled(False)
            self.selection_label.setText("No area selected.")
            return
        x0, y0, x1, y1 = (int(value) for value in rectangle)
        valid = x1 - x0 >= 8 and y1 - y0 >= 8
        self.generate_button.setEnabled(valid)
        self.selection_label.setText(
            f"Selected X {x0}–{x1 - 1}, Y {y0}–{y1 - 1} "
            f"({x1 - x0} × {y1 - y0} pixels)."
            + ("" if valid else " Select at least 8 × 8 pixels.")
        )

    def _accept_area(self) -> None:
        rectangle = self.selection_view.selected_rectangle()
        if rectangle is not None:
            self.area_selected.emit(rectangle)
            self.close()


class PreviewWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        path: Path,
        z_index: int,
        settings: PreprocessingSettings,
        xy_um_per_pixel: float,
        z_step_um: float,
        statistics: StackStatistics | None,
        cache_key: tuple[object, ...],
    ) -> None:
        super().__init__()
        self.path = path
        self.z_index = z_index
        self.settings = settings
        self.xy_um_per_pixel = xy_um_per_pixel
        self.z_step_um = z_step_um
        self.statistics = statistics
        self.cache_key = cache_key

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Preview", 0, 1, f"Reading Z {self.z_index + 1}")
            result = make_preview(
                self.path,
                self.z_index,
                self.settings,
                xy_um_per_pixel=self.xy_um_per_pixel,
                z_step_um=self.z_step_um,
                statistics=self.statistics,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.progress.emit("Preview", 1, 1, f"Z {self.z_index + 1} ready")
        self.completed.emit((self.cache_key, result))


class BatchPreprocessWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, manifest: dict[str, object], project_path: Path) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = process_project_cache(
                self.manifest,
                self.project_path,
                progress=lambda phase, current, total, detail: self.progress.emit(
                    phase, current, total, detail
                ),
                cancel_event=self.cancel_event,
            )
        except ProcessingCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class DetectionWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    pair_completed = Signal(int, object)

    def __init__(self, manifest: dict[str, object], project_path: Path) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = detect_project(
                self.manifest,
                self.project_path,
                progress=lambda phase, current, total, detail: self.progress.emit(
                    phase, current, total, detail
                ),
                pair_completed=lambda index, summary: self.pair_completed.emit(
                    index, summary
                ),
                cancel_event=self.cancel_event,
            )
        except ProcessingCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class MeasurementWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, manifest: dict[str, object], project_path: Path) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = measure_project(
                self.manifest,
                self.project_path,
                progress=lambda phase, current, total, detail: self.progress.emit(
                    phase, current, total, detail
                ),
                cancel_event=self.cancel_event,
            )
        except ProcessingCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class ClusterTrimPreviewWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self, manifest: dict[str, object], specimen_index: int, cluster_id: int
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.specimen_index = specimen_index
        self.cluster_id = cluster_id

    @Slot()
    def run(self) -> None:
        try:
            preview = load_cluster_trim_preview(
                self.manifest,
                self.specimen_index,
                self.cluster_id,
                progress=lambda phase, current, total, detail: self.progress.emit(
                    phase, current, total, detail
                ),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(preview)


class DistributionPreviewWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        specimen_index: int,
        spine_id: int,
        review_mode: str = "cluster_positive",
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.specimen_index = specimen_index
        self.spine_id = spine_id
        self.review_mode = review_mode

    @Slot()
    def run(self) -> None:
        try:
            preview = (
                load_distribution_preview(
                    self.manifest, self.specimen_index, self.spine_id, margin_um=1.0
                )
                if self.review_mode == "cluster_positive"
                else load_spine_review_preview(
                    self.manifest, self.specimen_index, self.spine_id, margin_um=1.0
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(preview)


class CenterlineHintWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        project_path: Path,
        specimen_index: int,
        spine_id: int,
        point_zyx: tuple[int, int, int] | None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.specimen_index = specimen_index
        self.spine_id = spine_id
        self.point_zyx = point_zyx

    @Slot()
    def run(self) -> None:
        try:
            if self.point_zyx is None:
                result = clear_centerline_endpoint_hint(
                    self.manifest,
                    self.project_path,
                    self.specimen_index,
                    self.spine_id,
                )
            else:
                result = set_centerline_endpoint_hint(
                    self.manifest,
                    self.project_path,
                    self.specimen_index,
                    self.spine_id,
                    self.point_zyx,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class ExportWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        workbook_path: Path,
        validation_pdf: bool,
        excluded_pdf: bool,
        invalid_pdf: bool,
        margin_um: float,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.workbook_path = workbook_path
        self.validation_pdf = validation_pdf
        self.excluded_pdf = excluded_pdf
        self.invalid_pdf = invalid_pdf
        self.margin_um = margin_um

    @Slot()
    def run(self) -> None:
        try:
            result = export_measurements(
                self.manifest,
                self.workbook_path,
                validation_pdf=self.validation_pdf,
                excluded_audit_pdf=self.excluded_pdf,
                invalid_audit_pdf=self.invalid_pdf,
                pdf_margin_um=self.margin_um,
                progress=lambda phase, current, total, detail: self.progress.emit(
                    phase, current, total, detail
                ),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class ReviewWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        project_path: Path,
        specimen_index: int,
        action: ReviewAction | None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.specimen_index = specimen_index
        self.action = action

    @Slot()
    def run(self) -> None:
        try:
            if self.action is None:
                result = undo_last_review_action(
                    self.manifest, self.project_path, self.specimen_index
                )
            else:
                result = apply_review_action(
                    self.manifest,
                    self.project_path,
                    self.specimen_index,
                    self.action,
                    progress=lambda phase, current, total, detail: self.progress.emit(
                        phase, current, total, detail
                    ),
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class ContextWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        specimen_index: int,
        background_channel: str,
        corrected: bool,
        include_3d: bool,
        roi_xy: tuple[int, int, int, int] | None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.specimen_index = specimen_index
        self.background_channel = background_channel
        self.corrected = corrected
        self.include_3d = include_3d
        self.roi_xy = roi_xy
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = generate_context_volume(
                self.manifest,
                self.specimen_index,
                self.background_channel,
                corrected=self.corrected,
                include_3d=self.include_3d,
                roi_xy=self.roi_xy,
                progress=lambda phase, current, total, detail: self.progress.emit(
                    phase, current, total, detail
                ),
                cancel_event=self.cancel_event,
            )
        except ProcessingCancelled as exc:
            self.cancelled.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class ScanWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        directory: Path,
        channel_markers: dict[str, str] | None = None,
        default_group: str = "Experiment",
        manual_paths: tuple[Path, Path] | None = None,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.channel_markers = channel_markers
        self.default_group = default_group
        self.manual_paths = manual_paths

    @Slot()
    def run(self) -> None:
        try:
            callback = lambda phase, current, total, detail: self.progress.emit(
                phase, current, total, detail
            )
            report = (
                inspect_manual_pair(
                    self.manual_paths[0],
                    self.manual_paths[1],
                    default_experimental_group=self.default_group,
                    include_checksums=True,
                    progress=callback,
                )
                if self.manual_paths is not None
                else scan_batch(
                    self.directory,
                    channel_markers=self.channel_markers,
                    default_experimental_group=self.default_group,
                    include_checksums=True,
                    progress=callback,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(report)


class VerifyWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        directory: Path | None,
        relink: bool,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.directory = directory
        self.relink = relink

    @Slot()
    def run(self) -> None:
        callback = lambda phase, current, total, detail: self.progress.emit(
            phase, current, total, detail
        )
        try:
            if self.relink:
                results = relink_project_sources(
                    self.manifest, self.directory, progress=callback  # type: ignore[arg-type]
                )
            else:
                results = verify_project_sources(
                    self.manifest,
                    source_directory=self.directory,
                    full_checksums=True,
                    progress=callback,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(results)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Synpo Microscopy Processor — Beta {__version__}")
        self.setMinimumSize(720, 500)
        self._was_maximized_before_fullscreen = True

        self.report: ScanReport | None = None
        self.manifest: dict[str, object] | None = None
        self.project_path: Path | None = None
        self._job_thread: QThread | None = None
        self._job_worker: QObject | None = None
        self._job_kind: str | None = None
        self._progress_started_at = 0.0
        self._progress_last_at = 0.0
        self._progress_last_current = 0
        self._progress_last_total = 0
        self._progress_rate_ema: float | None = None
        self._preview_statistics: dict[tuple[object, ...], StackStatistics] = {}
        self._last_preview: PreviewResult | None = None
        self._last_detection: DetectionSlice | None = None
        self._last_review: ReviewSlice | None = None
        self._last_review_context: ContextVolume | None = None
        self._last_trim_preview: ClusterTrimPreview | None = None
        self._last_distribution_preview: DistributionPreview | SpineReviewPreview | None = None
        self._preferred_distribution_spine_id: int | None = None
        self._preferred_spine_review_mode: str | None = None
        self._spine_map_focus_only = False
        self._spine_map_current_id: int | None = None
        self._spine_map_dialogs: list[SpineMapDialog] = []
        self._centerline_hint_pending_reload = False
        self._review_thread: QThread | None = None
        self._review_worker: ReviewWorker | None = None
        self._review_refresh_pending = False
        self._context_thread: QThread | None = None
        self._context_worker: ContextWorker | None = None
        self._context_request: tuple[object, ...] | None = None
        self._context_cache: dict[tuple[object, ...], ContextVolume] = {}
        self._context_dialogs: list[ContextViewerDialog] = []
        self._area_dialog: AreaSelectionDialog | None = None
        self._preview_requested_while_busy = False
        self._preprocess_view_specimen_key: tuple[int, int] | None = None
        self._detection_view_specimen_key: tuple[int, int] | None = None
        self._review_view_specimen_key: tuple[int, int] | None = None
        self._calibration_store = CalibrationStore()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._request_preview)

        self._build_actions()
        self._build_interface()
        self._load_presets()
        self._update_sensitivity_warnings()
        self._set_job_running(False)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self.new_action = QAction("New batch", self)
        self.new_action.triggered.connect(self._new_batch)
        file_menu.addAction(self.new_action)

        self.open_action = QAction("Open project…", self)
        self.open_action.triggered.connect(self._open_project)
        file_menu.addAction(self.open_action)

        self.save_action = QAction("Save project", self)
        self.save_action.triggered.connect(self._save_project)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        project_menu = self.menuBar().addMenu("&Project")
        self.verify_action = QAction("Verify sources", self)
        self.verify_action.triggered.connect(self._verify_sources)
        project_menu.addAction(self.verify_action)

        self.relink_action = QAction("Relink source folder…", self)
        self.relink_action.triggered.connect(self._relink_sources)
        project_menu.addAction(self.relink_action)

        view_menu = self.menuBar().addMenu("&View")
        self.fullscreen_action = QAction("Toggle full screen", self)
        self.fullscreen_action.setShortcut("F11")
        self.fullscreen_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(self.fullscreen_action)

    @staticmethod
    def _clamp_window_geometry(rect: QRect) -> QRect:
        screen = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
        if screen is None:
            return QRect(rect)
        available = screen.availableGeometry()
        width = min(max(320, rect.width()), available.width())
        height = min(max(240, rect.height()), available.height())
        left = max(available.left(), min(rect.left(), available.right() - width + 1))
        top = max(available.top(), min(rect.top(), available.bottom() - height + 1))
        return QRect(left, top, width, height)

    def show_initial(self) -> None:
        settings = QSettings()
        saved_geometry = settings.value("main_window/normal_geometry")
        has_saved_geometry = isinstance(saved_geometry, QRect) and saved_geometry.isValid()
        if has_saved_geometry:
            self.setGeometry(self._clamp_window_geometry(saved_geometry))
        else:
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                width = min(1380, round(available.width() * 0.9))
                height = min(860, round(available.height() * 0.9))
                self.setGeometry(
                    available.center().x() - width // 2,
                    available.center().y() - height // 2,
                    width,
                    height,
                )
        mode = str(settings.value("main_window/mode", "maximized"))
        if not has_saved_geometry:
            mode = "maximized"
        if mode == "fullscreen":
            self.showFullScreen()
        elif mode == "maximized":
            self.showMaximized()
        else:
            self.show()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            if self._was_maximized_before_fullscreen:
                self.showMaximized()
            else:
                self.showNormal()
            return
        self._was_maximized_before_fullscreen = self.isMaximized()
        self.showFullScreen()

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self._toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def _save_window_preferences(self) -> None:
        settings = QSettings()
        normal = self.normalGeometry() if (self.isMaximized() or self.isFullScreen()) else self.geometry()
        settings.setValue("main_window/normal_geometry", self._clamp_window_geometry(normal))
        settings.setValue(
            "main_window/mode",
            "fullscreen" if self.isFullScreen() else "maximized" if self.isMaximized() else "normal",
        )

    def _build_interface(self) -> None:
        central = QWidget()
        central_layout = QVBoxLayout(central)
        self.tabs = QTabWidget()
        central_layout.addWidget(self.tabs)
        setup_tab = QWidget()
        outer = QVBoxLayout(setup_tab)

        locations = QGroupBox("Batch locations")
        locations_form = QFormLayout(locations)
        self.source_edit = QLineEdit()
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_edit, 1)
        source_browse = QPushButton("Browse…")
        source_browse.clicked.connect(self._browse_source)
        source_row.addWidget(source_browse)
        self.scan_button = QPushButton("Scan and validate")
        self.scan_button.clicked.connect(self._scan_source)
        source_row.addWidget(self.scan_button)
        locations_form.addRow("TIFF folder:", source_row)

        self.output_edit = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output)
        output_row.addWidget(output_browse)
        locations_form.addRow("Output folder:", output_row)
        outer.addWidget(locations)

        import_group = QGroupBox("Filename pairing")
        import_form = QFormLayout(import_group)
        self.channel_a_marker = QLineEdit("ChanA")
        self.channel_a_marker.setPlaceholderText("Example: ChanA or cy")
        self.channel_b_marker = QLineEdit("ChanB")
        self.channel_b_marker.setPlaceholderText("Example: ChanB or cl")
        import_form.addRow("Channel A marker:", self.channel_a_marker)
        import_form.addRow("Channel B marker:", self.channel_b_marker)
        self.fallback_group_edit = QLineEdit("Experiment")
        self.fallback_group_edit.setToolTip(
            "Used only when filenames do not contain the original underscore metadata schema."
        )
        import_form.addRow("Fallback group:", self.fallback_group_edit)
        self.manual_pair_button = QPushButton(
            "Manually choose one Channel A file and one Channel B fileвЂ¦"
        )
        self.manual_pair_button.clicked.connect(self._choose_manual_pair)
        import_form.addRow(self.manual_pair_button)
        import_note = QLabel(
            "Automatic pairing removes the configured channel marker from each TIFF name. "
            "Files without the original metadata schema are placed in the fallback group; "
            "their specimen names can be edited in the table."
        )
        import_note.setWordWrap(True)
        import_form.addRow(import_note)
        outer.addWidget(import_group)

        settings_row = QHBoxLayout()
        channel_group = QGroupBox("Channel roles (confirm for this batch)")
        channel_form = QFormLayout(channel_group)
        self.channel_a_role = self._role_combo("protein_clusters")
        self.channel_b_role = self._role_combo("dendrite_spines")
        channel_form.addRow("ChanA:", self.channel_a_role)
        channel_form.addRow("ChanB:", self.channel_b_role)
        settings_row.addWidget(channel_group)

        calibration_group = QGroupBox("Physical calibration (confirm for this batch)")
        calibration_form = QFormLayout(calibration_group)
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.currentTextChanged.connect(self._preset_selected)
        calibration_form.addRow("Named preset:", self.preset_combo)
        self.xy_spin = QDoubleSpinBox()
        self.xy_spin.setDecimals(7)
        self.xy_spin.setRange(0.0000001, 1000.0)
        self.xy_spin.setValue(0.0462584)
        self.xy_spin.setSuffix(" µm/pixel")
        calibration_form.addRow("X/Y pixel size:", self.xy_spin)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setDecimals(7)
        self.z_spin.setRange(0.0000001, 1000.0)
        self.z_spin.setValue(0.5)
        self.z_spin.setSuffix(" µm")
        calibration_form.addRow("Z step:", self.z_spin)
        save_preset = QPushButton("Save/update preset")
        save_preset.clicked.connect(self._save_preset)
        calibration_form.addRow("", save_preset)
        settings_row.addWidget(calibration_group)
        outer.addLayout(settings_row)

        self.summary_label = QLabel("Select a folder and scan it to begin.")
        self.summary_label.setWordWrap(True)
        outer.addWidget(self.summary_label)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Status", "Experimental group", "Specimen", "ChanA", "ChanB", "Shape (Z × Y × X)", "Type", "Issues"]
        )
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        outer.addWidget(self.table, 1)

        progress_row = QHBoxLayout()
        self.progress_label = QLabel("")
        progress_row.addWidget(self.progress_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(320)
        self.progress_bar.setVisible(False)
        progress_row.addWidget(self.progress_bar)
        self.save_button = QPushButton("Save project…")
        self.save_button.clicked.connect(self._save_project)
        progress_row.addWidget(self.save_button)
        outer.addLayout(progress_row)

        self.tabs.addTab(setup_tab, "1. Batch setup")
        self._build_preprocessing_tab()
        self._build_detection_tab()
        self._build_review_tab()
        self._build_measurements_tab()
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
        self.setCentralWidget(central)
        self._build_context_status_line()

    def _build_context_status_line(self) -> None:
        self.context_status_widget = QWidget()
        layout = QHBoxLayout(self.context_status_widget)
        layout.setContentsMargins(4, 0, 4, 0)
        self.context_status_label = QLabel("Preparing image context…")
        layout.addWidget(self.context_status_label, 1)
        self.context_status_progress = QProgressBar()
        self.context_status_progress.setMinimumWidth(220)
        self.context_status_progress.setRange(0, 1)
        layout.addWidget(self.context_status_progress)
        self.cancel_context_button = QPushButton("Cancel")
        self.cancel_context_button.clicked.connect(self._cancel_context_generation)
        layout.addWidget(self.cancel_context_button)
        self.context_status_widget.setVisible(False)
        self.statusBar().addPermanentWidget(self.context_status_widget, 1)

    def _build_preprocessing_tab(self) -> None:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        selection = QGroupBox("Representative preview")
        selection_layout = QHBoxLayout(selection)
        self.preprocess_specimen = QComboBox()
        self.preprocess_specimen.currentIndexChanged.connect(
            self._preprocess_specimen_changed
        )
        selection_layout.addWidget(QLabel("Specimen:"))
        selection_layout.addWidget(self.preprocess_specimen, 2)
        self.preprocess_channel = QComboBox()
        self.preprocess_channel.addItem("ChanA", "ChanA")
        self.preprocess_channel.addItem("ChanB", "ChanB")
        self.preprocess_channel.currentIndexChanged.connect(
            self._preprocess_channel_changed
        )
        selection_layout.addWidget(QLabel("Channel:"))
        selection_layout.addWidget(self.preprocess_channel)
        self.representative_check = QCheckBox("Use as representative specimen")
        selection_layout.addWidget(self.representative_check)
        self.special_preprocessing_check = QCheckBox(
            "Use special preprocessing settings for this pair"
        )
        self.special_preprocessing_check.setToolTip(
            "When checked, ChanA and ChanB keep specimen-specific settings. "
            "Clearing the checkmark returns to batch defaults without deleting the saved override."
        )
        self.special_preprocessing_check.toggled.connect(
            self._special_preprocessing_toggled
        )
        selection_layout.addWidget(self.special_preprocessing_check)
        outer.addWidget(selection)

        controls = QGroupBox("Adaptive preprocessing settings for this channel")
        controls_layout = QHBoxLayout(controls)
        self.background_spin = QDoubleSpinBox()
        self.background_spin.setRange(0.0, 99.9)
        self.background_spin.setDecimals(1)
        self.background_spin.setSuffix(" %")
        controls_layout.addWidget(QLabel("Background percentile:"))
        controls_layout.addWidget(self.background_spin)
        self.sigma_xy_spin = QDoubleSpinBox()
        self.sigma_xy_spin.setRange(0.0, 5.0)
        self.sigma_xy_spin.setDecimals(3)
        self.sigma_xy_spin.setSingleStep(0.01)
        self.sigma_xy_spin.setSuffix(" µm")
        controls_layout.addWidget(QLabel("Gaussian XY:"))
        controls_layout.addWidget(self.sigma_xy_spin)
        self.sigma_z_spin = QDoubleSpinBox()
        self.sigma_z_spin.setRange(0.0, 5.0)
        self.sigma_z_spin.setDecimals(3)
        self.sigma_z_spin.setSingleStep(0.05)
        self.sigma_z_spin.setSuffix(" µm")
        controls_layout.addWidget(QLabel("Gaussian Z:"))
        controls_layout.addWidget(self.sigma_z_spin)
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.1, 10.0)
        self.sensitivity_spin.setDecimals(2)
        self.sensitivity_spin.setSingleStep(0.05)
        self.sensitivity_spin.setToolTip(
            "Higher values retain more candidate voxels; 1.00 uses the adaptive threshold."
        )
        controls_layout.addWidget(QLabel("Threshold sensitivity:"))
        controls_layout.addWidget(self.sensitivity_spin)
        self.sensitivity_spin.valueChanged.connect(self._update_sensitivity_warnings)
        self.apply_preprocessing_button = QPushButton("Apply channel settings")
        self.apply_preprocessing_button.clicked.connect(
            self._apply_preprocessing_settings
        )
        controls_layout.addWidget(self.apply_preprocessing_button)
        outer.addWidget(controls)
        self.preprocessing_sensitivity_warning = QLabel("")
        self.preprocessing_sensitivity_warning.setWordWrap(True)
        self.preprocessing_sensitivity_warning.setStyleSheet("color: #a65a00;")
        outer.addWidget(self.preprocessing_sensitivity_warning)

        navigation = QHBoxLayout()
        self.z_label = QLabel("Z: —")
        navigation.addWidget(self.z_label)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.valueChanged.connect(self._z_changed)
        navigation.addWidget(self.z_slider, 1)
        self.contrast_low = QSpinBox()
        self.contrast_low.setRange(0, 65535)
        self.contrast_low.setValue(0)
        self.contrast_low.valueChanged.connect(self._render_preview)
        navigation.addWidget(QLabel("Black:"))
        navigation.addWidget(self.contrast_low)
        self.contrast_high = QSpinBox()
        self.contrast_high.setRange(1, 65535)
        self.contrast_high.setValue(65535)
        self.contrast_high.valueChanged.connect(self._render_preview)
        navigation.addWidget(QLabel("White:"))
        navigation.addWidget(self.contrast_high)
        auto_contrast = QPushButton("Auto contrast")
        auto_contrast.clicked.connect(self._auto_contrast)
        navigation.addWidget(auto_contrast)
        self.threshold_overlay = QCheckBox("Threshold overlay")
        self.threshold_overlay.setChecked(True)
        self.threshold_overlay.toggled.connect(self._render_preview)
        navigation.addWidget(self.threshold_overlay)
        refresh = QPushButton("Refresh preview")
        refresh.clicked.connect(self._request_preview)
        navigation.addWidget(refresh)
        outer.addLayout(navigation)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        raw_container = QWidget()
        raw_layout = QVBoxLayout(raw_container)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.addWidget(
            QLabel("Original 16-bit slice (measurements remain tied to this data)")
        )
        self.raw_view = SliceView("Choose a specimen to load a slice")
        raw_layout.addWidget(ZoomControls(self.raw_view))
        raw_scroll = QScrollArea()
        raw_scroll.setWidgetResizable(True)
        raw_scroll.setWidget(self.raw_view)
        raw_layout.addWidget(raw_scroll, 1)
        splitter.addWidget(raw_container)

        processed_container = QWidget()
        processed_layout = QVBoxLayout(processed_container)
        processed_layout.setContentsMargins(0, 0, 0, 0)
        processed_layout.addWidget(
            QLabel("Background-subtracted and smoothed detection image")
        )
        self.processed_view = SliceView("Processed preview")
        processed_layout.addWidget(ZoomControls(self.processed_view))
        processed_scroll = QScrollArea()
        processed_scroll.setWidgetResizable(True)
        processed_scroll.setWidget(self.processed_view)
        processed_layout.addWidget(processed_scroll, 1)
        splitter.addWidget(processed_container)
        splitter.setSizes([680, 680])
        outer.addWidget(splitter, 1)

        batch_row = QHBoxLayout()
        self.preprocessing_status = QLabel(
            "Save or open a project, tune representative specimens, then preprocess the batch."
        )
        self.preprocessing_status.setWordWrap(True)
        batch_row.addWidget(self.preprocessing_status, 1)
        self.run_preprocessing_button = QPushButton(
            "Preprocess entire batch / resume"
        )
        self.run_preprocessing_button.clicked.connect(self._run_batch_preprocessing)
        batch_row.addWidget(self.run_preprocessing_button)
        self.cancel_preprocessing_button = QPushButton("Cancel after current slice")
        self.cancel_preprocessing_button.clicked.connect(
            self._cancel_batch_preprocessing
        )
        self.cancel_preprocessing_button.setEnabled(False)
        batch_row.addWidget(self.cancel_preprocessing_button)
        outer.addLayout(batch_row)
        self.preprocessing_progress_label = QLabel("Batch progress: not running")
        self.preprocessing_progress_label.setWordWrap(True)
        outer.addWidget(self.preprocessing_progress_label)
        self.preprocessing_progress_bar = QProgressBar()
        self.preprocessing_progress_bar.setRange(0, 100)
        self.preprocessing_progress_bar.setValue(0)
        self.preprocessing_progress_bar.setFormat("%p%")
        outer.addWidget(self.preprocessing_progress_bar)

        tab.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.tabs.addTab(tab, "2. Preprocessing")

    def _build_detection_tab(self) -> None:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        side_scroll.setMinimumWidth(370)
        side_scroll.setMaximumWidth(470)
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)

        view_group = QGroupBox("Specimen and display")
        view_form = QFormLayout(view_group)
        view_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        view_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.detection_specimen = QComboBox()
        self.detection_specimen.currentIndexChanged.connect(
            self._detection_specimen_changed
        )
        view_form.addRow("Specimen:", self.detection_specimen)
        self.detection_background_channel = QComboBox()
        self.detection_background_channel.addItem("ChanB", "ChanB")
        self.detection_background_channel.addItem("ChanA", "ChanA")
        self.detection_background_channel.currentIndexChanged.connect(
            self._load_detection_view
        )
        view_form.addRow("Image background:", self.detection_background_channel)
        self.detection_black = QSpinBox()
        self.detection_black.setRange(0, 65535)
        self.detection_black.valueChanged.connect(self._render_detection_view)
        view_form.addRow("Black level:", self.detection_black)
        self.detection_white = QSpinBox()
        self.detection_white.setRange(1, 65535)
        self.detection_white.setValue(65535)
        self.detection_white.valueChanged.connect(self._render_detection_view)
        view_form.addRow("White level:", self.detection_white)
        detection_auto = QPushButton("Set contrast automatically")
        detection_auto.setMinimumHeight(32)
        detection_auto.clicked.connect(self._auto_detection_contrast)
        view_form.addRow(detection_auto)
        side_layout.addWidget(view_group)

        overlay_group = QGroupBox("Colored overlays")
        overlay_layout = QVBoxLayout(overlay_group)
        self.show_dendrites = QCheckBox("Dendrite shafts — green")
        self.show_dendrites.setChecked(True)
        self.show_dendrites.toggled.connect(self._render_detection_view)
        overlay_layout.addWidget(self.show_dendrites)
        self.show_spines = QCheckBox("Spine candidates — cyan")
        self.show_spines.setChecked(True)
        self.show_spines.toggled.connect(self._render_detection_view)
        overlay_layout.addWidget(self.show_spines)
        self.show_clusters = QCheckBox("Protein-cluster candidates — magenta")
        self.show_clusters.setChecked(True)
        self.show_clusters.toggled.connect(self._render_detection_view)
        overlay_layout.addWidget(self.show_clusters)
        side_layout.addWidget(overlay_group)

        settings_group = QGroupBox("Primary candidate detection settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        settings_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.dendrite_detection_sensitivity = QDoubleSpinBox()
        self.dendrite_detection_sensitivity.setRange(0.25, 10.0)
        self.dendrite_detection_sensitivity.setDecimals(2)
        self.dendrite_detection_sensitivity.setSingleStep(0.05)
        self.dendrite_detection_sensitivity.setToolTip(
            "Higher values retain more dendrite/spine signal."
        )
        settings_layout.addRow(
            "Dendrite/spine sensitivity:", self.dendrite_detection_sensitivity
        )
        self.dendrite_detection_sensitivity.valueChanged.connect(
            self._update_sensitivity_warnings
        )
        self.cluster_detection_sensitivity = QDoubleSpinBox()
        self.cluster_detection_sensitivity.setRange(0.25, 10.0)
        self.cluster_detection_sensitivity.setDecimals(2)
        self.cluster_detection_sensitivity.setSingleStep(0.05)
        self.cluster_detection_sensitivity.setToolTip(
            "Higher values retain more protein-cluster candidates."
        )
        settings_layout.addRow(
            "Protein-cluster sensitivity:", self.cluster_detection_sensitivity
        )
        self.cluster_detection_sensitivity.valueChanged.connect(
            self._update_sensitivity_warnings
        )
        self.detection_sensitivity_warning = QLabel("")
        self.detection_sensitivity_warning.setWordWrap(True)
        self.detection_sensitivity_warning.setStyleSheet("color: #a65a00;")
        settings_layout.addRow(self.detection_sensitivity_warning)
        self.spine_branch_length = QDoubleSpinBox()
        self.spine_branch_length.setRange(0.5, 10.0)
        self.spine_branch_length.setDecimals(2)
        self.spine_branch_length.setSuffix(" µm")
        settings_layout.addRow("Maximum terminal branch:", self.spine_branch_length)
        self.minimum_dendrite_length = QDoubleSpinBox()
        self.minimum_dendrite_length.setRange(0.5, 1000.0)
        self.minimum_dendrite_length.setDecimals(1)
        self.minimum_dendrite_length.setSuffix(" µm")
        settings_layout.addRow("Minimum dendrite length:", self.minimum_dendrite_length)
        self.minimum_spine_pixels = QSpinBox()
        self.minimum_spine_pixels.setRange(1, 10000)
        settings_layout.addRow(
            "Minimum spine projection area (pixels):", self.minimum_spine_pixels
        )
        self.minimum_cluster_voxels = QSpinBox()
        self.minimum_cluster_voxels.setRange(1, 1000000)
        settings_layout.addRow(
            "Minimum protein-cluster volume (voxels):", self.minimum_cluster_voxels
        )
        self.detection_memory_mode = QComboBox()
        self.detection_memory_mode.addItem(
            "Automatic (fast when safe)", AUTOMATIC_MEMORY_MODE
        )
        self.detection_memory_mode.addItem(
            "Always use low-memory detection", ALWAYS_LOW_MEMORY_MODE
        )
        self.detection_memory_mode.setToolTip(
            "Automatic mode switches large specimens to slower disk-backed detection. "
            "This execution setting does not invalidate completed masks."
        )
        settings_layout.addRow("Memory strategy:", self.detection_memory_mode)
        self.apply_detection_button = QPushButton("Save these detection settings")
        self.apply_detection_button.setMinimumHeight(34)
        self.apply_detection_button.clicked.connect(self._apply_detection_settings)
        settings_layout.addRow(self.apply_detection_button)
        side_layout.addWidget(settings_group)

        results_group = QGroupBox("Detection status")
        results_layout = QVBoxLayout(results_group)
        self.detection_counts = QLabel("No completed detection for this specimen.")
        self.detection_counts.setWordWrap(True)
        results_layout.addWidget(self.detection_counts)
        self.detection_projections_button = QPushButton(
            "Generate XY/XZ/YZ maximum projections"
        )
        self.detection_projections_button.setMinimumHeight(34)
        self.detection_projections_button.setEnabled(False)
        self.detection_projections_button.clicked.connect(
            lambda: self._open_context_view(False, "projections")
        )
        results_layout.addWidget(self.detection_projections_button)
        self.detection_3d_button = QPushButton("Generate rotatable 3D object view")
        self.detection_3d_button.setMinimumHeight(34)
        self.detection_3d_button.setEnabled(False)
        self.detection_3d_button.clicked.connect(
            lambda: self._open_context_view(False, "3d")
        )
        results_layout.addWidget(self.detection_3d_button)
        self.detection_status = QLabel(
            "Detection can start when at least one specimen pair has completed preprocessing."
        )
        self.detection_status.setWordWrap(True)
        results_layout.addWidget(self.detection_status)
        self.detection_progress_label = QLabel("Batch progress: not running")
        self.detection_progress_label.setWordWrap(True)
        results_layout.addWidget(self.detection_progress_label)
        self.detection_progress_bar = QProgressBar()
        self.detection_progress_bar.setRange(0, 100)
        self.detection_progress_bar.setValue(0)
        self.detection_progress_bar.setFormat("%p%")
        results_layout.addWidget(self.detection_progress_bar)
        self.run_detection_button = QPushButton(
            "Run automatic detection or resume the batch"
        )
        self.run_detection_button.setMinimumHeight(38)
        self.run_detection_button.clicked.connect(self._run_detection)
        results_layout.addWidget(self.run_detection_button)
        self.cancel_detection_button = QPushButton(
            "Cancel safely after the current step"
        )
        self.cancel_detection_button.setMinimumHeight(34)
        self.cancel_detection_button.clicked.connect(self._cancel_detection)
        self.cancel_detection_button.setEnabled(False)
        results_layout.addWidget(self.cancel_detection_button)
        side_layout.addWidget(results_group)
        side_layout.addStretch(1)
        side_scroll.setWidget(side_panel)
        splitter.addWidget(side_scroll)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        z_row = QHBoxLayout()
        self.detection_z_label = QLabel("Z: —")
        self.detection_z_label.setMinimumWidth(72)
        z_row.addWidget(self.detection_z_label)
        self.detection_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.detection_z_slider.setRange(0, 0)
        self.detection_z_slider.valueChanged.connect(self._detection_z_changed)
        z_row.addWidget(self.detection_z_slider, 1)
        viewer_layout.addLayout(z_row)

        self.detection_view = SliceView("Run detection to inspect candidate masks")
        self.detection_zoom_controls = ZoomControls(self.detection_view)
        viewer_layout.addWidget(self.detection_zoom_controls)
        detection_scroll = QScrollArea()
        detection_scroll.setWidgetResizable(True)
        detection_scroll.setWidget(self.detection_view)
        viewer_layout.addWidget(detection_scroll, 1)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([410, 970])

        self.tabs.addTab(tab, "3. Automatic detection")

    def _build_review_tab(self) -> None:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        side_scroll.setMinimumWidth(390)
        side_scroll.setMaximumWidth(500)
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)

        queue_group = QGroupBox("Specimen review queue")
        queue_layout = QVBoxLayout(queue_group)
        self.review_specimen = QComboBox()
        self.review_specimen.currentIndexChanged.connect(
            self._review_specimen_changed
        )
        queue_layout.addWidget(self.review_specimen)
        queue_buttons = QHBoxLayout()
        self.previous_review_button = QPushButton("Previous specimen")
        self.previous_review_button.clicked.connect(
            lambda: self._move_review_specimen(-1)
        )
        queue_buttons.addWidget(self.previous_review_button)
        self.next_review_button = QPushButton("Next specimen")
        self.next_review_button.clicked.connect(lambda: self._move_review_specimen(1))
        queue_buttons.addWidget(self.next_review_button)
        queue_layout.addLayout(queue_buttons)
        self.review_queue_status = QLabel("No detected specimens are ready for review.")
        self.review_queue_status.setWordWrap(True)
        queue_layout.addWidget(self.review_queue_status)
        side_layout.addWidget(queue_group)

        display_group = QGroupBox("Display")
        display_form = QFormLayout(display_group)
        display_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.review_view_mode = QComboBox()
        self.review_view_mode.addItem("Individual Z slice", "slice")
        self.review_view_mode.addItem("Drawable XY maximum projection", "xy_max")
        self.review_view_mode.currentIndexChanged.connect(
            self._review_view_mode_changed
        )
        display_form.addRow("Main canvas:", self.review_view_mode)
        self.review_background_channel = QComboBox()
        self.review_background_channel.addItem("ChanB", "ChanB")
        self.review_background_channel.addItem("ChanA", "ChanA")
        self.review_background_channel.currentIndexChanged.connect(
            self._load_review_view
        )
        display_form.addRow("Image background:", self.review_background_channel)
        self.review_black = QSpinBox()
        self.review_black.setRange(0, 65535)
        self.review_black.valueChanged.connect(self._render_review_view)
        display_form.addRow("Black level:", self.review_black)
        self.review_white = QSpinBox()
        self.review_white.setRange(1, 65535)
        self.review_white.setValue(65535)
        self.review_white.valueChanged.connect(self._render_review_view)
        display_form.addRow("White level:", self.review_white)
        review_auto = QPushButton("Set contrast automatically")
        review_auto.clicked.connect(self._auto_review_contrast)
        display_form.addRow(review_auto)
        self.review_show_dendrites = QCheckBox("Dendrite shafts — green")
        self.review_show_dendrites.setChecked(True)
        self.review_show_dendrites.toggled.connect(self._render_review_view)
        display_form.addRow(self.review_show_dendrites)
        self.review_show_spines = QCheckBox("Spines — cyan")
        self.review_show_spines.setChecked(True)
        self.review_show_spines.toggled.connect(self._render_review_view)
        display_form.addRow(self.review_show_spines)
        self.review_show_clusters = QCheckBox("Protein clusters — magenta")
        self.review_show_clusters.setChecked(True)
        self.review_show_clusters.toggled.connect(self._render_review_view)
        display_form.addRow(self.review_show_clusters)
        self.review_projections_button = QPushButton(
            "Generate XY/XZ/YZ maximum projections"
        )
        self.review_projections_button.clicked.connect(
            lambda: self._open_context_view(True, "projections")
        )
        display_form.addRow(self.review_projections_button)
        self.review_3d_button = QPushButton("Generate rotatable 3D object view")
        self.review_3d_button.clicked.connect(
            lambda: self._open_context_view(True, "3d")
        )
        display_form.addRow(self.review_3d_button)
        side_layout.addWidget(display_group)

        correction_group = QGroupBox("Hint-driven local correction")
        correction_form = QFormLayout(correction_group)
        correction_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        correction_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.review_object_type = QComboBox()
        self.review_object_type.addItem("Dendrite", "dendrite")
        self.review_object_type.addItem("Spine", "spine")
        correction_form.addRow("Object type:", self.review_object_type)
        self.review_operation = QComboBox()
        for label, value in (
            ("Add missed object", "add"),
            ("Exclude object", "exclude"),
            ("Exclude as filopodium", "filopodium"),
            ("Split touching objects", "split"),
            ("Merge objects", "merge"),
            ("Expand boundary", "expand"),
            ("Trim boundary", "trim"),
            ("Accept object", "accept"),
            ("Flag object for attention", "needs_attention"),
        ):
            self.review_operation.addItem(label, value)
        self.review_operation.currentIndexChanged.connect(
            self._review_tool_changed
        )
        correction_form.addRow("Action:", self.review_operation)
        self.review_brush_radius = QSpinBox()
        self.review_brush_radius.setRange(1, 100)
        self.review_brush_radius.setValue(4)
        self.review_brush_radius.setSuffix(" px")
        self.review_brush_radius.valueChanged.connect(
            self._review_brush_changed
        )
        correction_form.addRow("Hint brush radius:", self.review_brush_radius)
        self.review_memory_mode = QComboBox()
        self.review_memory_mode.addItem(
            "Automatic (use slow mode when needed)",
            AUTOMATIC_REVIEW_MEMORY_MODE,
        )
        self.review_memory_mode.addItem(
            "Always use slow low-memory correction",
            ALWAYS_LOW_MEMORY_REVIEW_MODE,
        )
        self.review_memory_mode.setToolTip(
            "Automatic mode uses disk-backed correction when the selected object "
            "would exceed the RAM safety limit. Forced mode is slower but useful "
            "on computers with little available memory."
        )
        correction_form.addRow("Memory strategy:", self.review_memory_mode)
        self.review_instruction = QLabel()
        self.review_instruction.setWordWrap(True)
        correction_form.addRow(self.review_instruction)
        self.review_hint_status = QLabel("No hint drawn.")
        correction_form.addRow(self.review_hint_status)
        hint_buttons = QHBoxLayout()
        self.clear_review_hint_button = QPushButton("Clear hint")
        self.clear_review_hint_button.clicked.connect(self._clear_review_hint)
        hint_buttons.addWidget(self.clear_review_hint_button)
        self.undo_review_stroke_button = QPushButton("Undo drawn stroke")
        self.undo_review_stroke_button.clicked.connect(
            self._undo_review_stroke
        )
        hint_buttons.addWidget(self.undo_review_stroke_button)
        correction_form.addRow(hint_buttons)
        self.apply_review_button = QPushButton("Apply hint and resegment locally")
        self.apply_review_button.setMinimumHeight(38)
        self.apply_review_button.clicked.connect(self._apply_review_action)
        correction_form.addRow(self.apply_review_button)
        self.undo_review_action_button = QPushButton("Undo last applied correction")
        self.undo_review_action_button.clicked.connect(self._undo_review_action)
        correction_form.addRow(self.undo_review_action_button)
        side_layout.addWidget(correction_group)

        checkpoint_group = QGroupBox("Specimen checkpoint")
        checkpoint_layout = QVBoxLayout(checkpoint_group)
        self.review_comment = QLineEdit()
        self.review_comment.setPlaceholderText("Optional note for this specimen")
        checkpoint_layout.addWidget(self.review_comment)
        checkpoint_buttons = QHBoxLayout()
        self.save_review_progress_button = QPushButton("Save as in progress")
        self.save_review_progress_button.clicked.connect(
            lambda: self._save_review_state(False)
        )
        checkpoint_buttons.addWidget(self.save_review_progress_button)
        self.complete_review_button = QPushButton("Mark review complete")
        self.complete_review_button.clicked.connect(
            lambda: self._save_review_state(True)
        )
        checkpoint_buttons.addWidget(self.complete_review_button)
        checkpoint_layout.addLayout(checkpoint_buttons)
        self.review_progress = QProgressBar()
        self.review_progress.setVisible(False)
        checkpoint_layout.addWidget(self.review_progress)
        self.review_status = QLabel("Corrections affect only the selected specimen.")
        self.review_status.setWordWrap(True)
        checkpoint_layout.addWidget(self.review_status)
        side_layout.addWidget(checkpoint_group)
        side_layout.addStretch(1)
        side_scroll.setWidget(side_panel)
        splitter.addWidget(side_scroll)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        z_row = QHBoxLayout()
        self.review_z_label = QLabel("Z: —")
        self.review_z_label.setMinimumWidth(72)
        z_row.addWidget(self.review_z_label)
        self.review_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.review_z_slider.setRange(0, 0)
        self.review_z_slider.valueChanged.connect(self._review_z_changed)
        z_row.addWidget(self.review_z_slider, 1)
        viewer_layout.addLayout(z_row)
        self.review_view = ReviewCanvas(
            "A detected specimen will appear here for optional correction"
        )
        self.review_view.hint_changed.connect(self._review_hint_changed)
        self.review_zoom_controls = ZoomControls(self.review_view)
        viewer_layout.addWidget(self.review_zoom_controls)
        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setWidget(self.review_view)
        viewer_layout.addWidget(review_scroll, 1)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 950])

        tab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.tabs.addTab(tab, "4. Review and correction")
        self._review_tool_changed()

    def _build_measurements_tab(self) -> None:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        side_scroll.setMinimumWidth(390)
        side_scroll.setMaximumWidth(500)
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)

        settings_group = QGroupBox("Association and volume settings")
        settings_form = QFormLayout(settings_group)
        settings_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.measurement_overlap = QDoubleSpinBox()
        self.measurement_overlap.setRange(0.0, 100.0)
        self.measurement_overlap.setDecimals(1)
        self.measurement_overlap.setValue(80.0)
        self.measurement_overlap.setSuffix("%")
        self.measurement_overlap.setToolTip(
            "Minimum fraction of a retained cluster that must overlap one spine."
        )
        settings_form.addRow("Minimum cluster/spine overlap:", self.measurement_overlap)
        self.measurement_end_method = QComboBox()
        self.measurement_end_method.addItem("No end trimming", "untrimmed")
        self.measurement_end_method.addItem(
            "Fixed slices from larger terminal end", "fixed"
        )
        self.measurement_end_method.addItem(
            "Adaptive oversized terminal slices", "adaptive"
        )
        self.measurement_end_method.currentIndexChanged.connect(
            self._measurement_method_changed
        )
        settings_form.addRow("Blurry cluster-end method:", self.measurement_end_method)
        self.measurement_fixed_slices = QSpinBox()
        self.measurement_fixed_slices.setRange(0, 20)
        self.measurement_fixed_slices.setValue(3)
        self.measurement_fixed_slices.setSuffix(" slices")
        settings_form.addRow("Fixed slices removed:", self.measurement_fixed_slices)
        self.measurement_area_factor = QDoubleSpinBox()
        self.measurement_area_factor.setRange(1.0, 10.0)
        self.measurement_area_factor.setDecimals(2)
        self.measurement_area_factor.setSingleStep(0.1)
        self.measurement_area_factor.setValue(1.8)
        self.measurement_area_factor.setSuffix("× stable area")
        settings_form.addRow("Adaptive oversized threshold:", self.measurement_area_factor)
        self.measurement_min_slices = QSpinBox()
        self.measurement_min_slices.setRange(1, 20)
        self.measurement_min_slices.setValue(2)
        self.measurement_min_slices.setSuffix(" slices")
        settings_form.addRow("Always retain at least:", self.measurement_min_slices)
        self.save_measurement_settings_button = QPushButton(
            "Save these measurement settings"
        )
        self.save_measurement_settings_button.clicked.connect(
            self._save_measurement_settings
        )
        settings_form.addRow(self.save_measurement_settings_button)
        side_layout.addWidget(settings_group)

        run_group = QGroupBox("Batch measurement")
        run_layout = QVBoxLayout(run_group)
        self.measurement_status = QLabel(
            "Detected specimens can be measured with automatic or corrected masks."
        )
        self.measurement_status.setWordWrap(True)
        run_layout.addWidget(self.measurement_status)
        self.measurement_progress_label = QLabel("Batch progress: not running")
        self.measurement_progress_label.setWordWrap(True)
        run_layout.addWidget(self.measurement_progress_label)
        self.measurement_progress_bar = QProgressBar()
        self.measurement_progress_bar.setRange(0, 100)
        self.measurement_progress_bar.setValue(0)
        self.measurement_progress_bar.setFormat("%p%")
        run_layout.addWidget(self.measurement_progress_bar)
        self.run_measurements_button = QPushButton(
            "Calculate measurements for entire batch / resume"
        )
        self.run_measurements_button.setMinimumHeight(38)
        self.run_measurements_button.clicked.connect(self._run_measurements)
        run_layout.addWidget(self.run_measurements_button)
        self.cancel_measurements_button = QPushButton(
            "Cancel safely after the current slice"
        )
        self.cancel_measurements_button.clicked.connect(self._cancel_measurements)
        self.cancel_measurements_button.setEnabled(False)
        run_layout.addWidget(self.cancel_measurements_button)
        side_layout.addWidget(run_group)

        inspect_group = QGroupBox("Inspect saved measurements")
        inspect_form = QFormLayout(inspect_group)
        self.measurement_specimen = QComboBox()
        self.measurement_specimen.currentIndexChanged.connect(
            self._measurement_specimen_changed
        )
        inspect_form.addRow("Specimen:", self.measurement_specimen)
        self.measurement_table_level = QComboBox()
        self.measurement_table_level.addItem("Specimen", "specimen_rows")
        self.measurement_table_level.addItem("Dendrites", "dendrite_rows")
        self.measurement_table_level.addItem("Spines", "spine_rows")
        self.measurement_table_level.addItem("Clusters and spine sums", "cluster_rows")
        self.measurement_table_level.addItem("Spine distributions", "distribution_rows")
        self.measurement_table_level.addItem(
            "Compare cluster-end methods", "cluster_end_comparison"
        )
        self.measurement_table_level.currentIndexChanged.connect(
            self._populate_measurement_table
        )
        inspect_form.addRow("Table:", self.measurement_table_level)
        self.measurement_cluster = QComboBox()
        inspect_form.addRow("Cluster illustration:", self.measurement_cluster)
        self.load_trim_preview_button = QPushButton(
            "Show counted and discarded voxels"
        )
        self.load_trim_preview_button.clicked.connect(self._load_trim_preview)
        inspect_form.addRow(self.load_trim_preview_button)
        side_layout.addWidget(inspect_group)

        distribution_group = QGroupBox("Spine review")
        distribution_form = QFormLayout(distribution_group)
        self.distribution_review_mode = QComboBox()
        self.distribution_review_mode.addItem(
            "Protein-cluster-positive spines", "cluster_positive"
        )
        self.distribution_review_mode.addItem(
            "Cluster-less spines (optional)", "cluster_less"
        )
        self.distribution_review_mode.currentIndexChanged.connect(
            self._distribution_review_mode_changed
        )
        distribution_form.addRow("Queue:", self.distribution_review_mode)
        self.distribution_spine = QComboBox()
        self.distribution_spine.currentIndexChanged.connect(self._distribution_spine_changed)
        distribution_form.addRow("Review spine:", self.distribution_spine)
        self.centerline_hint_button = QPushButton("Centerline end hint")
        self.centerline_hint_button.setCheckable(True)
        self.centerline_hint_button.toggled.connect(self._centerline_hint_mode_changed)
        distribution_form.addRow(self.centerline_hint_button)
        self.clear_centerline_hint_button = QPushButton("Clear end hint")
        self.clear_centerline_hint_button.clicked.connect(self._clear_centerline_hint)
        distribution_form.addRow(self.clear_centerline_hint_button)
        self.distribution_include = QCheckBox("Include in distribution summaries")
        self.distribution_include.setChecked(True)
        distribution_form.addRow(self.distribution_include)
        self.distribution_invalid = QCheckBox("Invalid spine (exclude from all metrics)")
        self.distribution_invalid.toggled.connect(
            lambda checked: self.distribution_include.setEnabled(not checked)
        )
        distribution_form.addRow(self.distribution_invalid)
        self.distribution_note = QLineEdit()
        self.distribution_note.setPlaceholderText("Optional reason or review note")
        distribution_form.addRow("Note:", self.distribution_note)
        self.save_distribution_review_button = QPushButton("Checkpoint decision and advance")
        self.save_distribution_review_button.clicked.connect(self._save_distribution_review)
        distribution_form.addRow(self.save_distribution_review_button)
        navigation = QHBoxLayout()
        self.previous_distribution_button = QPushButton("Previous")
        self.previous_distribution_button.clicked.connect(lambda: self._move_distribution_spine(-1))
        self.next_distribution_button = QPushButton("Next")
        self.next_distribution_button.clicked.connect(lambda: self._move_distribution_spine(1))
        navigation.addWidget(self.previous_distribution_button)
        navigation.addWidget(self.next_distribution_button)
        distribution_form.addRow(navigation)
        self.open_spine_map_button = QPushButton("Open numbered spine mapвЂ¦")
        self.open_spine_map_button.clicked.connect(
            lambda: self._open_spine_map(False)
        )
        distribution_form.addRow(self.open_spine_map_button)
        side_layout.addWidget(distribution_group)

        chart_group = QGroupBox("Experimental-group profile")
        chart_form = QFormLayout(chart_group)
        self.distribution_group_combo = QComboBox()
        self.distribution_group_combo.currentIndexChanged.connect(self._update_distribution_chart)
        chart_form.addRow("Group:", self.distribution_group_combo)
        self.distribution_chart_mode = QComboBox()
        self.distribution_chart_mode.addItem("Line profile", "line")
        self.distribution_chart_mode.addItem("Bar chart", "bar")
        self.distribution_chart_mode.currentIndexChanged.connect(self._update_distribution_chart)
        chart_form.addRow("Chart type:", self.distribution_chart_mode)
        self.distribution_fixed_scale = QCheckBox("Fixed 0–1 Y-axis")
        self.distribution_fixed_scale.toggled.connect(self._update_distribution_chart)
        chart_form.addRow(self.distribution_fixed_scale)
        side_layout.addWidget(chart_group)

        export_group = QGroupBox("Excel, CSV, and optional PDF export")
        export_form = QFormLayout(export_group)
        self.export_validation_pdf = QCheckBox("Main validation PDF")
        self.export_excluded_pdf = QCheckBox("Excluded-distribution audit PDF")
        self.export_invalid_pdf = QCheckBox("Invalid-spine audit PDF")
        export_form.addRow(self.export_validation_pdf)
        export_form.addRow(self.export_excluded_pdf)
        export_form.addRow(self.export_invalid_pdf)
        self.export_pdf_margin = QDoubleSpinBox()
        self.export_pdf_margin.setRange(0.0, 20.0)
        self.export_pdf_margin.setDecimals(2)
        self.export_pdf_margin.setValue(1.0)
        self.export_pdf_margin.setSuffix(" µm")
        export_form.addRow("PDF crop margin:", self.export_pdf_margin)
        self.export_measurements_button = QPushButton("Export workbook and CSV files…")
        self.export_measurements_button.clicked.connect(self._export_measurements)
        export_form.addRow(self.export_measurements_button)
        side_layout.addWidget(export_group)

        note = QLabel(
            "Ten calibrated curved-axis bins run from shaft to distal tip. Group means "
            "are specimen-weighted and error bars are SEM; no statistical tests are performed."
        )
        note.setWordWrap(True)
        side_layout.addWidget(note)
        side_layout.addStretch(1)
        side_scroll.setWidget(side_panel)
        splitter.addWidget(side_scroll)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        self.measurement_summary = QLabel("No saved measurement result selected.")
        self.measurement_summary.setWordWrap(True)
        result_layout.addWidget(self.measurement_summary)
        self.measurement_result_tabs = QTabWidget()
        result_layout.addWidget(self.measurement_result_tabs, 1)
        raw_page = QWidget()
        raw_layout = QVBoxLayout(raw_page)
        self.measurement_table = QTableWidget(0, 0)
        self.measurement_table.setAlternatingRowColors(True)
        self.measurement_table.verticalHeader().setVisible(False)
        raw_layout.addWidget(self.measurement_table, 1)
        self.trim_preview_label = QLabel(
            "Cluster-end illustration: green voxels are counted; magenta voxels are discarded."
        )
        self.trim_preview_label.setWordWrap(True)
        raw_layout.addWidget(self.trim_preview_label)
        self.trim_preview_view = SliceView(
            "Run measurements, select a cluster, then generate its voxel illustration"
        )
        self.trim_preview_view.setMinimumSize(420, 300)
        raw_layout.addWidget(self.trim_preview_view, 1)
        self.measurement_result_tabs.addTab(raw_page, "Raw measurement tables")

        distribution_page = QWidget()
        distribution_layout = QVBoxLayout(distribution_page)
        self.distribution_preview_status = QLabel("The first unreviewed cluster-positive spine opens automatically.")
        self.distribution_preview_status.setWordWrap(True)
        distribution_layout.addWidget(self.distribution_preview_status)
        self.distribution_z_controls = QWidget()
        distribution_z_layout = QHBoxLayout(self.distribution_z_controls)
        distribution_z_layout.setContentsMargins(0, 0, 0, 0)
        self.distribution_z_label = QLabel("Spine Z")
        self.distribution_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.distribution_z_slider.valueChanged.connect(self._distribution_z_changed)
        distribution_z_layout.addWidget(self.distribution_z_label)
        distribution_z_layout.addWidget(self.distribution_z_slider, 1)
        self.distribution_z_controls.setVisible(False)
        distribution_layout.addWidget(self.distribution_z_controls)
        self.distribution_spine_header = QLabel("Protein-cluster-positive spine")
        self.distribution_spine_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distribution_spine_header.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 3px;"
        )
        distribution_layout.addWidget(self.distribution_spine_header)
        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.distribution_dendrite_view = EndpointHintView("Dendrite/spine distribution preview")
        self.distribution_dendrite_view.point_clicked.connect(self._centerline_hint_clicked)
        self.distribution_dendrite_view.context_requested.connect(
            lambda: self._open_spine_map(True)
        )
        self.distribution_protein_view = ClickableSliceView("Protein-cluster distribution preview")
        self.distribution_protein_view.context_requested.connect(
            lambda: self._open_spine_map(True)
        )
        self.distribution_dendrite_view.setMinimumSize(320, 260)
        self.distribution_protein_view.setMinimumSize(320, 260)
        preview_splitter.addWidget(self.distribution_dendrite_view)
        preview_splitter.addWidget(self.distribution_protein_view)
        distribution_layout.addWidget(preview_splitter, 2)
        preview_zoom_row = QHBoxLayout()
        preview_zoom_row.addWidget(ZoomControls(self.distribution_dendrite_view), 1)
        preview_zoom_row.addWidget(ZoomControls(self.distribution_protein_view), 1)
        distribution_layout.addLayout(preview_zoom_row)
        self.distribution_chart = DistributionChart()
        distribution_layout.addWidget(self.distribution_chart, 1)
        self.measurement_result_tabs.addTab(distribution_page, "Distribution review and group profiles")
        splitter.addWidget(result_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 950])
        self.tabs.addTab(tab, "5. Measurements")
        self._measurement_method_changed()

    def _role_combo(self, selected: str) -> QComboBox:
        combo = QComboBox()
        for value, label in ROLE_LABELS.items():
            combo.addItem(label, value)
        combo.setCurrentIndex(combo.findData(selected))
        return combo

    def _load_presets(self) -> None:
        try:
            presets = self._calibration_store.load()
        except ValueError as exc:
            QMessageBox.warning(self, "Calibration presets", str(exc))
            presets = {}
        current = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for name, calibration in presets.items():
            self.preset_combo.addItem(name, calibration)
        self.preset_combo.setEditText(current)
        self.preset_combo.blockSignals(False)

    def _preset_selected(self, name: str) -> None:
        index = self.preset_combo.findText(name)
        if index < 0:
            return
        calibration = self.preset_combo.itemData(index)
        if isinstance(calibration, Calibration):
            self.xy_spin.setValue(calibration.xy_um_per_pixel)
            self.z_spin.setValue(calibration.z_step_um)

    def _current_calibration(self) -> Calibration:
        calibration = Calibration(
            preset_name=self.preset_combo.currentText().strip(),
            xy_um_per_pixel=self.xy_spin.value(),
            z_step_um=self.z_spin.value(),
        )
        calibration.validate()
        return calibration

    def _save_preset(self) -> None:
        try:
            calibration = self._current_calibration()
            self._calibration_store.save_preset(calibration)
            self._load_presets()
            self.preset_combo.setCurrentText(calibration.preset_name)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save preset", str(exc))

    def _browse_source(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select TIFF folder", self.source_edit.text())
        if directory:
            self.source_edit.setText(directory)
            if not self.output_edit.text():
                self.output_edit.setText(str(Path(directory) / "Synpo Results"))

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_edit.text())
        if directory:
            self.output_edit.setText(directory)

    def _current_channel_markers(self) -> dict[str, str]:
        markers = {
            "ChanA": self.channel_a_marker.text().strip(),
            "ChanB": self.channel_b_marker.text().strip(),
        }
        if not markers["ChanA"] or not markers["ChanB"]:
            raise ValueError("Enter both channel filename markers.")
        if markers["ChanA"].casefold() == markers["ChanB"].casefold():
            raise ValueError("Channel filename markers must be different.")
        return markers

    def _choose_manual_pair(self) -> None:
        start = self.source_edit.text().strip()
        channel_a, _ = QFileDialog.getOpenFileName(
            self,
            "Choose the Channel A TIFF",
            start,
            "TIFF stacks (*.tif *.tiff)",
        )
        if not channel_a:
            return
        channel_b, _ = QFileDialog.getOpenFileName(
            self,
            "Choose the matching Channel B TIFF",
            str(Path(channel_a).parent),
            "TIFF stacks (*.tif *.tiff)",
        )
        if not channel_b:
            return
        group = self.fallback_group_edit.text().strip()
        if not group:
            QMessageBox.warning(
                self, "Missing group", "Enter a fallback experimental-group name."
            )
            return
        first_parent = Path(channel_a).resolve().parent
        self.source_edit.setText(str(first_parent))
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(first_parent / "Synpo Results"))
        worker = ScanWorker(
            first_parent,
            default_group=group,
            manual_paths=(Path(channel_a), Path(channel_b)),
        )
        self._begin_import_scan(worker)

    def _prepare_preprocessing_tab(self) -> None:
        if self.manifest is None:
            self.tabs.setTabEnabled(1, False)
            return
        self.tabs.setTabEnabled(1, True)
        if self._job_thread is None:
            self.run_preprocessing_button.setEnabled(True)
            self.apply_preprocessing_button.setEnabled(True)
        current = self.preprocess_specimen.currentData()
        self.preprocess_specimen.blockSignals(True)
        self.preprocess_specimen.clear()
        for index, specimen in enumerate(self.manifest["specimens"]):
            self.preprocess_specimen.addItem(
                f"{specimen['experimental_group']} — {specimen['specimen_id']}", index
            )
        if current is not None:
            found = self.preprocess_specimen.findData(current)
            self.preprocess_specimen.setCurrentIndex(max(0, found))
        self.preprocess_specimen.blockSignals(False)
        roles = self.manifest["channel_roles"]
        for index in range(self.preprocess_channel.count()):
            channel = str(self.preprocess_channel.itemData(index))
            self.preprocess_channel.setItemText(
                index, f"{channel} — {ROLE_LABELS[str(roles[channel])]}"
            )
        self._preprocess_specimen_changed()
        completed = sum(
            specimen["checkpoints"]["preprocessing"].get("state") == "complete"
            for specimen in self.manifest["specimens"]
        )
        cache_path = self.manifest.get("cache", {}).get("path")
        self.preprocessing_status.setText(
            f"Preprocessing checkpoints: {completed}/{len(self.manifest['specimens'])} pairs complete."
            + (f" Cache: {cache_path}" if cache_path else "")
        )

    def _selected_specimen_index(self) -> int:
        value = self.preprocess_specimen.currentData()
        if value is None:
            raise ValueError("Select a specimen.")
        return int(value)

    def _selected_preprocess_channel(self) -> str:
        value = self.preprocess_channel.currentData()
        if value not in {"ChanA", "ChanB"}:
            raise ValueError("Select a channel.")
        return str(value)

    def _preprocess_specimen_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.preprocess_specimen.currentData() is None:
            return
        index = self._selected_specimen_index()
        specimen_key = (id(self.manifest), index)
        if specimen_key != self._preprocess_view_specimen_key:
            self.raw_view.reset_view()
            self.processed_view.reset_view()
            self._preprocess_view_specimen_key = specimen_key
        channel = self._selected_preprocess_channel()
        shape = self.manifest["specimens"][index]["channels"][channel]["metadata"]["shape"]
        z_count = 1 if len(shape) == 2 else int(shape[0])
        representatives = set(
            int(value)
            for value in self.manifest["preprocessing"].get(
                "representative_specimens", []
            )
        )
        self.representative_check.setChecked(index in representatives)
        special = {
            int(value)
            for value in self.manifest["preprocessing"].get(
                "special_specimens", []
            )
        }
        self.special_preprocessing_check.blockSignals(True)
        self.special_preprocessing_check.setChecked(index in special)
        self.special_preprocessing_check.blockSignals(False)
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, max(0, z_count - 1))
        self.z_slider.setValue(max(0, (z_count - 1) // 2))
        self.z_slider.blockSignals(False)
        self.z_label.setText(f"Z: {self.z_slider.value() + 1}/{z_count}")
        self._last_preview = None
        self._load_channel_settings()
        self._preview_timer.start()

    def _preprocess_channel_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None:
            return
        self._last_preview = None
        self._load_channel_settings()
        self._preprocess_specimen_changed()

    def _special_preprocessing_toggled(self, _checked: bool) -> None:
        if self.manifest is None:
            return
        self._last_preview = None
        self._load_channel_settings()
        self._preview_timer.start()

    def _load_channel_settings(self) -> None:
        if self.manifest is None:
            return
        channel = self._selected_preprocess_channel()
        specimen_index = self._selected_specimen_index()
        preprocessing = self.manifest["preprocessing"]
        saved = preprocessing.get("settings_by_specimen", {}).get(
            str(specimen_index), {}
        )
        if self.special_preprocessing_check.isChecked() and channel in saved:
            value = saved[channel]
        else:
            value = preprocessing["settings_by_channel"][channel]
        settings = PreprocessingSettings.from_dict(value)
        for widget, value in (
            (self.background_spin, settings.background_percentile),
            (self.sigma_xy_spin, settings.gaussian_sigma_xy_um),
            (self.sigma_z_spin, settings.gaussian_sigma_z_um),
            (self.sensitivity_spin, settings.threshold_sensitivity),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._update_sensitivity_warnings()

    def _current_preprocessing_settings(self) -> PreprocessingSettings:
        settings = PreprocessingSettings(
            background_percentile=self.background_spin.value(),
            gaussian_sigma_xy_um=self.sigma_xy_spin.value(),
            gaussian_sigma_z_um=self.sigma_z_spin.value(),
            threshold_sensitivity=self.sensitivity_spin.value(),
        )
        settings.validate()
        return settings

    def _update_sensitivity_warnings(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self, "preprocessing_sensitivity_warning"):
            self.preprocessing_sensitivity_warning.setText(
                "High-sensitivity mode (>3.0): substantially more background may be retained."
                if self.sensitivity_spin.value() > 3.0
                else ""
            )
        if hasattr(self, "detection_sensitivity_warning"):
            high = []
            if self.dendrite_detection_sensitivity.value() > 3.0:
                high.append("dendrite/spine")
            if self.cluster_detection_sensitivity.value() > 3.0:
                high.append("protein-cluster")
            self.detection_sensitivity_warning.setText(
                "High-sensitivity mode (>3.0) for "
                + " and ".join(high)
                + ": review the additional background candidates carefully."
                if high
                else ""
            )

    def _apply_preprocessing_settings(self) -> None:
        if self.manifest is None or self.project_path is None:
            QMessageBox.information(self, "No project", "Save or open a project first.")
            return
        try:
            channel, invalidated = self._store_preprocessing_selection()
            save_project(self.project_path, self.manifest)
            self._last_preview = None
            self.preprocessing_status.setText(
                f"Saved {channel} "
                + (
                    "special settings for this pair"
                    if self.special_preprocessing_check.isChecked()
                    else "batch-default settings"
                )
                + (
                    f"; invalidated {invalidated} affected preprocessing checkpoint(s)."
                    if invalidated
                    else "."
                )
                + " Previewing with the updated parameters."
            )
            self._request_preview()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot apply settings", str(exc))

    def _store_preprocessing_selection(self) -> tuple[str, int]:
        if self.manifest is None:
            raise ValueError("Save or open a project first.")
        before = {
            (index, channel): effective_preprocessing_settings(
                self.manifest, index, channel
            ).to_dict()
            for index, _specimen in enumerate(self.manifest["specimens"])
            for channel in ("ChanA", "ChanB")
        }
        channel = self._selected_preprocess_channel()
        specimen_index = self._selected_specimen_index()
        settings = self._current_preprocessing_settings().to_dict()
        preprocessing = self.manifest["preprocessing"]
        representatives = {
            int(value)
            for value in preprocessing.get("representative_specimens", [])
        }
        if self.representative_check.isChecked():
            representatives.add(specimen_index)
        else:
            representatives.discard(specimen_index)
        preprocessing["representative_specimens"] = sorted(representatives)

        special = {
            int(value) for value in preprocessing.get("special_specimens", [])
        }
        if self.special_preprocessing_check.isChecked():
            special.add(specimen_index)
            by_specimen = preprocessing.setdefault("settings_by_specimen", {})
            specimen_settings = by_specimen.setdefault(str(specimen_index), {})
            for candidate_channel in ("ChanA", "ChanB"):
                specimen_settings.setdefault(
                    candidate_channel,
                    dict(preprocessing["settings_by_channel"][candidate_channel]),
                )
            specimen_settings[channel] = settings
        else:
            special.discard(specimen_index)
            preprocessing["settings_by_channel"][channel] = settings
        preprocessing["special_specimens"] = sorted(special)

        changed: dict[int, list[str]] = {}
        for key, prior in before.items():
            index, candidate_channel = key
            current = effective_preprocessing_settings(
                self.manifest, index, candidate_channel
            ).to_dict()
            if current != prior:
                changed.setdefault(index, []).append(candidate_channel)
        for index, channels in changed.items():
            self._invalidate_preprocessing_channels(index, channels)
        return channel, sum(len(channels) for channels in changed.values())

    def _invalidate_preprocessing_channels(
        self, specimen_index: int, channels: list[str]
    ) -> None:
        if self.manifest is None:
            return
        specimen = self.manifest["specimens"][specimen_index]
        now = time.time()
        preprocessing = specimen["checkpoints"]["preprocessing"]
        completed_channels = preprocessing.setdefault("channels", {})
        for channel in channels:
            completed_channels.pop(channel, None)
        preprocessing.update({"state": "not_started", "updated_at": now})
        for stage in ("detection", "review", "measurements"):
            checkpoint = specimen["checkpoints"].setdefault(stage, {})
            checkpoint.update({"state": "not_started", "updated_at": now})
        specimen["review"]["state"] = "needs_attention"
        self._context_cache = {
            key: value
            for key, value in self._context_cache.items()
            if int(key[0]) != specimen_index
        }

    def _preview_cache_key(self) -> tuple[object, ...]:
        settings = self._current_preprocessing_settings()
        return (
            self._selected_specimen_index(),
            self._selected_preprocess_channel(),
            settings.background_percentile,
            settings.gaussian_sigma_xy_um,
            settings.gaussian_sigma_z_um,
            settings.threshold_sensitivity,
        )

    def _z_changed(self, value: int) -> None:
        total = self.z_slider.maximum() + 1
        self.z_label.setText(f"Z: {value + 1}/{total}")
        self._preview_timer.start()

    def _request_preview(self) -> None:
        if self.manifest is None:
            return
        if self._job_thread is not None:
            if self._job_kind == "preview":
                self._preview_requested_while_busy = True
            return
        try:
            index = self._selected_specimen_index()
            channel = self._selected_preprocess_channel()
            settings = self._current_preprocessing_settings()
            calibration = self.manifest["calibration"]
            channel_data = self.manifest["specimens"][index]["channels"][channel]
            path = channel_source_path(self.manifest, channel_data)
            key = self._preview_cache_key()
            worker = PreviewWorker(
                path,
                self.z_slider.value(),
                settings,
                float(calibration["xy_um_per_pixel"]),
                float(calibration["z_step_um"]),
                self._preview_statistics.get(key),
                key,
            )
            worker.completed.connect(self._preview_completed)
            self._start_worker(worker, "preview")
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot preview", str(exc))

    @Slot(object)
    def _preview_completed(self, payload: tuple[tuple[object, ...], PreviewResult]) -> None:
        key, result = payload
        sensitivity = float(key[-1])
        self._preview_statistics[key] = StackStatistics(
            background=result.background,
            otsu_threshold=result.threshold * sensitivity,
            applied_threshold=result.threshold,
            raw_low=result.raw_low,
            raw_high=result.raw_high,
        )
        expected = self._preview_cache_key()
        if key != expected or result.z_index != self.z_slider.value():
            self._preview_requested_while_busy = True
            return
        self._last_preview = result
        self._auto_contrast()
        self.preprocessing_status.setText(
            f"Background {result.background:.1f}; suggested threshold "
            f"{result.threshold:.1f}. Magenta voxels pass the detection threshold."
        )

    def _auto_contrast(self) -> None:
        if self._last_preview is None:
            return
        low = int(max(0, min(65534, round(self._last_preview.raw_low))))
        high = int(max(low + 1, min(65535, round(self._last_preview.raw_high))))
        self.contrast_low.blockSignals(True)
        self.contrast_high.blockSignals(True)
        self.contrast_low.setValue(low)
        self.contrast_high.setValue(high)
        self.contrast_low.blockSignals(False)
        self.contrast_high.blockSignals(False)
        self._render_preview()

    def _render_preview(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self._last_preview is None:
            return
        low = self.contrast_low.value()
        high = max(low + 1, self.contrast_high.value())
        self.raw_view.show_array(self._last_preview.raw, low, high)
        processed_low = max(0.0, low - self._last_preview.background)
        processed_high = max(processed_low + 1.0, high - self._last_preview.background)
        mask = (
            self._last_preview.processed >= self._last_preview.threshold
            if self.threshold_overlay.isChecked()
            else None
        )
        self.processed_view.show_array(
            self._last_preview.processed, processed_low, processed_high, mask
        )

    def _run_batch_preprocessing(self) -> None:
        if self.manifest is None or self.project_path is None:
            QMessageBox.information(self, "No project", "Save or open a project first.")
            return
        try:
            self._sync_manifest_edits()
            self._store_preprocessing_selection()
            save_project(self.project_path, self.manifest)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot start preprocessing", str(exc))
            return
        worker = BatchPreprocessWorker(self.manifest, self.project_path)
        worker.completed.connect(self._batch_preprocessing_completed)
        worker.cancelled.connect(self._batch_preprocessing_cancelled)
        self._start_worker(worker, "preprocess")

    def _cancel_batch_preprocessing(self) -> None:
        if isinstance(self._job_worker, BatchPreprocessWorker):
            self._job_worker.cancel()
            self.cancel_preprocessing_button.setEnabled(False)
            self.preprocessing_status.setText(
                "Cancellation requested; finishing the current slice safely."
            )

    @Slot(object)
    def _batch_preprocessing_completed(self, result: dict[str, object]) -> None:
        elapsed = float(result["elapsed_seconds"])
        self.preprocessing_status.setText(
            f"Batch preprocessing complete in {elapsed / 60:.1f} min. "
            f"Compressed cache: {result['cache_path']}"
        )
        self._prepare_preprocessing_tab()
        self._prepare_detection_tab()
        self._prepare_review_tab()
        self._prepare_measurements_tab()

    @Slot(str)
    def _batch_preprocessing_cancelled(self, message: str) -> None:
        self.preprocessing_status.setText(message)
        if self.project_path is not None and self.manifest is not None:
            save_project(self.project_path, self.manifest)

    def _prepare_detection_tab(self) -> None:
        if self.manifest is None:
            self.tabs.setTabEnabled(2, False)
            return
        self.tabs.setTabEnabled(2, True)
        settings = DetectionSettings.from_dict(
            self.manifest["detection"]["settings"]
        )
        for widget, value in (
            (self.dendrite_detection_sensitivity, settings.dendrite_sensitivity),
            (self.cluster_detection_sensitivity, settings.cluster_sensitivity),
            (self.spine_branch_length, settings.spine_branch_length_um),
            (self.minimum_dendrite_length, settings.minimum_dendrite_length_um),
            (self.minimum_spine_pixels, settings.minimum_spine_projection_pixels),
            (self.minimum_cluster_voxels, settings.minimum_cluster_voxels),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._update_sensitivity_warnings()
        memory_mode = str(
            self.manifest["detection"].get("memory_mode", AUTOMATIC_MEMORY_MODE)
        )
        memory_index = self.detection_memory_mode.findData(memory_mode)
        self.detection_memory_mode.setCurrentIndex(max(0, memory_index))

        current = self.detection_specimen.currentData()
        self.detection_specimen.blockSignals(True)
        self.detection_specimen.clear()
        for index, specimen in enumerate(self.manifest["specimens"]):
            state = specimen["checkpoints"]["detection"].get("state", "not_started")
            self.detection_specimen.addItem(
                f"{specimen['experimental_group']} — {specimen['specimen_id']} [{state}]",
                index,
            )
        if current is not None:
            found = self.detection_specimen.findData(current)
            self.detection_specimen.setCurrentIndex(max(0, found))
        self.detection_specimen.blockSignals(False)

        dendrite_channel = next(
            channel
            for channel, role in self.manifest["channel_roles"].items()
            if role == "dendrite_spines"
        )
        self.detection_background_channel.setCurrentIndex(
            self.detection_background_channel.findData(dendrite_channel)
        )
        eligible = sum(
            specimen["checkpoints"]["preprocessing"].get("state") == "complete"
            for specimen in self.manifest["specimens"]
        )
        complete = sum(
            specimen["checkpoints"]["detection"].get("state") == "complete"
            for specimen in self.manifest["specimens"]
        )
        skipped = sum(
            specimen["checkpoints"]["detection"].get("state") == "skipped"
            for specimen in self.manifest["specimens"]
        )
        failed = sum(
            specimen["checkpoints"]["detection"].get("state") == "failed"
            for specimen in self.manifest["specimens"]
        )
        self.detection_status.setText(
            f"Detection checkpoints: {complete}/{len(self.manifest['specimens'])} complete; "
            f"{skipped} skipped; {failed} failed; {eligible} pair(s) currently eligible. "
            "Skipped and failed specimens are retried on the next run."
        )
        if self._job_thread is None:
            self.apply_detection_button.setEnabled(True)
            self.run_detection_button.setEnabled(eligible > 0)
        self._detection_specimen_changed()

    def _current_detection_settings(self) -> DetectionSettings:
        settings = DetectionSettings(
            dendrite_sensitivity=self.dendrite_detection_sensitivity.value(),
            cluster_sensitivity=self.cluster_detection_sensitivity.value(),
            spine_branch_length_um=self.spine_branch_length.value(),
            minimum_dendrite_length_um=self.minimum_dendrite_length.value(),
            minimum_spine_projection_pixels=self.minimum_spine_pixels.value(),
            minimum_cluster_voxels=self.minimum_cluster_voxels.value(),
        )
        settings.validate()
        return settings

    def _apply_detection_settings(self) -> None:
        if self.manifest is None or self.project_path is None:
            QMessageBox.information(self, "No project", "Save or open a project first.")
            return
        try:
            self.manifest["detection"]["settings"] = (
                self._current_detection_settings().to_dict()
            )
            self.manifest["detection"]["memory_mode"] = str(
                self.detection_memory_mode.currentData() or AUTOMATIC_MEMORY_MODE
            )
            save_project(self.project_path, self.manifest)
            self.detection_status.setText(
                "Detection settings saved. Running again will replace stale automatic masks."
            )
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save detection settings", str(exc))

    def _run_detection(self) -> None:
        if self.manifest is None or self.project_path is None:
            QMessageBox.information(self, "No project", "Save or open a project first.")
            return
        try:
            self._sync_manifest_edits()
            self.manifest["detection"]["settings"] = (
                self._current_detection_settings().to_dict()
            )
            self.manifest["detection"]["memory_mode"] = str(
                self.detection_memory_mode.currentData() or AUTOMATIC_MEMORY_MODE
            )
            save_project(self.project_path, self.manifest)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot start detection", str(exc))
            return
        worker = DetectionWorker(self.manifest, self.project_path)
        worker.pair_completed.connect(self._detection_pair_completed)
        worker.completed.connect(self._detection_completed)
        worker.cancelled.connect(self._detection_cancelled)
        self._start_worker(worker, "detection")

    def _cancel_detection(self) -> None:
        if isinstance(self._job_worker, DetectionWorker):
            self._job_worker.cancel()
            self.cancel_detection_button.setEnabled(False)
            self.detection_status.setText(
                "Cancellation requested. The current safe step will finish first."
            )

    @Slot(int, object)
    def _detection_pair_completed(
        self, specimen_index: int, summary: dict[str, object]
    ) -> None:
        outcome = str(summary.get("outcome", "complete"))
        if outcome != "complete":
            specimen = self.manifest["specimens"][specimen_index]
            self.detection_status.setText(
                f"Pair {specimen_index + 1} {outcome}: "
                f"{summary.get('reason', 'Unknown error')}. Detection continues."
            )
            row = self.detection_specimen.findData(specimen_index)
            if row >= 0:
                self.detection_specimen.setItemText(
                    row,
                    f"{specimen['experimental_group']} / {specimen['specimen_id']} "
                    f"[{outcome}]",
                )
            if self.detection_specimen.currentData() == specimen_index:
                self._detection_specimen_changed()
            return
        if not bool(summary.get("skipped", False)):
            self._invalidate_context_views(specimen_index, corrected_only=False)
        self.detection_status.setText(
            f"Completed pair {specimen_index + 1}: {summary['dendrite_count']} dendrite "
            f"field(s), {summary['spine_count']} spine candidates, "
            f"{summary['cluster_count']} cluster candidates. Detection continues in background."
        )
        mode = str(summary.get("processing_mode", "standard")).replace("_", " ")
        reused = "reused, " if bool(summary.get("reused", False)) else ""
        self.detection_status.setText(
            f"Completed pair {specimen_index + 1} ({reused}{mode}). "
            "Detection continues in the background."
        )
        row = self.detection_specimen.findData(specimen_index)
        if row >= 0:
            specimen = self.manifest["specimens"][specimen_index]
            self.detection_specimen.setItemText(
                row,
                f"{specimen['experimental_group']} — {specimen['specimen_id']} [complete]",
            )
        if self.detection_specimen.currentData() == specimen_index:
            self._detection_specimen_changed()
        self._prepare_review_tab()
        self._prepare_measurements_tab()

    @Slot(object)
    def _detection_completed(self, result: dict[str, object]) -> None:
        self.detection_status.setText(
            f"Detection batch finished in {float(result['elapsed_seconds']) / 60:.1f} min: "
            f"{result['completed_pairs']} complete "
            f"({result['low_memory_pairs']} newly processed in low-memory mode), "
            f"{result['skipped_pairs']} skipped, {result['failed_pairs']} failed."
        )
        self._prepare_detection_tab()
        self._prepare_review_tab()
        self._prepare_measurements_tab()

    @Slot(str)
    def _detection_cancelled(self, message: str) -> None:
        self.detection_status.setText(message)
        if self.project_path is not None and self.manifest is not None:
            save_project(self.project_path, self.manifest)

    def _detection_specimen_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.detection_specimen.currentData() is None:
            return
        index = int(self.detection_specimen.currentData())
        specimen_key = (id(self.manifest), index)
        if specimen_key != self._detection_view_specimen_key:
            self.detection_view.reset_view()
            self._detection_view_specimen_key = specimen_key
        channel = str(self.detection_background_channel.currentData() or "ChanB")
        shape = self.manifest["specimens"][index]["channels"][channel]["metadata"]["shape"]
        z_count = 1 if len(shape) == 2 else int(shape[0])
        self.detection_z_slider.blockSignals(True)
        self.detection_z_slider.setRange(0, max(0, z_count - 1))
        self.detection_z_slider.setValue(max(0, (z_count - 1) // 2))
        self.detection_z_slider.blockSignals(False)
        self.detection_z_label.setText(
            f"Z: {self.detection_z_slider.value() + 1}/{z_count}"
        )
        checkpoint = self.manifest["specimens"][index]["checkpoints"]["detection"]
        summary = checkpoint.get("summary", {})
        if checkpoint.get("state") == "complete":
            self.detection_projections_button.setEnabled(True)
            self.detection_3d_button.setEnabled(True)
            self.detection_counts.setText(
                f"{summary.get('dendrite_count', 0)} dendrites | "
                f"{summary.get('spine_count', 0)} spines "
                f"({summary.get('flagged_spine_count', 0)} flagged) | "
                f"{summary.get('cluster_count', 0)} clusters "
                f"({summary.get('flagged_cluster_count', 0)} flagged)"
            )
            self._load_detection_view()
        else:
            self.detection_projections_button.setEnabled(False)
            self.detection_3d_button.setEnabled(False)
            self._last_detection = None
            state = str(checkpoint.get("state", "not_started"))
            reason = str(checkpoint.get("reason", "")).strip()
            detail = f" Reason: {reason}" if reason else ""
            self.detection_counts.setText(f"Detection state: {state}.{detail}")
            self.detection_view.setText(
                f"Detection is not complete for this specimen.{detail}"
            )

    def _detection_z_changed(self, value: int) -> None:
        self.detection_z_label.setText(
            f"Z: {value + 1}/{self.detection_z_slider.maximum() + 1}"
        )
        self._load_detection_view()

    def _load_detection_view(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.detection_specimen.currentData() is None:
            return
        index = int(self.detection_specimen.currentData())
        checkpoint = self.manifest["specimens"][index]["checkpoints"]["detection"]
        if checkpoint.get("state") != "complete":
            return
        try:
            self._last_detection = load_detection_slice(
                self.manifest,
                index,
                self.detection_z_slider.value(),
                str(self.detection_background_channel.currentData()),
            )
            self._auto_detection_contrast()
        except (OSError, ValueError, KeyError, IndexError) as exc:
            self.detection_view.setText(f"Cannot load detection slice: {exc}")

    def _auto_detection_contrast(self) -> None:
        if self._last_detection is None:
            return
        low, high = np.percentile(self._last_detection.raw, (0.5, 99.8))
        low_value = int(max(0, min(65534, round(float(low)))))
        high_value = int(max(low_value + 1, min(65535, round(float(high)))))
        self.detection_black.blockSignals(True)
        self.detection_white.blockSignals(True)
        self.detection_black.setValue(low_value)
        self.detection_white.setValue(high_value)
        self.detection_black.blockSignals(False)
        self.detection_white.blockSignals(False)
        self._render_detection_view()

    def _render_detection_view(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self._last_detection is None:
            return
        self.detection_view.show_detection(
            self._last_detection.raw,
            self.detection_black.value(),
            max(self.detection_black.value() + 1, self.detection_white.value()),
            dendrites=(
                self._last_detection.dendrites if self.show_dendrites.isChecked() else None
            ),
            spines=self._last_detection.spines if self.show_spines.isChecked() else None,
            clusters=(
                self._last_detection.clusters if self.show_clusters.isChecked() else None
            ),
        )

    def _prepare_review_tab(self) -> None:
        if self._review_thread is not None:
            self._review_refresh_pending = True
            return
        self._review_refresh_pending = False
        if self.manifest is None:
            self.tabs.setTabEnabled(3, False)
            return
        detected = [
            index
            for index, specimen in enumerate(self.manifest["specimens"])
            if specimen["checkpoints"]["detection"].get("state") == "complete"
        ]
        self.tabs.setTabEnabled(3, bool(detected))
        current = self.review_specimen.currentData()
        self.review_specimen.blockSignals(True)
        self.review_specimen.clear()
        complete_count = 0
        for index in detected:
            specimen = self.manifest["specimens"][index]
            state = str(specimen["review"].get("state", "needs_attention"))
            if state == "complete":
                complete_count += 1
            self.review_specimen.addItem(
                f"{specimen['experimental_group']} — {specimen['specimen_id']} [{state}]",
                index,
            )
        if current is not None:
            found = self.review_specimen.findData(current)
            self.review_specimen.setCurrentIndex(max(0, found))
        self.review_specimen.blockSignals(False)
        self.review_queue_status.setText(
            f"{len(detected)} detected specimen(s) available; "
            f"{complete_count} marked review complete. New detections appear here immediately."
        )
        memory_mode = str(
            self.manifest["review_settings"].get(
                "memory_mode", AUTOMATIC_REVIEW_MEMORY_MODE
            )
        )
        memory_index = self.review_memory_mode.findData(memory_mode)
        self.review_memory_mode.setCurrentIndex(max(0, memory_index))
        if detected:
            self._review_specimen_changed()
        else:
            self._last_review = None
            self.review_view.setText("Waiting for automatic detection to finish a specimen.")
        self._set_review_busy(self._review_thread is not None)

    def _current_measurement_settings(self) -> MeasurementSettings:
        return MeasurementSettings(
            minimum_cluster_spine_overlap_percent=self.measurement_overlap.value(),
            cluster_end_method=str(self.measurement_end_method.currentData()),  # type: ignore[arg-type]
            fixed_end_slices=self.measurement_fixed_slices.value(),
            adaptive_area_factor=self.measurement_area_factor.value(),
            minimum_retained_slices=self.measurement_min_slices.value(),
        )

    def _measurement_method_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        method = self.measurement_end_method.currentData()
        self.measurement_fixed_slices.setEnabled(method == "fixed")
        self.measurement_area_factor.setEnabled(method == "adaptive")

    def _prepare_measurements_tab(self) -> None:
        if self.manifest is None:
            self.tabs.setTabEnabled(4, False)
            return
        reviewed = [
            index
            for index, specimen in enumerate(self.manifest["specimens"])
            if specimen["checkpoints"]["preprocessing"].get("state") == "complete"
            and specimen["checkpoints"]["detection"].get("state") == "complete"
            and specimen["checkpoints"]["review"].get("state") == "complete"
        ]
        self.tabs.setTabEnabled(4, bool(reviewed))
        settings = MeasurementSettings.from_dict(
            self.manifest["measurements"]["settings"]
        )
        controls = (
            self.measurement_overlap,
            self.measurement_fixed_slices,
            self.measurement_area_factor,
            self.measurement_min_slices,
        )
        for control in controls:
            control.blockSignals(True)
        self.measurement_end_method.blockSignals(True)
        self.measurement_overlap.setValue(
            settings.minimum_cluster_spine_overlap_percent
        )
        self.measurement_end_method.setCurrentIndex(
            self.measurement_end_method.findData(settings.cluster_end_method)
        )
        self.measurement_fixed_slices.setValue(settings.fixed_end_slices)
        self.measurement_area_factor.setValue(settings.adaptive_area_factor)
        self.measurement_min_slices.setValue(settings.minimum_retained_slices)
        self.measurement_end_method.blockSignals(False)
        for control in controls:
            control.blockSignals(False)
        self._measurement_method_changed()

        current = self.measurement_specimen.currentData()
        self.measurement_specimen.blockSignals(True)
        self.measurement_specimen.clear()
        complete = 0
        for index in reviewed:
            specimen = self.manifest["specimens"][index]
            checkpoint = specimen["checkpoints"].get("measurements", {})
            state = str(checkpoint.get("state", "not_started"))
            complete += state == "complete"
            self.measurement_specimen.addItem(
                f"{specimen['experimental_group']} — {specimen['specimen_id']} [{state}]",
                index,
            )
        if current is not None:
            found = self.measurement_specimen.findData(current)
            self.measurement_specimen.setCurrentIndex(max(0, found))
        self.measurement_specimen.blockSignals(False)
        self.measurement_status.setText(
            f"{len(reviewed)} fully reviewed specimen(s) eligible; "
            f"{complete} measurement checkpoint(s) complete. "
            "Incomplete pairs are skipped and can be measured later with Resume."
        )
        self._refresh_distribution_groups()
        if reviewed:
            self._measurement_specimen_changed()

    def _save_measurement_settings(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            settings = self._current_measurement_settings()
            self.manifest["measurements"]["settings"] = settings.to_dict()
            for specimen in self.manifest["specimens"]:
                checkpoint = specimen["checkpoints"].setdefault("measurements", {})
                checkpoint["state"] = "not_started"
            save_project(self.project_path, self.manifest)
            self._prepare_measurements_tab()
            self.measurement_status.setText(
                "Measurement settings saved. Existing results will be recomputed or resumed by signature."
            )
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save measurement settings", str(exc))

    def _run_measurements(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            settings = self._current_measurement_settings()
            self.manifest["measurements"]["settings"] = settings.to_dict()
            save_project(self.project_path, self.manifest)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot start measurements", str(exc))
            return
        worker = MeasurementWorker(self.manifest, self.project_path)
        worker.completed.connect(self._measurements_completed)
        worker.cancelled.connect(self._measurements_cancelled)
        self._start_worker(worker, "measurements")

    def _cancel_measurements(self) -> None:
        if isinstance(self._job_worker, MeasurementWorker):
            self._job_worker.cancel()
            self.measurement_status.setText(
                "Cancellation requested; finishing the current safe slice."
            )

    @Slot(object)
    def _measurements_completed(self, result: dict[str, object]) -> None:
        summaries = result.get("summaries", [])
        self.measurement_status.setText(
            f"Measurement batch complete: {len(summaries)} specimen checkpoint(s) ready."
        )
        self._prepare_measurements_tab()

    @Slot(str)
    def _measurements_cancelled(self, message: str) -> None:
        self.measurement_status.setText(message)
        if self.manifest is not None and self.project_path is not None:
            save_project(self.project_path, self.manifest)
        self._prepare_measurements_tab()

    def _measurement_specimen_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        self._last_trim_preview = None
        self.measurement_cluster.clear()
        value = self.measurement_specimen.currentData()
        if self.manifest is None or value is None:
            return
        specimen_index = int(value)
        checkpoint = self.manifest["specimens"][specimen_index]["checkpoints"].get(
            "measurements", {}
        )
        if checkpoint.get("state") != "complete":
            self.measurement_summary.setText(
                "This specimen has not been measured with the current saved settings."
            )
            self.measurement_table.setRowCount(0)
            self.measurement_table.setColumnCount(0)
            return
        try:
            result = load_measurement_result(self.manifest, specimen_index)
        except (OSError, ValueError, KeyError) as exc:
            self.measurement_summary.setText(f"Cannot load measurements: {exc}")
            return
        self._populate_spine_review_queue(result)
        specimen_row = result["specimen_rows"][0]
        source = "corrected" if result.get("corrected_masks") else "automatic"
        self.measurement_summary.setText(
            f"{specimen_row['dendrite_count']} dendrite(s), "
            f"{specimen_row['spine_count']} spine(s), "
            f"{specimen_row['included_cluster_count']} included cluster(s) | "
            f"{source} masks | overlap threshold "
            f"{result['settings']['minimum_cluster_spine_overlap_percent']:.1f}%."
        )
        for cluster_id, details in result.get("cluster_trim_details", {}).items():
            self.measurement_cluster.addItem(
                f"Cluster {cluster_id}: {len(details.get('discarded_z_slices', []))} discarded Z slice(s)",
                int(cluster_id),
            )
        self._populate_measurement_table()
        if self.distribution_spine.count():
            self._distribution_spine_changed()

    def _current_spine_review_mode(self) -> str:
        return str(
            self._preferred_spine_review_mode
            or self.distribution_review_mode.currentData()
            or "cluster_positive"
        )

    def _populate_spine_review_queue(self, result: dict[str, object]) -> None:
        mode = self._current_spine_review_mode()
        if self._preferred_spine_review_mode is not None:
            index = self.distribution_review_mode.findData(mode)
            if index >= 0:
                self.distribution_review_mode.blockSignals(True)
                self.distribution_review_mode.setCurrentIndex(index)
                self.distribution_review_mode.blockSignals(False)
            self._preferred_spine_review_mode = None
        previous_spine = (
            self._preferred_distribution_spine_id
            if self._preferred_distribution_spine_id is not None
            else self.distribution_spine.currentData()
        )
        self._preferred_distribution_spine_id = None
        rows = (
            list(result.get("distribution_rows", []))
            if mode == "cluster_positive"
            else [
                row
                for row in result.get("spine_rows", [])
                if not bool(row.get("has_protein_cluster", False))
            ]
        )
        self.distribution_spine.blockSignals(True)
        self.distribution_spine.clear()
        review_key = (
            "distribution_reviewed"
            if mode == "cluster_positive"
            else "validity_reviewed"
        )
        for row in rows:
            reviewed = bool(row.get(review_key, False))
            valid = bool(row.get("spine_valid", True))
            if mode == "cluster_positive":
                state = (
                    "invalid"
                    if not valid
                    else (
                        "included"
                        if bool(row.get("distribution_included", False))
                        else "distribution excluded"
                    )
                )
                detail = f"; {row.get('distribution_axis_status', '')}"
            else:
                state = "invalid" if not valid else "valid"
                detail = ""
            self.distribution_spine.addItem(
                f"Spine {row['spine_id']} — "
                f"{'reviewed' if reviewed else 'unreviewed'}; {state}{detail}",
                int(row["spine_id"]),
            )
        selected = (
            self.distribution_spine.findData(previous_spine)
            if previous_spine is not None
            else -1
        )
        if selected < 0:
            selected = next(
                (
                    index
                    for index, row in enumerate(rows)
                    if not bool(row.get(review_key, False))
                ),
                0,
            )
        if self.distribution_spine.count():
            self.distribution_spine.setCurrentIndex(selected)
        self.distribution_spine.blockSignals(False)
        positive_mode = mode == "cluster_positive"
        for widget in (
            self.centerline_hint_button,
            self.clear_centerline_hint_button,
            self.distribution_include,
        ):
            widget.setVisible(positive_mode)
        if not self.distribution_spine.count():
            self.distribution_spine_header.setText(
                "No protein-cluster-positive spines"
                if positive_mode
                else "No cluster-less spines"
            )
            self.distribution_preview_status.setText(
                "This optional review queue is empty for the selected specimen."
            )

    def _distribution_review_mode_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.measurement_specimen.currentData() is None:
            return
        try:
            result = load_measurement_result(
                self.manifest, int(self.measurement_specimen.currentData())
            )
        except (OSError, ValueError, KeyError):
            return
        self._preferred_distribution_spine_id = None
        self._populate_spine_review_queue(result)
        if self.distribution_spine.count():
            self._distribution_spine_changed()

    def _refresh_distribution_groups(self) -> None:
        if self.manifest is None:
            return
        results = []
        for index, specimen in enumerate(self.manifest["specimens"]):
            if specimen["checkpoints"].get("measurements", {}).get("state") != "complete":
                continue
            try:
                results.append(load_measurement_result(self.manifest, index))
            except (ValueError, OSError):
                continue
        _specimen_rows, group_rows = distribution_summary_rows(results)
        current = self.distribution_group_combo.currentData()
        self.distribution_group_combo.blockSignals(True)
        self.distribution_group_combo.clear()
        for row in group_rows:
            self.distribution_group_combo.addItem(str(row["experimental_group"]), row)
        if current is not None:
            index = self.distribution_group_combo.findData(current)
            if index >= 0:
                self.distribution_group_combo.setCurrentIndex(index)
        self.distribution_group_combo.blockSignals(False)
        self._update_distribution_chart()

    def _update_distribution_chart(self, *_args) -> None:  # type: ignore[no-untyped-def]
        row = self.distribution_group_combo.currentData()
        self.distribution_chart.set_mode(str(self.distribution_chart_mode.currentData() or "line"))
        self.distribution_chart.set_fixed_scale(self.distribution_fixed_scale.isChecked())
        self.distribution_chart.set_profile(row if isinstance(row, dict) else None)

    def _distribution_spine_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if (
            self.manifest is None
            or self.measurement_specimen.currentData() is None
            or self.distribution_spine.currentData() is None
        ):
            return
        specimen_index = int(self.measurement_specimen.currentData())
        spine_id = int(self.distribution_spine.currentData())
        review_mode = self._current_spine_review_mode()
        positive_mode = review_mode == "cluster_positive"
        self._last_distribution_preview = None
        self.centerline_hint_button.setEnabled(False)
        self.centerline_hint_button.blockSignals(True)
        self.centerline_hint_button.setChecked(False)
        self.centerline_hint_button.blockSignals(False)
        self.distribution_dendrite_view.point_mode = False
        self.distribution_z_controls.setVisible(False)
        try:
            result = load_measurement_result(self.manifest, specimen_index)
            rows = (
                result.get("distribution_rows", [])
                if positive_mode
                else result.get("spine_rows", [])
            )
            row = next(item for item in rows if int(item["spine_id"]) == spine_id)
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            self.distribution_preview_status.setText(f"Cannot load spine review row: {exc}")
            return
        self.distribution_include.blockSignals(True)
        self.distribution_invalid.blockSignals(True)
        self.distribution_include.setChecked(
            bool(row.get("distribution_included", False)) if positive_mode else False
        )
        self.distribution_invalid.setChecked(not bool(row.get("spine_valid", True)))
        self.distribution_include.setEnabled(
            positive_mode and bool(row.get("spine_valid", True))
        )
        self.distribution_note.setText(
            str(row.get("review_note", row.get("validity_note", "")))
        )
        self.distribution_include.blockSignals(False)
        self.distribution_invalid.blockSignals(False)
        self.clear_centerline_hint_button.setEnabled(
            positive_mode and bool(row.get("centerline_endpoint_hint_present", False))
        )
        position = self.distribution_spine.currentIndex() + 1
        count = self.distribution_spine.count()
        kind_label = (
            "Protein-cluster-positive spine"
            if positive_mode
            else "Cluster-less spine (optional quality review)"
        )
        self.distribution_spine_header.setText(
            f"{kind_label} — Spine {spine_id} — {position} of {count}"
        )
        self.distribution_preview_status.setText(
            f"Loading Spine {spine_id}. "
            + (
                f"{row.get('distribution_axis_status', '')}. {row.get('distribution_axis_note', '')}"
                if positive_mode
                else "This queue contains spines with no cluster meeting the current overlap threshold."
            )
        )
        if self._job_thread is not None:
            self.distribution_preview_status.setText(
                "Distribution row is ready; its image preview will load when the current background operation finishes."
            )
            return
        worker = DistributionPreviewWorker(
            self.manifest, specimen_index, spine_id, review_mode
        )
        worker.completed.connect(self._distribution_preview_completed)
        self._start_worker(worker, "distribution_preview")

    @Slot(object)
    def _distribution_preview_completed(
        self, preview: DistributionPreview | SpineReviewPreview
    ) -> None:
        self._last_distribution_preview = preview
        if isinstance(preview, SpineReviewPreview):
            self.centerline_hint_button.setEnabled(False)
            spine_id = int(preview.row["spine_id"])
            spine_overlay = np.where(
                preview.spine_projection == spine_id, 1, 0
            ).astype(np.uint8)
            cluster_overlay = np.where(
                preview.cluster_projection > 0, 1, 0
            ).astype(np.uint8)
            self.distribution_dendrite_view.show_rgb(
                self._distribution_overlay(
                    preview.dendrite_projection, spine_overlay, ()
                )
            )
            self.distribution_protein_view.show_rgb(
                self._distribution_overlay(
                    preview.protein_projection, cluster_overlay, ()
                )
            )
            self.clear_centerline_hint_button.setEnabled(False)
            self.distribution_preview_status.setText(
                f"Spine {spine_id} has no protein cluster meeting the current overlap rule. "
                "Mark it invalid only if the spine segmentation itself is unusable. "
                "Click either crop to locate it in the full specimen."
            )
            self.measurement_result_tabs.setCurrentIndex(1)
            return
        self.centerline_hint_button.setEnabled(True)
        self.distribution_z_slider.blockSignals(True)
        self.distribution_z_slider.setRange(*preview.spine_z_range)
        preferred_z = (
            preview.endpoint_local_zyx[0]
            if preview.endpoint_local_zyx is not None
            else preview.spine_z_range[0]
        )
        self.distribution_z_slider.setValue(
            min(preview.spine_z_range[1], max(preview.spine_z_range[0], preferred_z))
        )
        self.distribution_z_slider.blockSignals(False)
        self._render_distribution_dendrite_view()
        self.distribution_protein_view.show_rgb(
            self._distribution_overlay(preview.protein_projection, preview.cluster_bins_projection, ())
        )
        self.clear_centerline_hint_button.setEnabled(
            bool(preview.row.get("centerline_endpoint_hint_present", False))
        )
        ratios = [preview.row.get(f"bin_{index:02d}_ratio") for index in range(1, 11)]
        self.distribution_preview_status.setText(
            f"Spine {preview.row['spine_id']} | {preview.row['distribution_axis_status']} | "
            f"protein-cluster-positive | shaft-to-tip ratios: "
            + ", ".join("blank" if value is None else f"{float(value):.3g}" for value in ratios)
            + ". Click either crop to locate it in the full specimen."
        )
        self.measurement_result_tabs.setCurrentIndex(1)

    @staticmethod
    def _distribution_overlay(
        raw: np.ndarray,
        bins: np.ndarray,
        axis_xy: tuple[tuple[int, int], ...],
        base_xy: tuple[int, int] | None = None,
        endpoint_xy: tuple[int, int] | None = None,
    ) -> np.ndarray:
        low, high = np.percentile(raw, (0.5, 99.8))
        scale = max(1.0, float(high - low))
        gray = np.clip((raw.astype(np.float32) - low) * 255.0 / scale, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        for index, color in enumerate(DISTRIBUTION_COLORS, start=1):
            mask = bins == index
            if np.any(mask):
                rgb[mask] = np.clip(rgb[mask].astype(np.float32) * 0.30 + np.asarray(color) * 0.70, 0, 255).astype(np.uint8)
        for x, y in axis_xy:
            if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                rgb[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 255
        for point, color in (
            (base_xy, np.asarray((32, 220, 88), dtype=np.uint8)),
            (endpoint_xy, np.asarray((238, 50, 200), dtype=np.uint8)),
        ):
            if point is None:
                continue
            x, y = point
            yy, xx = np.ogrid[: rgb.shape[0], : rgb.shape[1]]
            circle = (xx - x) ** 2 + (yy - y) ** 2 <= 16
            rgb[circle] = color
        return rgb

    def _render_distribution_dendrite_view(self) -> None:
        preview = self._last_distribution_preview
        if not isinstance(preview, DistributionPreview):
            return
        if self.centerline_hint_button.isChecked():
            z_index = self.distribution_z_slider.value()
            raw = preview.dendrite_stack[z_index]
            spine_mask = preview.spine_mask_stack[z_index]
            bins = np.where(spine_mask, 1, 0).astype(np.uint8)
            axis = tuple(
                (point[2], point[1])
                for point in preview.axis_points_local_zyx
                if point[0] == z_index
            )
            base = (
                (preview.base_point_local_zyx[2], preview.base_point_local_zyx[1])
                if preview.base_point_local_zyx is not None
                and preview.base_point_local_zyx[0] == z_index
                else None
            )
            endpoint = (
                (preview.endpoint_local_zyx[2], preview.endpoint_local_zyx[1])
                if preview.endpoint_local_zyx is not None
                and preview.endpoint_local_zyx[0] == z_index
                else None
            )
            rgb = self._distribution_overlay(raw, bins, axis, base, endpoint)
            self.distribution_z_label.setText(
                f"Spine Z: {z_index + 1} "
                f"({preview.spine_z_range[0] + 1}–{preview.spine_z_range[1] + 1})"
            )
        else:
            base = (
                (preview.base_point_local_zyx[2], preview.base_point_local_zyx[1])
                if preview.base_point_local_zyx is not None
                else None
            )
            endpoint = (
                (preview.endpoint_local_zyx[2], preview.endpoint_local_zyx[1])
                if preview.endpoint_local_zyx is not None
                else None
            )
            rgb = self._distribution_overlay(
                preview.dendrite_projection,
                preview.spine_bins_projection,
                preview.axis_xy,
                base,
                endpoint,
            )
        self.distribution_dendrite_view.show_rgb(rgb)

    def _centerline_hint_mode_changed(self, enabled: bool) -> None:
        if enabled and not isinstance(
            self._last_distribution_preview, DistributionPreview
        ):
            self.centerline_hint_button.blockSignals(True)
            self.centerline_hint_button.setChecked(False)
            self.centerline_hint_button.blockSignals(False)
            self.distribution_preview_status.setText(
                "Wait for the cropped spine preview before placing an endpoint hint."
            )
            return
        self.distribution_z_controls.setVisible(enabled)
        self.distribution_dendrite_view.point_mode = enabled
        self.distribution_dendrite_view.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self._render_distribution_dendrite_view()
        if enabled:
            self.distribution_preview_status.setText(
                "Choose a spine-only Z slice, then click near a spine voxel to set the centerline endpoint."
            )

    def _distribution_z_changed(self, _value: int) -> None:
        if self.centerline_hint_button.isChecked():
            self._render_distribution_dendrite_view()

    def _centerline_hint_clicked(self, x: int, y: int) -> None:
        preview = self._last_distribution_preview
        if (
            not self.centerline_hint_button.isChecked()
            or not isinstance(preview, DistributionPreview)
            or self._job_thread is not None
        ):
            return
        z_index = self.distribution_z_slider.value()
        mask = preview.spine_mask_stack[z_index]
        yy, xx = np.nonzero(mask)
        if not len(xx):
            self.distribution_preview_status.setText(
                "No selected-spine voxels exist on this slice. Choose another spine Z slice."
            )
            return
        distances = (xx - x) ** 2 + (yy - y) ** 2
        nearest = int(np.argmin(distances))
        search_radius_pixels = 12
        if int(distances[nearest]) > search_radius_pixels**2:
            self.distribution_preview_status.setText(
                "Endpoint not placed: click closer to the colored spine on this slice."
            )
            return
        local_x, local_y = int(xx[nearest]), int(yy[nearest])
        global_point = (
            z_index,
            preview.crop_origin_yx[0] + local_y,
            preview.crop_origin_yx[1] + local_x,
        )
        if (
            self.manifest is None
            or self.project_path is None
            or self.measurement_specimen.currentData() is None
            or self.distribution_spine.currentData() is None
        ):
            return
        self.distribution_preview_status.setText(
            f"Applying endpoint at Z {z_index + 1}; rebuilding this spine’s centerline…"
        )
        worker = CenterlineHintWorker(
            self.manifest,
            self.project_path,
            int(self.measurement_specimen.currentData()),
            int(self.distribution_spine.currentData()),
            global_point,
        )
        worker.completed.connect(self._centerline_hint_completed)
        self._start_worker(worker, "centerline_hint")

    def _clear_centerline_hint(self) -> None:
        if (
            self.manifest is None
            or self.project_path is None
            or self.measurement_specimen.currentData() is None
            or self.distribution_spine.currentData() is None
            or self._job_thread is not None
        ):
            return
        self.distribution_preview_status.setText(
            "Clearing the manual endpoint and restoring automatic endpoint detection…"
        )
        worker = CenterlineHintWorker(
            self.manifest,
            self.project_path,
            int(self.measurement_specimen.currentData()),
            int(self.distribution_spine.currentData()),
            None,
        )
        worker.completed.connect(self._centerline_hint_completed)
        self._start_worker(worker, "centerline_hint")

    @Slot(object)
    def _centerline_hint_completed(self, _result: dict[str, object]) -> None:
        self.centerline_hint_button.blockSignals(True)
        self.centerline_hint_button.setChecked(False)
        self.centerline_hint_button.blockSignals(False)
        self.distribution_dendrite_view.point_mode = False
        self.distribution_dendrite_view.setCursor(Qt.CursorShape.ArrowCursor)
        self.distribution_z_controls.setVisible(False)
        self._centerline_hint_pending_reload = True
        self._refresh_distribution_groups()
        self._populate_measurement_table()
        self.distribution_preview_status.setText(
            "Centerline endpoint checkpoint saved. Reloading the cropped maximum projection…"
        )

    def _move_distribution_spine(self, offset: int) -> None:
        count = self.distribution_spine.count()
        if count:
            self.distribution_spine.setCurrentIndex((self.distribution_spine.currentIndex() + offset) % count)

    def _save_distribution_review(self) -> None:
        if (
            self.manifest is None
            or self.project_path is None
            or self.measurement_specimen.currentData() is None
            or self.distribution_spine.currentData() is None
        ):
            return
        specimen_index = int(self.measurement_specimen.currentData())
        spine_id = int(self.distribution_spine.currentData())
        review_mode = self._current_spine_review_mode()
        try:
            if review_mode == "cluster_positive":
                updated = set_distribution_review(
                    self.manifest,
                    self.project_path,
                    specimen_index,
                    spine_id,
                    distribution_included=self.distribution_include.isChecked(),
                    invalid_spine=self.distribution_invalid.isChecked(),
                    note=self.distribution_note.text(),
                )
            else:
                updated = set_spine_quality_review(
                    self.manifest,
                    self.project_path,
                    specimen_index,
                    spine_id,
                    invalid_spine=self.distribution_invalid.isChecked(),
                    note=self.distribution_note.text(),
                )
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save spine review", str(exc))
            return
        if review_mode == "cluster_positive":
            candidates = updated.get("distribution_rows", [])
            review_key = "distribution_reviewed"
        else:
            candidates = [
                row
                for row in updated.get("spine_rows", [])
                if not bool(row.get("has_protein_cluster", False))
            ]
            review_key = "validity_reviewed"
        next_id = next(
            (
                int(row["spine_id"])
                for row in candidates
                if not bool(row.get(review_key, False))
            ),
            None,
        )
        self._preferred_distribution_spine_id = next_id
        self._preferred_spine_review_mode = review_mode
        self._refresh_distribution_groups()
        # Refresh labels, then advance to the first remaining unreviewed spine.
        self._measurement_specimen_changed()
        if next_id is None:
            self.distribution_preview_status.setText(
                "All cluster-positive spines in this specimen have been reviewed."
                if review_mode == "cluster_positive"
                else "Optional cluster-less spine review is complete for this specimen."
            )

    def _export_measurements(self) -> None:
        if self.manifest is None:
            return
        complete = sum(
            specimen["checkpoints"].get("measurements", {}).get("state")
            == "complete"
            for specimen in self.manifest["specimens"]
        )
        omitted = len(self.manifest["specimens"]) - complete
        self.measurement_status.setText(
            f"Partial export: {complete} completed pair(s) included; "
            f"{omitted} incomplete pair(s) omitted."
            if omitted
            else f"Export includes all {complete} completed pair(s)."
        )
        output = Path(str(self.manifest["output_directory"]))
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Synpo measurement workbook",
            str(output / "Synpo_measurements.xlsx"),
            "Excel workbook (*.xlsx)",
        )
        if not selected:
            return
        worker = ExportWorker(
            self.manifest,
            Path(selected),
            self.export_validation_pdf.isChecked(),
            self.export_excluded_pdf.isChecked(),
            self.export_invalid_pdf.isChecked(),
            self.export_pdf_margin.value(),
        )
        worker.completed.connect(self._export_completed)
        self._start_worker(worker, "export")

    @Slot(object)
    def _export_completed(self, result: dict[str, object]) -> None:
        self.measurement_status.setText(
            f"Verified export complete: {result['workbook']} and CSV files in {result['csv_directory']}."
        )
        QMessageBox.information(
            self,
            "Export verified",
            f"Workbook and CSV files were written and reopened successfully.\n\n{result['workbook']}",
        )

    def _populate_measurement_table(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.measurement_specimen.currentData() is None:
            return
        specimen_index = int(self.measurement_specimen.currentData())
        try:
            result = load_measurement_result(self.manifest, specimen_index)
        except (OSError, ValueError, KeyError):
            return
        key = str(self.measurement_table_level.currentData())
        rows = (
            cluster_end_comparison_rows(result)
            if key == "cluster_end_comparison"
            else list(result.get(key, []))
        )
        columns = list(rows[0].keys()) if rows else []
        self.measurement_table.setColumnCount(len(columns))
        self.measurement_table.setHorizontalHeaderLabels(
            [column.replace("_", " ") for column in columns]
        )
        self.measurement_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                value = row.get(column)
                if value is None:
                    text_value = "pending" if "distribution" in column else "—"
                elif isinstance(value, bool):
                    text_value = "yes" if value else "no"
                elif isinstance(value, float):
                    text_value = f"{value:.6g}"
                elif isinstance(value, list):
                    text_value = ", ".join(str(item + 1) for item in value) or "none"
                else:
                    text_value = str(value)
                self.measurement_table.setItem(
                    row_index, column_index, QTableWidgetItem(text_value)
                )
        header = self.measurement_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def _load_trim_preview(self) -> None:
        if (
            self.manifest is None
            or self.measurement_specimen.currentData() is None
            or self.measurement_cluster.currentData() is None
        ):
            QMessageBox.information(
                self, "No cluster selected", "Select a measured specimen and cluster first."
            )
            return
        worker = ClusterTrimPreviewWorker(
            self.manifest,
            int(self.measurement_specimen.currentData()),
            int(self.measurement_cluster.currentData()),
        )
        worker.completed.connect(self._trim_preview_completed)
        self._start_worker(worker, "trim_preview")

    @Slot(object)
    def _trim_preview_completed(self, preview: ClusterTrimPreview) -> None:
        self._last_trim_preview = preview
        low, high = np.percentile(preview.raw_projection, (0.5, 99.8))
        black = int(max(0, min(65534, round(float(low)))))
        white = int(max(black + 1, min(65535, round(float(high)))))
        self.trim_preview_view.show_detection(
            preview.raw_projection,
            black,
            white,
            dendrites=preview.counted_projection,
            spines=None,
            clusters=preview.discarded_projection,
        )
        retained = ", ".join(str(value + 1) for value in preview.retained_z_slices)
        discarded = ", ".join(str(value + 1) for value in preview.discarded_z_slices)
        self.trim_preview_label.setText(
            f"Cluster {preview.cluster_id}: green = counted voxels (Z {retained or 'none'}); "
            f"magenta = discarded terminal voxels (Z {discarded or 'none'})."
        )

    def _selected_review_specimen(self) -> int:
        value = self.review_specimen.currentData()
        if value is None:
            raise ValueError("Select a detected specimen to review.")
        return int(value)

    def _review_specimen_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.review_specimen.currentData() is None:
            return
        index = self._selected_review_specimen()
        specimen_key = (id(self.manifest), index)
        if specimen_key != self._review_view_specimen_key:
            self.review_view.reset_view()
            self._review_view_specimen_key = specimen_key
        channel = str(self.review_background_channel.currentData() or "ChanB")
        shape = self.manifest["specimens"][index]["channels"][channel]["metadata"][
            "shape"
        ]
        z_count = 1 if len(shape) == 2 else int(shape[0])
        self.review_z_slider.blockSignals(True)
        self.review_z_slider.setRange(0, max(0, z_count - 1))
        self.review_z_slider.setValue(max(0, (z_count - 1) // 2))
        self.review_z_slider.blockSignals(False)
        self._update_review_z_label()
        specimen = self.manifest["specimens"][index]
        self.review_comment.setText(str(specimen["review"].get("comment", "")))
        self.review_view.clear_hint()
        self._load_review_view(auto_contrast=True)
        checkpoint = specimen["checkpoints"]["review"]
        detection_summary = specimen["checkpoints"]["detection"].get("summary", {})
        self.review_status.setText(
            f"State: {specimen['review'].get('state', 'needs_attention')} | "
            f"{checkpoint.get('edit_count', 0)} active edit(s) | "
            f"automatic candidates: {detection_summary.get('dendrite_count', 0)} dendrites, "
            f"{detection_summary.get('spine_count', 0)} spines, "
            f"{detection_summary.get('cluster_count', 0)} clusters."
        )

    def _move_review_specimen(self, offset: int) -> None:
        count = self.review_specimen.count()
        if not count:
            return
        self.review_specimen.setCurrentIndex(
            (self.review_specimen.currentIndex() + offset) % count
        )

    def _review_z_changed(self, value: int) -> None:
        self._update_review_z_label()
        self.review_view.clear_hint()
        if self.review_view_mode.currentData() == "slice":
            self._load_review_view(auto_contrast=False)

    def _update_review_z_label(self) -> None:
        prefix = (
            "Reference Z"
            if self.review_view_mode.currentData() == "xy_max"
            else "Z"
        )
        self.review_z_label.setText(
            f"{prefix}: {self.review_z_slider.value() + 1}/"
            f"{self.review_z_slider.maximum() + 1}"
        )

    def _review_view_mode_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        self.review_view.clear_hint()
        self._update_review_z_label()
        self._load_review_view(auto_contrast=True)

    def _load_review_view(
        self, *_args, auto_contrast: bool = False
    ) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.review_specimen.currentData() is None:
            return
        if self.review_view_mode.currentData() == "xy_max":
            self._load_review_projection(auto_contrast=auto_contrast)
            return
        try:
            self._last_review = load_review_slice(
                self.manifest,
                self._selected_review_specimen(),
                self.review_z_slider.value(),
                str(self.review_background_channel.currentData()),
            )
            if auto_contrast:
                self._auto_review_contrast()
            else:
                self._render_review_view()
        except (OSError, ValueError, KeyError, IndexError) as exc:
            self.review_view.setText(f"Cannot load review slice: {exc}")

    def _load_review_projection(self, *, auto_contrast: bool) -> None:
        if self.manifest is None or self.review_specimen.currentData() is None:
            return
        specimen_index = self._selected_review_specimen()
        background_channel = str(self.review_background_channel.currentData())
        signature = context_signature(
            self.manifest, specimen_index, corrected=True
        )
        cache_key = (
            specimen_index,
            background_channel,
            True,
            signature,
            False,
            None,
        )
        if cache_key in self._context_cache:
            self._display_review_projection(
                self._context_cache[cache_key], auto_contrast=auto_contrast
            )
            return
        request = (
            cache_key,
            specimen_index,
            background_channel,
            True,
            "review_main",
            self.review_z_slider.value(),
            False,
            None,
            auto_contrast,
        )
        if self._context_thread is not None:
            self.review_status.setText(
                "Waiting for the current projection/3D generation to finish."
            )
            return
        self._start_context_generation(request)

    def _display_review_projection(
        self, volume: ContextVolume, *, auto_contrast: bool
    ) -> None:
        self._last_review_context = volume
        projection = volume.xy
        self._last_review = ReviewSlice(
            raw=projection.raw,
            dendrites=projection.dendrites,
            spines=projection.spines,
            clusters=projection.clusters,
            z_index=self.review_z_slider.value(),
            z_count=volume.z_count,
            corrected=volume.corrected,
        )
        self.review_status.setText(
            "Drawable XY maximum projection ready. Projection hints infer their "
            "3D Z location from objects or image signal."
        )
        if auto_contrast:
            self._auto_review_contrast()
        else:
            self._render_review_view()

    def _auto_review_contrast(self) -> None:
        if self._last_review is None:
            return
        low, high = np.percentile(self._last_review.raw, (0.5, 99.8))
        low_value = int(max(0, min(65534, round(float(low)))))
        high_value = int(max(low_value + 1, min(65535, round(float(high)))))
        self.review_black.blockSignals(True)
        self.review_white.blockSignals(True)
        self.review_black.setValue(low_value)
        self.review_white.setValue(high_value)
        self.review_black.blockSignals(False)
        self.review_white.blockSignals(False)
        self._render_review_view()

    def _render_review_view(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self._last_review is None:
            return
        self.review_view.show_detection(
            self._last_review.raw,
            self.review_black.value(),
            max(self.review_black.value() + 1, self.review_white.value()),
            dendrites=(
                self._last_review.dendrites
                if self.review_show_dendrites.isChecked()
                else None
            ),
            spines=(
                self._last_review.spines if self.review_show_spines.isChecked() else None
            ),
            clusters=(
                self._last_review.clusters
                if self.review_show_clusters.isChecked()
                else None
            ),
        )

    def _review_tool_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        operation = str(self.review_operation.currentData())
        self.review_view.set_hint_color(
            REVIEW_BRUSH_COLORS.get(operation, QColor("#ffe119"))
        )
        if operation == "filopodium":
            self.review_object_type.setCurrentIndex(
                self.review_object_type.findData("spine")
            )
        instructions = {
            "add": (
                "Draw one separate stroke inside each missed object. Every stroke seeds one "
                "independent 3D object; the saved preprocessed image determines its boundary. "
                "Measurements still use the original 16-bit voxels."
            ),
            "exclude": "Touch an unwanted object to exclude the complete 3D object.",
            "filopodium": "Touch a spine candidate to exclude and record it as a filopodium.",
            "split": "Draw across the contact or neck that should separate one object into two.",
            "merge": "Draw through at least two objects that should be one object.",
            "expand": "Draw toward missing signal from an existing object; the boundary is regrown locally.",
            "trim": "Draw across the excess part; the connected object is retained locally.",
            "accept": "Touch an object to mark it accepted without changing its mask.",
            "needs_attention": "Touch an object to retain it but flag it for later attention.",
        }
        self.review_instruction.setText(instructions.get(operation, "Draw a hint."))

    def _review_brush_changed(self, value: int) -> None:
        self.review_view.set_brush_radius(value)

    def _review_hint_changed(self, count: int) -> None:
        stroke_count = len(self.review_view.hint_strokes())
        self.review_hint_status.setText(
            f"{stroke_count} separate hint(s), {count} sampled point(s)."
            if count
            else "No hint drawn."
        )

    def _clear_review_hint(self) -> None:
        self.review_view.clear_hint()

    def _undo_review_stroke(self) -> None:
        self.review_view.undo_stroke()

    def _apply_review_action(self) -> None:
        if self.manifest is None or self.project_path is None:
            QMessageBox.information(self, "No project", "Save or open a project first.")
            return
        points = self.review_view.hint_points()
        if not points:
            QMessageBox.information(
                self,
                "Draw a hint",
                "Draw or click on the current Z slice or XY maximum projection first.",
            )
            return
        action = ReviewAction(
            object_type=str(self.review_object_type.currentData()),
            operation=str(self.review_operation.currentData()),
            z_index=self.review_z_slider.value(),
            points=points,
            brush_radius_pixels=self.review_brush_radius.value(),
            projection_hint=self.review_view_mode.currentData() == "xy_max",
            strokes=self.review_view.hint_strokes(),
        )
        self.manifest["review_settings"]["memory_mode"] = str(
            self.review_memory_mode.currentData()
            or AUTOMATIC_REVIEW_MEMORY_MODE
        )
        save_project(self.project_path, self.manifest)
        self._start_review_worker(action)

    def _undo_review_action(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        self._start_review_worker(None)

    def _start_review_worker(self, action: ReviewAction | None) -> None:
        if (
            self._review_thread is not None
            or self.manifest is None
            or self.project_path is None
        ):
            return
        try:
            specimen_index = self._selected_review_specimen()
        except ValueError as exc:
            QMessageBox.information(self, "No specimen", str(exc))
            return
        worker = ReviewWorker(
            self.manifest, self.project_path, specimen_index, action
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_review_progress)
        worker.completed.connect(self._review_action_completed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(self._review_action_failed)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._review_worker_finished)
        thread.finished.connect(thread.deleteLater)
        self._review_thread = thread
        self._review_worker = worker
        self._set_review_busy(True)
        thread.start()

    @Slot(str, int, int, str)
    def _update_review_progress(
        self, phase: str, current: int, total: int, detail: str
    ) -> None:
        self.review_progress.setVisible(True)
        self.review_progress.setRange(0, max(1, total))
        self.review_progress.setValue(current)
        self.review_status.setText(f"{phase}: {detail}")

    @Slot(object)
    def _review_action_completed(self, result) -> None:  # type: ignore[no-untyped-def]
        if result.checkpoint_written:
            self._invalidate_context_views(result.specimen_index, corrected_only=True)
            self.review_view.clear_hint()
            self._load_review_view(auto_contrast=False)
        if result.operation == "add" and result.hint_results:
            created = [
                item for item in result.hint_results if item.get("status") == "created"
            ]
            skipped = [
                item for item in result.hint_results if item.get("status") != "created"
            ]
            details = "; ".join(
                f"hint {item['hint_index']}: {item['message']}" for item in skipped
            )
            checkpoint = (
                "One atomic checkpoint was written."
                if result.checkpoint_written
                else "No mask change or checkpoint was written; adjust the hint(s) and retry."
            )
            self.review_status.setText(
                f"Add hints: {len(created)}/{len(result.hint_results)} object(s) created. "
                + (f"{details}. " if details else "")
                + (
                    "Slow low-memory correction was used. "
                    if result.processing_mode == "low_memory"
                    else ""
                )
                + checkpoint
            )
            return
        self.review_status.setText(
            f"Saved {result.operation}: {result.dendrite_count} dendrites, "
            f"{result.spine_count} spines; {result.edit_count} active edit(s). "
            + (
                "Slow low-memory correction was used. "
                if result.processing_mode == "low_memory"
                else ""
            )
            + "An automatic specimen checkpoint was written."
        )

    @Slot(str)
    def _review_action_failed(self, message: str) -> None:
        self.review_status.setText("Correction was not applied; the previous mask is intact.")
        QMessageBox.warning(self, "Cannot apply correction", message)

    @Slot()
    def _review_worker_finished(self) -> None:
        if self._review_worker is not None:
            self._review_worker.deleteLater()
        self._review_worker = None
        self._review_thread = None
        self.review_progress.setVisible(False)
        self._set_review_busy(False)
        if self._review_refresh_pending:
            self._prepare_review_tab()
        else:
            self._refresh_review_specimen_label()
        self._prepare_measurements_tab()

    def _set_review_busy(self, busy: bool) -> None:
        has_specimen = self.review_specimen.count() > 0
        for widget in (
            self.review_specimen,
            self.previous_review_button,
            self.next_review_button,
            self.review_view_mode,
            self.review_background_channel,
            self.review_z_slider,
            self.review_object_type,
            self.review_operation,
            self.review_brush_radius,
            self.review_memory_mode,
            self.review_projections_button,
            self.review_3d_button,
            self.clear_review_hint_button,
            self.undo_review_stroke_button,
            self.apply_review_button,
            self.undo_review_action_button,
            self.save_review_progress_button,
            self.complete_review_button,
        ):
            widget.setEnabled(has_specimen and not busy)
        self.review_view.setEnabled(has_specimen and not busy)

    def _refresh_review_specimen_label(self) -> None:
        if self.manifest is None or self.review_specimen.currentData() is None:
            return
        index = self._selected_review_specimen()
        specimen = self.manifest["specimens"][index]
        self.review_specimen.setItemText(
            self.review_specimen.currentIndex(),
            f"{specimen['experimental_group']} — {specimen['specimen_id']} "
            f"[{specimen['review'].get('state', 'needs_attention')}]",
        )

    def _save_review_state(self, complete: bool) -> None:
        if (
            self.manifest is None
            or self.project_path is None
            or self._review_thread is not None
        ):
            return
        try:
            set_specimen_review_state(
                self.manifest,
                self.project_path,
                self._selected_review_specimen(),
                complete=complete,
                comment=self.review_comment.text(),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot save review checkpoint", str(exc))
            return
        self._refresh_review_specimen_label()
        state = "complete" if complete else "in progress"
        self.review_status.setText(
            f"Specimen review marked {state}; comment and checkpoint saved."
        )
        self._prepare_measurements_tab()
        if complete:
            self._move_review_specimen(1)

    def _open_spine_map(self, focus_only: bool) -> None:
        if (
            self.manifest is None
            or self.measurement_specimen.currentData() is None
            or self.distribution_spine.currentData() is None
        ):
            self.distribution_preview_status.setText(
                "Select a measured specimen and spine before opening its full-field map."
            )
            return
        specimen_index = int(self.measurement_specimen.currentData())
        spine_id = int(self.distribution_spine.currentData())
        role_channels = {
            role: channel for channel, role in self.manifest["channel_roles"].items()
        }
        background_channel = str(role_channels["dendrite_spines"])
        signature = context_signature(
            self.manifest, specimen_index, corrected=True
        )
        cache_key = (
            specimen_index,
            background_channel,
            True,
            signature,
            False,
            None,
        )
        request = (
            cache_key,
            specimen_index,
            background_channel,
            True,
            "spine_map",
            0,
            False,
            None,
            False,
        )
        self._spine_map_focus_only = bool(focus_only)
        self._spine_map_current_id = spine_id
        if cache_key in self._context_cache:
            self._show_spine_map(request, self._context_cache[cache_key])
        elif self._context_thread is None:
            self._start_context_generation(request)
        else:
            self.distribution_preview_status.setText(
                "A projection is already being generated; try the spine map again when it finishes."
            )

    def _show_spine_map(
        self, request: tuple[object, ...], volume: ContextVolume
    ) -> None:
        if self.manifest is None:
            return
        _, specimen_index, background_channel, _, _, _, _, _, _ = request
        try:
            result = load_measurement_result(self.manifest, int(specimen_index))
        except (OSError, ValueError, KeyError) as exc:
            self.distribution_preview_status.setText(
                f"Cannot open numbered spine map: {exc}"
            )
            return
        specimen = self.manifest["specimens"][int(specimen_index)]
        dialog = SpineMapDialog(
            (
                f"Spine {self._spine_map_current_id} in full specimen"
                if self._spine_map_focus_only
                else "Numbered spine map"
            )
            + f" — {specimen['specimen_id']}",
            volume,
            result,
            lambda z, index=int(specimen_index), channel=str(background_channel): load_review_slice(
                self.manifest, index, z, channel
            ),
            self._spine_map_current_id,
            focus_only=self._spine_map_focus_only,
            parent=self,
        )
        dialog.spine_selected.connect(self._spine_map_selected)
        dialog.destroyed.connect(
            lambda *_args, target=dialog: self._forget_spine_map_dialog(target)
        )
        self._spine_map_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @Slot(int)
    def _spine_map_selected(self, spine_id: int) -> None:
        if self.manifest is None or self.measurement_specimen.currentData() is None:
            return
        try:
            result = load_measurement_result(
                self.manifest, int(self.measurement_specimen.currentData())
            )
            row = next(
                item
                for item in result.get("spine_rows", [])
                if int(item["spine_id"]) == spine_id
            )
        except (OSError, ValueError, KeyError, StopIteration):
            return
        self._preferred_spine_review_mode = (
            "cluster_positive"
            if bool(row.get("has_protein_cluster", False))
            else "cluster_less"
        )
        self._preferred_distribution_spine_id = spine_id
        self._measurement_specimen_changed()

    def _forget_spine_map_dialog(self, dialog: SpineMapDialog) -> None:
        if dialog in self._spine_map_dialogs:
            self._spine_map_dialogs.remove(dialog)

    def _open_context_view(self, corrected: bool, view: str) -> None:
        if self.manifest is None:
            return
        if corrected and self._review_thread is not None:
            QMessageBox.information(
                self,
                "Correction in progress",
                "Wait for the current correction to finish before generating 3D context.",
            )
            return
        try:
            specimen_index = (
                self._selected_review_specimen()
                if corrected
                else int(self.detection_specimen.currentData())
            )
        except (TypeError, ValueError):
            QMessageBox.information(
                self, "No specimen", "Select a completed detection first."
            )
            return
        specimen = self.manifest["specimens"][specimen_index]
        if specimen["checkpoints"]["detection"].get("state") != "complete":
            QMessageBox.information(
                self, "Detection incomplete", "This specimen is not ready for 3D viewing."
            )
            return
        background_channel = str(
            self.review_background_channel.currentData()
            if corrected
            else self.detection_background_channel.currentData()
        )
        signature = context_signature(
            self.manifest, specimen_index, corrected=corrected
        )
        cache_key = (
            specimen_index,
            background_channel,
            corrected,
            signature,
            False,
            None,
        )
        request_view = "select_area" if view == "3d" else "projections"
        request = (
            cache_key,
            specimen_index,
            background_channel,
            corrected,
            request_view,
            self.review_z_slider.value()
            if corrected
            else self.detection_z_slider.value(),
            False,
            None,
            False,
        )
        if cache_key in self._context_cache:
            if request_view == "select_area":
                self._show_area_selection(request, self._context_cache[cache_key])
            else:
                self._show_context_dialog(request, self._context_cache[cache_key])
            return
        if self._context_thread is not None:
            self.context_status_label.setText(
                "A projection or 3D view is already being generated."
            )
            return
        self._start_context_generation(request)

    def _start_context_generation(self, request: tuple[object, ...]) -> None:
        if self.manifest is None or self._context_thread is not None:
            return
        (
            _,
            specimen_index,
            background_channel,
            corrected,
            _,
            _,
            include_3d,
            roi_xy,
            _,
        ) = request
        worker = ContextWorker(
            self.manifest,
            int(specimen_index),
            str(background_channel),
            bool(corrected),
            bool(include_3d),
            roi_xy,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._context_progress_updated)
        worker.completed.connect(self._context_completed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(self._context_failed)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(self._context_cancelled)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._context_finished)
        thread.finished.connect(thread.deleteLater)
        self._context_thread = thread
        self._context_worker = worker
        self._context_request = request
        self.context_status_label.setText("Preparing projections and 3D objects…")
        self.context_status_progress.setRange(0, 1)
        self.context_status_progress.setValue(0)
        self.cancel_context_button.setEnabled(True)
        self.context_status_widget.setVisible(True)
        thread.start()

    @Slot(str, int, int, str)
    def _context_progress_updated(
        self, phase: str, current: int, total: int, detail: str
    ) -> None:
        self.context_status_progress.setRange(0, max(1, total))
        self.context_status_progress.setValue(current)
        self.context_status_label.setText(f"{phase}: {detail}")

    def _cancel_context_generation(self) -> None:
        if self._context_worker is None:
            return
        self._context_worker.cancel()
        self.cancel_context_button.setEnabled(False)
        self.context_status_label.setText(
            "Cancellation requested; finishing the current safe step…"
        )

    @Slot(object)
    def _context_completed(self, volume: ContextVolume) -> None:
        if self._context_request is None:
            return
        cache_key = self._context_request[0]
        self._context_cache[cache_key] = volume
        while len(self._context_cache) > 2:
            oldest = next(iter(self._context_cache))
            del self._context_cache[oldest]
        view = str(self._context_request[4])
        if view == "select_area":
            self._show_area_selection(self._context_request, volume)
        elif view == "spine_map":
            self._show_spine_map(self._context_request, volume)
        elif view == "review_main":
            self._display_review_projection(
                volume, auto_contrast=bool(self._context_request[8])
            )
        else:
            self._show_context_dialog(self._context_request, volume)

    def _show_context_dialog(
        self, request: tuple[object, ...], volume: ContextVolume
    ) -> None:
        if self.manifest is None:
            return
        _, specimen_index, _, corrected, view, initial_z, _, _, _ = request
        specimen = self.manifest["specimens"][int(specimen_index)]
        source_label = (
            "corrected review masks"
            if bool(corrected) and volume.corrected
            else "automatic detection masks"
        )
        dialog = ContextViewerDialog(
            f"{specimen['experimental_group']} — {specimen['specimen_id']} | {source_label}",
            volume,
            int(initial_z),
            self,
        )
        dialog.setProperty("specimen_index", int(specimen_index))
        dialog.setProperty("corrected", bool(corrected))
        dialog.z_selected.connect(
            lambda z, index=int(specimen_index), review=bool(corrected): self._context_z_selected(
                index, review, z
            )
        )
        dialog.destroyed.connect(
            lambda *_args, target=dialog: self._forget_context_dialog(target)
        )
        self._context_dialogs.append(dialog)
        dialog.select_view(str(view))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _show_area_selection(
        self, request: tuple[object, ...], volume: ContextVolume
    ) -> None:
        if self.manifest is None:
            return
        _, specimen_index, _, corrected, _, _, _, _, _ = request
        specimen = self.manifest["specimens"][int(specimen_index)]
        if self._area_dialog is not None:
            self._area_dialog.close()
        dialog = AreaSelectionDialog(
            f"Select area for 3D — {specimen['experimental_group']} — "
            f"{specimen['specimen_id']}",
            volume,
            self,
        )
        dialog.area_selected.connect(
            lambda roi, source_request=request: self._generate_cropped_3d(
                source_request, roi
            )
        )
        dialog.destroyed.connect(lambda *_args: self._clear_area_dialog(dialog))
        self._area_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _clear_area_dialog(self, dialog: AreaSelectionDialog) -> None:
        if self._area_dialog is dialog:
            self._area_dialog = None

    def _generate_cropped_3d(
        self, source_request: tuple[object, ...], roi
    ) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None:
            return
        if self._context_thread is not None:
            QTimer.singleShot(
                100, lambda: self._generate_cropped_3d(source_request, roi)
            )
            return
        (
            _,
            specimen_index,
            background_channel,
            corrected,
            _,
            initial_z,
            _,
            _,
            _,
        ) = source_request
        rectangle = tuple(int(value) for value in roi)
        signature = context_signature(
            self.manifest, int(specimen_index), corrected=bool(corrected)
        )
        cache_key = (
            int(specimen_index),
            str(background_channel),
            bool(corrected),
            signature,
            True,
            rectangle,
        )
        request = (
            cache_key,
            int(specimen_index),
            str(background_channel),
            bool(corrected),
            "3d",
            int(initial_z),
            True,
            rectangle,
            False,
        )
        if cache_key in self._context_cache:
            self._show_context_dialog(request, self._context_cache[cache_key])
        else:
            self._start_context_generation(request)

    def _forget_context_dialog(self, dialog: ContextViewerDialog) -> None:
        if dialog in self._context_dialogs:
            self._context_dialogs.remove(dialog)

    def _context_z_selected(self, specimen_index: int, corrected: bool, z: int) -> None:
        combo = self.review_specimen if corrected else self.detection_specimen
        if combo.currentData() != specimen_index:
            return
        slider = self.review_z_slider if corrected else self.detection_z_slider
        slider.setValue(max(slider.minimum(), min(slider.maximum(), z)))

    @Slot(str)
    def _context_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Cannot generate 3D context", message)

    @Slot(str)
    def _context_cancelled(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    @Slot()
    def _context_finished(self) -> None:
        if self._context_worker is not None:
            self._context_worker.deleteLater()
        self._context_worker = None
        self._context_thread = None
        self._context_request = None
        self.context_status_widget.setVisible(False)

    def _invalidate_context_views(
        self, specimen_index: int, *, corrected_only: bool
    ) -> None:
        for key in list(self._context_cache):
            if int(key[0]) == specimen_index and (not corrected_only or bool(key[2])):
                del self._context_cache[key]
        for dialog in list(self._context_dialogs):
            if int(dialog.property("specimen_index")) == specimen_index and (
                not corrected_only or bool(dialog.property("corrected"))
            ):
                dialog.close()

    def _scan_source(self) -> None:
        if self._review_thread is not None:
            QMessageBox.information(
                self, "Review in progress", "Wait for the current correction to finish."
            )
            return
        directory = Path(self.source_edit.text().strip())
        if not directory.is_dir():
            QMessageBox.warning(self, "Invalid folder", "Select an existing TIFF folder.")
            return
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(directory / "Synpo Results"))
        try:
            markers = self._current_channel_markers()
            group = self.fallback_group_edit.text().strip()
            if not group:
                raise ValueError("Enter a fallback experimental-group name.")
        except ValueError as exc:
            QMessageBox.warning(self, "Filename pairing", str(exc))
            return
        self._begin_import_scan(
            ScanWorker(directory, markers, group)
        )

    def _begin_import_scan(self, worker: ScanWorker) -> None:
        for dialog in list(self._context_dialogs):
            dialog.close()
        for dialog in list(self._spine_map_dialogs):
            dialog.close()
        self._context_cache.clear()
        self.report = None
        self.manifest = None
        self.project_path = None
        self._last_review = None
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
        worker.completed.connect(self._scan_completed)
        self._start_worker(worker, "scan")

    @staticmethod
    def _format_progress_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        if seconds < 60:
            return f"{seconds} s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes} min {seconds:02d} s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours} h {minutes:02d} min"

    def _batch_progress_widgets(self, kind: str | None):
        return {
            "preprocess": (
                self.preprocessing_progress_bar,
                self.preprocessing_progress_label,
            ),
            "detection": (
                self.detection_progress_bar,
                self.detection_progress_label,
            ),
            "measurements": (
                self.measurement_progress_bar,
                self.measurement_progress_label,
            ),
        }.get(str(kind))

    def _start_worker(self, worker: QObject, kind: str) -> None:
        if self._job_thread is not None or self._review_thread is not None:
            QMessageBox.information(self, "Work in progress", "Wait for the current operation to finish.")
            return
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_progress)
        worker.failed.connect(self._job_failed)
        worker.failed.connect(thread.quit)
        worker.completed.connect(thread.quit)
        if isinstance(worker, (BatchPreprocessWorker, DetectionWorker, MeasurementWorker)):
            worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._worker_finished)
        thread.finished.connect(thread.deleteLater)
        self._job_thread = thread
        self._job_worker = worker
        self._job_kind = kind
        now = time.monotonic()
        self._progress_started_at = now
        self._progress_last_at = now
        self._progress_last_current = 0
        self._progress_last_total = 0
        self._progress_rate_ema = None
        batch_widgets = self._batch_progress_widgets(kind)
        if batch_widgets is not None:
            batch_bar, batch_label = batch_widgets
            batch_bar.setRange(0, 100)
            batch_bar.setValue(0)
            batch_bar.setFormat("Starting…")
            batch_label.setText("Batch progress: starting; estimating remaining time…")
        self._set_job_running(True)
        thread.start()

    @Slot(str, int, int, str)
    def _update_progress(self, phase: str, current: int, total: int, detail: str) -> None:
        total = max(1, int(total))
        current = min(total, max(0, int(current)))
        now = time.monotonic()
        if self._progress_started_at <= 0:
            self._progress_started_at = now
            self._progress_last_at = now
        if total != self._progress_last_total or current < self._progress_last_current:
            self._progress_rate_ema = None
            self._progress_last_at = self._progress_started_at
            self._progress_last_current = 0
            self._progress_last_total = total
        delta_work = current - self._progress_last_current
        delta_time = now - self._progress_last_at
        if delta_work > 0 and delta_time > 0.01:
            instantaneous = delta_work / delta_time
            self._progress_rate_ema = (
                instantaneous
                if self._progress_rate_ema is None
                else self._progress_rate_ema * 0.75 + instantaneous * 0.25
            )
        self._progress_last_current = current
        self._progress_last_at = now
        self._progress_last_total = total
        elapsed = max(0.0, now - self._progress_started_at)
        if current >= total:
            remaining_text = "complete"
        elif self._progress_rate_ema and self._progress_rate_ema > 0:
            remaining = (total - current) / self._progress_rate_ema
            remaining_text = (
                f"about {self._format_progress_duration(remaining)} remaining"
            )
        else:
            remaining_text = "estimating remaining time…"
        percent = round(current * 100 / total)
        progress_text = (
            f"{phase}: {detail} | {current}/{total} ({percent}%) | "
            f"elapsed {self._format_progress_duration(elapsed)} | {remaining_text}"
        )
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat("%p% — %v/%m")
        self.progress_label.setText(progress_text)
        batch_widgets = self._batch_progress_widgets(self._job_kind)
        if batch_widgets is not None:
            batch_bar, batch_label = batch_widgets
            batch_bar.setRange(0, total)
            batch_bar.setValue(current)
            batch_bar.setFormat("%p% — %v/%m")
            batch_label.setText(progress_text)

    @Slot(str)
    def _job_failed(self, message: str) -> None:
        if self._job_kind == "preprocess" and self.manifest is not None:
            self.preprocessing_status.setText(
                "Preprocessing stopped with an error. Completed cache slices remain resumable."
            )
            if self.project_path is not None:
                save_project(self.project_path, self.manifest)
        elif self._job_kind == "detection" and self.manifest is not None:
            self.detection_status.setText(
                "Detection stopped with an error. Completed specimen checkpoints remain usable."
            )
            if self.project_path is not None:
                save_project(self.project_path, self.manifest)
        elif self._job_kind in {"measurements", "trim_preview", "distribution_preview", "centerline_hint", "export"}:
            self.measurement_status.setText(
                "Measurement operation stopped with an error; completed checkpoints remain usable."
            )
        QMessageBox.critical(self, "Operation failed", message)

    @Slot()
    def _worker_finished(self) -> None:
        finished_kind = self._job_kind
        finished_widgets = self._batch_progress_widgets(finished_kind)
        if finished_widgets is not None:
            finished_bar, finished_label = finished_widgets
            elapsed = self._format_progress_duration(
                max(0.0, time.monotonic() - self._progress_started_at)
            )
            if finished_bar.value() >= finished_bar.maximum():
                finished_label.setText(f"Batch complete in {elapsed}.")
            else:
                finished_label.setText(
                    f"Batch stopped at {finished_bar.value()}/{finished_bar.maximum()} "
                    f"after {elapsed}; completed checkpoints remain available."
                )
        if self._job_worker is not None:
            self._job_worker.deleteLater()
        self._job_worker = None
        self._job_thread = None
        self._job_kind = None
        self._set_job_running(False)
        if finished_kind == "preview" and self._preview_requested_while_busy:
            self._preview_requested_while_busy = False
            self._preview_timer.start()
        elif finished_kind == "measurements" and self.distribution_spine.count():
            QTimer.singleShot(0, self._distribution_spine_changed)
        elif finished_kind == "centerline_hint" and self._centerline_hint_pending_reload:
            self._centerline_hint_pending_reload = False
            QTimer.singleShot(0, self._distribution_spine_changed)

    @Slot(object)
    def _scan_completed(self, report: ScanReport) -> None:
        self.report = report
        self.source_edit.setText(str(report.source_directory))
        self._populate_scan_table(report)
        warnings = self._report_issue_count(report, "warning")
        mode_text = {
            "strict": "original metadata filenames",
            "flexible": "configured channel markers with a single fallback group",
            "manual": "manually selected channel files",
        }.get(report.import_mode, report.import_mode)
        self.summary_label.setText(
            f"Found {len(report.pairs)} specimen pair(s). "
            f"Errors: {report.error_count}; warnings: {warnings}. "
            f"Pairing mode: {mode_text}. Experimental group and specimen labels "
            "may be edited before saving."
        )
        self.progress_label.setText("Scan complete")

    @staticmethod
    def _report_issue_count(report: ScanReport, severity: str) -> int:
        issues = list(report.issues)
        for pair in report.pairs:
            issues.extend(pair.issues)
            for channel_file in pair.channels.values():
                issues.extend(channel_file.issues)
        return sum(issue.severity == severity for issue in issues)

    def _populate_scan_table(self, report: ScanReport) -> None:
        self.table.setRowCount(len(report.pairs))
        for row, pair in enumerate(report.pairs):
            issues = list(pair.issues)
            for channel_file in pair.channels.values():
                issues.extend(channel_file.issues)
            issue_text = "; ".join(issue.message for issue in issues)
            values = [
                (f"Ready ({pair.import_mode})" if pair.valid else "Error"),
                pair.experimental_group,
                pair.specimen_id,
                pair.channels.get("ChanA").filename if "ChanA" in pair.channels else "—",
                pair.channels.get("ChanB").filename if "ChanB" in pair.channels else "—",
                pair.shape_text,
                pair.dtype_text,
                issue_text,
            ]
            for column, value in enumerate(values):
                editable = column in {1, 2}
                self._set_table_item(row, column, value, editable=editable)
            if "ChanA" in pair.channels:
                self.table.item(row, 3).setToolTip(
                    str(pair.channels["ChanA"].path)
                )
            if "ChanB" in pair.channels:
                self.table.item(row, 4).setToolTip(
                    str(pair.channels["ChanB"].path)
                )
            self.table.item(row, 0).setBackground(
                QColor("#dff3e4") if pair.valid else QColor("#f8d7da")
            )

    def _set_table_item(self, row: int, column: int, value: str, *, editable: bool = False) -> None:
        item = QTableWidgetItem(value)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        self.table.setItem(row, column, item)

    def _sync_scan_edits(self) -> None:
        if self.report is None:
            return
        for row, pair in enumerate(self.report.pairs):
            group = self.table.item(row, 1).text().strip()
            specimen = self.table.item(row, 2).text().strip()
            if not group or not specimen:
                raise ValueError("Experimental group and specimen labels cannot be empty.")
            pair.experimental_group = group
            pair.specimen_id = specimen

    def _sync_manifest_edits(self) -> None:
        if self.manifest is None:
            return
        if not self.output_edit.text().strip():
            raise ValueError("Select an output folder.")
        seen: set[tuple[str, str]] = set()
        for row, specimen_data in enumerate(self.manifest["specimens"]):
            group = self.table.item(row, 1).text().strip()
            specimen = self.table.item(row, 2).text().strip()
            if not group or not specimen:
                raise ValueError("Experimental group and specimen labels cannot be empty.")
            key = (group.casefold(), specimen.casefold())
            if key in seen:
                raise ValueError(f"Duplicate group/specimen label: {group} / {specimen}")
            seen.add(key)
            specimen_data["experimental_group"] = group
            specimen_data["specimen_id"] = specimen
        self.manifest["source_directory"] = self.source_edit.text().strip()
        self.manifest["output_directory"] = str(Path(self.output_edit.text().strip()).resolve())
        self.manifest["channel_roles"] = self._current_roles()
        self.manifest["calibration"] = self._current_calibration().to_dict()
        self.manifest.setdefault("import_settings", {}).update(
            {
                "channel_markers": self._current_channel_markers(),
                "default_experimental_group": self.fallback_group_edit.text().strip()
                or "Experiment",
            }
        )

    def _current_roles(self) -> dict[str, str]:
        roles = {
            "ChanA": str(self.channel_a_role.currentData()),
            "ChanB": str(self.channel_b_role.currentData()),
        }
        if len(set(roles.values())) != 2:
            raise ValueError("ChanA and ChanB must have different roles.")
        return roles

    def _save_project(self) -> None:
        if self._review_thread is not None:
            QMessageBox.information(
                self, "Review in progress", "Wait for the current correction to finish."
            )
            return
        try:
            if self.report is not None:
                self._sync_scan_edits()
                output = Path(self.output_edit.text().strip())
                if not self.output_edit.text().strip():
                    raise ValueError("Select an output folder.")
                manifest = create_project_manifest(
                    self.report,
                    output_directory=output,
                    channel_roles=self._current_roles(),
                    calibration=self._current_calibration(),
                )
                if self.project_path is None:
                    safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", str(manifest["batch_prefix"])).strip("-")
                    suggested = output / f"{safe_prefix or 'batch'}.synpo.json"
                    selected, _ = QFileDialog.getSaveFileName(
                        self,
                        "Save Synpo project",
                        str(suggested),
                        "Synpo project (*.synpo.json)",
                    )
                    if not selected:
                        return
                    self.project_path = Path(selected)
                self.manifest = manifest
                self.report = None
            elif self.manifest is not None:
                self._sync_manifest_edits()
            else:
                raise ValueError("Scan a batch or open a project before saving.")

            if self.project_path is None:
                raise ValueError("No project filename was selected.")
            self.project_path = save_project(self.project_path, self.manifest)
            self._prepare_preprocessing_tab()
            self._prepare_detection_tab()
            self._prepare_review_tab()
            self._prepare_measurements_tab()
            self.statusBar().showMessage(f"Saved {self.project_path}", 8000)
            self.setWindowTitle(f"Synpo Microscopy Processor — {self.project_path.name}")
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save project", str(exc))

    def _open_project(self) -> None:
        if self._review_thread is not None:
            QMessageBox.information(
                self, "Review in progress", "Wait for the current correction to finish."
            )
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Synpo project",
            "",
            "Synpo project (*.synpo.json);;JSON files (*.json)",
        )
        if not selected:
            return
        try:
            manifest = load_project(selected)
            quick_results = verify_project_sources(manifest, full_checksums=False)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))
            return
        self.report = None
        for dialog in list(self._context_dialogs):
            dialog.close()
        for dialog in list(self._spine_map_dialogs):
            dialog.close()
        self._context_cache.clear()
        self.manifest = manifest
        self.project_path = Path(selected).resolve()
        self._populate_manifest(manifest)
        self._prepare_preprocessing_tab()
        self._prepare_detection_tab()
        self._prepare_review_tab()
        self._prepare_measurements_tab()
        missing = sum(item["status"] != "ok" for item in quick_results)
        if missing:
            self.summary_label.setText(
                f"Project opened, but {missing} source file(s) are missing or changed. "
                "Use Project → Relink source folder."
            )
        else:
            self.summary_label.setText(
                "Project opened. Source filenames and sizes match; use Verify sources "
                "for full SHA-256 verification."
            )
        self.setWindowTitle(f"Synpo Microscopy Processor — {self.project_path.name}")

    def _populate_manifest(self, manifest: dict[str, object]) -> None:
        self.source_edit.setText(str(manifest["source_directory"]))
        self.output_edit.setText(str(manifest["output_directory"]))
        import_settings = manifest.get("import_settings", {})
        markers = import_settings.get(
            "channel_markers", {"ChanA": "ChanA", "ChanB": "ChanB"}
        )
        self.channel_a_marker.setText(str(markers.get("ChanA", "ChanA")))
        self.channel_b_marker.setText(str(markers.get("ChanB", "ChanB")))
        self.fallback_group_edit.setText(
            str(import_settings.get("default_experimental_group", "Experiment"))
        )
        roles = manifest["channel_roles"]
        self.channel_a_role.setCurrentIndex(self.channel_a_role.findData(roles["ChanA"]))
        self.channel_b_role.setCurrentIndex(self.channel_b_role.findData(roles["ChanB"]))
        calibration = Calibration.from_dict(manifest["calibration"])
        self.preset_combo.setCurrentText(calibration.preset_name)
        self.xy_spin.setValue(calibration.xy_um_per_pixel)
        self.z_spin.setValue(calibration.z_step_um)

        specimens = manifest["specimens"]
        self.table.setRowCount(len(specimens))
        for row, specimen in enumerate(specimens):
            channel_a = specimen["channels"]["ChanA"]
            channel_b = specimen["channels"]["ChanB"]
            metadata_a = channel_a["metadata"]
            metadata_b = channel_b["metadata"]
            same_shape = metadata_a["shape"] == metadata_b["shape"]
            values = [
                "Saved",
                specimen["experimental_group"],
                specimen["specimen_id"],
                channel_a["filename"],
                channel_b["filename"],
                " × ".join(str(value) for value in metadata_a["shape"]) if same_shape else "mismatch",
                metadata_a["dtype"],
                "",
            ]
            for column, value in enumerate(values):
                self._set_table_item(row, column, str(value), editable=column in {1, 2})
            self.table.item(row, 3).setToolTip(
                str(channel_source_path(manifest, channel_a))
            )
            self.table.item(row, 4).setToolTip(
                str(channel_source_path(manifest, channel_b))
            )

    def _verify_sources(self) -> None:
        if self.manifest is None:
            QMessageBox.information(self, "No project", "Open or save a project first.")
            return
        worker = VerifyWorker(
            self.manifest,
            None,
            False,
        )
        worker.completed.connect(lambda results: self._verification_completed(results, False))
        self._start_worker(worker, "verify")

    def _relink_sources(self) -> None:
        if self.manifest is None:
            QMessageBox.information(self, "No project", "Open or save a project first.")
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the relocated TIFF folder or common parent folder",
            str(self.manifest["source_directory"]),
        )
        if not directory:
            return
        worker = VerifyWorker(self.manifest, Path(directory), True)
        worker.completed.connect(lambda results: self._verification_completed(results, True))
        self._start_worker(worker, "relink")

    @Slot(object)
    def _verification_completed(self, results: list[dict[str, str]], relink: bool) -> None:
        failures = [item for item in results if item["status"] != "ok"]
        if failures:
            details = "\n".join(
                f"{item['filename']}: {item['detail']}" for item in failures[:12]
            )
            if len(failures) > 12:
                details += f"\n…and {len(failures) - 12} more"
            QMessageBox.warning(
                self,
                "Source verification failed",
                f"{len(failures)} of {len(results)} file(s) did not match.\n\n{details}",
            )
        else:
            if relink:
                self.source_edit.setText(str(self.manifest["source_directory"]))
                if self.project_path is not None:
                    save_project(self.project_path, self.manifest)
                message = (
                    "All checksums match. Each source file path was relinked and saved; "
                    "subfolders were searched when necessary."
                )
            else:
                message = "All source files passed full SHA-256 verification."
            QMessageBox.information(self, "Source verification", message)
        self.progress_label.setText("Verification complete")

    def _new_batch(self) -> None:
        if (
            self._job_thread is not None
            or self._review_thread is not None
            or self._context_thread is not None
        ):
            return
        for dialog in list(self._context_dialogs):
            dialog.close()
        self._context_cache.clear()
        self.report = None
        self.manifest = None
        self.project_path = None
        self._last_preview = None
        self._last_detection = None
        self._last_review = None
        self._last_distribution_preview = None
        self._preview_statistics.clear()
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
        self.source_edit.clear()
        self.output_edit.clear()
        self.table.setRowCount(0)
        self.summary_label.setText("Select a folder and scan it to begin.")
        self.setWindowTitle(f"Synpo Microscopy Processor — Beta {__version__}")

    def _set_job_running(self, running: bool) -> None:
        self.progress_bar.setVisible(running)
        for widget in (self.scan_button, self.manual_pair_button, self.save_button):
            widget.setEnabled(not running)
        for action in (
            self.new_action,
            self.open_action,
            self.save_action,
            self.verify_action,
            self.relink_action,
        ):
            action.setEnabled(not running)
        if hasattr(self, "run_preprocessing_button"):
            self.run_preprocessing_button.setEnabled(not running and self.manifest is not None)
            self.apply_preprocessing_button.setEnabled(not running and self.manifest is not None)
            self.cancel_preprocessing_button.setEnabled(
                running and self._job_kind == "preprocess"
            )
        if hasattr(self, "run_detection_button"):
            eligible = bool(
                self.manifest
                and any(
                    specimen["checkpoints"]["preprocessing"].get("state")
                    == "complete"
                    for specimen in self.manifest["specimens"]
                )
            )
            self.run_detection_button.setEnabled(not running and eligible)
            self.apply_detection_button.setEnabled(
                not running and self.manifest is not None
            )
            self.cancel_detection_button.setEnabled(
                running and self._job_kind == "detection"
            )
        if hasattr(self, "run_measurements_button"):
            eligible = bool(
                self.manifest
                and any(
                    specimen["checkpoints"]["preprocessing"].get("state")
                    == "complete"
                    and specimen["checkpoints"]["detection"].get("state")
                    == "complete"
                    and specimen["checkpoints"]["review"].get("state")
                    == "complete"
                    for specimen in self.manifest["specimens"]
                )
            )
            measured = bool(
                self.manifest
                and any(
                    specimen["checkpoints"].get("measurements", {}).get("state")
                    == "complete"
                    for specimen in self.manifest["specimens"]
                )
            )
            self.run_measurements_button.setEnabled(not running and eligible)
            self.save_measurement_settings_button.setEnabled(
                not running and self.manifest is not None
            )
            self.load_trim_preview_button.setEnabled(not running and measured)
            self.cancel_measurements_button.setEnabled(
                running and self._job_kind == "measurements"
            )
            self.save_distribution_review_button.setEnabled(not running)
            self.export_measurements_button.setEnabled(not running and measured)
            self.centerline_hint_button.setEnabled(
                not running
                and measured
                and self._current_spine_review_mode() == "cluster_positive"
            )
            self.clear_centerline_hint_button.setEnabled(
                not running
                and measured
                and self._last_distribution_preview is not None
                and bool(
                    self._last_distribution_preview.row.get(
                        "centerline_endpoint_hint_present", False
                    )
                )
            )
            self.open_spine_map_button.setEnabled(not running and measured)
        if not running and not self.progress_label.text():
            self.progress_label.setText("Ready")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            self._job_thread is not None
            or self._review_thread is not None
            or self._context_thread is not None
        ):
            QMessageBox.information(
                self,
                "Operation in progress",
                "Cancel the batch operation if needed and wait for the current safe "
                "step or correction to finish before closing Synpo.",
            )
            event.ignore()
            return
        self._save_window_preferences()
        super().closeEvent(event)


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("Synpo Microscopy Processor")
    application.setOrganizationName("Synpo")
    assets_path = Path(__file__).resolve().parent / "assets"
    icon_name = "synpo.ico" if sys.platform == "win32" else "synpo-icon.png"
    icon_path = assets_path / icon_name
    if icon_path.is_file():
        application.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    if icon_path.is_file():
        window.setWindowIcon(QIcon(str(icon_path)))
    window.show_initial()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
