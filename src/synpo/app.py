from __future__ import annotations

import re
import sys
import time
from threading import Event
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from PySide6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, QSettings, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices, QGuiApplication, QIcon, QImage, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

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
from .transfer import (
    TRANSFER_SUFFIX,
    TransferCacheError,
    TransferImportResult,
    create_transfer_archive,
    import_transfer_archive,
    inspect_transfer_archive,
)
from .preprocessing import (
    PreprocessingSettings,
    PreviewResult,
    ProcessingCancelled,
    StackStatistics,
    effective_preprocessing_settings,
    make_preview,
    process_project_cache,
    preprocessing_parameters_set,
)
from .detection import (
    ALWAYS_LOW_MEMORY_MODE,
    AUTOMATIC_MEMORY_MODE,
    DetectionSettings,
    DetectionSlice,
    detect_project,
    effective_detection_settings,
    load_detection_slice,
)
from .diagnostics import (
    DIAGNOSTIC_LEVEL_CORRECTION,
    DIAGNOSTIC_LEVEL_FULL,
    DiagnosticCallback,
    DiagnosticSession,
    diagnostic_protocol_path,
    diagnostic_session_parent,
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
    discard_specimen_review,
)
from .regions import normalize_rectangles, rectangle_records, specimen_rectangles
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
    MorphologyPreview,
    accept_all_eligible_distribution_spines,
    apply_morphology_review_edit,
    cluster_end_comparison_rows,
    clear_centerline_endpoint_hint,
    distribution_profile_available,
    distribution_summary_rows,
    filtered_measurement_result,
    load_cluster_trim_preview,
    load_distribution_preview,
    load_spine_review_preview,
    load_measurement_result,
    load_morphology_preview,
    measure_project,
    recover_compatible_measurement_checkpoints,
    set_centerline_endpoint_hint,
    set_distribution_review,
    set_spine_quality_review,
    spine_volume_filter_settings,
    redo_morphology_review,
    undo_morphology_review,
)
from .exporting import (
    export_measurements,
    filter_exported_measurement_workbook,
    inspect_exported_measurement_workbook,
)
from .morphology import (
    DEFAULT_FEATURES,
    DEFAULT_PLOT_STYLE,
    MORPHOLOGY_FEATURES,
    MorphologyClusteringSettings,
    draw_pca_3d_feature_axes,
    draw_pca_interpretation,
    draw_feature_correlation,
    draw_protein_puncta_volume,
    export_morphology_analysis,
    load_morphology_run,
    morphology_feature_value,
    feature_correlation_data,
    run_morphology_clustering_from_workbook,
    save_named_morphology_run,
    update_run_colors,
    update_run_plot_style,
)


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
    "dendrite_to_spine": QColor("#b8ff3d"),
    "spine_to_dendrite": QColor("#ff6f61"),
    "trim": QColor("#ff00ff"),
    "expand": QColor("#2389ff"),
    "split": QColor("#ffe119"),
    "filopodium": QColor("#9b59b6"),
    "merge": QColor("#ED6291"),
    "accept": QColor("#22d3ee"),
    "needs_attention": QColor("#ff8c1a"),
    "erase": QColor("#ffffff"),
}

DISTINCT_OBJECT_PALETTE = np.asarray(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ],
    dtype=np.uint8,
)


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


def _distinct_object_colors(labels: np.ndarray, kind: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.uint64)
    indices = (values * np.uint64(2654435761)) % np.uint64(len(DISTINCT_OBJECT_PALETTE))
    base = DISTINCT_OBJECT_PALETTE[indices.astype(np.intp)].astype(np.uint16)
    if kind == 0:
        # Reserve a disjoint blue range so a dendrite can never equal a spine color.
        base[..., 2] //= 2
        return base.astype(np.uint8)
    if kind == 1:
        base[..., 2] = 160 + base[..., 2] * 95 // 255
        return np.clip(base, 0, 255).astype(np.uint8)
    return base.astype(np.uint8)


def _screen_limited_size(widget: QWidget, width: int, height: int) -> tuple[int, int]:
    screen = widget.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return width, height
    available = screen.availableGeometry()
    return (
        max(320, min(width, round(available.width() * 0.9))),
        max(240, min(height, round(available.height() * 0.9))),
    )


class AbsoluteSlider(QSlider):
    """A slider whose mouse click maps directly to the clicked value."""

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.orientation() == Qt.Orientation.Horizontal
            and self.maximum() > self.minimum()
        ):
            fraction = max(0.0, min(1.0, event.position().x() / max(1, self.width() - 1)))
            self.setValue(
                self.minimum()
                + round(fraction * (self.maximum() - self.minimum()))
            )
            event.accept()
            return
        super().mousePressEvent(event)


class SliceView(QLabel):
    zoom_changed = Signal(int)
    navigate_requested = Signal(int)

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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() in {Qt.Key.Key_Up, Qt.Key.Key_Down}:
            self.navigate_requested.emit(-1 if event.key() == Qt.Key.Key_Up else 1)
            event.accept()
            return
        super().keyPressEvent(event)

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
        distinct_dendrites_spines: bool = False,
    ) -> None:
        scale = max(1.0, float(high) - float(low))
        gray = np.clip(
            (np.asarray(array, dtype=np.float32) - low) * 255.0 / scale, 0, 255
        ).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        for kind, (labels, color) in enumerate(
            (
                (dendrites, np.array([35, 220, 70], dtype=np.float32)),
                (spines, np.array([0, 205, 255], dtype=np.float32)),
                (clusters, np.array([255, 40, 205], dtype=np.float32)),
            )
        ):
            if labels is None:
                continue
            mask = np.asarray(labels) > 0
            overlay_colors = (
                _distinct_object_colors(np.asarray(labels), kind).astype(np.float32)
                if distinct_dendrites_spines and kind < 2
                else np.broadcast_to(color, rgb.shape)
            )
            rgb[mask] = np.clip(
                rgb[mask].astype(np.float32) * 0.3
                + overlay_colors[mask] * 0.7,
                0,
                255,
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


class MultiRoiView(SliceView):
    rectangle_drawn = Signal(object)

    def __init__(self) -> None:
        super().__init__("Analysis ROI projection")
        self._rectangles: list[tuple[int, int, int, int]] = []
        self._drawing_start: tuple[int, int] | None = None
        self._drawing_end: tuple[int, int] | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_rectangles(self, rectangles: list[tuple[int, int, int, int]]) -> None:
        self._rectangles = list(rectangles)
        self._render()

    def _render(self) -> None:
        super()._render()
        current = self.pixmap()
        if self._image is None or current is None:
            return
        canvas = current.copy()
        painter = QPainter(canvas)
        target = self._target_rect()
        sx = target.width() / max(1, self._image.width())
        sy = target.height() / max(1, self._image.height())
        for index, (x0, y0, x1, y1) in enumerate(self._rectangles, start=1):
            painter.setPen(QPen(QColor(40, 230, 100), 3))
            rectangle = QRectF(
                target.left() + x0 * sx,
                target.top() + y0 * sy,
                (x1 - x0) * sx,
                (y1 - y0) * sy,
            )
            painter.drawRect(rectangle)
            painter.drawText(rectangle.topLeft() + QPointF(5, 17), f"ROI {index}")
        if self._drawing_start is not None and self._drawing_end is not None:
            x0, y0 = self._drawing_start
            x1, y1 = self._drawing_end
            painter.setPen(QPen(QColor(255, 145, 20), 3))
            painter.drawRect(
                QRectF(
                    target.left() + min(x0, x1) * sx,
                    target.top() + min(y0, y1) * sy,
                    (abs(x1 - x0) + 1) * sx,
                    (abs(y1 - y0) + 1) * sy,
                )
            )
        painter.end()
        self.setPixmap(canvas)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.image_coordinate(event.position())
            if point is not None:
                self._drawing_start = point
                self._drawing_end = point
                self._render()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drawing_start is not None:
            point = self.image_coordinate(event.position())
            if point is not None:
                self._drawing_end = point
                self._render()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self._drawing_start is not None:
            point = self.image_coordinate(event.position())
            if point is not None:
                self._drawing_end = point
            start, end = self._drawing_start, self._drawing_end
            self._drawing_start = None
            self._drawing_end = None
            self._render()
            if start is not None and end is not None:
                x0, x1 = sorted((start[0], end[0]))
                y0, y1 = sorted((start[1], end[1]))
                if x1 - x0 >= 3 and y1 - y0 >= 3:
                    self.rectangle_drawn.emit((x0, y0, x1 + 1, y1 + 1))
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AnalysisRoiDialog(QDialog):
    def __init__(
        self,
        projection: np.ndarray,
        rectangles: list[tuple[int, int, int, int]],
        parent=None,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("Analysis regions")
        self.resize(*_screen_limited_size(self, 1050, 820))
        self._shape_yx = tuple(int(value) for value in projection.shape)
        full = [(0, 0, self._shape_yx[1], self._shape_yx[0])]
        normalized = normalize_rectangles(rectangles, self._shape_yx)
        self._rectangles = [] if normalized == full else normalized
        self._replace_index: int | None = None
        layout = QVBoxLayout(self)
        instructions = QLabel(
            "Draw one or more rectangles on the XY maximum projection. Separate regions "
            "are analyzed independently; touching or overlapping regions merge automatically."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.view = MultiRoiView()
        low, high = np.percentile(projection, (0.5, 99.8))
        self.view.show_array(projection, float(low), float(max(low + 1, high)))
        self.view.rectangle_drawn.connect(self._rectangle_drawn)
        layout.addWidget(self.view, 1)
        layout.addWidget(ZoomControls(self.view))
        controls = QHBoxLayout()
        self.roi_list = QComboBox()
        controls.addWidget(self.roi_list, 1)
        redraw = QPushButton("Redraw selected")
        redraw.clicked.connect(self._redraw_selected)
        controls.addWidget(redraw)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        controls.addWidget(remove)
        full_button = QPushButton("Use full image")
        full_button.clicked.connect(self._use_full_image)
        controls.addWidget(full_button)
        layout.addLayout(controls)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        apply_button = QPushButton("Save regions")
        apply_button.clicked.connect(self.accept)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        self._refresh()

    def rectangles(self) -> list[tuple[int, int, int, int]]:
        return list(self._rectangles)

    def _refresh(self) -> None:
        self._rectangles = normalize_rectangles(self._rectangles, self._shape_yx)
        self.view.set_rectangles(self._rectangles)
        self.roi_list.clear()
        if not self._rectangles:
            self.roi_list.addItem("Full image", -1)
        else:
            for index, (x0, y0, x1, y1) in enumerate(self._rectangles):
                self.roi_list.addItem(
                    f"ROI {index + 1}: X {x0}-{x1 - 1}, Y {y0}-{y1 - 1}", index
                )

    @Slot(object)
    def _rectangle_drawn(self, rectangle) -> None:  # type: ignore[no-untyped-def]
        if self._replace_index is not None and self._replace_index < len(self._rectangles):
            self._rectangles[self._replace_index] = tuple(rectangle)
        else:
            self._rectangles.append(tuple(rectangle))
        self._replace_index = None
        self._refresh()

    def _redraw_selected(self) -> None:
        value = self.roi_list.currentData()
        self._replace_index = int(value) if value is not None and int(value) >= 0 else None

    def _remove_selected(self) -> None:
        value = self.roi_list.currentData()
        if value is not None and 0 <= int(value) < len(self._rectangles):
            self._rectangles.pop(int(value))
        self._replace_index = None
        self._refresh()

    def _use_full_image(self) -> None:
        self._rectangles = []
        self._replace_index = None
        self._refresh()


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
        self._brush_diameter = 9
        self._hint_color = QColor(REVIEW_BRUSH_COLORS["add"])
        self._last_hint_render = 0.0
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))
        self._brush_diameter = self._brush_radius * 2 + 1
        self._draw_hints()

    def set_brush_diameter(self, diameter: int) -> None:
        self._brush_diameter = max(1, int(diameter))
        self._brush_radius = self._brush_diameter // 2
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
                self._last_hint_render = time.monotonic()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._drawing:
            point = self._image_position(event.position())
            if point is not None:
                self._append_to_stroke(point)
                now = time.monotonic()
                if now - self._last_hint_render >= 1.0 / 30.0:
                    self._draw_hints()
                    self._last_hint_render = now
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
            self._brush_diameter,
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


class MorphologyReviewCanvas(ReviewCanvas):
    """All-spine geometry overlay with paintable head/neck annotations."""

    def show_morphology(self, preview: MorphologyPreview, z_index: int, maximum: bool) -> None:
        raw_stack = preview.dendrite_stack
        raw = np.max(raw_stack, axis=0) if maximum else raw_stack[z_index]
        low, high = np.percentile(raw, (0.5, 99.8))
        scale = max(1.0, float(high) - float(low))
        gray = np.clip((raw.astype(np.float32) - low) * 255.0 / scale, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        spine = np.any(preview.spine_mask_stack, axis=0) if maximum else preview.spine_mask_stack[z_index]
        head = np.any(preview.head_mask_stack, axis=0) if maximum else preview.head_mask_stack[z_index]
        clusters = np.any(preview.cluster_mask_stack, axis=0) if maximum else preview.cluster_mask_stack[z_index]
        neck = spine & ~head
        for mask, color in (
            (neck, np.asarray([25, 190, 240])),
            (head, np.asarray([255, 145, 35])),
            (clusters, np.asarray([245, 40, 205])),
        ):
            rgb[mask] = np.clip(rgb[mask].astype(np.float32) * 0.25 + color * 0.75, 0, 255).astype(np.uint8)
        self.show_rgb(rgb)
        image = self._image.copy() if self._image is not None else None
        if image is not None:
            painter = QPainter(image)
            visible_axis = [point for point in preview.axis_points_local_zyx if maximum or point[0] == z_index]
            painter.setPen(QPen(QColor("#ffffff"), 1))
            for first, second in zip(visible_axis, visible_axis[1:]):
                painter.drawLine(QPoint(first[2], first[1]), QPoint(second[2], second[1]))
            for point, color in ((preview.base_point_local_zyx, QColor("#2cff60")), (preview.tip_point_local_zyx, QColor("#fff000"))):
                if point is not None and (maximum or point[0] == z_index):
                    painter.setPen(QPen(color, 2))
                    painter.drawEllipse(QPoint(point[2], point[1]), 4, 4)
            painter.end()
            self._image = image
        self._base_image = self._image.copy() if self._image is not None else None
        self.clear_hint()


class MorphologyPlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.5, 5.5), tight_layout=True)
        super().__init__(self.figure)
        self.setMinimumSize(480, 360)

    def show_run(self, result: dict[str, object], mode: str) -> None:
        from matplotlib.colors import to_rgba

        self.figure.clear()
        if mode == "pca_interpretation":
            draw_pca_interpretation(self.figure, result)
            self.draw_idle()
            return
        if mode == "pca_3d_features":
            draw_pca_3d_feature_axes(self.figure, result)
            self.draw_idle()
            return
        assignments = list(result.get("assignments", []))
        definitions = list(result.get("cluster_definitions", []))
        colors = {int(row["morphology_cluster_id"]): str(row.get("color", "#457b9d")) for row in definitions}
        dimensions = (
            3
            if int(result.get("settings", {}).get("pca_dimensions", 2)) == 3
            and assignments
            and "pca_3" in assignments[0]
            else 2
        )
        embedding_dimensions = int(
            result.get("settings", {}).get("embedding_plot_dimensions", 2)
        )
        is_3d = (mode == "pca" and dimensions == 3) or (
            mode == "embedding" and embedding_dimensions == 3
        )
        axis = self.figure.add_subplot(111, projection="3d" if is_3d else None)
        style = {**DEFAULT_PLOT_STYLE, **dict(result.get("plot_style", {}))}
        axes_rgba = to_rgba(
            str(style["axes_color"]), alpha=float(style["axes_alpha"])
        )
        background_rgba = to_rgba(
            str(style["background_color"]), alpha=float(style["background_alpha"])
        )
        self.figure.patch.set_facecolor(background_rgba)
        axis.set_facecolor(background_rgba)
        group_markers = ("o", "^", "s", "D", "P", "X", "v", "<", ">", "*")
        groups = sorted({str(row.get("experimental_group", "")) for row in assignments})
        marker_by_group = {
            group: group_markers[index % len(group_markers)]
            for index, group in enumerate(groups)
        }
        if mode in {"volume_length", "volume_straight", "pca", "embedding"}:
            for cluster in sorted(colors):
                for group in groups:
                    members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster and str(row.get("experimental_group", "")) == group]
                    if not members:
                        continue
                    label = f"Cluster {cluster} · {group}"
                    marker = marker_by_group[group]
                    if mode in {"volume_length", "volume_straight"}:
                        length_key = "spine_base_to_tip_distance_um" if mode == "volume_straight" else "spine_curvilinear_length_um"
                        members = [row for row in members if row.get("volume_um3") is not None and row.get(length_key) is not None]
                        axis.scatter([row.get("volume_um3") for row in members], [row.get(length_key) for row in members], color=colors[cluster], marker=marker, alpha=0.72, label=label)
                    elif mode == "embedding" and embedding_dimensions == 3:
                        axis.scatter([row.get("embedding_1") for row in members], [row.get("embedding_2") for row in members], [row.get("embedding_3") for row in members], color=colors[cluster], marker=marker, alpha=0.72, label=label)
                    elif mode == "embedding":
                        axis.scatter([row.get("embedding_1") for row in members], [row.get("embedding_2") for row in members], color=colors[cluster], marker=marker, alpha=0.72, label=label)
                    elif dimensions == 3:
                        axis.scatter([row.get("pca_1") for row in members], [row.get("pca_2") for row in members], [row.get("pca_3") for row in members], color=colors[cluster], marker=marker, alpha=0.72, label=label)
                    else:
                        axis.scatter([row.get("pca_1") for row in members], [row.get("pca_2") for row in members], color=colors[cluster], marker=marker, alpha=0.72, label=label)
            if mode in {"volume_length", "volume_straight"}:
                axis.set_xscale("log")
                length_label = "Base-to-tip distance (µm)" if mode == "volume_straight" else "Curvilinear length (µm)"
                axis.set(xlabel="Spine volume (µm³)", ylabel=length_label, title=f"Volume versus {length_label.removesuffix(' (µm)').lower()}")
            elif mode == "pca":
                axis.set(xlabel="PCA 1", ylabel="PCA 2", title=f"{dimensions}D PCA morphology plot")
                if dimensions == 3:
                    axis.set_zlabel("PCA 3")
            else:
                method = str(result.get("settings", {}).get("reduction_method", "umap"))
                method_label = "PCC/PCUMAP" if method == "pcumap" else "UMAP"
                clustering_dimensions = int(
                    result.get("settings", {}).get("embedding_dimensions", embedding_dimensions)
                )
                axis.set(
                    xlabel=f"{method_label} 1",
                    ylabel=f"{method_label} 2",
                    title=(
                        f"{embedding_dimensions}D view of the {clustering_dimensions}D "
                        f"{method_label} clustering space"
                    ),
                )
                if embedding_dimensions == 3:
                    axis.set_zlabel(f"{method_label} 3")
            axis.legend()
        elif mode == "custom_features":
            features = list(result.get("settings", {}).get("features", []))
            x_feature = str(style.get("custom_x_feature", ""))
            y_feature = str(style.get("custom_y_feature", ""))
            if x_feature not in features:
                x_feature = features[0] if features else ""
            if y_feature not in features:
                y_feature = features[1] if len(features) > 1 else x_feature
            plotted = False
            if x_feature and y_feature:
                for cluster in sorted(colors):
                    for group in groups:
                        points = []
                        for row in assignments:
                            if (
                                int(row["morphology_cluster_id"]) != cluster
                                or str(row.get("experimental_group", "")) != group
                            ):
                                continue
                            x_value = morphology_feature_value(row, x_feature)
                            y_value = morphology_feature_value(row, y_feature)
                            if x_value is not None and y_value is not None:
                                points.append((x_value, y_value))
                        if points:
                            axis.scatter(
                                [point[0] for point in points],
                                [point[1] for point in points],
                                color=colors[cluster],
                                marker=marker_by_group[group],
                                alpha=0.72,
                                label=f"Cluster {cluster} · {group}",
                            )
                            plotted = True
                axis.set(
                    xlabel=MORPHOLOGY_FEATURES[x_feature][0],
                    ylabel=MORPHOLOGY_FEATURES[y_feature][0],
                    title=(
                        f"{MORPHOLOGY_FEATURES[y_feature][0]} versus "
                        f"{MORPHOLOGY_FEATURES[x_feature][0]}"
                    ),
                )
            if not plotted:
                axis.text(
                    0.5,
                    0.5,
                    "No complete values are available for this feature pair.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
        elif mode == "protein_positive":
            clusters = sorted(colors)
            values = []
            for cluster in clusters:
                members = [row for row in assignments if int(row["morphology_cluster_id"]) == cluster]
                values.append(100.0 * sum(bool(row.get("has_protein_cluster")) for row in members) / len(members) if members else 0.0)
            axis.bar([str(value) for value in clusters], values, color=[colors[value] for value in clusters])
            axis.set(xlabel="Morphology cluster", ylabel="Protein-positive spines (%)", title="Protein-positive fraction")
        elif mode == "protein_volume":
            draw_protein_puncta_volume(axis, result)
        elif mode == "protein_position":
            summary = list(result.get("protein_summary", []))
            for cluster in sorted(colors):
                row = next((value for value in summary if int(value.get("morphology_cluster_id", 0)) == cluster and value.get("subset") == "protein_positive"), None)
                if row:
                    values = [row.get(f"bin_{index:02d}_mean") for index in range(1, 11)]
                    axis.plot(range(1, 11), [np.nan if value is None else float(value) for value in values], marker="o", color=colors[cluster], label=f"Cluster {cluster}")
            axis.set(xlabel="Normalized shaft-to-tip bin", ylabel="Mean protein distribution", title="Protein position profiles")
            axis.legend()
        elif mode == "group_proportions":
            rows = list(result.get("group_summary", []))
            groups = sorted({str(row.get("experimental_group", "")) for row in rows})
            clusters = sorted(colors)
            positions = np.arange(len(groups), dtype=float)
            width = 0.8 / max(1, len(clusters))
            for offset, cluster in enumerate(clusters):
                values = [next((float(row.get("specimen_percentage_mean") or 0.0) for row in rows if str(row.get("experimental_group", "")) == group and int(row.get("morphology_cluster_id", 0)) == cluster), 0.0) for group in groups]
                axis.bar(positions + (offset - (len(clusters) - 1) / 2.0) * width, values, width=width, color=colors[cluster], label=f"Cluster {cluster}")
            axis.set_xticks(positions, groups, rotation=25, ha="right")
            axis.set(xlabel="Experimental group", ylabel="Mean specimen proportion (%)", title="Cluster proportions by group")
            axis.legend()
        else:
            features = list(result.get("settings", {}).get("features", []))
            matrix = np.asarray([[float(row.get(f"{feature}_median") or 0.0) for feature in features] for row in definitions], dtype=np.float64)
            center = np.mean(matrix, axis=0)
            spread = np.std(matrix, axis=0)
            image = axis.imshow((matrix - center) / np.where(spread > 1e-12, spread, 1.0), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
            axis.set_xticks(np.arange(len(features)), [MORPHOLOGY_FEATURES[value][0] for value in features], rotation=35, ha="right")
            axis.set_yticks(np.arange(len(definitions)), [f"Cluster {row['morphology_cluster_id']}" for row in definitions])
            axis.set_title("Standardized cluster median profiles")
            self.figure.colorbar(image, ax=axis)
        axis.tick_params(colors=axes_rgba)
        axis.xaxis.label.set_color(axes_rgba)
        axis.yaxis.label.set_color(axes_rgba)
        axis.title.set_color(axes_rgba)
        if is_3d:
            axis.zaxis.label.set_color(axes_rgba)
        for spine in axis.spines.values():
            spine.set_color(axes_rgba)
        handles, labels = axis.get_legend_handles_labels()
        existing_legend = axis.get_legend()
        if existing_legend is not None:
            existing_legend.remove()
        legend = None
        if bool(style.get("show_legend", True)) and handles:
            legend_position = str(style.get("legend_position", "outside_right"))
            if legend_position == "outside_bottom":
                legend = axis.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.16),
                    ncols=min(3, len(handles)),
                    borderaxespad=0.0,
                )
            elif legend_position == "inside":
                legend = axis.legend(handles, labels, loc="upper right")
            else:
                legend = axis.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                )
        if legend is not None:
            legend.get_frame().set_facecolor(background_rgba)
            for text_item in legend.get_texts():
                text_item.set_color(axes_rgba)
        axis.grid(alpha=0.2, color=axes_rgba)
        self.draw_idle()


class ClusterCountScorePanel(QWidget):
    """Plot and tabulate the candidate scores used to choose a cluster count."""

    _METHOD_LABELS = {
        "information_criterion": "Information criterion",
        "silhouette": "Silhouette score",
        "elbow": "Within-cluster SSE (elbow)",
    }

    def __init__(self) -> None:
        super().__init__()
        self._result: dict[str, object] | None = None
        self._result_identity: int | None = None
        layout = QVBoxLayout(self)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Score to display:"))
        self.score_method = QComboBox()
        self.score_method.addItem(
            "Information criterion (lower is better)", "information_criterion"
        )
        self.score_method.addItem(
            "Silhouette score (higher is better)", "silhouette"
        )
        self.score_method.addItem(
            "Within-cluster SSE elbow", "elbow"
        )
        self.score_method.currentIndexChanged.connect(self._render)
        selector_row.addWidget(self.score_method, 1)
        layout.addLayout(selector_row)
        self.summary = QLabel(
            "Run or load a clustering analysis to inspect its candidate scores."
        )
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.figure = Figure(figsize=(7.5, 4.1), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(480, 300)
        layout.addWidget(self.canvas, 1)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            [
                "Clusters",
                "Displayed score",
                "Elbow distance",
                "Accepted",
                "Cluster sizes",
                "Selected",
            ]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(220)
        layout.addWidget(self.table)

    def show_run(self, result: dict[str, object]) -> None:
        is_new_result = id(result) != self._result_identity
        self._result = result
        self._result_identity = id(result)
        if is_new_result:
            method = str(
                dict(result.get("settings", {})).get(
                    "cluster_count_selection", "information_criterion"
                )
            )
            index = self.score_method.findData(method)
            self.score_method.blockSignals(True)
            self.score_method.setCurrentIndex(max(0, index))
            self.score_method.blockSignals(False)
        self._render()

    @staticmethod
    def _number(value: object) -> str:
        if value is None:
            return "—"
        number = float(value)
        if not np.isfinite(number):
            return "—"
        magnitude = abs(number)
        if magnitude != 0.0 and (magnitude >= 100_000 or magnitude < 0.001):
            return f"{number:.4e}"
        return f"{number:.5g}"

    def _render(self, *_args) -> None:  # type: ignore[no-untyped-def]
        self.figure.clear()
        self.table.setRowCount(0)
        if self._result is None:
            self.canvas.draw_idle()
            return
        diagnostics = sorted(
            (dict(row) for row in self._result.get("candidate_diagnostics", [])),
            key=lambda row: int(row.get("cluster_count", 0)),
        )
        if not diagnostics:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "This saved run has no candidate-score diagnostics.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.summary.setText(
                "Candidate scores are unavailable for this saved run. Re-run the "
                "analysis to generate them."
            )
            self.canvas.draw_idle()
            return

        method = str(self.score_method.currentData())
        if method == "silhouette":
            score_key = "silhouette"
            y_label = "Mean silhouette score"
            direction = "Higher values indicate better separated clusters."
        elif method == "elbow":
            score_key = "within_cluster_sse"
            y_label = "Within-cluster SSE"
            direction = "Choose the knee where further SSE reduction begins to level off."
        else:
            score_key = "criterion"
            criterion_names = {
                str(row.get("criterion_name", "Information criterion"))
                for row in diagnostics
            }
            y_label = (
                next(iter(criterion_names))
                if len(criterion_names) == 1
                else "Information criterion"
            )
            direction = "Lower values indicate the preferred candidate."

        axis = self.figure.add_subplot(111)
        plotted = [
            row
            for row in diagnostics
            if row.get(score_key) is not None
            and np.isfinite(float(row[score_key]))
        ]
        if plotted:
            counts = [int(row["cluster_count"]) for row in plotted]
            scores = [float(row[score_key]) for row in plotted]
            axis.plot(counts, scores, color="#8a99a8", linewidth=1.4, zorder=1)
            accepted = [row for row in plotted if bool(row.get("accepted", False))]
            rejected = [row for row in plotted if not bool(row.get("accepted", False))]
            if accepted:
                axis.scatter(
                    [int(row["cluster_count"]) for row in accepted],
                    [float(row[score_key]) for row in accepted],
                    color="#2878b5",
                    s=48,
                    label="Accepted candidate",
                    zorder=3,
                )
            if rejected:
                axis.scatter(
                    [int(row["cluster_count"]) for row in rejected],
                    [float(row[score_key]) for row in rejected],
                    color="#7f7f7f",
                    marker="x",
                    s=58,
                    label="Rejected by minimum cluster size",
                    zorder=3,
                )
            selected = [row for row in plotted if bool(row.get("selected", False))]
            if selected:
                axis.scatter(
                    [int(row["cluster_count"]) for row in selected],
                    [float(row[score_key]) for row in selected],
                    color="#d62728",
                    edgecolor="white",
                    marker="*",
                    s=190,
                    linewidth=0.8,
                    label="Selected for this run",
                    zorder=5,
                )
            for row in plotted:
                axis.annotate(
                    self._number(row[score_key]),
                    (int(row["cluster_count"]), float(row[score_key])),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            axis.set_xticks(counts)
            axis.legend(loc="best")
        else:
            axis.text(
                0.5,
                0.5,
                "No score is available for this method.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set(
            xlabel="Number of clusters",
            ylabel=y_label,
            title=f"{self._METHOD_LABELS.get(method, method)} by cluster count",
        )
        axis.grid(alpha=0.22)

        self.table.setRowCount(len(diagnostics))
        for row_index, row in enumerate(diagnostics):
            values = (
                str(row.get("cluster_count", "")),
                self._number(row.get(score_key)),
                self._number(row.get("elbow_distance")),
                "Yes" if bool(row.get("accepted", False)) else "No",
                ", ".join(str(value) for value in row.get("cluster_sizes", [])),
                "Yes" if bool(row.get("selected", False)) else "",
            )
            for column_index, value in enumerate(values):
                self.table.setItem(
                    row_index, column_index, QTableWidgetItem(value)
                )

        saved_method = str(
            dict(self._result.get("settings", {})).get(
                "cluster_count_selection", "information_criterion"
            )
        )
        selected_count = self._result.get("selected_cluster_count", "—")
        self.summary.setText(
            f"This run selected {selected_count} clusters using "
            f"{self._METHOD_LABELS.get(saved_method, saved_method)}. {direction} "
            "The red star always marks the count actually used by the saved run; "
            "gray crosses failed the configured minimum cluster-size rule."
        )
        self.canvas.draw_idle()


class CorrelationMatrixPanel(QWidget):
    """Interactive Pearson feature-redundancy matrix for a saved analysis run."""

    settings_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._result: dict[str, object] | None = None
        self._result_identity: int | None = None
        self._data: dict[str, object] | None = None
        self._matrix_axis: object | None = None
        self._selected_pair: tuple[str, str] | None = None
        self._negative_color = QColor("#2166ac")
        self._zero_color = QColor("#f7f7f7")
        self._positive_color = QColor("#b2182b")

        layout = QVBoxLayout(self)
        feature_group = QGroupBox("Metrics included in this matrix")
        feature_layout = QGridLayout(feature_group)
        self.feature_checks: dict[str, QCheckBox] = {}
        for index, (feature, (label, _column)) in enumerate(
            MORPHOLOGY_FEATURES.items()
        ):
            checkbox = QCheckBox(label)
            checkbox.setChecked(feature in DEFAULT_FEATURES)
            self.feature_checks[feature] = checkbox
            feature_layout.addWidget(checkbox, index // 3, index % 3)
        layout.addWidget(feature_group)

        selection_row = QHBoxLayout()
        copy_button = QPushButton("Copy clustering metrics")
        copy_button.clicked.connect(self._copy_clustering_features)
        selection_row.addWidget(copy_button)
        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(
            lambda: self._set_all_features(True)
        )
        selection_row.addWidget(select_all_button)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self._set_all_features(False))
        selection_row.addWidget(clear_button)
        selection_row.addStretch(1)
        layout.addLayout(selection_row)

        options = QGridLayout()
        options.addWidget(QLabel("Spine population:"), 0, 0)
        self.scope = QComboBox()
        self.scope.setToolTip(
            "Pooled ignores group labels. Choosing one group simply restricts the "
            "spines included in the same feature-to-feature Pearson calculation."
        )
        options.addWidget(self.scope, 0, 1)
        options.addWidget(QLabel("Flag |r| at or above:"), 0, 2)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setDecimals(2)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.80)
        options.addWidget(self.threshold, 0, 3)
        self.reorder = QCheckBox("Group similar metrics")
        self.reorder.setToolTip(
            "Hierarchically reorder metrics by absolute Pearson correlation."
        )
        options.addWidget(self.reorder, 1, 0, 1, 2)
        self.export_groups = QCheckBox(
            "Include one matrix per experimental group in full export"
        )
        options.addWidget(self.export_groups, 1, 2, 1, 2)

        options.addWidget(QLabel("Colormap:"), 2, 0)
        self.colormap = QComboBox()
        for label, value in (
            ("Coolwarm", "coolwarm"),
            ("Red / blue", "RdBu_r"),
            ("Purple / orange", "PuOr"),
            ("Brown / blue-green", "BrBG"),
            ("Pink / green", "PiYG"),
            ("Grayscale", "Greys"),
            ("Custom three-color", "custom"),
        ):
            self.colormap.addItem(label, value)
        self.colormap.currentIndexChanged.connect(self._colormap_changed)
        options.addWidget(self.colormap, 2, 1)
        options.addWidget(QLabel("Heatmap alpha:"), 2, 2)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setDecimals(2)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(1.0)
        options.addWidget(self.alpha, 2, 3)

        custom_colors = QHBoxLayout()
        self.negative_color_button = QPushButton("Negative color…")
        self.negative_color_button.clicked.connect(
            lambda: self._choose_color("negative")
        )
        custom_colors.addWidget(self.negative_color_button)
        self.zero_color_button = QPushButton("Zero color…")
        self.zero_color_button.clicked.connect(
            lambda: self._choose_color("zero")
        )
        custom_colors.addWidget(self.zero_color_button)
        self.positive_color_button = QPushButton("Positive color…")
        self.positive_color_button.clicked.connect(
            lambda: self._choose_color("positive")
        )
        custom_colors.addWidget(self.positive_color_button)
        custom_colors.addStretch(1)
        custom_widget = QWidget()
        custom_widget.setLayout(custom_colors)
        options.addWidget(custom_widget, 3, 0, 1, 4)
        layout.addLayout(options)

        action_row = QHBoxLayout()
        update_button = QPushButton("Update matrix")
        update_button.clicked.connect(self._apply_and_render)
        action_row.addWidget(update_button)
        action_row.addStretch(1)
        action_row.addWidget(QLabel("Export format:"))
        self.export_format = QComboBox()
        self.export_format.addItem("SVG (vector)", "svg")
        self.export_format.addItem("PDF (vector)", "pdf")
        self.export_format.addItem("PNG", "png")
        action_row.addWidget(self.export_format)
        action_row.addWidget(QLabel("PNG DPI:"))
        self.export_dpi = QComboBox()
        for dpi in (300, 600, 1200):
            self.export_dpi.addItem(str(dpi), dpi)
        self.export_dpi.setCurrentIndex(1)
        action_row.addWidget(self.export_dpi)
        export_button = QPushButton("Export correlation figure…")
        export_button.clicked.connect(self._export_figure)
        action_row.addWidget(export_button)
        layout.addLayout(action_row)

        self.summary = QLabel(
            "Run or load a clustering analysis to inspect feature redundancy."
        )
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.figure = Figure(figsize=(9.5, 6.2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(560, 390)
        self.canvas.mpl_connect("button_press_event", self._matrix_clicked)
        layout.addWidget(self.canvas, 1)
        self._refresh_color_buttons()
        self._colormap_changed()

    def show_run(self, result: dict[str, object]) -> None:
        is_new_result = id(result) != self._result_identity
        self._result = result
        self._result_identity = id(result)
        if is_new_result:
            style = {
                **DEFAULT_PLOT_STYLE,
                **dict(result.get("plot_style", {})),
            }
            controls = (
                self.scope,
                self.threshold,
                self.reorder,
                self.export_groups,
                self.colormap,
                self.alpha,
            )
            for control in controls:
                control.blockSignals(True)
            groups = sorted(
                {
                    str(row.get("experimental_group", ""))
                    for row in result.get("assignments", [])
                }
            )
            self.scope.clear()
            self.scope.addItem("All included spines (pooled)", "__pooled__")
            for group in groups:
                self.scope.addItem(group or "(blank group)", group)
            scope_index = self.scope.findData(
                str(style.get("correlation_scope", "__pooled__"))
            )
            self.scope.setCurrentIndex(max(0, scope_index))
            selected = {
                str(feature)
                for feature in style.get(
                    "correlation_features", DEFAULT_FEATURES
                )
            }
            for feature, checkbox in self.feature_checks.items():
                checkbox.setChecked(feature in selected)
            self.threshold.setValue(float(style["correlation_threshold"]))
            self.reorder.setChecked(bool(style["correlation_reorder"]))
            self.export_groups.setChecked(
                bool(style["correlation_export_group_matrices"])
            )
            self.colormap.setCurrentIndex(
                max(
                    0,
                    self.colormap.findData(str(style["correlation_colormap"])),
                )
            )
            self.alpha.setValue(float(style["correlation_alpha"]))
            self._negative_color = QColor(
                str(style["correlation_negative_color"])
            )
            self._zero_color = QColor(str(style["correlation_zero_color"]))
            self._positive_color = QColor(
                str(style["correlation_positive_color"])
            )
            for control in controls:
                control.blockSignals(False)
            self._selected_pair = None
            self._refresh_color_buttons()
            self._colormap_changed()
        self._render()

    def settings(self) -> dict[str, object]:
        scope_data = self.scope.currentData()
        return {
            "correlation_features": [
                feature
                for feature, checkbox in self.feature_checks.items()
                if checkbox.isChecked()
            ],
            "correlation_scope": (
                "__pooled__" if scope_data is None else str(scope_data)
            ),
            "correlation_threshold": self.threshold.value(),
            "correlation_colormap": str(self.colormap.currentData() or "coolwarm"),
            "correlation_negative_color": self._negative_color.name(),
            "correlation_zero_color": self._zero_color.name(),
            "correlation_positive_color": self._positive_color.name(),
            "correlation_alpha": self.alpha.value(),
            "correlation_reorder": self.reorder.isChecked(),
            "correlation_export_group_matrices": self.export_groups.isChecked(),
        }

    def _set_all_features(self, selected: bool) -> None:
        for checkbox in self.feature_checks.values():
            checkbox.setChecked(selected)

    def _copy_clustering_features(self) -> None:
        if self._result is None:
            return
        selected = {
            str(feature)
            for feature in dict(self._result.get("settings", {})).get(
                "features", []
            )
        }
        for feature, checkbox in self.feature_checks.items():
            checkbox.setChecked(feature in selected)

    def _choose_color(self, target: str) -> None:
        current = {
            "negative": self._negative_color,
            "zero": self._zero_color,
            "positive": self._positive_color,
        }[target]
        chosen = QColorDialog.getColor(current, self, f"Choose {target} correlation color")
        if not chosen.isValid():
            return
        if target == "negative":
            self._negative_color = chosen
        elif target == "zero":
            self._zero_color = chosen
        else:
            self._positive_color = chosen
        self._refresh_color_buttons()
        self.colormap.setCurrentIndex(self.colormap.findData("custom"))

    def _refresh_color_buttons(self) -> None:
        for button, color in (
            (self.negative_color_button, self._negative_color),
            (self.zero_color_button, self._zero_color),
            (self.positive_color_button, self._positive_color),
        ):
            button.setStyleSheet(f"background-color: {color.name()};")

    def _colormap_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        enabled = self.colormap.currentData() == "custom"
        self.negative_color_button.setEnabled(enabled)
        self.zero_color_button.setEnabled(enabled)
        self.positive_color_button.setEnabled(enabled)

    def _current_features(self) -> list[str]:
        return [
            feature
            for feature, checkbox in self.feature_checks.items()
            if checkbox.isChecked()
        ]

    def _apply_and_render(self) -> None:
        if len(self._current_features()) < 2:
            QMessageBox.warning(
                self,
                "Cannot calculate correlation matrix",
                "Select at least two metrics.",
            )
            return
        self._selected_pair = None
        if self._render():
            self.settings_changed.emit(self.settings())

    def _render(self) -> bool:
        self.figure.clear()
        self._matrix_axis = None
        self._data = None
        if self._result is None:
            self.canvas.draw_idle()
            return False
        scope_data = self.scope.currentData()
        scope = "__pooled__" if scope_data is None else str(scope_data)
        group = None if scope == "__pooled__" else scope
        try:
            self._data = draw_feature_correlation(
                self.figure,
                self._result,
                features=self._current_features(),
                experimental_group=group,
                threshold=self.threshold.value(),
                reorder=self.reorder.isChecked(),
                colormap=str(self.colormap.currentData() or "coolwarm"),
                negative_color=self._negative_color.name(),
                zero_color=self._zero_color.name(),
                positive_color=self._positive_color.name(),
                alpha=self.alpha.value(),
                scatter_pair=self._selected_pair,
            )
        except ValueError as exc:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                str(exc),
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.summary.setText(str(exc))
            self.canvas.draw_idle()
            return False
        self._matrix_axis = self.figure.axes[0] if self.figure.axes else None
        flagged = [
            row
            for row in self._data["pairs"]
            if row.get("high_correlation") or row.get("same_source_measurement")
        ]
        warnings = "; ".join(
            f"{row['feature_1_label']} / {row['feature_2_label']}: "
            + (
                f"r={float(row['pearson_r']):.3f}, "
                if row.get("pearson_r") is not None
                else "r=N/A, "
            )
            + str(row["warning"])
            for row in flagged
        )
        count = int(self._data["included_spine_count"])
        missing = int(self._data["missing_spine_count"])
        small_sample = " Warning: fewer than 30 complete spines." if count < 30 else ""
        self.summary.setText(
            f"Pearson matrix: {count} listwise-complete spines; {missing} omitted for "
            f"missing selected metrics.{small_sample} Click a matrix cell to inspect "
            f"its scatter plot."
            + (f" Flagged pairs: {warnings}" if warnings else " No pairs are flagged.")
        )
        self.canvas.draw_idle()
        return True

    def _matrix_clicked(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            self._data is None
            or event.inaxes is not self._matrix_axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        column = int(round(float(event.xdata)))
        row = int(round(float(event.ydata)))
        features = list(self._data["features"])
        if not (0 <= row < len(features) and 0 <= column < len(features)):
            return
        self._selected_pair = (features[column], features[row])
        self._render()

    def _export_figure(self) -> None:
        if self._data is None and not self._render():
            return
        extension = str(self.export_format.currentData() or "svg")
        filters = {
            "svg": "SVG vector figure (*.svg)",
            "pdf": "PDF vector figure (*.pdf)",
            "png": "PNG image (*.png)",
        }
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export correlation matrix",
            f"feature_correlation_matrix.{extension}",
            filters[extension],
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.casefold() != f".{extension}":
            path = path.with_suffix(f".{extension}")
        arguments: dict[str, object] = {"bbox_inches": "tight"}
        if extension == "png":
            arguments["dpi"] = int(self.export_dpi.currentData())
        try:
            self.figure.savefig(path, **arguments)
        except OSError as exc:
            QMessageBox.warning(self, "Cannot export correlation figure", str(exc))
            return
        QMessageBox.information(
            self, "Correlation figure exported", f"Saved: {path}"
        )


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
        rois_xy: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.z_index = z_index
        self.settings = settings
        self.xy_um_per_pixel = xy_um_per_pixel
        self.z_step_um = z_step_um
        self.statistics = statistics
        self.cache_key = cache_key
        self.rois_xy = rois_xy

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
                rois_xy=self.rois_xy,
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

    def __init__(
        self,
        manifest: dict[str, object],
        project_path: Path,
        specimen_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.cancel_event = Event()
        self.specimen_indices = specimen_indices

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
                specimen_indices=self.specimen_indices,
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

    def __init__(
        self,
        manifest: dict[str, object],
        project_path: Path,
        specimen_indices: list[int] | None = None,
        force: bool = False,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.cancel_event = Event()
        self.specimen_indices = specimen_indices
        self.force = force

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
                specimen_indices=self.specimen_indices,
                force=self.force,
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


class MorphologyClusteringWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, manifest: dict[str, object], project_path: Path, name: str, settings: MorphologyClusteringSettings) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.name = name
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Morphology clustering", 0, 1, "Preparing all-spine feature matrix")
            result = save_named_morphology_run(self.manifest, self.project_path, self.name, self.settings)
            self.progress.emit("Morphology clustering", 1, 1, "Saved named analysis run")
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class MorphologyEditWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        project_path: Path,
        specimen_index: int,
        spine_id: int,
        mode: str,
        arguments: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.specimen_index = specimen_index
        self.spine_id = spine_id
        self.mode = mode
        self.arguments = arguments or {}

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Geometry review", 0, 1, f"Recalculating spine {self.spine_id}")
            if self.mode == "undo":
                result = undo_morphology_review(
                    self.manifest, self.project_path, self.specimen_index, self.spine_id
                )
            elif self.mode == "redo":
                result = redo_morphology_review(
                    self.manifest, self.project_path, self.specimen_index, self.spine_id
                )
            else:
                call_arguments = {
                    key: value for key, value in self.arguments.items() if key != "advance"
                }
                result = apply_morphology_review_edit(
                    self.manifest,
                    self.project_path,
                    self.specimen_index,
                    self.spine_id,
                    **call_arguments,
                )
            self.progress.emit("Geometry review", 1, 1, f"Spine {self.spine_id} saved")
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(
            {
                "result": result,
                "specimen_index": self.specimen_index,
                "spine_id": self.spine_id,
                "advance": bool(self.arguments.get("advance", False)),
            }
        )


class MorphologyExportWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, result: dict[str, object], path: Path) -> None:
        super().__init__()
        self.result = result
        self.path = path

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Morphology export", 0, 1, "Writing workbook, CSV, PDF, SVG, and PNG files")
            exported = export_morphology_analysis(self.result, self.path)
            self.progress.emit("Morphology export", 1, 1, "Export verified")
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(exported)


class StandaloneMorphologyWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, source: Path, output: Path, settings: MorphologyClusteringSettings, name: str) -> None:
        super().__init__()
        self.source = source
        self.output = output
        self.settings = settings
        self.name = name

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Standalone morphology", 0, 1, "Loading workbook and fitting clusters")
            result = run_morphology_clustering_from_workbook(self.source, self.output, self.settings, run_name=self.name)
            self.progress.emit("Standalone morphology", 1, 1, "Analysis package verified")
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


class SpineVolumeHistogram(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.values = np.asarray([], dtype=np.float64)
        self.cutoff = 0.0
        self.setMinimumHeight(180)

    def set_data(self, values: list[float], cutoff: float) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.cutoff = float(cutoff)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15181d"))
        area = self.rect().adjusted(48, 14, -18, -34)
        painter.setPen(QColor("#d7dce2"))
        painter.drawText(8, 18, "Spine count")
        if not self.values.size or area.width() <= 1 or area.height() <= 1:
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "No manually valid spines")
            return
        minimum = min(float(np.min(self.values)), self.cutoff)
        maximum = max(float(np.max(self.values)), self.cutoff)
        if maximum <= minimum:
            maximum = minimum + max(abs(minimum) * 0.01, 1e-9)
        q1, q3 = np.percentile(self.values, [25.0, 75.0])
        width = 2.0 * float(q3 - q1) * float(len(self.values)) ** (-1.0 / 3.0)
        bins = (
            int(np.ceil((float(np.max(self.values)) - float(np.min(self.values))) / width))
            if len(self.values) >= 4 and width > 0 and np.isfinite(width)
            else int(np.ceil(np.log2(len(self.values)) + 1.0))
        )
        bins = min(60, max(1, bins))
        counts, edges = np.histogram(self.values, bins=bins, range=(minimum, maximum))
        peak = max(1, int(np.max(counts)))
        bar_width = area.width() / len(counts)
        for index, count in enumerate(counts):
            left, right = float(edges[index]), float(edges[index + 1])
            height = area.height() * int(count) / peak
            color = QColor("#e05252") if right <= self.cutoff else QColor("#4d9de0")
            painter.fillRect(
                QRectF(area.left() + index * bar_width, area.bottom() - height, max(1.0, bar_width - 1.0), height),
                color,
            )
        x_cutoff = area.left() + area.width() * (self.cutoff - minimum) / (maximum - minimum)
        painter.setPen(QPen(QColor("#ffd166"), 2))
        painter.drawLine(int(x_cutoff), area.top(), int(x_cutoff), area.bottom())
        painter.setPen(QColor("#d7dce2"))
        painter.drawText(area.left(), area.bottom() + 20, f"{minimum:.4g}")
        painter.drawText(area.right() - 70, area.bottom() + 20, 70, 20, Qt.AlignmentFlag.AlignRight, f"{maximum:.4g} µm³")


class SpineVolumeFilterDialog(QDialog):
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        cutoff_um3: float,
        enabled: bool,
        parent: QWidget | None = None,
        allow_disable: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spine-volume filter preview and overrides")
        self.resize(980, 720)
        self.rows = [
            dict(row)
            for row in rows
            if bool(row.get("manual_spine_valid", row.get("spine_valid", True)))
        ]
        self.force_keep = {
            (
                str(row.get("experimental_group", "")),
                str(row.get("specimen_id", "")),
                int(row.get("spine_id") or 0),
            )
            for row in self.rows
            if bool(row.get("volume_filter_force_keep", False))
        }
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.enabled = QCheckBox("Exclude spines whose volume is strictly below the cutoff")
        self.enabled.setChecked(enabled)
        self.enabled.setEnabled(allow_disable)
        form.addRow(self.enabled)
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0.0, 1_000_000.0)
        self.cutoff.setDecimals(6)
        self.cutoff.setValue(max(0.0, cutoff_um3))
        self.cutoff.setSuffix(" µm³")
        form.addRow("Volume cutoff:", self.cutoff)
        layout.addLayout(form)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.histogram = SpineVolumeHistogram()
        layout.addWidget(self.histogram)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Group", "Specimen", "ROI", "Dendrite", "Spine", "Volume µm³", "Clusters", "Keep"]
        )
        self.table.setSortingEnabled(True)
        self.table.itemChanged.connect(self._item_changed)
        layout.addWidget(self.table, 1)
        note = QLabel(
            "Keep overrides only the volume rule. It never restores a spine marked manually invalid. "
            "Equality with the cutoff is retained."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.cutoff.valueChanged.connect(self._refresh)
        self.enabled.toggled.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args) -> None:  # type: ignore[no-untyped-def]
        cutoff = self.cutoff.value()
        active = self.enabled.isChecked()
        below = [row for row in self.rows if float(row.get("volume_um3") or 0.0) < cutoff]
        removed = [
            row
            for row in below
            if (
                str(row.get("experimental_group", "")),
                str(row.get("specimen_id", "")),
                int(row.get("spine_id") or 0),
            )
            not in self.force_keep
        ] if active else []
        cluster_positive = sum(bool(row.get("has_protein_cluster", False)) for row in removed)
        kept_below = len(below) - len(removed) if active else 0
        by_group: dict[str, int] = {}
        by_specimen: dict[str, int] = {}
        specimen_totals: dict[str, int] = {}
        for row in self.rows:
            label = (
                f"{row.get('experimental_group', '')}/{row.get('specimen_id', '')}"
            )
            specimen_totals[label] = specimen_totals.get(label, 0) + 1
        for row in removed:
            group = str(row.get("experimental_group", ""))
            specimen = str(row.get("specimen_id", ""))
            by_group[group] = by_group.get(group, 0) + 1
            label = f"{group}/{specimen}"
            by_specimen[label] = by_specimen.get(label, 0) + 1
        retained = len(self.rows) - len(removed)
        warning = " WARNING: no valid spines would remain." if self.rows and retained == 0 else ""
        emptied = sorted(
            label
            for label, total in specimen_totals.items()
            if by_specimen.get(label, 0) == total
        )
        specimen_warning = (
            " WARNING: these specimens would have zero valid spines: "
            + ", ".join(emptied)
            + "."
            if emptied
            else ""
        )
        groups = ", ".join(f"{key}: {value}" for key, value in sorted(by_group.items())) or "none"
        specimens = ", ".join(f"{key}: {value}" for key, value in sorted(by_specimen.items())) or "none"
        percent = 100.0 * len(removed) / len(self.rows) if self.rows else 0.0
        self.summary.setText(
            f"Would exclude {len(removed)} of {len(self.rows)} manually valid spines ({percent:.1f}%); "
            f"{cluster_positive} are cluster-positive; {kept_below} below-cutoff spine(s) have a Keep override. "
            f"Remaining: {retained}.{warning}{specimen_warning}\n"
            f"By group: {groups}\nBy specimen: {specimens}"
        )
        self.histogram.set_data(
            [float(row.get("volume_um3") or 0.0) for row in self.rows], cutoff
        )
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(below))
        for row_index, row in enumerate(below):
            key = (
                str(row.get("experimental_group", "")),
                str(row.get("specimen_id", "")),
                int(row.get("spine_id") or 0),
            )
            values = [
                key[0], key[1], row.get("roi_id", ""), row.get("dendrite_id", ""), key[2],
                float(row.get("volume_um3") or 0.0),
                row.get(
                    "included_cluster_count",
                    int(bool(row["has_protein_cluster"]))
                    if "has_protein_cluster" in row
                    else "unknown",
                ),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in {2, 3, 4, 5, 6}:
                    item.setData(Qt.ItemDataRole.EditRole, value)
                self.table.setItem(row_index, column, item)
            keep = QTableWidgetItem()
            keep.setFlags(keep.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            keep.setCheckState(Qt.CheckState.Checked if key in self.force_keep else Qt.CheckState.Unchecked)
            keep.setData(Qt.ItemDataRole.UserRole, key)
            self.table.setItem(row_index, 7, keep)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 7:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(key, tuple):
            return
        if item.checkState() == Qt.CheckState.Checked:
            self.force_keep.add(key)
        else:
            self.force_keep.discard(key)
        self._refresh()

    def values(self) -> tuple[bool, float, set[tuple[str, str, int]]]:
        return self.enabled.isChecked(), self.cutoff.value(), set(self.force_keep)


class WorkbookVolumeFilterWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        source: Path,
        output: Path,
        cutoff_um3: float,
        force_keep: set[tuple[str, str, int]],
    ) -> None:
        super().__init__()
        self.source = source
        self.output = output
        self.cutoff_um3 = cutoff_um3
        self.force_keep = force_keep

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Filtering workbook", 0, 1, self.source.name)
            result = filter_exported_measurement_workbook(
                self.source,
                self.output,
                cutoff_um3=self.cutoff_um3,
                force_keep_keys=self.force_keep,
            )
            self.progress.emit("Filtering workbook", 1, 1, self.output.name)
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
        diagnostic: DiagnosticCallback | None = None,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.specimen_index = specimen_index
        self.action = action
        self.diagnostic = diagnostic

    @Slot()
    def run(self) -> None:
        started = time.monotonic()
        operation = self.action.operation if self.action is not None else "undo"
        if self.diagnostic is not None:
            self.diagnostic(
                "correction_started",
                "correction",
                {
                    "specimen_index": self.specimen_index,
                    "operation": operation,
                    "object_type": (
                        self.action.object_type if self.action is not None else None
                    ),
                    "projection_hint": (
                        self.action.projection_hint if self.action is not None else None
                    ),
                    "brush_radius_pixels": (
                        self.action.brush_radius_pixels
                        if self.action is not None
                        else None
                    ),
                },
            )
        try:
            if self.action is None:
                result = undo_last_review_action(
                    self.manifest,
                    self.project_path,
                    self.specimen_index,
                    diagnostic=self.diagnostic,
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
                    diagnostic=self.diagnostic,
                )
        except Exception as exc:
            if self.diagnostic is not None:
                self.diagnostic(
                    "correction_finished",
                    "correction",
                    {
                        "specimen_index": self.specimen_index,
                        "operation": operation,
                        "status": "failed",
                        "duration_seconds": round(time.monotonic() - started, 6),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            self.failed.emit(str(exc))
            return
        if self.diagnostic is not None:
            self.diagnostic(
                "correction_finished",
                "correction",
                {
                    "specimen_index": self.specimen_index,
                    "operation": operation,
                    "status": "completed",
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "processing_mode": result.processing_mode,
                    "checkpoint_written": result.checkpoint_written,
                },
            )
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


class TransferCreateWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        manifest: dict[str, object],
        project_path: Path,
        archive_path: Path,
        mode: str,
        include_raw: bool,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.project_path = project_path
        self.archive_path = archive_path
        self.mode = mode
        self.include_raw = include_raw

    @Slot()
    def run(self) -> None:
        def report(phase: str, current: int, total: int, detail: str) -> None:
            units = 10_000
            scaled = min(units, int(current * units / max(1, total)))
            self.progress.emit(phase, scaled, units, detail)

        try:
            result = create_transfer_archive(
                self.manifest,
                self.project_path,
                self.archive_path,
                mode=self.mode,
                include_raw=self.include_raw,
                progress=report,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class TransferImportWorker(QObject):
    progress = Signal(str, int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        archive_path: Path,
        destination_parent: Path,
        external_raw_directory: Path | None,
        conflict_policy: str,
        recover_as_settings_only: bool = False,
    ) -> None:
        super().__init__()
        self.archive_path = archive_path
        self.destination_parent = destination_parent
        self.external_raw_directory = external_raw_directory
        self.conflict_policy = conflict_policy
        self.recover_as_settings_only = recover_as_settings_only

    @Slot()
    def run(self) -> None:
        def report(phase: str, current: int, total: int, detail: str) -> None:
            units = 10_000
            scaled = min(units, int(current * units / max(1, total)))
            self.progress.emit(phase, scaled, units, detail)

        try:
            result = import_transfer_archive(
                self.archive_path,
                self.destination_parent,
                external_raw_directory=self.external_raw_directory,
                conflict_policy=self.conflict_policy,
                recover_as_settings_only=self.recover_as_settings_only,
                progress=report,
            )
        except TransferCacheError as exc:
            self.completed.emit(exc)
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


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
        self._diagnostic_job_phase: str | None = None
        self._preview_statistics: dict[tuple[object, ...], StackStatistics] = {}
        self._last_preview: PreviewResult | None = None
        self._last_detection: DetectionSlice | None = None
        self._last_review: ReviewSlice | None = None
        self._last_review_context: ContextVolume | None = None
        self._review_projection_loading = False
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
        self._diagnostic_review_phase: str | None = None
        self._context_thread: QThread | None = None
        self._context_worker: ContextWorker | None = None
        self._context_request: tuple[object, ...] | None = None
        self._diagnostic_context_started_at = 0.0
        self._diagnostic_context_phase: str | None = None
        self._context_cache: dict[tuple[object, ...], ContextVolume] = {}
        self._context_dialogs: list[ContextViewerDialog] = []
        self._area_dialog: AreaSelectionDialog | None = None
        self._preview_requested_while_busy = False
        self._preprocess_view_specimen_key: tuple[int, int] | None = None
        self._detection_view_specimen_key: tuple[int, int] | None = None
        self._review_view_specimen_key: tuple[int, int] | None = None
        self._diagnostics: DiagnosticSession | None = None
        self._pending_transfer_recovery: tuple[Path, Path, Path | None, str] | None = None
        self._diagnostic_last_directory: Path | None = None
        self._calibration_store = CalibrationStore()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._request_preview)
        self._diagnostic_timer = QTimer(self)
        self._diagnostic_timer.setInterval(2000)
        self._diagnostic_timer.timeout.connect(self._sample_diagnostics)

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

        self.open_action = QAction("Open project or transfer ZIP…", self)
        self.open_action.triggered.connect(self._open_project)
        file_menu.addAction(self.open_action)

        self.filter_export_action = QAction(
            "Filter exported measurement workbook…", self
        )
        self.filter_export_action.triggered.connect(
            self._filter_exported_measurement_workbook
        )
        file_menu.addAction(self.filter_export_action)

        self.cluster_export_action = QAction(
            "Cluster exported morphology workbook…", self
        )
        self.cluster_export_action.triggered.connect(
            self._cluster_exported_morphology_workbook
        )
        file_menu.addAction(self.cluster_export_action)

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

        project_menu.addSeparator()
        self.create_transfer_action = QAction("Create transfer ZIP…", self)
        self.create_transfer_action.triggered.connect(self._create_transfer_zip)
        project_menu.addAction(self.create_transfer_action)

        advanced_menu = self.menuBar().addMenu("&Advanced")
        self.diagnostic_mode_action = QAction("Diagnostic mode", self)
        self.diagnostic_mode_action.setCheckable(True)
        self.diagnostic_mode_action.setToolTip(
            "Record privacy-conscious performance timings and system samples."
        )
        self.diagnostic_mode_action.toggled.connect(self._toggle_diagnostic_mode)
        advanced_menu.addAction(self.diagnostic_mode_action)

        diagnostic_level_menu = advanced_menu.addMenu("Diagnostic level")
        self.diagnostic_level_group = QActionGroup(self)
        self.diagnostic_level_group.setExclusive(True)
        self.diagnostic_correction_action = QAction(
            "Correction diagnostics only", self
        )
        self.diagnostic_correction_action.setCheckable(True)
        self.diagnostic_correction_action.setData(DIAGNOSTIC_LEVEL_CORRECTION)
        self.diagnostic_level_group.addAction(self.diagnostic_correction_action)
        diagnostic_level_menu.addAction(self.diagnostic_correction_action)
        self.diagnostic_full_action = QAction("Full-session diagnostics", self)
        self.diagnostic_full_action.setCheckable(True)
        self.diagnostic_full_action.setData(DIAGNOSTIC_LEVEL_FULL)
        self.diagnostic_full_action.setChecked(True)
        self.diagnostic_level_group.addAction(self.diagnostic_full_action)
        diagnostic_level_menu.addAction(self.diagnostic_full_action)
        self.diagnostic_level_menu = diagnostic_level_menu

        advanced_menu.addSeparator()
        self.finish_diagnostic_action = QAction(
            "Finish and package diagnostic session…", self
        )
        self.finish_diagnostic_action.setEnabled(False)
        self.finish_diagnostic_action.triggered.connect(
            self._finish_and_package_diagnostics
        )
        advanced_menu.addAction(self.finish_diagnostic_action)

        self.open_diagnostic_folder_action = QAction(
            "Open diagnostic folder", self
        )
        self.open_diagnostic_folder_action.setEnabled(False)
        self.open_diagnostic_folder_action.triggered.connect(
            self._open_diagnostic_folder
        )
        advanced_menu.addAction(self.open_diagnostic_folder_action)

        self.open_diagnostic_protocol_action = QAction(
            "Open diagnostic protocol PDF", self
        )
        self.open_diagnostic_protocol_action.triggered.connect(
            self._open_diagnostic_protocol
        )
        advanced_menu.addAction(self.open_diagnostic_protocol_action)

        view_menu = self.menuBar().addMenu("&View")
        self.fullscreen_action = QAction("Toggle full screen", self)
        self.fullscreen_action.setShortcut("F11")
        self.fullscreen_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(self.fullscreen_action)

    def _selected_diagnostic_level(self) -> str:
        checked = self.diagnostic_level_group.checkedAction()
        return str(checked.data()) if checked is not None else DIAGNOSTIC_LEVEL_FULL

    def _diagnostic_parent_directory(self) -> Path | None:
        if self.manifest is not None:
            return diagnostic_session_parent(self.manifest["output_directory"])
        output_text = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if output_text:
            return diagnostic_session_parent(output_text)
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select a folder for Synpo diagnostics",
            "",
        )
        return Path(selected).resolve() / "Synpo diagnostics" if selected else None

    @Slot(bool)
    def _toggle_diagnostic_mode(self, enabled: bool) -> None:
        if enabled:
            if self._diagnostics is not None and self._diagnostics.active:
                return
            parent = self._diagnostic_parent_directory()
            if parent is None:
                self.diagnostic_mode_action.blockSignals(True)
                self.diagnostic_mode_action.setChecked(False)
                self.diagnostic_mode_action.blockSignals(False)
                return
            try:
                session = DiagnosticSession.create(
                    parent, level=self._selected_diagnostic_level()
                )
            except (OSError, ValueError) as exc:
                self.diagnostic_mode_action.blockSignals(True)
                self.diagnostic_mode_action.setChecked(False)
                self.diagnostic_mode_action.blockSignals(False)
                QMessageBox.warning(
                    self, "Cannot start diagnostics", str(exc)
                )
                return
            self._diagnostics = session
            self._diagnostic_last_directory = session.directory
            self.diagnostic_level_menu.setEnabled(False)
            self.finish_diagnostic_action.setEnabled(True)
            self.open_diagnostic_folder_action.setEnabled(True)
            self._diagnostic_timer.start()
            self._sample_diagnostics()
            self._record_diagnostic_project_context()
            self.statusBar().showMessage(
                f"Diagnostic mode started: {session.directory}", 10000
            )
            return

        if self._diagnostics is None:
            return
        if self._operation_in_progress():
            self.diagnostic_mode_action.blockSignals(True)
            self.diagnostic_mode_action.setChecked(True)
            self.diagnostic_mode_action.blockSignals(False)
            QMessageBox.information(
                self,
                "Diagnostic operation in progress",
                "Wait for the current operation to finish before stopping diagnostics, "
                "so its final timing records are preserved.",
            )
            return
        self._finish_diagnostics(package=False, reason="diagnostic_mode_disabled")

    def _operation_in_progress(self) -> bool:
        return any(
            worker is not None
            for worker in (
                self._job_thread,
                self._review_thread,
                self._context_thread,
            )
        )

    def _record_diagnostic(
        self, event: str, *, scope: str = "full", **details: object
    ) -> None:
        if self._diagnostics is not None:
            self._diagnostics.record(event, scope=scope, **details)

    def _diagnostic_callback(self) -> DiagnosticCallback | None:
        if self._diagnostics is None or not self._diagnostics.active:
            return None
        return self._diagnostics.callback

    def _sample_diagnostics(self) -> None:
        if self._diagnostics is not None:
            self._diagnostics.sample_system()

    def _record_diagnostic_project_context(self) -> None:
        if self._diagnostics is None or self.manifest is None:
            return
        try:
            cache_path: Path | None = project_cache_path(self.manifest)
        except (KeyError, TypeError, ValueError):
            cache_path = None
        self._diagnostics.record_project_context(
            output_directory=self.manifest.get("output_directory"),
            cache_path=cache_path,
            specimen_count=len(self.manifest.get("specimens", [])),
        )

    def _finish_diagnostics(self, *, package: bool, reason: str) -> Path | None:
        session = self._diagnostics
        if session is None:
            return None
        self._diagnostic_timer.stop()
        try:
            result = session.package(reason=reason) if package else session.finish(reason=reason)
        except OSError as exc:
            QMessageBox.warning(self, "Cannot finish diagnostics", str(exc))
            return None
        self._diagnostic_last_directory = session.directory
        self._diagnostics = None
        self.diagnostic_mode_action.blockSignals(True)
        self.diagnostic_mode_action.setChecked(False)
        self.diagnostic_mode_action.blockSignals(False)
        self.diagnostic_level_menu.setEnabled(True)
        self.finish_diagnostic_action.setEnabled(False)
        self.open_diagnostic_folder_action.setEnabled(True)
        self.statusBar().showMessage(
            f"Diagnostic {'bundle' if package else 'session'} saved: {result}", 12000
        )
        return result

    def _finish_and_package_diagnostics(self) -> None:
        if self._operation_in_progress():
            QMessageBox.information(
                self,
                "Diagnostic operation in progress",
                "Wait for the current operation to finish before packaging diagnostics.",
            )
            return
        result = self._finish_diagnostics(
            package=True, reason="user_finished_and_packaged"
        )
        if result is not None:
            QMessageBox.information(
                self,
                "Diagnostic bundle ready",
                f"The diagnostic ZIP was saved to:\n{result}",
            )

    def _open_diagnostic_folder(self) -> None:
        directory = (
            self._diagnostics.directory
            if self._diagnostics is not None
            else self._diagnostic_last_directory
        )
        if directory is None or not directory.is_dir():
            QMessageBox.information(
                self, "No diagnostic folder", "No diagnostic session folder is available."
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _open_diagnostic_protocol(self) -> None:
        protocol = diagnostic_protocol_path()
        if not protocol.is_file():
            QMessageBox.warning(
                self,
                "Diagnostic protocol missing",
                f"The bundled diagnostic protocol was not found:\n{protocol}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(protocol)))

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
        outer.addLayout(progress_row)

        self.tabs.addTab(setup_tab, "1. Batch setup")
        self._build_preprocessing_tab()
        self._build_detection_tab()
        self._build_review_tab()
        self._build_measurements_tab()
        self._build_morphology_tab()
        self._build_advanced_clustering_tab()
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)
        self.tabs.setTabEnabled(5, False)
        self.tabs.setTabEnabled(6, False)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self.save_button = QPushButton("Save project…")
        self.save_button.setToolTip(
            "Save the project from any workflow step. Unapplied processing and correction controls are not committed."
        )
        self.save_button.clicked.connect(self._save_project)
        save_row.addWidget(self.save_button)
        central_layout.addLayout(save_row)
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

        selection = QGroupBox("Specimen parameter review")
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
        self.fill_unset_preprocessing_check = QCheckBox(
            "Make all unset images in this channel use these parameters"
        )
        self.fill_unset_preprocessing_check.setToolTip(
            "Copies the current values to this channel only for included specimens whose "
            "parameters have not been set. Existing settings are never overwritten."
        )
        selection_layout.addWidget(self.fill_unset_preprocessing_check)
        outer.addWidget(selection)

        analysis_row = QHBoxLayout()
        self.exclude_specimen_check = QCheckBox("Exclude this specimen pair from analysis")
        self.exclude_specimen_check.toggled.connect(self._preprocess_exclusion_changed)
        analysis_row.addWidget(self.exclude_specimen_check)
        self.exclusion_reason = QLineEdit()
        self.exclusion_reason.setPlaceholderText("Optional exclusion reason")
        self.exclusion_reason.editingFinished.connect(self._save_exclusion_reason)
        analysis_row.addWidget(self.exclusion_reason, 1)
        self.manage_rois_button = QPushButton("Manage analysis ROIs…")
        self.manage_rois_button.clicked.connect(self._manage_analysis_rois)
        analysis_row.addWidget(self.manage_rois_button)
        self.roi_status = QLabel("Full image")
        analysis_row.addWidget(self.roi_status)
        outer.addLayout(analysis_row)

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
        self.z_slider = AbsoluteSlider(Qt.Orientation.Horizontal)
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
        self.raw_view.navigate_requested.connect(self._move_preprocess_specimen)
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
        self.processed_view.navigate_requested.connect(self._move_preprocess_specimen)
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
            "Save or open a project, mark both channels for each included specimen, then preprocess the batch."
        )
        self.preprocessing_status.setWordWrap(True)
        batch_row.addWidget(self.preprocessing_status, 1)
        self.run_preprocessing_button = QPushButton(
            "Preprocess entire batch / resume"
        )
        self.run_preprocessing_button.clicked.connect(self._run_batch_preprocessing)
        batch_row.addWidget(self.run_preprocessing_button)
        self.run_selected_preprocessing_button = QPushButton(
            "Preprocess selected specimen"
        )
        self.run_selected_preprocessing_button.clicked.connect(
            self._run_selected_preprocessing
        )
        batch_row.addWidget(self.run_selected_preprocessing_button)
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
        self.apply_detection_button = QPushButton("Save settings for selected specimen")
        self.apply_detection_button.setMinimumHeight(34)
        self.apply_detection_button.clicked.connect(self._apply_detection_settings)
        settings_layout.addRow(self.apply_detection_button)
        self.apply_detection_defaults_button = QPushButton("Save as batch defaults")
        self.apply_detection_defaults_button.clicked.connect(
            self._apply_detection_default_settings
        )
        settings_layout.addRow(self.apply_detection_defaults_button)
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
        self.run_selected_detection_button = QPushButton(
            "Redo detection for selected specimen"
        )
        self.run_selected_detection_button.clicked.connect(
            lambda: self._run_selected_detection(None, force=True)
        )
        results_layout.addWidget(self.run_selected_detection_button)
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
        reprocess_buttons = QHBoxLayout()
        self.review_reprocess_preprocessing_button = QPushButton(
            "Edit preprocessing and rerun…"
        )
        self.review_reprocess_preprocessing_button.clicked.connect(
            self._review_edit_preprocessing
        )
        reprocess_buttons.addWidget(self.review_reprocess_preprocessing_button)
        self.review_redetect_button = QPushButton("Edit detection and rerun…")
        self.review_redetect_button.clicked.connect(self._review_edit_detection)
        reprocess_buttons.addWidget(self.review_redetect_button)
        queue_layout.addLayout(reprocess_buttons)
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
        self.review_distinct_object_colors = QCheckBox(
            "Distinct colors for individual dendrites and spines"
        )
        self.review_distinct_object_colors.setToolTip(
            "Assign stable contrasting colors to object IDs so touching borders are visible."
        )
        self.review_distinct_object_colors.toggled.connect(self._render_review_view)
        display_form.addRow(self.review_distinct_object_colors)
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
            ("Erase painted dendrite and spine mask", "erase"),
            ("Assign painted dendrite area to spine", "dendrite_to_spine"),
            ("Assign painted spine area to dendrite", "spine_to_dendrite"),
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
        self.review_brush_diameter = QSpinBox()
        self.review_brush_diameter.setRange(1, 1000)
        self.review_brush_diameter.setValue(9)
        self.review_brush_diameter.setSuffix(" px")
        self.review_brush_diameter.setToolTip(
            "Displayed brush width. A 1000 px maximum is available for clearing large debris fields."
        )
        self.review_brush_diameter.valueChanged.connect(
            self._review_brush_changed
        )
        correction_form.addRow("Hint brush diameter:", self.review_brush_diameter)
        self.review_sensitivity_label = QLabel("Resegmentation sensitivity:")
        self.review_sensitivity_widget = QWidget()
        sensitivity_layout = QHBoxLayout(self.review_sensitivity_widget)
        sensitivity_layout.setContentsMargins(0, 0, 0, 0)
        self.review_sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.review_sensitivity.setRange(25, 1000)
        self.review_sensitivity.setValue(100)
        self.review_sensitivity.setSingleStep(5)
        self.review_sensitivity.setPageStep(25)
        self.review_sensitivity.valueChanged.connect(
            self._review_sensitivity_changed
        )
        sensitivity_layout.addWidget(self.review_sensitivity, 1)
        self.review_sensitivity_value = QLabel("1.00")
        self.review_sensitivity_value.setMinimumWidth(38)
        sensitivity_layout.addWidget(self.review_sensitivity_value)
        correction_form.addRow(
            self.review_sensitivity_label, self.review_sensitivity_widget
        )
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
        self.apply_review_button = QPushButton("Apply correction")
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
        shortcut_map = {
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
        self.review_shortcut_actions = []
        for shortcut, operation in shortcut_map.items():
            shortcut_action = QAction(f"Select {operation}", tab)
            shortcut_action.setShortcut(shortcut)
            shortcut_action.setShortcutContext(
                Qt.ShortcutContext.WidgetWithChildrenShortcut
            )
            shortcut_action.triggered.connect(
                lambda _checked=False, value=operation: self._select_review_operation(value)
            )
            tab.addAction(shortcut_action)
            self.review_shortcut_actions.append(shortcut_action)
        apply_action = QAction("Apply correction", tab)
        apply_action.setShortcut("Return")
        apply_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        apply_action.triggered.connect(self._apply_review_shortcut)
        tab.addAction(apply_action)
        self.review_shortcut_actions.append(apply_action)
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
        self.maximum_centerline_gap = QDoubleSpinBox()
        self.maximum_centerline_gap.setRange(0.0, 10.0)
        self.maximum_centerline_gap.setDecimals(2)
        self.maximum_centerline_gap.setSingleStep(0.1)
        self.maximum_centerline_gap.setValue(1.0)
        self.maximum_centerline_gap.setSuffix(" µm")
        self.maximum_centerline_gap.setToolTip(
            "Maximum signal-guided virtual bridge for a spine split into exactly two components."
        )
        settings_form.addRow("Maximum centerline gap:", self.maximum_centerline_gap)
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
        self.accept_all_distribution_spines = QCheckBox(
            "Accept all eligible spines in all measured specimens"
        )
        self.accept_all_distribution_spines.setToolTip(
            "Marks every non-invalidated, non-volume-filtered spine with a usable "
            "distribution path as reviewed and included. Existing review notes are preserved."
        )
        self.accept_all_distribution_spines.toggled.connect(
            self._accept_all_distribution_spines
        )
        distribution_form.addRow(self.accept_all_distribution_spines)
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

        volume_filter_group = QGroupBox("Spine-volume export filter")
        volume_filter_form = QFormLayout(volume_filter_group)
        self.volume_filter_enabled = QCheckBox(
            "Exclude spines below the volume cutoff"
        )
        self.volume_filter_enabled.toggled.connect(
            self._save_volume_filter_controls
        )
        volume_filter_form.addRow(self.volume_filter_enabled)
        self.volume_filter_cutoff = QDoubleSpinBox()
        self.volume_filter_cutoff.setRange(0.0, 1_000_000.0)
        self.volume_filter_cutoff.setDecimals(6)
        self.volume_filter_cutoff.setSuffix(" µm³")
        self.volume_filter_cutoff.editingFinished.connect(
            self._save_volume_filter_controls
        )
        volume_filter_form.addRow("Cutoff:", self.volume_filter_cutoff)
        self.volume_filter_preview_button = QPushButton(
            "Preview distribution and set overrides…"
        )
        self.volume_filter_preview_button.clicked.connect(
            self._open_volume_filter_preview
        )
        volume_filter_form.addRow(self.volume_filter_preview_button)
        side_layout.addWidget(volume_filter_group)

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

    def _build_morphology_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(390)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        geometry = QGroupBox("All-spine geometry review")
        self.morphology_geometry_group = geometry
        geometry_form = QFormLayout(geometry)
        self.morphology_specimen = QComboBox()
        self.morphology_specimen.currentIndexChanged.connect(self._morphology_specimen_changed)
        geometry_form.addRow("Specimen:", self.morphology_specimen)
        self.morphology_spine = QComboBox()
        self.morphology_spine.currentIndexChanged.connect(self._morphology_spine_changed)
        geometry_form.addRow("Spine:", self.morphology_spine)
        self.morphology_view_mode = QComboBox()
        self.morphology_view_mode.addItem("Maximum projection", "maximum")
        self.morphology_view_mode.addItem("Single Z slice", "slice")
        self.morphology_view_mode.currentIndexChanged.connect(self._render_morphology_preview)
        geometry_form.addRow("View:", self.morphology_view_mode)
        self.morphology_z = AbsoluteSlider(Qt.Orientation.Horizontal)
        self.morphology_z.valueChanged.connect(self._render_morphology_preview)
        geometry_form.addRow("Z slice:", self.morphology_z)
        self.morphology_tool = QComboBox()
        self.morphology_tool.addItem("Set shaft-contact base", "set_base")
        self.morphology_tool.addItem("Set distal tip", "set_tip")
        self.morphology_tool.addItem("Paint as head", "paint_head")
        self.morphology_tool.addItem("Paint as neck", "paint_neck")
        self.morphology_tool.currentIndexChanged.connect(self._morphology_tool_changed)
        geometry_form.addRow("Tool:", self.morphology_tool)
        self.morphology_brush = QSpinBox()
        self.morphology_brush.setRange(1, 101)
        self.morphology_brush.setSingleStep(2)
        self.morphology_brush.setValue(7)
        self.morphology_brush.valueChanged.connect(lambda value: self.morphology_canvas.set_brush_diameter(value))
        geometry_form.addRow("Brush diameter:", self.morphology_brush)
        self.morphology_z_radius = QSpinBox()
        self.morphology_z_radius.setRange(0, 20)
        geometry_form.addRow("Slice brush Z radius:", self.morphology_z_radius)
        self.morphology_reviewed = QCheckBox("Geometry checked")
        geometry_form.addRow(self.morphology_reviewed)
        self.morphology_invalid = QCheckBox("Invalid spine — exclude from all metrics")
        geometry_form.addRow(self.morphology_invalid)
        self.morphology_note = QLineEdit()
        geometry_form.addRow("Review note:", self.morphology_note)
        apply_button = QPushButton("Apply geometry edit")
        apply_button.clicked.connect(self._apply_morphology_edit)
        geometry_form.addRow(apply_button)
        undo_row = QHBoxLayout()
        self.morphology_undo = QPushButton("Undo")
        self.morphology_undo.clicked.connect(lambda: self._undo_redo_morphology(False))
        self.morphology_redo = QPushButton("Redo")
        self.morphology_redo.clicked.connect(lambda: self._undo_redo_morphology(True))
        undo_row.addWidget(self.morphology_undo)
        undo_row.addWidget(self.morphology_redo)
        geometry_form.addRow(undo_row)
        reset_row = QHBoxLayout()
        reset_border = QPushButton("Reset head/neck")
        reset_border.clicked.connect(lambda: self._reset_morphology("reset_border"))
        reset_anchors = QPushButton("Reset anchors")
        reset_anchors.clicked.connect(lambda: self._reset_morphology("reset_anchors"))
        reset_row.addWidget(reset_border)
        reset_row.addWidget(reset_anchors)
        geometry_form.addRow(reset_row)
        checkpoint = QPushButton("Checkpoint review and advance")
        checkpoint.clicked.connect(self._checkpoint_morphology_review)
        geometry_form.addRow(checkpoint)
        controls_layout.addWidget(geometry)

        clustering = QGroupBox("Named morphology clustering run")
        clustering_form = QFormLayout(clustering)
        self.morphology_saved_run = QComboBox()
        self.morphology_saved_run.currentIndexChanged.connect(self._morphology_saved_run_changed)
        clustering_form.addRow("Saved run:", self.morphology_saved_run)
        self.morphology_run_name = QLineEdit("Default morphology analysis")
        clustering_form.addRow("Run name:", self.morphology_run_name)
        self.morphology_algorithm = QComboBox()
        self.morphology_algorithm.addItem("Gaussian mixture", "gaussian_mixture")
        self.morphology_algorithm.addItem("Ward hierarchical", "ward")
        self.morphology_algorithm.addItem("K-means", "kmeans")
        clustering_form.addRow("Algorithm:", self.morphology_algorithm)
        self.morphology_features: dict[str, QCheckBox] = {}
        feature_widget = QWidget()
        feature_layout = QVBoxLayout(feature_widget)
        feature_layout.setContentsMargins(0, 0, 0, 0)
        for key, (label, _column) in MORPHOLOGY_FEATURES.items():
            checkbox = QCheckBox(label)
            checkbox.setChecked(key in DEFAULT_FEATURES)
            self.morphology_features[key] = checkbox
            feature_layout.addWidget(checkbox)
        clustering_form.addRow("Morphology features:", feature_widget)
        protein_note = QLabel("Protein puncta are never clustering or PCA inputs; they appear only in plots and descriptive summaries.")
        protein_note.setWordWrap(True)
        clustering_form.addRow(protein_note)
        self.morphology_groups = QListWidget()
        self.morphology_groups.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.morphology_groups.setMaximumHeight(115)
        self.morphology_groups.setToolTip(
            "Select one or more experimental groups to include in this clustering run."
        )
        clustering_form.addRow("Experimental groups:", self.morphology_groups)
        self.morphology_reviewed_only = QCheckBox(
            "Cluster geometry-reviewed spines only"
        )
        self.morphology_reviewed_only.setToolTip(
            "When checked, only spines marked Geometry checked are included in PCA and clustering. "
            "Unchecked includes every otherwise valid spine."
        )
        clustering_form.addRow(self.morphology_reviewed_only)
        self.morphology_pca_dimensions = QComboBox()
        self.morphology_pca_dimensions.addItem("2D", 2)
        self.morphology_pca_dimensions.addItem("3D", 3)
        clustering_form.addRow("PCA plot:", self.morphology_pca_dimensions)
        self.morphology_use_pca = QCheckBox("Cluster in PCA space")
        self.morphology_use_pca.setChecked(True)
        clustering_form.addRow(self.morphology_use_pca)
        cluster_range = QHBoxLayout()
        self.morphology_min_clusters = QSpinBox()
        self.morphology_min_clusters.setRange(1, 10)
        self.morphology_min_clusters.setValue(1)
        self.morphology_max_clusters = QSpinBox()
        self.morphology_max_clusters.setRange(1, 10)
        self.morphology_max_clusters.setValue(6)
        cluster_range.addWidget(QLabel("Min"))
        cluster_range.addWidget(self.morphology_min_clusters)
        cluster_range.addWidget(QLabel("Max"))
        cluster_range.addWidget(self.morphology_max_clusters)
        clustering_form.addRow("Candidate clusters:", cluster_range)
        self.morphology_cluster_count_selection = QComboBox()
        self.morphology_cluster_count_selection.addItem(
            "Information criterion (BIC / penalized SSE)",
            "information_criterion",
        )
        self.morphology_cluster_count_selection.addItem(
            "Maximum silhouette score", "silhouette"
        )
        self.morphology_cluster_count_selection.addItem(
            "Elbow of within-cluster SSE", "elbow"
        )
        self.morphology_cluster_count_selection.setToolTip(
            "Used only when Fixed count is Automatic. Elbow requires at least "
            "three accepted candidate cluster counts."
        )
        clustering_form.addRow(
            "Automatic selection:", self.morphology_cluster_count_selection
        )
        self.morphology_fixed_clusters = QSpinBox()
        self.morphology_fixed_clusters.setRange(0, 10)
        self.morphology_fixed_clusters.setSpecialValueText("Automatic")
        self.morphology_fixed_clusters.valueChanged.connect(
            self._cluster_count_controls_changed
        )
        clustering_form.addRow("Fixed count:", self.morphology_fixed_clusters)
        self.morphology_scaling = QComboBox()
        self.morphology_scaling.addItem("Robust median / IQR", "robust")
        self.morphology_scaling.addItem("Z-score", "zscore")
        self.morphology_scaling.addItem("None", "none")
        clustering_form.addRow("Scaling:", self.morphology_scaling)
        self.morphology_seed = QSpinBox()
        self.morphology_seed.setRange(0, 2_000_000_000)
        self.morphology_seed.setValue(42)
        clustering_form.addRow("Random seed:", self.morphology_seed)
        self.morphology_min_cluster_spines = QSpinBox()
        self.morphology_min_cluster_spines.setRange(2, 1_000_000)
        self.morphology_min_cluster_spines.setValue(10)
        clustering_form.addRow("Minimum spines/cluster:", self.morphology_min_cluster_spines)
        self.morphology_min_cluster_fraction = QDoubleSpinBox()
        self.morphology_min_cluster_fraction.setRange(0.0, 50.0)
        self.morphology_min_cluster_fraction.setDecimals(1)
        self.morphology_min_cluster_fraction.setValue(5.0)
        self.morphology_min_cluster_fraction.setSuffix(" %")
        clustering_form.addRow("Minimum cluster fraction:", self.morphology_min_cluster_fraction)
        run_button = QPushButton("Run / replace named analysis")
        run_button.clicked.connect(self._run_morphology_clustering)
        self.run_morphology_button = run_button
        clustering_form.addRow(run_button)
        color_button = QPushButton("Change cluster colors…")
        color_button.clicked.connect(self._change_morphology_colors)
        self.morphology_color_button = color_button
        clustering_form.addRow(color_button)
        self.morphology_axes_color = QColor(DEFAULT_PLOT_STYLE["axes_color"])
        axes_color_button = QPushButton("Change axes color…")
        axes_color_button.clicked.connect(
            lambda: self._change_morphology_plot_color("axes")
        )
        self.morphology_axes_color_button = axes_color_button
        clustering_form.addRow(axes_color_button)
        self.morphology_axes_alpha = QDoubleSpinBox()
        self.morphology_axes_alpha.setRange(0.0, 1.0)
        self.morphology_axes_alpha.setSingleStep(0.05)
        self.morphology_axes_alpha.setValue(1.0)
        self.morphology_axes_alpha.editingFinished.connect(
            self._save_morphology_plot_style
        )
        clustering_form.addRow("Axes alpha:", self.morphology_axes_alpha)
        self.morphology_background_color = QColor(DEFAULT_PLOT_STYLE["background_color"])
        background_button = QPushButton("Change plot background…")
        background_button.clicked.connect(
            lambda: self._change_morphology_plot_color("background")
        )
        self.morphology_background_color_button = background_button
        clustering_form.addRow(background_button)
        self.morphology_background_alpha = QDoubleSpinBox()
        self.morphology_background_alpha.setRange(0.0, 1.0)
        self.morphology_background_alpha.setSingleStep(0.05)
        self.morphology_background_alpha.setValue(1.0)
        self.morphology_background_alpha.editingFinished.connect(
            self._save_morphology_plot_style
        )
        clustering_form.addRow("Background alpha:", self.morphology_background_alpha)
        export_button = QPushButton("Export separate analysis package…")
        export_button.clicked.connect(self._export_morphology_run)
        self.export_morphology_button = export_button
        clustering_form.addRow(export_button)
        controls_layout.addWidget(clustering)
        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        output = QWidget()
        output_layout = QVBoxLayout(output)
        self.morphology_status = QLabel("Calculate measurements before reviewing all-spine geometry.")
        self.morphology_status.setWordWrap(True)
        output_layout.addWidget(self.morphology_status)
        self.morphology_display_tabs = QTabWidget()
        geometry_page = QWidget()
        geometry_layout = QVBoxLayout(geometry_page)
        self.morphology_canvas = MorphologyReviewCanvas("Select a measured spine")
        self.morphology_canvas.navigate_requested.connect(self._move_morphology_spine)
        geometry_layout.addWidget(self.morphology_canvas, 1)
        geometry_layout.addWidget(ZoomControls(self.morphology_canvas))
        self.morphology_metrics = QLabel("")
        self.morphology_metrics.setWordWrap(True)
        geometry_layout.addWidget(self.morphology_metrics)
        self.morphology_display_tabs.addTab(geometry_page, "Geometry review")
        plots_page = QWidget()
        plots_layout = QVBoxLayout(plots_page)
        self.morphology_plot_mode = QComboBox()
        self.morphology_plot_mode.addItem("Volume vs curvilinear length", "volume_length")
        self.morphology_plot_mode.addItem("Volume vs base-to-tip distance", "volume_straight")
        self.morphology_plot_mode.addItem("Selected morphology features", "custom_features")
        self.morphology_plot_mode.addItem("PCA", "pca")
        self.morphology_plot_mode.addItem("3D PCA with feature axes", "pca_3d_features")
        self.morphology_plot_mode.addItem("PCA interpretation", "pca_interpretation")
        self.morphology_plot_mode.addItem("Protein-positive fraction", "protein_positive")
        self.morphology_plot_mode.addItem("Protein puncta volume", "protein_volume")
        self.morphology_plot_mode.addItem("Protein position profiles", "protein_position")
        self.morphology_plot_mode.addItem("Group cluster proportions", "group_proportions")
        self.morphology_plot_mode.addItem("Cluster profile heatmap", "cluster_heatmap")
        self.morphology_plot_mode.currentIndexChanged.connect(
            self._morphology_plot_mode_changed
        )
        plots_layout.addWidget(self.morphology_plot_mode)
        feature_plot_row = QHBoxLayout()
        feature_plot_row.addWidget(QLabel("X:"))
        self.morphology_custom_x = QComboBox()
        feature_plot_row.addWidget(self.morphology_custom_x, 1)
        feature_plot_row.addWidget(QLabel("Y:"))
        self.morphology_custom_y = QComboBox()
        feature_plot_row.addWidget(self.morphology_custom_y, 1)
        self.morphology_custom_x.currentIndexChanged.connect(
            self._save_morphology_plot_style
        )
        self.morphology_custom_y.currentIndexChanged.connect(
            self._save_morphology_plot_style
        )
        self.morphology_feature_plot_row = QWidget()
        self.morphology_feature_plot_row.setLayout(feature_plot_row)
        plots_layout.addWidget(self.morphology_feature_plot_row)
        pca_point_row = QHBoxLayout()
        self.morphology_pca_show_points = QCheckBox("Show spine points")
        self.morphology_pca_show_points.setChecked(True)
        self.morphology_pca_show_points.setToolTip(
            "Hide the PCA score points to view the morphology-feature vectors alone."
        )
        self.morphology_pca_show_points.toggled.connect(
            self._save_morphology_plot_style
        )
        pca_point_row.addWidget(self.morphology_pca_show_points)
        pca_point_row.addWidget(QLabel("Point opacity:"))
        self.morphology_pca_point_alpha = QDoubleSpinBox()
        self.morphology_pca_point_alpha.setRange(0.0, 1.0)
        self.morphology_pca_point_alpha.setDecimals(2)
        self.morphology_pca_point_alpha.setSingleStep(0.05)
        self.morphology_pca_point_alpha.setValue(
            float(DEFAULT_PLOT_STYLE["pca_point_alpha"])
        )
        self.morphology_pca_point_alpha.setToolTip(
            "Opacity of spine points in the PCA feature-vector views. "
            "Feature arrows remain fully opaque."
        )
        self.morphology_pca_point_alpha.editingFinished.connect(
            self._save_morphology_plot_style
        )
        pca_point_row.addWidget(self.morphology_pca_point_alpha)
        pca_point_row.addStretch(1)
        self.morphology_pca_point_row = QWidget()
        self.morphology_pca_point_row.setLayout(pca_point_row)
        plots_layout.addWidget(self.morphology_pca_point_row)
        legend_row = QHBoxLayout()
        self.morphology_show_legend = QCheckBox("Show legend")
        self.morphology_show_legend.setChecked(True)
        self.morphology_show_legend.toggled.connect(
            self._save_morphology_plot_style
        )
        legend_row.addWidget(self.morphology_show_legend)
        legend_row.addWidget(QLabel("Position:"))
        self.morphology_legend_position = QComboBox()
        self.morphology_legend_position.addItem("Outside right", "outside_right")
        self.morphology_legend_position.addItem("Below plot", "outside_bottom")
        self.morphology_legend_position.addItem("Inside plot", "inside")
        self.morphology_legend_position.currentIndexChanged.connect(
            self._save_morphology_plot_style
        )
        legend_row.addWidget(self.morphology_legend_position)
        legend_row.addStretch(1)
        plots_layout.addLayout(legend_row)
        self.morphology_plot = MorphologyPlotCanvas()
        plots_layout.addWidget(self.morphology_plot, 1)
        self.morphology_display_tabs.addTab(plots_page, "Interactive clustering plots")
        self.morphology_cluster_score_panel = ClusterCountScorePanel()
        self.morphology_display_tabs.addTab(
            self.morphology_cluster_score_panel, "Cluster-count scores"
        )
        self.morphology_correlation_panel = CorrelationMatrixPanel()
        self.morphology_correlation_panel.settings_changed.connect(
            self._save_morphology_correlation_settings
        )
        self.morphology_display_tabs.addTab(
            self.morphology_correlation_panel, "Feature correlations"
        )
        output_layout.addWidget(self.morphology_display_tabs, 1)
        splitter.addWidget(output)
        splitter.setSizes([420, 980])
        self._last_morphology_preview: MorphologyPreview | None = None
        self._active_morphology_run: dict[str, object] | None = None
        self._morphology_plot_mode_changed()
        self.tabs.addTab(tab, "6. Morphology clustering")
        self._morphology_tool_changed()

    def _build_advanced_clustering_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(410)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        run_group = QGroupBox("Named nonlinear morphology analysis")
        form = QFormLayout(run_group)
        note = QLabel(
            "PCA remains the interpretable default in tab 6. Here, the selected "
            "higher-dimensional UMAP or PCC/PCUMAP embedding is the actual input "
            "to Gaussian mixture, Ward, or K-means clustering."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.advanced_saved_run = QComboBox()
        self.advanced_saved_run.currentIndexChanged.connect(
            self._advanced_saved_run_changed
        )
        form.addRow("Saved run:", self.advanced_saved_run)
        self.advanced_run_name = QLineEdit("Advanced nonlinear analysis")
        form.addRow("Run name:", self.advanced_run_name)
        self.advanced_reduction = QComboBox()
        self.advanced_reduction.addItem("UMAP", "umap")
        self.advanced_reduction.addItem("PCC/PCUMAP (Gildenblat & Pahnke)", "pcumap")
        self.advanced_reduction.currentIndexChanged.connect(
            self._advanced_reduction_changed
        )
        form.addRow("Reduction:", self.advanced_reduction)

        self.advanced_features: dict[str, QCheckBox] = {}
        feature_widget = QWidget()
        feature_layout = QGridLayout(feature_widget)
        feature_layout.setContentsMargins(0, 0, 0, 0)
        for index, (key, (label, _column)) in enumerate(MORPHOLOGY_FEATURES.items()):
            checkbox = QCheckBox(label)
            checkbox.setChecked(key in DEFAULT_FEATURES)
            self.advanced_features[key] = checkbox
            feature_layout.addWidget(checkbox, index // 2, index % 2)
        form.addRow("Morphology features:", feature_widget)
        protein_note = QLabel(
            "Protein puncta remain descriptive outputs only; they never influence "
            "the nonlinear embedding or cluster assignments."
        )
        protein_note.setWordWrap(True)
        form.addRow(protein_note)
        self.advanced_groups = QListWidget()
        self.advanced_groups.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.advanced_groups.setMaximumHeight(110)
        form.addRow("Experimental groups:", self.advanced_groups)
        self.advanced_reviewed_only = QCheckBox(
            "Cluster geometry-reviewed spines only"
        )
        form.addRow(self.advanced_reviewed_only)
        self.advanced_algorithm = QComboBox()
        self.advanced_algorithm.addItem("Gaussian mixture", "gaussian_mixture")
        self.advanced_algorithm.addItem("Ward hierarchical", "ward")
        self.advanced_algorithm.addItem("K-means", "kmeans")
        form.addRow("Cluster algorithm:", self.advanced_algorithm)

        cluster_range = QHBoxLayout()
        self.advanced_min_clusters = QSpinBox()
        self.advanced_min_clusters.setRange(1, 10)
        self.advanced_min_clusters.setValue(1)
        self.advanced_max_clusters = QSpinBox()
        self.advanced_max_clusters.setRange(1, 10)
        self.advanced_max_clusters.setValue(6)
        cluster_range.addWidget(QLabel("Min"))
        cluster_range.addWidget(self.advanced_min_clusters)
        cluster_range.addWidget(QLabel("Max"))
        cluster_range.addWidget(self.advanced_max_clusters)
        form.addRow("Candidate clusters:", cluster_range)
        self.advanced_cluster_count_selection = QComboBox()
        self.advanced_cluster_count_selection.addItem(
            "Information criterion (BIC / penalized SSE)",
            "information_criterion",
        )
        self.advanced_cluster_count_selection.addItem(
            "Maximum silhouette score", "silhouette"
        )
        self.advanced_cluster_count_selection.addItem(
            "Elbow of within-cluster SSE", "elbow"
        )
        self.advanced_cluster_count_selection.setToolTip(
            "Used only when Fixed count is Automatic. Elbow requires at least "
            "three accepted candidate cluster counts."
        )
        form.addRow(
            "Automatic selection:", self.advanced_cluster_count_selection
        )
        self.advanced_fixed_clusters = QSpinBox()
        self.advanced_fixed_clusters.setRange(0, 10)
        self.advanced_fixed_clusters.setSpecialValueText("Automatic")
        self.advanced_fixed_clusters.valueChanged.connect(
            self._cluster_count_controls_changed
        )
        form.addRow("Fixed count:", self.advanced_fixed_clusters)
        self.advanced_scaling = QComboBox()
        self.advanced_scaling.addItem("Robust median / IQR", "robust")
        self.advanced_scaling.addItem("Z-score", "zscore")
        self.advanced_scaling.addItem("None", "none")
        form.addRow("Input scaling:", self.advanced_scaling)
        self.advanced_seed = QSpinBox()
        self.advanced_seed.setRange(0, 2_000_000_000)
        self.advanced_seed.setValue(42)
        form.addRow("Random seed:", self.advanced_seed)
        self.advanced_min_cluster_spines = QSpinBox()
        self.advanced_min_cluster_spines.setRange(2, 1_000_000)
        self.advanced_min_cluster_spines.setValue(10)
        form.addRow("Minimum spines/cluster:", self.advanced_min_cluster_spines)
        self.advanced_min_cluster_fraction = QDoubleSpinBox()
        self.advanced_min_cluster_fraction.setRange(0.0, 50.0)
        self.advanced_min_cluster_fraction.setDecimals(1)
        self.advanced_min_cluster_fraction.setValue(5.0)
        self.advanced_min_cluster_fraction.setSuffix(" %")
        form.addRow("Minimum cluster fraction:", self.advanced_min_cluster_fraction)
        controls_layout.addWidget(run_group)

        embedding_group = QGroupBox("Embedding and reproducibility")
        embedding_form = QFormLayout(embedding_group)
        self.advanced_embedding_dimensions = QSpinBox()
        self.advanced_embedding_dimensions.setRange(2, 20)
        self.advanced_embedding_dimensions.setValue(5)
        embedding_form.addRow("Clustering dimensions:", self.advanced_embedding_dimensions)
        self.advanced_plot_dimensions = QComboBox()
        self.advanced_plot_dimensions.addItem("2D", 2)
        self.advanced_plot_dimensions.addItem("3D", 3)
        embedding_form.addRow("Interactive plot:", self.advanced_plot_dimensions)
        self.advanced_neighbors = QSpinBox()
        self.advanced_neighbors.setRange(2, 500)
        self.advanced_neighbors.setValue(15)
        embedding_form.addRow("Nearest neighbors:", self.advanced_neighbors)
        self.advanced_min_dist = QDoubleSpinBox()
        self.advanced_min_dist.setRange(0.0, 0.99)
        self.advanced_min_dist.setDecimals(3)
        self.advanced_min_dist.setSingleStep(0.05)
        self.advanced_min_dist.setValue(0.1)
        embedding_form.addRow("Minimum distance:", self.advanced_min_dist)
        self.advanced_metric = QComboBox()
        self.advanced_metric.addItem("Euclidean", "euclidean")
        self.advanced_metric.addItem("Manhattan", "manhattan")
        embedding_form.addRow("Distance metric:", self.advanced_metric)
        self.advanced_iterations = QSpinBox()
        self.advanced_iterations.setRange(50, 10_000)
        self.advanced_iterations.setSingleStep(50)
        self.advanced_iterations.setValue(500)
        embedding_form.addRow("Optimization iterations:", self.advanced_iterations)
        self.advanced_stability_repetitions = QSpinBox()
        self.advanced_stability_repetitions.setRange(1, 10)
        self.advanced_stability_repetitions.setValue(3)
        self.advanced_stability_repetitions.setToolTip(
            "Includes the primary fit. Additional fits quantify embedding-distance "
            "and cluster-assignment stability across random seeds."
        )
        embedding_form.addRow("Seed repetitions:", self.advanced_stability_repetitions)
        controls_layout.addWidget(embedding_group)

        self.advanced_pcumap_group = QGroupBox("PCC/PCUMAP correlation controls")
        pcumap_form = QFormLayout(self.advanced_pcumap_group)
        self.advanced_pcumap_reference_points = QSpinBox()
        self.advanced_pcumap_reference_points.setRange(2, 1_000_000)
        self.advanced_pcumap_reference_points.setValue(100)
        pcumap_form.addRow("Reference points:", self.advanced_pcumap_reference_points)
        self.advanced_pcumap_beta = QDoubleSpinBox()
        self.advanced_pcumap_beta.setRange(0.001, 1_000_000.0)
        self.advanced_pcumap_beta.setDecimals(3)
        self.advanced_pcumap_beta.setValue(10.0)
        pcumap_form.addRow("Beta:", self.advanced_pcumap_beta)
        self.advanced_pcumap_weight = QDoubleSpinBox()
        self.advanced_pcumap_weight.setRange(0.0, 1_000_000_000.0)
        self.advanced_pcumap_weight.setDecimals(1)
        self.advanced_pcumap_weight.setValue(90_000.0)
        pcumap_form.addRow("Correlation-loss weight:", self.advanced_pcumap_weight)
        self.advanced_pcumap_start = QSpinBox()
        self.advanced_pcumap_start.setRange(0, 10_000)
        self.advanced_pcumap_start.setValue(10)
        pcumap_form.addRow("Start correlation epoch:", self.advanced_pcumap_start)
        self.advanced_pcumap_device = QComboBox()
        self.advanced_pcumap_device.addItem("CPU (reproducible default)", "cpu")
        self.advanced_pcumap_device.addItem("Automatic", "auto")
        self.advanced_pcumap_device.addItem("CUDA GPU", "cuda")
        pcumap_form.addRow("Compute device:", self.advanced_pcumap_device)
        controls_layout.addWidget(self.advanced_pcumap_group)

        action_group = QGroupBox("Run, appearance, and export")
        action_form = QFormLayout(action_group)
        self.run_advanced_button = QPushButton("Run / replace named advanced analysis")
        self.run_advanced_button.clicked.connect(self._run_advanced_clustering)
        action_form.addRow(self.run_advanced_button)
        advanced_color_button = QPushButton("Change cluster colorsвЂ¦")
        advanced_color_button.clicked.connect(self._change_advanced_colors)
        action_form.addRow(advanced_color_button)
        self.advanced_axes_color = QColor(DEFAULT_PLOT_STYLE["axes_color"])
        self.advanced_axes_color_button = QPushButton("Change axes colorвЂ¦")
        self.advanced_axes_color_button.clicked.connect(
            lambda: self._change_advanced_plot_color("axes")
        )
        action_form.addRow(self.advanced_axes_color_button)
        self.advanced_axes_alpha = QDoubleSpinBox()
        self.advanced_axes_alpha.setRange(0.0, 1.0)
        self.advanced_axes_alpha.setSingleStep(0.05)
        self.advanced_axes_alpha.setValue(1.0)
        self.advanced_axes_alpha.editingFinished.connect(
            self._save_advanced_plot_style
        )
        action_form.addRow("Axes alpha:", self.advanced_axes_alpha)
        self.advanced_background_color = QColor(DEFAULT_PLOT_STYLE["background_color"])
        self.advanced_background_color_button = QPushButton("Change plot backgroundвЂ¦")
        self.advanced_background_color_button.clicked.connect(
            lambda: self._change_advanced_plot_color("background")
        )
        action_form.addRow(self.advanced_background_color_button)
        self.advanced_background_alpha = QDoubleSpinBox()
        self.advanced_background_alpha.setRange(0.0, 1.0)
        self.advanced_background_alpha.setSingleStep(0.05)
        self.advanced_background_alpha.setValue(1.0)
        self.advanced_background_alpha.editingFinished.connect(
            self._save_advanced_plot_style
        )
        action_form.addRow("Background alpha:", self.advanced_background_alpha)
        export_button = QPushButton("Export separate advanced analysis packageвЂ¦")
        export_button.clicked.connect(self._export_advanced_run)
        action_form.addRow(export_button)
        controls_layout.addWidget(action_group)
        controls_layout.addStretch(1)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)

        output = QWidget()
        output_layout = QVBoxLayout(output)
        self.advanced_status = QLabel(
            "Complete spine measurements before fitting a nonlinear embedding."
        )
        self.advanced_status.setWordWrap(True)
        output_layout.addWidget(self.advanced_status)
        self.advanced_display_tabs = QTabWidget()
        advanced_plots_page = QWidget()
        advanced_plots_layout = QVBoxLayout(advanced_plots_page)
        plot_row = QHBoxLayout()
        self.advanced_plot_mode = QComboBox()
        self.advanced_plot_mode.addItem("Nonlinear embedding", "embedding")
        self.advanced_plot_mode.addItem("Volume vs curvilinear length", "volume_length")
        self.advanced_plot_mode.addItem("Protein-positive fraction", "protein_positive")
        self.advanced_plot_mode.addItem("Protein puncta volume", "protein_volume")
        self.advanced_plot_mode.addItem("Protein position profiles", "protein_position")
        self.advanced_plot_mode.addItem("Group cluster proportions", "group_proportions")
        self.advanced_plot_mode.addItem("Cluster profile heatmap", "cluster_heatmap")
        self.advanced_plot_mode.currentIndexChanged.connect(
            self._render_advanced_plot
        )
        plot_row.addWidget(self.advanced_plot_mode, 1)
        self.advanced_show_legend = QCheckBox("Show legend")
        self.advanced_show_legend.setChecked(True)
        self.advanced_show_legend.toggled.connect(self._save_advanced_plot_style)
        plot_row.addWidget(self.advanced_show_legend)
        self.advanced_legend_position = QComboBox()
        self.advanced_legend_position.addItem("Outside right", "outside_right")
        self.advanced_legend_position.addItem("Below plot", "outside_bottom")
        self.advanced_legend_position.addItem("Inside plot", "inside")
        self.advanced_legend_position.currentIndexChanged.connect(
            self._save_advanced_plot_style
        )
        plot_row.addWidget(self.advanced_legend_position)
        advanced_plots_layout.addLayout(plot_row)
        self.advanced_diagnostics = QLabel("")
        self.advanced_diagnostics.setWordWrap(True)
        advanced_plots_layout.addWidget(self.advanced_diagnostics)
        self.advanced_plot = MorphologyPlotCanvas()
        advanced_plots_layout.addWidget(self.advanced_plot, 1)
        self.advanced_display_tabs.addTab(
            advanced_plots_page, "Interactive clustering plots"
        )
        self.advanced_cluster_score_panel = ClusterCountScorePanel()
        self.advanced_display_tabs.addTab(
            self.advanced_cluster_score_panel, "Cluster-count scores"
        )
        self.advanced_correlation_panel = CorrelationMatrixPanel()
        self.advanced_correlation_panel.settings_changed.connect(
            self._save_advanced_correlation_settings
        )
        self.advanced_display_tabs.addTab(
            self.advanced_correlation_panel, "Feature correlations"
        )
        output_layout.addWidget(self.advanced_display_tabs, 1)
        splitter.addWidget(output)
        splitter.setSizes([440, 960])
        self._active_advanced_run: dict[str, object] | None = None
        self.tabs.addTab(tab, "7. Advanced clustering")
        self._advanced_reduction_changed()
        self._cluster_count_controls_changed()

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
            set_count = sum(
                preprocessing_parameters_set(self.manifest, index, channel)
                for channel in ("ChanA", "ChanB")
            )
            excluded = bool(specimen.get("analysis", {}).get("excluded", False))
            suffix = (
                "excluded"
                if excluded
                else "reviewed ✓"
                if set_count == 2
                else f"marked {set_count}/2"
            )
            self.preprocess_specimen.addItem(
                f"{specimen['experimental_group']} — {specimen['specimen_id']} [{suffix}]",
                index,
            )
            self.preprocess_specimen.setItemData(
                self.preprocess_specimen.count() - 1,
                QColor("#777777")
                if excluded
                else QColor("#16823b")
                if set_count == 2
                else QColor("#a65a00"),
                Qt.ItemDataRole.ForegroundRole,
            )
        if current is not None:
            found = self.preprocess_specimen.findData(current)
            self.preprocess_specimen.setCurrentIndex(max(0, found))
        self.preprocess_specimen.blockSignals(False)
        roles = self.manifest["channel_roles"]
        for index in range(self.preprocess_channel.count()):
            channel = str(self.preprocess_channel.itemData(index))
            specimen_index = int(self.preprocess_specimen.currentData() or 0)
            state = (
                "marked ✓"
                if preprocessing_parameters_set(self.manifest, specimen_index, channel)
                else "not marked"
            )
            self.preprocess_channel.setItemText(
                index, f"{channel} — {ROLE_LABELS[str(roles[channel])]} [{state}]"
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
        specimen_changed = specimen_key != self._preprocess_view_specimen_key
        if specimen_changed:
            self.raw_view.reset_view()
            self.processed_view.reset_view()
            self._preprocess_view_specimen_key = specimen_key
        channel = self._selected_preprocess_channel()
        shape = self.manifest["specimens"][index]["channels"][channel]["metadata"]["shape"]
        z_count = 1 if len(shape) == 2 else int(shape[0])
        specimen = self.manifest["specimens"][index]
        analysis = specimen.get("analysis", {})
        self.exclude_specimen_check.blockSignals(True)
        self.exclude_specimen_check.setChecked(bool(analysis.get("excluded", False)))
        self.exclude_specimen_check.blockSignals(False)
        self.exclusion_reason.setText(str(analysis.get("exclusion_reason", "")))
        rectangles = specimen_rectangles(self.manifest, index, full_if_empty=False)
        self.roi_status.setText(f"{len(rectangles)} ROI(s)" if rectangles else "Full image")
        self._preprocess_exclusion_changed(self.exclude_specimen_check.isChecked())
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, max(0, z_count - 1))
        if specimen_changed:
            self.z_slider.setValue(max(0, (z_count - 1) // 2))
        self.z_slider.blockSignals(False)
        self.z_label.setText(f"Z: {self.z_slider.value() + 1}/{z_count}")
        self._last_preview = None
        self._load_channel_settings()
        roles = self.manifest["channel_roles"]
        for combo_index in range(self.preprocess_channel.count()):
            candidate = str(self.preprocess_channel.itemData(combo_index))
            state = (
                "marked ✓"
                if preprocessing_parameters_set(self.manifest, index, candidate)
                else "not marked"
            )
            self.preprocess_channel.setItemText(
                combo_index,
                f"{candidate} — {ROLE_LABELS[str(roles[candidate])]} [{state}]",
            )
        self._preview_timer.start()

    def _preprocess_channel_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None:
            return
        self._last_preview = None
        self._load_channel_settings()
        self._preprocess_specimen_changed()

    def _move_preprocess_specimen(self, delta: int) -> None:
        if self.preprocess_specimen.count() <= 0:
            return
        self.preprocess_specimen.setCurrentIndex(
            max(0, min(self.preprocess_specimen.count() - 1, self.preprocess_specimen.currentIndex() + delta))
        )

    def _preprocess_exclusion_changed(self, checked: bool) -> None:
        self.exclusion_reason.setEnabled(checked)
        self.manage_rois_button.setEnabled(not checked)
        if self.manifest is None or self.preprocess_specimen.currentData() is None:
            return
        index = self._selected_specimen_index()
        analysis = self.manifest["specimens"][index].setdefault("analysis", {})
        previous = bool(analysis.get("excluded", False))
        analysis["excluded"] = bool(checked)
        if previous != bool(checked):
            self._invalidate_preprocessing_channels(index, ["ChanA", "ChanB"])
            if checked:
                self.manifest["specimens"][index]["checkpoints"]["preprocessing"]["state"] = "excluded"
            if self.project_path is not None:
                save_project(self.project_path, self.manifest)

    def _save_exclusion_reason(self) -> None:
        if self.manifest is None or self.project_path is None or self.preprocess_specimen.currentData() is None:
            return
        analysis = self.manifest["specimens"][self._selected_specimen_index()].setdefault("analysis", {})
        analysis["exclusion_reason"] = self.exclusion_reason.text().strip()
        save_project(self.project_path, self.manifest)

    @staticmethod
    def _raw_xy_projection(path: Path) -> np.ndarray:
        with tifffile.TiffFile(path) as tiff:
            series = tiff.series[0]
            shape = tuple(int(value) for value in series.shape)
            if len(shape) == 2:
                return np.squeeze(np.asarray(series.asarray()))
            projection = np.zeros(shape[-2:], dtype=np.uint16)
            for z_index in range(shape[0]):
                plane = np.squeeze(np.asarray(series.asarray(key=z_index)))
                np.maximum(projection, plane, out=projection)
            return projection

    def _manage_analysis_rois(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        index = self._selected_specimen_index()
        channel = self._selected_preprocess_channel()
        source = channel_source_path(
            self.manifest, self.manifest["specimens"][index]["channels"][channel]
        )
        try:
            projection = self._raw_xy_projection(source)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot load ROI projection", str(exc))
            return
        current = specimen_rectangles(self.manifest, index, full_if_empty=False)
        dialog = AnalysisRoiDialog(projection, current, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updated = normalize_rectangles(dialog.rectangles(), projection.shape)
        if updated == current:
            return
        specimen = self.manifest["specimens"][index]
        specimen.setdefault("analysis", {})["rois_xy"] = rectangle_records(updated)
        self._invalidate_preprocessing_channels(index, ["ChanA", "ChanB"])
        save_project(self.project_path, self.manifest)
        self.roi_status.setText(f"{len(updated)} ROI(s)" if updated else "Full image")
        self._last_preview = None
        self._request_preview()

    def _load_channel_settings(self) -> None:
        if self.manifest is None:
            return
        channel = self._selected_preprocess_channel()
        specimen_index = self._selected_specimen_index()
        settings = effective_preprocessing_settings(self.manifest, specimen_index, channel)
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
                f"Saved independent {channel} settings"
                + (
                    f"; invalidated {invalidated} affected preprocessing checkpoint(s)."
                    if invalidated
                    else "."
                )
                + " Previewing with the updated parameters."
            )
            self._request_preview()
            self._prepare_preprocessing_tab()
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
        by_specimen = preprocessing.setdefault("settings_by_specimen", {})
        set_by_specimen = preprocessing.setdefault("parameters_set_by_specimen", {})

        def save_for(index: int) -> None:
            by_specimen.setdefault(str(index), {})[channel] = dict(settings)
            set_values = set(set_by_specimen.setdefault(str(index), []))
            set_values.add(channel)
            set_by_specimen[str(index)] = sorted(set_values)

        save_for(specimen_index)
        if self.fill_unset_preprocessing_check.isChecked():
            for index, specimen in enumerate(self.manifest["specimens"]):
                if bool(specimen.get("analysis", {}).get("excluded", False)):
                    continue
                if not preprocessing_parameters_set(self.manifest, index, channel):
                    save_for(index)
            self.fill_unset_preprocessing_check.setChecked(False)

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
        rectangles = specimen_rectangles(
            self.manifest, self._selected_specimen_index()
        ) if self.manifest is not None else []
        return (
            self._selected_specimen_index(),
            self._selected_preprocess_channel(),
            settings.background_percentile,
            settings.gaussian_sigma_xy_um,
            settings.gaussian_sigma_z_um,
            settings.threshold_sensitivity,
            tuple(rectangles),
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
                specimen_rectangles(self.manifest, index),
            )
            worker.completed.connect(self._preview_completed)
            self._start_worker(worker, "preview")
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot preview", str(exc))

    @Slot(object)
    def _preview_completed(self, payload: tuple[tuple[object, ...], PreviewResult]) -> None:
        key, result = payload
        sensitivity = float(key[-2])
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
            unset = [
                f"{specimen['specimen_id']} {channel}"
                for index, specimen in enumerate(self.manifest["specimens"])
                if not bool(specimen.get("analysis", {}).get("excluded", False))
                for channel in ("ChanA", "ChanB")
                if not preprocessing_parameters_set(self.manifest, index, channel)
            ]
            if unset:
                raise ValueError(
                    "Set preprocessing parameters for every included image first. Unset: "
                    + ", ".join(unset[:12])
                    + ("…" if len(unset) > 12 else "")
                )
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot start preprocessing", str(exc))
            return
        worker = BatchPreprocessWorker(self.manifest, self.project_path)
        worker.completed.connect(self._batch_preprocessing_completed)
        worker.cancelled.connect(self._batch_preprocessing_cancelled)
        self._start_worker(worker, "preprocess")

    def _run_selected_preprocessing(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            index = self._selected_specimen_index()
            specimen = self.manifest["specimens"][index]
            has_review_work = bool(specimen.get("review", {}).get("history")) or (
                specimen.get("checkpoints", {}).get("review", {}).get("state")
                not in {None, "not_started"}
            )
            if has_review_work:
                answer = QMessageBox.warning(
                    self,
                    "Discard manual corrections?",
                    "Reprocessing permanently discards all manual corrections and their history "
                    "for this specimen. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self._store_preprocessing_selection()
            missing = [
                channel
                for channel in ("ChanA", "ChanB")
                if not preprocessing_parameters_set(self.manifest, index, channel)
            ]
            if missing:
                raise ValueError(
                    "Set both channel parameters before reprocessing this specimen: "
                    + ", ".join(missing)
                )
            if bool(specimen.get("analysis", {}).get("excluded", False)):
                raise ValueError("This specimen is excluded from analysis.")
            if has_review_work:
                discard_specimen_review(self.manifest, self.project_path, index)
            save_project(self.project_path, self.manifest)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot reprocess specimen", str(exc))
            return
        self._pending_detection_after_preprocess = index
        worker = BatchPreprocessWorker(self.manifest, self.project_path, [index])
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
        self._pending_detection_ready = getattr(
            self, "_pending_detection_after_preprocess", None
        )
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
        self._pending_detection_after_preprocess = None
        self._pending_detection_ready = None
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
            if self.detection_specimen.currentData() is None:
                raise ValueError("Select a specimen first.")
            index = int(self.detection_specimen.currentData())
            self.manifest["detection"].setdefault("settings_by_specimen", {})[
                str(index)
            ] = self._current_detection_settings().to_dict()
            self.manifest["detection"]["memory_mode"] = str(
                self.detection_memory_mode.currentData() or AUTOMATIC_MEMORY_MODE
            )
            save_project(self.project_path, self.manifest)
            self.detection_status.setText(
                "Specimen-specific detection settings saved. Redo detection to apply them."
            )
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save detection settings", str(exc))

    def _apply_detection_default_settings(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            self.manifest["detection"]["settings"] = self._current_detection_settings().to_dict()
            self.manifest["detection"]["memory_mode"] = str(
                self.detection_memory_mode.currentData() or AUTOMATIC_MEMORY_MODE
            )
            save_project(self.project_path, self.manifest)
            self.detection_status.setText(
                "Batch defaults saved. Existing specimen-specific overrides were preserved."
            )
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot save detection defaults", str(exc))

    def _run_detection(self) -> None:
        if self.manifest is None or self.project_path is None:
            QMessageBox.information(self, "No project", "Save or open a project first.")
            return
        try:
            self._sync_manifest_edits()
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

    def _run_selected_detection(
        self, specimen_index: int | None = None, *, force: bool = True
    ) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            index = (
                int(self.detection_specimen.currentData())
                if specimen_index is None
                else int(specimen_index)
            )
            specimen = self.manifest["specimens"][index]
            if specimen["checkpoints"]["preprocessing"].get("state") != "complete":
                raise ValueError("Preprocessing must be complete for this specimen.")
            if bool(specimen.get("analysis", {}).get("excluded", False)):
                raise ValueError("This specimen is excluded from analysis.")
            has_review_work = bool(specimen.get("review", {}).get("history")) or (
                specimen.get("checkpoints", {}).get("review", {}).get("state")
                not in {None, "not_started"}
            )
            if has_review_work:
                answer = QMessageBox.warning(
                    self,
                    "Discard manual corrections?",
                    "Redoing detection permanently discards all manual corrections and their "
                    "history for this specimen. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            if specimen_index is None:
                self.manifest["detection"].setdefault("settings_by_specimen", {})[
                    str(index)
                ] = self._current_detection_settings().to_dict()
            discard_specimen_review(self.manifest, self.project_path, index)
            save_project(self.project_path, self.manifest)
        except (TypeError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot redo detection", str(exc))
            return
        worker = DetectionWorker(
            self.manifest, self.project_path, [index], force=force
        )
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
        settings = effective_detection_settings(self.manifest, index)
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
        correction_sensitivity = float(
            self.manifest["review_settings"].get("correction_sensitivity", 1.0)
        )
        self.review_sensitivity.setValue(
            max(25, min(1000, round(correction_sensitivity * 100)))
        )
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
            maximum_centerline_gap_um=self.maximum_centerline_gap.value(),
        )

    def _measurement_method_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        method = self.measurement_end_method.currentData()
        self.measurement_fixed_slices.setEnabled(method == "fixed")
        self.measurement_area_factor.setEnabled(method == "adaptive")

    def _prepare_morphology_tab(self) -> None:
        measured: list[int] = []
        if self.manifest is not None:
            measured = [
                index
                for index, specimen in enumerate(self.manifest["specimens"])
                if specimen["checkpoints"].get("measurements", {}).get("state") == "complete"
            ]
        self.tabs.setTabEnabled(5, bool(measured))
        current = self.morphology_specimen.currentData()
        self.morphology_specimen.blockSignals(True)
        self.morphology_specimen.clear()
        if self.manifest is not None:
            for index in measured:
                specimen = self.manifest["specimens"][index]
                reviews = specimen.get("morphology_review", {}).get("spines", {})
                reviewed = sum(bool(value.get("reviewed")) for value in reviews.values())
                self.morphology_specimen.addItem(
                    f"{specimen['experimental_group']} — {specimen['specimen_id']} [{reviewed} geometry checked]",
                    index,
                )
        if current is not None:
            found = self.morphology_specimen.findData(current)
            self.morphology_specimen.setCurrentIndex(max(0, found))
        self.morphology_specimen.blockSignals(False)
        selected_groups = {item.text() for item in self.morphology_groups.selectedItems()}
        had_group_choices = self.morphology_groups.count() > 0
        self.morphology_groups.clear()
        if self.manifest is not None:
            groups = sorted(
                {
                    str(self.manifest["specimens"][index]["experimental_group"])
                    for index in measured
                }
            )
            for group in groups:
                self.morphology_groups.addItem(group)
                item = self.morphology_groups.item(self.morphology_groups.count() - 1)
                item.setSelected(group in selected_groups if had_group_choices else True)
        current_run = self.morphology_saved_run.currentData()
        self.morphology_saved_run.blockSignals(True)
        self.morphology_saved_run.clear()
        if self.manifest is not None:
            for record in self.manifest.get("morphology_analysis", {}).get("runs", []):
                if str(record.get("settings", {}).get("reduction_method", "pca")) != "pca":
                    continue
                suffix = " [stale—refit needed]" if record.get("stale") else ""
                self.morphology_saved_run.addItem(f"{record.get('name', 'Analysis')}{suffix}", record.get("run_id"))
        if current_run is not None:
            found = self.morphology_saved_run.findData(current_run)
            self.morphology_saved_run.setCurrentIndex(max(0, found))
        self.morphology_saved_run.blockSignals(False)
        if measured:
            self._morphology_specimen_changed()
        if self.morphology_saved_run.count():
            self._morphology_saved_run_changed()
        self._prepare_advanced_clustering_tab(measured)

    def _prepare_advanced_clustering_tab(
        self, measured: list[int] | None = None
    ) -> None:
        if measured is None:
            measured = []
            if self.manifest is not None:
                measured = [
                    index
                    for index, specimen in enumerate(self.manifest["specimens"])
                    if specimen["checkpoints"].get("measurements", {}).get("state")
                    == "complete"
                ]
        self.tabs.setTabEnabled(6, bool(measured))
        selected_groups = {item.text() for item in self.advanced_groups.selectedItems()}
        had_groups = self.advanced_groups.count() > 0
        self.advanced_groups.clear()
        if self.manifest is not None:
            groups = sorted(
                {
                    str(self.manifest["specimens"][index]["experimental_group"])
                    for index in measured
                }
            )
            for group in groups:
                self.advanced_groups.addItem(group)
                item = self.advanced_groups.item(self.advanced_groups.count() - 1)
                item.setSelected(group in selected_groups if had_groups else True)
        current_run = self.advanced_saved_run.currentData()
        self.advanced_saved_run.blockSignals(True)
        self.advanced_saved_run.clear()
        if self.manifest is not None:
            for record in self.manifest.get("morphology_analysis", {}).get("runs", []):
                method = str(record.get("settings", {}).get("reduction_method", "pca"))
                if method not in {"umap", "pcumap"}:
                    continue
                suffix = " [stale - refit needed]" if record.get("stale") else ""
                label = "PCC/PCUMAP" if method == "pcumap" else "UMAP"
                self.advanced_saved_run.addItem(
                    f"{record.get('name', 'Analysis')} [{label}]{suffix}",
                    record.get("run_id"),
                )
        if current_run is not None:
            found = self.advanced_saved_run.findData(current_run)
            if found >= 0:
                self.advanced_saved_run.setCurrentIndex(found)
        self.advanced_saved_run.blockSignals(False)
        if self.advanced_saved_run.count():
            self._advanced_saved_run_changed()

    def _advanced_reduction_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        self.advanced_pcumap_group.setVisible(
            self.advanced_reduction.currentData() == "pcumap"
        )

    def _cluster_count_controls_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self, "morphology_cluster_count_selection"):
            self.morphology_cluster_count_selection.setEnabled(
                self.morphology_fixed_clusters.value() == 0
            )
        if hasattr(self, "advanced_cluster_count_selection"):
            self.advanced_cluster_count_selection.setEnabled(
                self.advanced_fixed_clusters.value() == 0
            )

    def _current_advanced_settings(self) -> MorphologyClusteringSettings:
        features = tuple(
            key for key, checkbox in self.advanced_features.items()
            if checkbox.isChecked()
        )
        groups = tuple(item.text() for item in self.advanced_groups.selectedItems())
        if self.advanced_groups.count() and not groups:
            raise ValueError("Select at least one experimental group for clustering.")
        return MorphologyClusteringSettings(
            algorithm=str(self.advanced_algorithm.currentData()),
            features=features,
            pca_dimensions=3,
            use_pca_for_clustering=False,
            minimum_clusters=self.advanced_min_clusters.value(),
            maximum_clusters=self.advanced_max_clusters.value(),
            fixed_cluster_count=self.advanced_fixed_clusters.value(),
            cluster_count_selection=str(
                self.advanced_cluster_count_selection.currentData()
            ),
            scaling=str(self.advanced_scaling.currentData()),
            random_seed=self.advanced_seed.value(),
            minimum_cluster_spines=self.advanced_min_cluster_spines.value(),
            minimum_cluster_fraction=self.advanced_min_cluster_fraction.value() / 100.0,
            included_groups=groups,
            reviewed_only=self.advanced_reviewed_only.isChecked(),
            reduction_method=str(self.advanced_reduction.currentData()),
            embedding_dimensions=self.advanced_embedding_dimensions.value(),
            embedding_plot_dimensions=int(self.advanced_plot_dimensions.currentData()),
            umap_n_neighbors=self.advanced_neighbors.value(),
            umap_min_dist=self.advanced_min_dist.value(),
            umap_metric=str(self.advanced_metric.currentData()),
            umap_iterations=self.advanced_iterations.value(),
            pcumap_reference_points=self.advanced_pcumap_reference_points.value(),
            pcumap_beta=self.advanced_pcumap_beta.value(),
            pcumap_correlation_weight=self.advanced_pcumap_weight.value(),
            pcumap_correlation_start=self.advanced_pcumap_start.value(),
            pcumap_device=str(self.advanced_pcumap_device.currentData()),
            embedding_stability_repetitions=self.advanced_stability_repetitions.value(),
        )

    def _run_advanced_clustering(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            settings = self._current_advanced_settings()
            settings.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot run advanced clustering", str(exc))
            return
        worker = MorphologyClusteringWorker(
            self.manifest,
            self.project_path,
            self.advanced_run_name.text(),
            settings,
        )
        worker.completed.connect(self._advanced_clustering_completed)
        self._start_worker(worker, "advanced_morphology")

    @Slot(object)
    def _advanced_clustering_completed(self, result: dict[str, object]) -> None:
        self._active_advanced_run = result
        self._prepare_morphology_tab()
        run_index = self.advanced_saved_run.findData(result.get("run_id"))
        if run_index >= 0:
            self.advanced_saved_run.setCurrentIndex(run_index)
        self._active_advanced_run = result
        self._refresh_advanced_plot_style_controls()
        self._render_advanced_plot()
        self._show_advanced_diagnostics(result)

    def _show_advanced_diagnostics(self, result: dict[str, object]) -> None:
        metadata = dict(result.get("embedding_metadata", {}))
        stability = dict(result.get("embedding_stability", {}))
        method = (
            "PCC/PCUMAP"
            if metadata.get("method") == "pcumap"
            else str(metadata.get("method", "embedding")).upper()
        )
        trust = metadata.get("trustworthiness")
        distance = metadata.get("distance_rank_correlation")
        repeat_distance = stability.get("mean_distance_rank_correlation")
        repeat_clusters = stability.get("mean_cluster_adjusted_rand")
        values = [
            f"Saved {result.get('name')}: {result.get('selected_cluster_count')} clusters from "
            f"{result.get('included_spine_count')} valid complete spines in a "
            f"{metadata.get('dimensions')}D {method} space.",
            f"Trustworthiness: {float(trust):.3f}." if trust is not None else "",
            f"Input/embedding distance-rank correlation: {float(distance):.3f}."
            if distance is not None else "",
            f"Across-seed embedding stability: {float(repeat_distance):.3f}."
            if repeat_distance is not None else "",
            f"Across-seed cluster adjusted Rand: {float(repeat_clusters):.3f}."
            if repeat_clusters is not None else "",
            "Protein puncta were descriptive only.",
        ]
        self.advanced_status.setText(" ".join(value for value in values if value))
        self.advanced_diagnostics.setText(
            f"Implementation: {metadata.get('implementation', '')} "
            f"{metadata.get('implementation_version', '')}; neighbors used: "
            f"{metadata.get('effective_neighbors')} (requested "
            f"{metadata.get('requested_neighbors')}); stability fits completed: "
            f"{stability.get('completed_repetitions', 1)}."
        )

    def _advanced_saved_run_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.advanced_saved_run.currentData() is None:
            return
        try:
            result = load_morphology_run(
                self.manifest, str(self.advanced_saved_run.currentData())
            )
            settings = MorphologyClusteringSettings.from_dict(
                dict(result.get("settings", {}))
            )
            if settings.reduction_method not in {"umap", "pcumap"}:
                return
            self._active_advanced_run = result
            self.advanced_run_name.setText(str(result.get("name", "")))
            self.advanced_reduction.setCurrentIndex(
                max(0, self.advanced_reduction.findData(settings.reduction_method))
            )
            for key, checkbox in self.advanced_features.items():
                checkbox.setChecked(key in settings.features)
            self.advanced_algorithm.setCurrentIndex(
                max(0, self.advanced_algorithm.findData(settings.algorithm))
            )
            self.advanced_min_clusters.setValue(settings.minimum_clusters)
            self.advanced_max_clusters.setValue(settings.maximum_clusters)
            self.advanced_fixed_clusters.setValue(settings.fixed_cluster_count)
            self.advanced_cluster_count_selection.setCurrentIndex(
                max(
                    0,
                    self.advanced_cluster_count_selection.findData(
                        settings.cluster_count_selection
                    ),
                )
            )
            self.advanced_scaling.setCurrentIndex(
                max(0, self.advanced_scaling.findData(settings.scaling))
            )
            self.advanced_seed.setValue(settings.random_seed)
            self.advanced_min_cluster_spines.setValue(settings.minimum_cluster_spines)
            self.advanced_min_cluster_fraction.setValue(
                settings.minimum_cluster_fraction * 100.0
            )
            selected_groups = set(settings.included_groups)
            for index in range(self.advanced_groups.count()):
                item = self.advanced_groups.item(index)
                item.setSelected(not selected_groups or item.text() in selected_groups)
            self.advanced_reviewed_only.setChecked(settings.reviewed_only)
            self.advanced_embedding_dimensions.setValue(settings.embedding_dimensions)
            self.advanced_plot_dimensions.setCurrentIndex(
                max(
                    0,
                    self.advanced_plot_dimensions.findData(
                        settings.embedding_plot_dimensions
                    ),
                )
            )
            self.advanced_neighbors.setValue(settings.umap_n_neighbors)
            self.advanced_min_dist.setValue(settings.umap_min_dist)
            self.advanced_metric.setCurrentIndex(
                max(0, self.advanced_metric.findData(settings.umap_metric))
            )
            self.advanced_iterations.setValue(settings.umap_iterations)
            self.advanced_stability_repetitions.setValue(
                settings.embedding_stability_repetitions
            )
            self.advanced_pcumap_reference_points.setValue(
                settings.pcumap_reference_points
            )
            self.advanced_pcumap_beta.setValue(settings.pcumap_beta)
            self.advanced_pcumap_weight.setValue(settings.pcumap_correlation_weight)
            self.advanced_pcumap_start.setValue(settings.pcumap_correlation_start)
            self.advanced_pcumap_device.setCurrentIndex(
                max(0, self.advanced_pcumap_device.findData(settings.pcumap_device))
            )
            self._advanced_reduction_changed()
            self._refresh_advanced_plot_style_controls()
            self._render_advanced_plot()
            self._show_advanced_diagnostics(result)
        except (OSError, ValueError) as exc:
            self.advanced_status.setText(f"Cannot load saved advanced analysis: {exc}")

    def _render_advanced_plot(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self._active_advanced_run is not None:
            self.advanced_plot.show_run(
                self._active_advanced_run,
                str(self.advanced_plot_mode.currentData()),
            )
            self.advanced_cluster_score_panel.show_run(
                self._active_advanced_run
            )
            self.advanced_correlation_panel.show_run(
                self._active_advanced_run
            )

    def _save_advanced_correlation_settings(
        self, correlation_settings: dict[str, object]
    ) -> None:
        if self._active_advanced_run is None:
            return
        style = {
            **DEFAULT_PLOT_STYLE,
            **dict(self._active_advanced_run.get("plot_style", {})),
            **dict(correlation_settings),
        }
        self._active_advanced_run["plot_style"] = style
        try:
            run_id = str(self._active_advanced_run.get("run_id", ""))
            known = bool(
                self.manifest
                and any(
                    str(record.get("run_id", "")) == run_id
                    for record in self.manifest.get("morphology_analysis", {}).get("runs", [])
                )
            )
            if known and self.manifest is not None and self.project_path is not None:
                self._active_advanced_run = update_run_plot_style(
                    self.manifest, self.project_path, run_id, style
                )
            self.advanced_correlation_panel.show_run(self._active_advanced_run)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(
                self, "Cannot save correlation settings", str(exc)
            )

    def _change_advanced_colors(self) -> None:
        if self._active_advanced_run is None:
            return
        colors: list[str] = []
        for definition in self._active_advanced_run.get("cluster_definitions", []):
            chosen = QColorDialog.getColor(
                QColor(str(definition.get("color", "#457b9d"))),
                self,
                f"Color for morphology cluster {definition.get('morphology_cluster_id')}",
            )
            if not chosen.isValid():
                return
            colors.append(chosen.name())
        try:
            run_id = str(self._active_advanced_run.get("run_id", ""))
            known = bool(
                self.manifest
                and any(
                    str(record.get("run_id", "")) == run_id
                    for record in self.manifest.get("morphology_analysis", {}).get("runs", [])
                )
            )
            if known and self.manifest is not None and self.project_path is not None:
                self._active_advanced_run = update_run_colors(
                    self.manifest, self.project_path, run_id, colors
                )
            else:
                for definition, color in zip(
                    self._active_advanced_run.get("cluster_definitions", []), colors
                ):
                    definition["color"] = color
            self._render_advanced_plot()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot save colors", str(exc))

    def _refresh_advanced_plot_style_controls(self) -> None:
        style = {
            **DEFAULT_PLOT_STYLE,
            **dict((self._active_advanced_run or {}).get("plot_style", {})),
        }
        self.advanced_axes_color = QColor(str(style["axes_color"]))
        self.advanced_background_color = QColor(str(style["background_color"]))
        controls = (
            self.advanced_axes_alpha,
            self.advanced_background_alpha,
            self.advanced_show_legend,
            self.advanced_legend_position,
        )
        for control in controls:
            control.blockSignals(True)
        self.advanced_axes_alpha.setValue(float(style["axes_alpha"]))
        self.advanced_background_alpha.setValue(float(style["background_alpha"]))
        self.advanced_show_legend.setChecked(bool(style["show_legend"]))
        self.advanced_legend_position.setCurrentIndex(
            max(
                0,
                self.advanced_legend_position.findData(str(style["legend_position"])),
            )
        )
        for control in controls:
            control.blockSignals(False)
        self.advanced_axes_color_button.setStyleSheet(
            f"background-color: {self.advanced_axes_color.name()};"
        )
        self.advanced_background_color_button.setStyleSheet(
            f"background-color: {self.advanced_background_color.name()};"
        )

    def _change_advanced_plot_color(self, target: str) -> None:
        current = QColor(
            self.advanced_axes_color
            if target == "axes"
            else self.advanced_background_color
        )
        alpha = (
            self.advanced_axes_alpha
            if target == "axes"
            else self.advanced_background_alpha
        )
        current.setAlphaF(alpha.value())
        chosen = QColorDialog.getColor(
            current,
            self,
            "Choose axes color" if target == "axes" else "Choose plot background",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not chosen.isValid():
            return
        if target == "axes":
            self.advanced_axes_color = QColor(chosen)
        else:
            self.advanced_background_color = QColor(chosen)
        alpha.setValue(chosen.alphaF())
        self._save_advanced_plot_style()

    def _save_advanced_plot_style(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self._active_advanced_run is None:
            return
        style = {
            **DEFAULT_PLOT_STYLE,
            **dict(self._active_advanced_run.get("plot_style", {})),
            "axes_color": self.advanced_axes_color.name(),
            "axes_alpha": self.advanced_axes_alpha.value(),
            "background_color": self.advanced_background_color.name(),
            "background_alpha": self.advanced_background_alpha.value(),
            "show_legend": self.advanced_show_legend.isChecked(),
            "legend_position": str(
                self.advanced_legend_position.currentData() or "outside_right"
            ),
        }
        self._active_advanced_run["plot_style"] = style
        try:
            run_id = str(self._active_advanced_run.get("run_id", ""))
            known = bool(
                self.manifest
                and any(
                    str(record.get("run_id", "")) == run_id
                    for record in self.manifest.get("morphology_analysis", {}).get("runs", [])
                )
            )
            if known and self.manifest is not None and self.project_path is not None:
                self._active_advanced_run = update_run_plot_style(
                    self.manifest, self.project_path, run_id, style
                )
            self._refresh_advanced_plot_style_controls()
            self._render_advanced_plot()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot save plot style", str(exc))

    def _export_advanced_run(self) -> None:
        if self._active_advanced_run is None:
            QMessageBox.information(
                self, "No advanced run", "Run or select an advanced analysis first."
            )
            return
        suggested = (
            str(Path(self.manifest["output_directory"]) / "Synpo_advanced_clustering.xlsx")
            if self.manifest
            else "Synpo_advanced_clustering.xlsx"
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export advanced clustering analysis",
            suggested,
            "Excel workbook (*.xlsx)",
        )
        if not selected:
            return
        worker = MorphologyExportWorker(self._active_advanced_run, Path(selected))
        worker.completed.connect(self._morphology_export_completed)
        self._start_worker(worker, "advanced_morphology_export")

    def _morphology_specimen_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        current_spine = self.morphology_spine.currentData()
        self.morphology_spine.blockSignals(True)
        self.morphology_spine.clear()
        specimen_value = self.morphology_specimen.currentData()
        if self.manifest is not None and specimen_value is not None:
            try:
                result = load_measurement_result(self.manifest, int(specimen_value))
                decisions = self.manifest["specimens"][int(specimen_value)].get("morphology_review", {}).get("spines", {})
                validity_decisions = self.manifest["specimens"][int(specimen_value)].get("distribution_review", {}).get("spines", {})
                for row in result.get("morphology_rows", []):
                    spine_id = int(row["spine_id"])
                    marker = (
                        "invalid"
                        if validity_decisions.get(str(spine_id), {}).get("invalid_spine")
                        else "checked ✓"
                        if decisions.get(str(spine_id), {}).get("reviewed")
                        else "unreviewed"
                    )
                    status = row.get("spine_length_status", "unknown")
                    self.morphology_spine.addItem(f"Spine {spine_id} [{marker}; {status}]", spine_id)
            except (OSError, ValueError, KeyError) as exc:
                self.morphology_status.setText(f"Cannot load morphology measurements: {exc}")
        if current_spine is not None:
            found = self.morphology_spine.findData(current_spine)
            if found >= 0:
                self.morphology_spine.setCurrentIndex(found)
        self.morphology_spine.blockSignals(False)
        if self.morphology_spine.count():
            self._morphology_spine_changed()

    def _morphology_spine_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        specimen_value = self.morphology_specimen.currentData()
        spine_value = self.morphology_spine.currentData()
        if self.manifest is None or specimen_value is None or spine_value is None:
            return
        try:
            preview = load_morphology_preview(self.manifest, int(specimen_value), int(spine_value))
        except (OSError, ValueError, KeyError) as exc:
            self.morphology_status.setText(f"Cannot load spine geometry: {exc}")
            return
        self._last_morphology_preview = preview
        self.morphology_z.setRange(0, max(0, preview.spine_mask_stack.shape[0] - 1))
        self.morphology_z.setValue((preview.spine_z_range[0] + preview.spine_z_range[1]) // 2)
        decision = self.manifest["specimens"][int(specimen_value)].get("morphology_review", {}).get("spines", {}).get(str(int(spine_value)), {})
        self.morphology_reviewed.setChecked(bool(decision.get("reviewed", False)))
        validity = self.manifest["specimens"][int(specimen_value)].get("distribution_review", {}).get("spines", {}).get(str(int(spine_value)), {})
        self.morphology_invalid.setChecked(bool(validity.get("invalid_spine", False)))
        self.morphology_note.setText(str(decision.get("note", "")))
        row = preview.row
        specimen = self.manifest["specimens"][int(specimen_value)]
        self.morphology_metrics.setText(
            f"Group: {specimen['experimental_group']} | specimen: {specimen['specimen_id']} | "
            f"curvilinear length: {row.get('spine_curvilinear_length_um')} µm | "
            f"base-to-tip: {row.get('spine_base_to_tip_distance_um')} µm | "
            f"tortuosity: {row.get('centerline_tortuosity')} | "
            f"head/neck split: {row.get('head_neck_split_status')}"
        )
        self.morphology_status.setText("Orange = head; cyan = neck; magenta = protein puncta; white = centerline; green/yellow = base/tip.")
        self._render_morphology_preview()

    def _render_morphology_preview(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self._last_morphology_preview is None:
            return
        maximum = self.morphology_view_mode.currentData() == "maximum"
        self.morphology_z.setEnabled(not maximum)
        self.morphology_canvas.show_morphology(self._last_morphology_preview, self.morphology_z.value(), maximum)
        self._morphology_tool_changed()

    def _morphology_tool_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        tool = self.morphology_tool.currentData()
        colors = {"set_base": "#2cff60", "set_tip": "#fff000", "paint_head": "#ff9123", "paint_neck": "#19bef0"}
        if hasattr(self, "morphology_canvas"):
            self.morphology_canvas.set_hint_color(QColor(colors.get(str(tool), "#ffffff")))
            self.morphology_canvas.set_brush_diameter(1 if tool in {"set_base", "set_tip"} else self.morphology_brush.value())
            self.morphology_brush.setEnabled(tool in {"paint_head", "paint_neck"})

    def _morphology_anchor_point(self) -> tuple[int, int, int]:
        preview = self._last_morphology_preview
        points = self.morphology_canvas.hint_points()
        if preview is None or not points:
            raise ValueError("Click the desired spine voxel first.")
        x, y = points[-1]
        if not (0 <= y < preview.spine_mask_stack.shape[1] and 0 <= x < preview.spine_mask_stack.shape[2]):
            raise ValueError("The selected point is outside the spine crop.")
        possible = np.flatnonzero(preview.spine_mask_stack[:, y, x])
        if not len(possible):
            raise ValueError("Select a point on the colored spine mask.")
        current = self.morphology_z.value()
        z_index = int(possible[np.argmin(np.abs(possible - current))]) if self.morphology_view_mode.currentData() == "maximum" else current
        if not preview.spine_mask_stack[z_index, y, x]:
            raise ValueError("Select a spine voxel on the current Z slice.")
        return z_index, y + preview.crop_origin_yx[0], x + preview.crop_origin_yx[1]

    def _apply_morphology_edit(self) -> None:
        if self.manifest is None or self.project_path is None or self._last_morphology_preview is None:
            return
        specimen_index = int(self.morphology_specimen.currentData())
        spine_id = int(self.morphology_spine.currentData())
        operation = str(self.morphology_tool.currentData())
        try:
            strokes = self.morphology_canvas.hint_strokes()
            if operation in {"paint_head", "paint_neck"} and not strokes:
                raise ValueError("Draw on the selected spine before applying the brush.")
            arguments = {
                "operation": operation,
                "point_zyx": self._morphology_anchor_point() if operation in {"set_base", "set_tip"} else None,
                "strokes_xy": strokes,
                "maximum_projection": self.morphology_view_mode.currentData() == "maximum",
                "z_index": self.morphology_z.value(),
                "z_radius": self.morphology_z_radius.value(),
                "brush_radius": max(0, self.morphology_brush.value() // 2),
                "reviewed": self.morphology_reviewed.isChecked(),
                "note": self.morphology_note.text(),
                "invalid_spine": self.morphology_invalid.isChecked(),
            }
            self._start_morphology_edit_worker(specimen_index, spine_id, "apply", arguments)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot apply geometry edit", str(exc))

    def _undo_redo_morphology(self, redo: bool) -> None:
        if self.manifest is None or self.project_path is None or self.morphology_spine.currentData() is None:
            return
        self._start_morphology_edit_worker(
            int(self.morphology_specimen.currentData()),
            int(self.morphology_spine.currentData()),
            "redo" if redo else "undo",
        )

    def _reset_morphology(self, operation: str) -> None:
        if self.manifest is None or self.project_path is None or self.morphology_spine.currentData() is None:
            return
        self._start_morphology_edit_worker(
            int(self.morphology_specimen.currentData()),
            int(self.morphology_spine.currentData()),
            "apply",
            {"operation": operation},
        )

    def _checkpoint_morphology_review(self) -> None:
        if self.manifest is None or self.project_path is None or self.morphology_spine.currentData() is None:
            return
        self._start_morphology_edit_worker(
            int(self.morphology_specimen.currentData()),
            int(self.morphology_spine.currentData()),
            "apply",
            {
                "operation": "checkpoint",
                "reviewed": True,
                "note": self.morphology_note.text(),
                "invalid_spine": self.morphology_invalid.isChecked(),
                "advance": True,
            },
        )

    def _start_morphology_edit_worker(
        self,
        specimen_index: int,
        spine_id: int,
        mode: str,
        arguments: dict[str, object] | None = None,
    ) -> None:
        if self.manifest is None or self.project_path is None:
            return
        worker = MorphologyEditWorker(
            self.manifest,
            self.project_path,
            specimen_index,
            spine_id,
            mode,
            arguments,
        )
        worker.completed.connect(self._morphology_edit_completed)
        self._start_worker(worker, "morphology_edit")

    @Slot(object)
    def _morphology_edit_completed(self, payload: dict[str, object]) -> None:
        edited_spine = int(payload["spine_id"])
        advance = bool(payload.get("advance", False))
        self._prepare_measurements_tab()
        self._prepare_morphology_tab()
        current_index = self.morphology_spine.findData(edited_spine)
        if current_index >= 0:
            target = min(self.morphology_spine.count() - 1, current_index + (1 if advance else 0))
            self.morphology_spine.setCurrentIndex(target)
        self.morphology_status.setText(
            f"Spine {edited_spine} geometry saved."
            + (" Advanced to the next spine." if advance and current_index + 1 < self.morphology_spine.count() else "")
        )

    def _move_morphology_spine(self, offset: int) -> None:
        if self.morphology_spine.count():
            self.morphology_spine.setCurrentIndex(max(0, min(self.morphology_spine.count() - 1, self.morphology_spine.currentIndex() + int(offset))))

    def _current_morphology_settings(self) -> MorphologyClusteringSettings:
        features = tuple(key for key, checkbox in self.morphology_features.items() if checkbox.isChecked())
        groups = tuple(item.text() for item in self.morphology_groups.selectedItems())
        if self.morphology_groups.count() and not groups:
            raise ValueError("Select at least one experimental group for clustering.")
        return MorphologyClusteringSettings(
            algorithm=str(self.morphology_algorithm.currentData()),
            features=features,
            pca_dimensions=int(self.morphology_pca_dimensions.currentData()),
            use_pca_for_clustering=self.morphology_use_pca.isChecked(),
            minimum_clusters=self.morphology_min_clusters.value(),
            maximum_clusters=self.morphology_max_clusters.value(),
            fixed_cluster_count=self.morphology_fixed_clusters.value(),
            cluster_count_selection=str(
                self.morphology_cluster_count_selection.currentData()
            ),
            scaling=str(self.morphology_scaling.currentData()),
            random_seed=self.morphology_seed.value(),
            minimum_cluster_spines=self.morphology_min_cluster_spines.value(),
            minimum_cluster_fraction=self.morphology_min_cluster_fraction.value() / 100.0,
            included_groups=groups,
            reviewed_only=self.morphology_reviewed_only.isChecked(),
        )

    def _run_morphology_clustering(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            settings = self._current_morphology_settings()
            settings.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot run clustering", str(exc))
            return
        worker = MorphologyClusteringWorker(self.manifest, self.project_path, self.morphology_run_name.text(), settings)
        worker.completed.connect(self._morphology_clustering_completed)
        self._start_worker(worker, "morphology")

    @Slot(object)
    def _morphology_clustering_completed(self, result: dict[str, object]) -> None:
        self._active_morphology_run = result
        self._prepare_morphology_tab()
        self.morphology_display_tabs.setCurrentIndex(1)
        self._render_morphology_plot()
        self._refresh_morphology_plot_style_controls()
        stability = result.get("bootstrap_stability", {}).get("mean_adjusted_rand")
        stability_text = f" Mean specimen-bootstrap adjusted Rand: {float(stability):.3f}." if stability is not None else ""
        reviewed_text = (
            " geometry-reviewed"
            if bool(result.get("settings", {}).get("reviewed_only", False))
            else ""
        )
        self.morphology_status.setText(f"Saved {result.get('name')}: {result.get('selected_cluster_count')} clusters from {result.get('included_spine_count')}{reviewed_text} valid complete spines.{stability_text} Protein puncta were descriptive only.")

    def _morphology_saved_run_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.morphology_saved_run.currentData() is None:
            return
        try:
            self._active_morphology_run = load_morphology_run(self.manifest, str(self.morphology_saved_run.currentData()))
            self.morphology_run_name.setText(str(self._active_morphology_run.get("name", "")))
            settings = MorphologyClusteringSettings.from_dict(dict(self._active_morphology_run.get("settings", {})))
            self.morphology_algorithm.setCurrentIndex(max(0, self.morphology_algorithm.findData(settings.algorithm)))
            for key, checkbox in self.morphology_features.items():
                checkbox.setChecked(key in settings.features)
            self.morphology_pca_dimensions.setCurrentIndex(max(0, self.morphology_pca_dimensions.findData(settings.pca_dimensions)))
            self.morphology_use_pca.setChecked(settings.use_pca_for_clustering)
            self.morphology_min_clusters.setValue(settings.minimum_clusters)
            self.morphology_max_clusters.setValue(settings.maximum_clusters)
            self.morphology_fixed_clusters.setValue(settings.fixed_cluster_count)
            self.morphology_cluster_count_selection.setCurrentIndex(
                max(
                    0,
                    self.morphology_cluster_count_selection.findData(
                        settings.cluster_count_selection
                    ),
                )
            )
            self.morphology_scaling.setCurrentIndex(max(0, self.morphology_scaling.findData(settings.scaling)))
            self.morphology_seed.setValue(settings.random_seed)
            self.morphology_min_cluster_spines.setValue(settings.minimum_cluster_spines)
            self.morphology_min_cluster_fraction.setValue(settings.minimum_cluster_fraction * 100.0)
            selected_groups = set(settings.included_groups)
            for index in range(self.morphology_groups.count()):
                item = self.morphology_groups.item(index)
                item.setSelected(not selected_groups or item.text() in selected_groups)
            self.morphology_reviewed_only.setChecked(settings.reviewed_only)
            self._refresh_morphology_plot_style_controls()
            self._render_morphology_plot()
        except (OSError, ValueError) as exc:
            self.morphology_status.setText(f"Cannot load saved analysis: {exc}")

    def _render_morphology_plot(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self._active_morphology_run is not None:
            self.morphology_plot.show_run(self._active_morphology_run, str(self.morphology_plot_mode.currentData()))
            self.morphology_cluster_score_panel.show_run(
                self._active_morphology_run
            )
            self.morphology_correlation_panel.show_run(
                self._active_morphology_run
            )

    def _save_morphology_correlation_settings(
        self, correlation_settings: dict[str, object]
    ) -> None:
        if self._active_morphology_run is None:
            return
        style = {
            **DEFAULT_PLOT_STYLE,
            **dict(self._active_morphology_run.get("plot_style", {})),
            **dict(correlation_settings),
        }
        self._active_morphology_run["plot_style"] = style
        try:
            run_id = str(self._active_morphology_run.get("run_id", ""))
            known = bool(
                self.manifest
                and any(
                    str(record.get("run_id", "")) == run_id
                    for record in self.manifest.get("morphology_analysis", {}).get("runs", [])
                )
            )
            if known and self.manifest is not None and self.project_path is not None:
                self._active_morphology_run = update_run_plot_style(
                    self.manifest, self.project_path, run_id, style
                )
            self.morphology_correlation_panel.show_run(
                self._active_morphology_run
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(
                self, "Cannot save correlation settings", str(exc)
            )

    def _morphology_plot_mode_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        self.morphology_feature_plot_row.setVisible(
            self.morphology_plot_mode.currentData() == "custom_features"
        )
        self.morphology_pca_point_row.setVisible(
            self.morphology_plot_mode.currentData()
            in {"pca_interpretation", "pca_3d_features"}
        )
        self._render_morphology_plot()

    def _change_morphology_colors(self) -> None:
        if self._active_morphology_run is None:
            return
        colors = []
        for definition in self._active_morphology_run.get("cluster_definitions", []):
            current = QColor(str(definition.get("color", "#457b9d")))
            chosen = QColorDialog.getColor(current, self, f"Color for morphology cluster {definition.get('morphology_cluster_id')}")
            if not chosen.isValid():
                return
            colors.append(chosen.name())
        try:
            run_id = str(self._active_morphology_run.get("run_id", ""))
            known_run = bool(self.manifest and any(
                str(record.get("run_id", "")) == run_id
                for record in self.manifest.get("morphology_analysis", {}).get("runs", [])
            ))
            if known_run and self.manifest is not None and self.project_path is not None:
                self._active_morphology_run = update_run_colors(
                    self.manifest, self.project_path, run_id, colors
                )
            else:
                for definition, color in zip(
                    self._active_morphology_run.get("cluster_definitions", []), colors
                ):
                    definition["color"] = color
            self._render_morphology_plot()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot save colors", str(exc))

    def _refresh_morphology_plot_style_controls(self) -> None:
        style = {
            **DEFAULT_PLOT_STYLE,
            **dict((self._active_morphology_run or {}).get("plot_style", {})),
        }
        self.morphology_axes_color = QColor(str(style["axes_color"]))
        self.morphology_background_color = QColor(str(style["background_color"]))
        self.morphology_axes_alpha.blockSignals(True)
        self.morphology_background_alpha.blockSignals(True)
        self.morphology_show_legend.blockSignals(True)
        self.morphology_legend_position.blockSignals(True)
        self.morphology_custom_x.blockSignals(True)
        self.morphology_custom_y.blockSignals(True)
        self.morphology_pca_show_points.blockSignals(True)
        self.morphology_pca_point_alpha.blockSignals(True)
        self.morphology_axes_alpha.setValue(float(style["axes_alpha"]))
        self.morphology_background_alpha.setValue(float(style["background_alpha"]))
        self.morphology_show_legend.setChecked(bool(style["show_legend"]))
        self.morphology_pca_show_points.setChecked(
            bool(style["pca_show_points"])
        )
        self.morphology_pca_point_alpha.setValue(
            float(style["pca_point_alpha"])
        )
        legend_index = self.morphology_legend_position.findData(
            str(style["legend_position"])
        )
        self.morphology_legend_position.setCurrentIndex(max(0, legend_index))
        features = list(
            (self._active_morphology_run or {}).get("settings", {}).get(
                "features", []
            )
        )
        self.morphology_custom_x.clear()
        self.morphology_custom_y.clear()
        for feature in features:
            label = MORPHOLOGY_FEATURES.get(feature, (feature, ""))[0]
            self.morphology_custom_x.addItem(label, feature)
            self.morphology_custom_y.addItem(label, feature)
        x_feature = str(style.get("custom_x_feature", ""))
        y_feature = str(style.get("custom_y_feature", ""))
        x_index = self.morphology_custom_x.findData(x_feature)
        y_index = self.morphology_custom_y.findData(y_feature)
        self.morphology_custom_x.setCurrentIndex(max(0, x_index))
        self.morphology_custom_y.setCurrentIndex(
            max(0, y_index if y_index >= 0 else min(1, len(features) - 1))
        )
        self.morphology_axes_alpha.blockSignals(False)
        self.morphology_background_alpha.blockSignals(False)
        self.morphology_show_legend.blockSignals(False)
        self.morphology_legend_position.blockSignals(False)
        self.morphology_custom_x.blockSignals(False)
        self.morphology_custom_y.blockSignals(False)
        self.morphology_pca_show_points.blockSignals(False)
        self.morphology_pca_point_alpha.blockSignals(False)
        self.morphology_feature_plot_row.setVisible(
            self.morphology_plot_mode.currentData() == "custom_features"
        )
        self.morphology_pca_point_row.setVisible(
            self.morphology_plot_mode.currentData()
            in {"pca_interpretation", "pca_3d_features"}
        )
        self.morphology_axes_color_button.setStyleSheet(
            f"background-color: {self.morphology_axes_color.name()};"
        )
        self.morphology_background_color_button.setStyleSheet(
            f"background-color: {self.morphology_background_color.name()};"
        )

    def _change_morphology_plot_color(self, target: str) -> None:
        current = QColor(
            self.morphology_axes_color
            if target == "axes"
            else self.morphology_background_color
        )
        alpha_control = (
            self.morphology_axes_alpha
            if target == "axes"
            else self.morphology_background_alpha
        )
        current.setAlphaF(alpha_control.value())
        chosen = QColorDialog.getColor(
            current,
            self,
            "Choose axes color" if target == "axes" else "Choose plot background",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not chosen.isValid():
            return
        if target == "axes":
            self.morphology_axes_color = QColor(chosen)
        else:
            self.morphology_background_color = QColor(chosen)
        alpha_control.setValue(chosen.alphaF())
        self._save_morphology_plot_style()

    def _save_morphology_plot_style(self) -> None:
        if self._active_morphology_run is None:
            return
        style = {
            **DEFAULT_PLOT_STYLE,
            **dict(self._active_morphology_run.get("plot_style", {})),
            "axes_color": self.morphology_axes_color.name(),
            "axes_alpha": self.morphology_axes_alpha.value(),
            "background_color": self.morphology_background_color.name(),
            "background_alpha": self.morphology_background_alpha.value(),
            "show_legend": self.morphology_show_legend.isChecked(),
            "legend_position": str(
                self.morphology_legend_position.currentData() or "outside_right"
            ),
            "custom_x_feature": str(self.morphology_custom_x.currentData() or ""),
            "custom_y_feature": str(self.morphology_custom_y.currentData() or ""),
            "pca_show_points": self.morphology_pca_show_points.isChecked(),
            "pca_point_alpha": self.morphology_pca_point_alpha.value(),
        }
        self._active_morphology_run["plot_style"] = dict(style)
        try:
            run_id = str(self._active_morphology_run.get("run_id", ""))
            known_run = bool(
                self.manifest
                and any(
                    str(record.get("run_id", "")) == run_id
                    for record in self.manifest.get("morphology_analysis", {}).get("runs", [])
                )
            )
            if known_run and self.manifest is not None and self.project_path is not None:
                self._active_morphology_run = update_run_plot_style(
                    self.manifest, self.project_path, run_id, style
                )
            self._refresh_morphology_plot_style_controls()
            self._render_morphology_plot()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot save plot style", str(exc))

    def _export_morphology_run(self) -> None:
        if self._active_morphology_run is None:
            QMessageBox.information(self, "No analysis run", "Run or select a named morphology analysis first.")
            return
        selected, _ = QFileDialog.getSaveFileName(self, "Export morphology analysis", str(Path(self.manifest["output_directory"]) / "Synpo_morphology_analysis.xlsx") if self.manifest else "Synpo_morphology_analysis.xlsx", "Excel workbook (*.xlsx)")
        if not selected:
            return
        worker = MorphologyExportWorker(self._active_morphology_run, Path(selected))
        worker.completed.connect(self._morphology_export_completed)
        self._start_worker(worker, "morphology_export")

    @Slot(object)
    def _morphology_export_completed(self, result: dict[str, object]) -> None:
        QMessageBox.information(self, "Morphology analysis exported", f"Workbook: {result.get('workbook')}\nPlots: {result.get('plot_directory')}\nReport: {result.get('report_pdf')}")

    def _cluster_exported_morphology_workbook(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self,
            "Open Synpo morphology or measurement workbook",
            "",
            "Excel workbook (*.xlsx)",
        )
        if not source:
            return
        output, _ = QFileDialog.getSaveFileName(
            self,
            "Save standalone morphology analysis",
            str(Path(source).with_name(f"{Path(source).stem}_morphology_analysis.xlsx")),
            "Excel workbook (*.xlsx)",
        )
        if not output:
            return
        try:
            advanced = self.tabs.currentIndex() == 6
            settings = (
                self._current_advanced_settings()
                if advanced
                else self._current_morphology_settings()
            )
            settings.validate()
            standalone_settings = settings.to_dict()
            standalone_settings["included_groups"] = []
            settings = MorphologyClusteringSettings.from_dict(standalone_settings)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot start standalone analysis", str(exc))
            return
        worker = StandaloneMorphologyWorker(
            Path(source), Path(output), settings,
            (
                self.advanced_run_name.text().strip()
                if advanced
                else self.morphology_run_name.text().strip()
            ) or "Standalone morphology analysis",
        )
        worker.completed.connect(self._standalone_morphology_completed)
        self._start_worker(worker, "standalone_morphology")

    @Slot(object)
    def _standalone_morphology_completed(self, payload: dict[str, object]) -> None:
        result = dict(payload.get("result", {}))
        method = str(result.get("settings", {}).get("reduction_method", "pca"))
        if method in {"umap", "pcumap"}:
            self._active_advanced_run = result
            self._refresh_advanced_plot_style_controls()
            self._render_advanced_plot()
            self._show_advanced_diagnostics(result)
        else:
            self._active_morphology_run = result
            self._render_morphology_plot()
        exported = payload.get("export", {})
        QMessageBox.information(
            self,
            "Standalone morphology analysis complete",
            f"Workbook: {exported.get('workbook')}\nPlots: {exported.get('plot_directory')}\nReport: {exported.get('report_pdf')}",
        )

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
            self.maximum_centerline_gap,
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
        self.maximum_centerline_gap.setValue(settings.maximum_centerline_gap_um)
        self.measurement_end_method.blockSignals(False)
        for control in controls:
            control.blockSignals(False)
        self._measurement_method_changed()
        filter_enabled, filter_cutoff = spine_volume_filter_settings(self.manifest)
        self.volume_filter_enabled.blockSignals(True)
        self.volume_filter_cutoff.blockSignals(True)
        self.volume_filter_enabled.setChecked(filter_enabled)
        self.volume_filter_cutoff.setValue(filter_cutoff)
        self.volume_filter_enabled.blockSignals(False)
        self.volume_filter_cutoff.blockSignals(False)

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
        else:
            self._sync_distribution_bulk_acceptance()

    def _save_measurement_settings(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        try:
            settings = self._current_measurement_settings()
            filter_enabled, filter_cutoff = spine_volume_filter_settings(self.manifest)
            self.manifest["measurements"]["settings"] = {
                **settings.to_dict(),
                "spine_volume_filter_enabled": filter_enabled,
                "spine_volume_filter_cutoff_um3": filter_cutoff,
            }
            for specimen in self.manifest["specimens"]:
                checkpoint = specimen["checkpoints"].setdefault("measurements", {})
                checkpoint["state"] = "not_started"
            for run in self.manifest.get("morphology_analysis", {}).get("runs", []):
                run["stale"] = True
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
            filter_enabled, filter_cutoff = spine_volume_filter_settings(self.manifest)
            self.manifest["measurements"]["settings"] = {
                **settings.to_dict(),
                "spine_volume_filter_enabled": filter_enabled,
                "spine_volume_filter_cutoff_um3": filter_cutoff,
            }
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
        self._prepare_morphology_tab()

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
            self._sync_distribution_bulk_acceptance()
            return
        try:
            raw_result = load_measurement_result(self.manifest, specimen_index)
            result, _volume_audit = filtered_measurement_result(
                self.manifest, specimen_index
            )
        except (OSError, ValueError, KeyError) as exc:
            self.measurement_summary.setText(f"Cannot load measurements: {exc}")
            return
        self._populate_spine_review_queue(raw_result)
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
        self._sync_distribution_bulk_acceptance()

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
                filtered, _audit = filtered_measurement_result(self.manifest, index)
                results.append(filtered)
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
            + (
                f". Orange marks show a {float(preview.row.get('centerline_bridge_length_um', 0.0)):.3f} µm virtual bridge; review before inclusion."
                if bool(preview.row.get("centerline_bridge_used", False))
                else ""
            )
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
        bridge_xy: tuple[tuple[int, int], ...] = (),
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
        for index, (x, y) in enumerate(bridge_xy):
            if index % 2 == 0 and 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                rgb[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = (255, 145, 20)
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
            bridge = tuple(
                (point[2], point[1])
                for point in preview.bridge_points_local_zyx
                if point[0] == z_index
            )
            rgb = self._distribution_overlay(raw, bins, axis, base, endpoint, bridge)
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
                tuple((point[2], point[1]) for point in preview.bridge_points_local_zyx),
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

    def _distribution_bulk_acceptance_state(self) -> tuple[int, int]:
        if self.manifest is None:
            return 0, 0
        eligible = 0
        reviewed = 0
        for specimen_index, specimen in enumerate(self.manifest.get("specimens", [])):
            checkpoint = specimen.get("checkpoints", {}).get("measurements", {})
            if checkpoint.get("state") != "complete":
                continue
            try:
                result, _audit = filtered_measurement_result(
                    self.manifest, specimen_index
                )
            except (OSError, ValueError, KeyError):
                continue
            for row in result.get("distribution_rows", []):
                if (
                    bool(row.get("volume_filter_excluded", False))
                    or not bool(row.get("spine_valid", True))
                    or not distribution_profile_available(row)
                ):
                    continue
                eligible += 1
                if bool(row.get("distribution_reviewed", False)):
                    reviewed += 1
        return eligible, reviewed

    def _sync_distribution_bulk_acceptance(self) -> None:
        eligible, reviewed = self._distribution_bulk_acceptance_state()
        self._distribution_bulk_eligible = eligible
        self._distribution_bulk_remaining = max(0, eligible - reviewed)
        self.accept_all_distribution_spines.blockSignals(True)
        self.accept_all_distribution_spines.setChecked(
            eligible > 0 and reviewed == eligible
        )
        self.accept_all_distribution_spines.blockSignals(False)
        self.accept_all_distribution_spines.setEnabled(
            self._job_worker is None and self._distribution_bulk_eligible > 0
        )
        self.accept_all_distribution_spines.setToolTip(
            "Marks every non-invalidated, non-volume-filtered spine with a usable "
            "distribution path as reviewed and included. Existing review notes are preserved. "
            f"Current status: {reviewed} of {eligible} eligible spines reviewed."
        )

    def _accept_all_distribution_spines(self, checked: bool) -> None:
        if not checked:
            QTimer.singleShot(0, self._sync_distribution_bulk_acceptance)
            return
        if self.manifest is None or self.project_path is None:
            return
        try:
            counts = accept_all_eligible_distribution_spines(
                self.manifest, self.project_path
            )
        except (OSError, ValueError, KeyError) as exc:
            self.accept_all_distribution_spines.blockSignals(True)
            self.accept_all_distribution_spines.setChecked(False)
            self.accept_all_distribution_spines.blockSignals(False)
            QMessageBox.warning(
                self, "Cannot accept distribution spines", str(exc)
            )
            return
        self._refresh_distribution_groups()
        self._measurement_specimen_changed()
        self._prepare_morphology_tab()
        self._sync_distribution_bulk_acceptance()
        self.measurement_status.setText(
            f"Accepted {counts['accepted_count']} previously unreviewed distribution spine(s) "
            f"across {counts['measured_specimen_count']} measured specimen(s) "
            f"and preserved {counts['already_reviewed_count']} existing review decision(s). Skipped "
            f"{counts['invalid_count']} invalid, "
            f"{counts['volume_filtered_count']} volume-filtered, and "
            f"{counts['unusable_path_count']} without a usable path."
        )

    def _save_volume_filter_controls(self, *_args) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.project_path is None:
            return
        settings = self.manifest["measurements"].setdefault("settings", {})
        settings["spine_volume_filter_enabled"] = self.volume_filter_enabled.isChecked()
        settings["spine_volume_filter_cutoff_um3"] = self.volume_filter_cutoff.value()
        for run in self.manifest.get("morphology_analysis", {}).get("runs", []):
            run["stale"] = True
        try:
            save_project(self.project_path, self.manifest)
        except OSError as exc:
            QMessageBox.warning(self, "Cannot save volume filter", str(exc))
            return
        self._refresh_distribution_groups()
        self._measurement_specimen_changed()
        self._prepare_morphology_tab()

    def _open_volume_filter_preview(self) -> None:
        if self.manifest is None or self.project_path is None:
            return
        rows: list[dict[str, object]] = []
        for specimen_index, specimen in enumerate(self.manifest["specimens"]):
            if specimen["checkpoints"].get("measurements", {}).get("state") != "complete":
                continue
            try:
                result = load_measurement_result(self.manifest, specimen_index)
            except (OSError, ValueError):
                continue
            decisions = specimen.get("distribution_review", {}).get("spines", {})
            for source in result.get("spine_rows", []):
                row = dict(source)
                decision = decisions.get(str(int(row.get("spine_id") or 0)), {})
                row["volume_filter_force_keep"] = bool(
                    decision.get("volume_filter_force_keep", False)
                )
                rows.append(row)
        enabled, cutoff = spine_volume_filter_settings(self.manifest)
        dialog = SpineVolumeFilterDialog(
            rows,
            cutoff_um3=cutoff,
            enabled=enabled,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        enabled, cutoff, force_keep = dialog.values()
        represented = {
            (
                str(row.get("experimental_group", "")),
                str(row.get("specimen_id", "")),
                int(row.get("spine_id") or 0),
            )
            for row in rows
        }
        settings = self.manifest["measurements"].setdefault("settings", {})
        settings["spine_volume_filter_enabled"] = enabled
        settings["spine_volume_filter_cutoff_um3"] = cutoff
        for run in self.manifest.get("morphology_analysis", {}).get("runs", []):
            run["stale"] = True
        for specimen in self.manifest["specimens"]:
            group = str(specimen.get("experimental_group", ""))
            specimen_id = str(specimen.get("specimen_id", ""))
            decisions = specimen.setdefault(
                "distribution_review", {"spines": {}, "updated_at": None}
            ).setdefault("spines", {})
            known_ids = {
                key[2] for key in force_keep if key[0] == group and key[1] == specimen_id
            }
            represented_ids = {
                key[2] for key in represented if key[0] == group and key[1] == specimen_id
            }
            for spine_id, decision in list(decisions.items()):
                if int(spine_id) in represented_ids and int(spine_id) not in known_ids:
                    decision.pop("volume_filter_force_keep", None)
            for spine_id in known_ids:
                decisions.setdefault(str(spine_id), {})[
                    "volume_filter_force_keep"
                ] = True
            specimen["distribution_review"]["updated_at"] = time.time()
        try:
            save_project(self.project_path, self.manifest)
        except OSError as exc:
            QMessageBox.warning(self, "Cannot save volume filter", str(exc))
            return
        self.volume_filter_enabled.blockSignals(True)
        self.volume_filter_cutoff.blockSignals(True)
        self.volume_filter_enabled.setChecked(enabled)
        self.volume_filter_cutoff.setValue(cutoff)
        self.volume_filter_enabled.blockSignals(False)
        self.volume_filter_cutoff.blockSignals(False)
        self._refresh_distribution_groups()
        self._measurement_specimen_changed()
        self._prepare_morphology_tab()
        self.measurement_status.setText(
            "Spine-volume filter saved; summaries were refreshed without remeasurement."
        )

    def _filter_exported_measurement_workbook(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Filter a Synpo measurement workbook",
            "",
            "Excel workbook (*.xlsx)",
        )
        if not selected:
            return
        try:
            inspection = inspect_exported_measurement_workbook(selected)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot use workbook", str(exc))
            return
        if bool(inspection.get("filter_enabled", False)):
            QMessageBox.warning(
                self,
                "Use the unfiltered source workbook",
                "This workbook is already volume-filtered and no longer contains the omitted "
                "detail rows. Select the original unfiltered workbook so the cutoff remains reversible.",
            )
            return
        dialog = SpineVolumeFilterDialog(
            list(inspection["spines"]),
            cutoff_um3=float(inspection.get("cutoff_um3", 0.0)),
            enabled=True,
            parent=self,
            allow_disable=False,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        _enabled, cutoff, force_keep = dialog.values()
        source = Path(selected).resolve()
        suggested = source.with_name(f"{source.stem}_volume-filtered.xlsx")
        output, _ = QFileDialog.getSaveFileName(
            self,
            "Save filtered Synpo workbook",
            str(suggested),
            "Excel workbook (*.xlsx)",
        )
        if not output:
            return
        if Path(output).resolve() == source:
            QMessageBox.warning(
                self,
                "Choose another filename",
                "The source workbook is the reversible original and cannot be overwritten.",
            )
            return
        worker = WorkbookVolumeFilterWorker(
            source, Path(output), cutoff, force_keep
        )
        worker.completed.connect(self._workbook_filter_completed)
        self._start_worker(worker, "workbook_filter")

    @Slot(object)
    def _workbook_filter_completed(self, result: dict[str, object]) -> None:
        unavailable = [
            row["table"]
            for row in result.get("compatibility_report", [])
            if row.get("status") == "unavailable"
            and row.get("table") != "PDF exports"
        ]
        report = (
            "Unavailable exact recalculations: " + ", ".join(unavailable)
            if unavailable
            else "Every supported scientific table was recalculated exactly."
        )
        QMessageBox.information(
            self,
            "Filtered workbook verified",
            f"Excluded {result['excluded_spine_count']} spine(s).\n"
            f"Workbook: {result['workbook']}\nCSV folder: {result['csv_directory']}\n\n"
            f"{report}\nPDFs were not regenerated because standalone mode has no image/cache data.",
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
            result, _audit = filtered_measurement_result(
                self.manifest, specimen_index
            )
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
        self._update_review_tool_controls()
        self._load_review_view(auto_contrast=True)

    def _load_review_view(
        self, *_args, auto_contrast: bool = False
    ) -> None:  # type: ignore[no-untyped-def]
        if self.manifest is None or self.review_specimen.currentData() is None:
            return
        if self.review_view_mode.currentData() == "xy_max":
            self._load_review_projection(auto_contrast=auto_contrast)
            return
        started = time.monotonic()
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
            self._record_diagnostic(
                "review_slice_loaded",
                scope="correction",
                specimen_index=self._selected_review_specimen(),
                z_index=self.review_z_slider.value(),
                duration_seconds=round(time.monotonic() - started, 6),
                corrected=self._last_review.corrected,
            )
        except (OSError, ValueError, KeyError, IndexError) as exc:
            self._record_diagnostic(
                "review_slice_load_failed",
                scope="correction",
                duration_seconds=round(time.monotonic() - started, 6),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
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
        self._last_review_context = None
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
            self._update_review_tool_controls()
            return
        self._review_projection_loading = True
        self.review_view.setEnabled(False)
        self._update_review_tool_controls()
        self._start_context_generation(request)

    def _display_review_projection(
        self, volume: ContextVolume, *, auto_contrast: bool
    ) -> None:
        self._review_projection_loading = False
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
        self.review_view.clear_hint()
        self._update_review_tool_controls()

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
            distinct_dendrites_spines=self.review_distinct_object_colors.isChecked(),
        )

    def _review_tool_changed(self, *_args) -> None:  # type: ignore[no-untyped-def]
        operation = str(self.review_operation.currentData())
        self.review_view.set_hint_color(
            REVIEW_BRUSH_COLORS.get(operation, QColor("#ffe119"))
        )
        if operation in {"filopodium", "dendrite_to_spine"}:
            self.review_object_type.setCurrentIndex(
                self.review_object_type.findData("spine")
            )
        elif operation == "spine_to_dendrite":
            self.review_object_type.setCurrentIndex(
                self.review_object_type.findData("dendrite")
            )
        if (
            operation in {"dendrite_to_spine", "spine_to_dendrite"}
            and self.review_view_mode.currentData() != "xy_max"
        ):
            self.review_view_mode.setCurrentIndex(
                self.review_view_mode.findData("xy_max")
            )
        instructions = {
            "add": (
                "Draw one separate stroke inside each missed object. Every stroke seeds one "
                "3D region. Touching new regions become one object; a region touching exactly "
                "one existing object joins its stable ID. Ambiguous contact is skipped."
            ),
            "erase": (
                "Erase literal painted mask voxels from both dendrites and spines. In a single "
                "Z slice only that slice is changed; on the XY projection all painted Z columns "
                "are cleared. Object IDs are preserved."
            ),
            "dendrite_to_spine": (
                "On the XY maximum projection, paint across exactly one spine and any dendrite "
                "area that belongs to it. Covered dendrite voxels in every Z slice are transferred "
                "literally to that spine; image intensity is ignored."
            ),
            "spine_to_dendrite": (
                "On the XY maximum projection, paint across exactly one dendrite and any spine "
                "area that belongs to it. Covered spine voxels from every touched spine ID in "
                "every Z slice are transferred literally to that dendrite; image intensity is ignored."
            ),
            "exclude": "Touch an unwanted object to exclude the complete 3D object.",
            "filopodium": "Touch a spine candidate to exclude and record it as a filopodium.",
            "split": "Draw across the contact or neck that should separate one object into two.",
            "merge": "Draw through at least two objects that should be one object.",
            "expand": "Draw toward missing signal from an existing object; the boundary is regrown locally.",
            "trim": (
                "Draw across the excess part. The connected image-supported remainder is retained "
                "locally; higher sensitivity removes more weak signal."
            ),
            "accept": "Touch an object to mark it accepted without changing its mask.",
            "needs_attention": "Touch an object to retain it but flag it for later attention.",
        }
        self.review_instruction.setText(instructions.get(operation, "Draw a hint."))
        self._update_review_tool_controls()

    def _update_review_tool_controls(self) -> None:
        operation = str(self.review_operation.currentData())
        available = self.review_specimen.count() > 0 and self._review_thread is None
        has_fixed_type = operation in {
            "filopodium",
            "dendrite_to_spine",
            "spine_to_dendrite",
            "erase",
        }
        self.review_object_type.setEnabled(available and not has_fixed_type)
        sensitivity_enabled = available and operation in {"add", "expand", "trim"}
        self.review_sensitivity_label.setEnabled(sensitivity_enabled)
        self.review_sensitivity_widget.setEnabled(sensitivity_enabled)
        projection_valid = (
            operation not in {"dendrite_to_spine", "spine_to_dendrite"}
            or (
                self.review_view_mode.currentData() == "xy_max"
                and self._last_review_context is not None
            )
        )
        drawing_available = available and not self._review_projection_loading
        self.review_view.setEnabled(drawing_available)
        self.apply_review_button.setEnabled(drawing_available and projection_valid)

    def _select_review_operation(self, operation: str) -> None:
        index = self.review_operation.findData(operation)
        if index >= 0:
            self.review_operation.setCurrentIndex(index)

    def _apply_review_shortcut(self) -> None:
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return
        self._apply_review_action()

    def _review_edit_preprocessing(self) -> None:
        if self.review_specimen.currentData() is None:
            return
        specimen_index = int(self.review_specimen.currentData())
        self._prepare_preprocessing_tab()
        row = self.preprocess_specimen.findData(specimen_index)
        if row >= 0:
            self.preprocess_specimen.setCurrentIndex(row)
        self.tabs.setCurrentIndex(1)
        self.preprocessing_status.setText(
            "Edit both channel parameters or ROIs, then choose ‘Preprocess selected specimen’. "
            "Detection will rerun automatically and existing corrections will be discarded."
        )

    def _review_edit_detection(self) -> None:
        if self.review_specimen.currentData() is None:
            return
        specimen_index = int(self.review_specimen.currentData())
        self._prepare_detection_tab()
        row = self.detection_specimen.findData(specimen_index)
        if row >= 0:
            self.detection_specimen.setCurrentIndex(row)
        self.tabs.setCurrentIndex(2)
        self.detection_status.setText(
            "Edit specimen-specific settings, then choose ‘Redo detection for selected specimen’. "
            "Existing corrections will be permanently discarded."
        )

    def _review_brush_changed(self, value: int) -> None:
        self.review_view.set_brush_diameter(value)

    def _review_sensitivity_changed(self, value: int) -> None:
        self.review_sensitivity_value.setText(f"{value / 100.0:.2f}")

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
        operation = str(self.review_operation.currentData())
        sensitivity = self.review_sensitivity.value() / 100.0
        action = ReviewAction(
            object_type=str(self.review_object_type.currentData()),
            operation=operation,
            z_index=self.review_z_slider.value(),
            points=points,
            brush_radius_pixels=max(0, self.review_brush_diameter.value() // 2),
            sensitivity=sensitivity,
            projection_hint=self.review_view_mode.currentData() == "xy_max",
            strokes=self.review_view.hint_strokes(),
        )
        self.manifest["review_settings"]["memory_mode"] = str(
            self.review_memory_mode.currentData()
            or AUTOMATIC_REVIEW_MEMORY_MODE
        )
        if operation in {"add", "expand", "trim"}:
            self.manifest["review_settings"]["correction_sensitivity"] = sensitivity
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
            self.manifest,
            self.project_path,
            specimen_index,
            action,
            diagnostic=self._diagnostic_callback(),
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
        self._diagnostic_review_phase = None
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
        if phase != self._diagnostic_review_phase or current >= total:
            self._record_diagnostic(
                "correction_progress",
                scope="correction",
                phase=phase,
                current=current,
                total=total,
                detail=detail,
            )
            self._diagnostic_review_phase = phase

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
            joined = [
                item for item in result.hint_results if item.get("status") == "joined"
            ]
            skipped = [
                item for item in result.hint_results if item.get("status") == "skipped"
            ]
            details = "; ".join(
                f"hint {item['hint_index']}: {item['message']}"
                for item in (*joined, *skipped)
            )
            checkpoint = (
                "One atomic checkpoint was written."
                if result.checkpoint_written
                else "No mask change or checkpoint was written; adjust the hint(s) and retry."
            )
            self.review_status.setText(
                f"Add hints: {len(created)} new, {len(joined)} joined, "
                f"{len(skipped)} skipped. "
                + (f"{details}. " if details else "")
                + (
                    "Slow low-memory correction was used. "
                    if result.processing_mode == "low_memory"
                    else ""
                )
                + checkpoint
            )
            return
        if result.operation in {"dendrite_to_spine", "spine_to_dendrite"}:
            source_type = (
                "dendrite"
                if result.operation == "dendrite_to_spine"
                else "spine"
            )
            destination_type = (
                "spine"
                if result.operation == "dendrite_to_spine"
                else "dendrite"
            )
            self.review_status.setText(
                f"Transferred {result.transferred_voxel_count} {source_type} voxel(s) to "
                f"{destination_type} "
                f"{result.affected_ids[0]}; {result.edit_count} active edit(s). "
                + (
                    "Slow low-memory correction was used. "
                    if result.processing_mode == "low_memory"
                    else ""
                )
                + "An automatic specimen checkpoint was written."
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
        self.review_status.setText(
            f"Correction was not applied; the previous mask is intact. {message}"
        )
        if self.review_operation.currentData() not in {
            "dendrite_to_spine",
            "spine_to_dendrite",
        }:
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
            self.review_brush_diameter,
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
        self._update_review_tool_controls()

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
        self._diagnostic_context_started_at = time.monotonic()
        self._diagnostic_context_phase = None
        self._record_diagnostic(
            "context_generation_started",
            scope="correction" if bool(corrected) else "full",
            specimen_index=int(specimen_index),
            corrected=bool(corrected),
            include_3d=bool(include_3d),
            roi_xy=list(roi_xy) if roi_xy is not None else None,
            request_view=str(request[4]),
        )
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
        if phase != self._diagnostic_context_phase or current >= total:
            corrected = bool(self._context_request[3]) if self._context_request else False
            self._record_diagnostic(
                "context_generation_progress",
                scope="correction" if corrected else "full",
                phase=phase,
                current=current,
                total=total,
                detail=detail,
            )
            self._diagnostic_context_phase = phase

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
        self._record_diagnostic(
            "context_generation_completed",
            scope="correction" if bool(self._context_request[3]) else "full",
            duration_seconds=round(
                time.monotonic() - self._diagnostic_context_started_at, 6
            ),
            corrected=bool(self._context_request[3]),
            include_3d=bool(self._context_request[6]),
            request_view=str(self._context_request[4]),
        )
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
        corrected = bool(self._context_request[3]) if self._context_request else False
        self._record_diagnostic(
            "context_generation_failed",
            scope="correction" if corrected else "full",
            duration_seconds=round(
                time.monotonic() - self._diagnostic_context_started_at, 6
            ),
            error_message=message,
        )
        QMessageBox.warning(self, "Cannot generate 3D context", message)

    @Slot(str)
    def _context_cancelled(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    @Slot()
    def _context_finished(self) -> None:
        review_projection_finished = bool(
            self._context_request is not None
            and self._context_request[4] == "review_main"
        )
        if self._context_worker is not None:
            self._context_worker.deleteLater()
        self._context_worker = None
        self._context_thread = None
        self._context_request = None
        self.context_status_widget.setVisible(False)
        if review_projection_finished:
            self._review_projection_loading = False
            self._update_review_tool_controls()
        elif (
            self.review_view_mode.currentData() == "xy_max"
            and self._last_review_context is None
            and self.manifest is not None
            and self.review_specimen.currentData() is not None
        ):
            QTimer.singleShot(0, lambda: self._load_review_view(auto_contrast=False))

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
        self.tabs.setTabEnabled(5, False)
        self.tabs.setTabEnabled(6, False)
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
        self._diagnostic_job_phase = None
        batch_widgets = self._batch_progress_widgets(kind)
        if batch_widgets is not None:
            batch_bar, batch_label = batch_widgets
            batch_bar.setRange(0, 100)
            batch_bar.setValue(0)
            batch_bar.setFormat("Starting…")
            batch_label.setText("Batch progress: starting; estimating remaining time…")
        self._set_job_running(True)
        self._record_diagnostic("operation_started", kind=kind)
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
        if phase != self._diagnostic_job_phase or current >= total:
            self._record_diagnostic(
                "operation_progress",
                kind=self._job_kind,
                phase=phase,
                current=current,
                total=total,
                detail=detail,
            )
            self._diagnostic_job_phase = phase

    @Slot(str)
    def _job_failed(self, message: str) -> None:
        self._record_diagnostic(
            "operation_failed",
            kind=self._job_kind,
            duration_seconds=round(
                max(0.0, time.monotonic() - self._progress_started_at), 6
            ),
            error_message=message,
        )
        if self._job_kind == "preprocess" and self.manifest is not None:
            self._pending_detection_after_preprocess = None
            self._pending_detection_ready = None
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
        self._record_diagnostic(
            "operation_finished",
            kind=finished_kind,
            duration_seconds=round(
                max(0.0, time.monotonic() - self._progress_started_at), 6
            ),
        )
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
        elif finished_kind == "preprocess":
            pending = getattr(self, "_pending_detection_ready", None)
            self._pending_detection_after_preprocess = None
            self._pending_detection_ready = None
            if pending is not None:
                QTimer.singleShot(
                    0, lambda index=pending: self._run_selected_detection(index, force=True)
                )
        elif finished_kind == "transfer_import" and self._pending_transfer_recovery:
            request = self._pending_transfer_recovery
            self._pending_transfer_recovery = None
            QTimer.singleShot(
                0,
                lambda saved=request: self._start_transfer_import(
                    saved[0],
                    saved[1],
                    saved[2],
                    saved[3],
                    recover_as_settings_only=True,
                ),
            )

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
            self._record_diagnostic_project_context()
            self._record_diagnostic(
                "project_saved", project_path=self.project_path
            )
            self._prepare_preprocessing_tab()
            self._prepare_detection_tab()
            self._prepare_review_tab()
            self._prepare_measurements_tab()
            self._prepare_morphology_tab()
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
            "Open Synpo project or transfer ZIP",
            "",
            "Synpo projects and transfers (*.synpo.json *.synpo-transfer.zip);;"
            "Synpo project (*.synpo.json);;Synpo transfer (*.synpo-transfer.zip);;"
            "JSON files (*.json)",
        )
        if not selected:
            return
        if selected.casefold().endswith(TRANSFER_SUFFIX):
            self._open_transfer_archive(Path(selected))
            return
        self._activate_project(Path(selected))

    def _activate_project(self, selected: Path) -> None:
        try:
            manifest = load_project(selected)
            recovered_measurements = recover_compatible_measurement_checkpoints(
                manifest
            )
            recovery_save_error = ""
            if recovered_measurements:
                try:
                    save_project(selected, manifest)
                except OSError as exc:
                    recovery_save_error = str(exc)
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
        self.project_path = selected.resolve()
        self._record_diagnostic_project_context()
        self._record_diagnostic(
            "project_opened", project_path=self.project_path
        )
        self._populate_manifest(manifest)
        self._prepare_preprocessing_tab()
        self._prepare_detection_tab()
        self._prepare_review_tab()
        self._prepare_measurements_tab()
        self._prepare_morphology_tab()
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
        if recovered_measurements:
            recovery_text = (
                f" Recovered {len(recovered_measurements)} exact compatible "
                "measurement checkpoint(s) from the existing cache."
            )
            if recovery_save_error:
                recovery_text += (
                    " The recovery is active for this session but could not be saved: "
                    f"{recovery_save_error}"
                )
            self.summary_label.setText(self.summary_label.text() + recovery_text)
        self.setWindowTitle(f"Synpo Microscopy Processor — {self.project_path.name}")

    def _create_transfer_zip(self) -> None:
        if self.manifest is None or self.project_path is None:
            QMessageBox.information(
                self, "No saved project", "Open or save a project before creating a transfer ZIP."
            )
            return
        try:
            self._sync_manifest_edits()
            self.project_path = save_project(self.project_path, self.manifest)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot create transfer", str(exc))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Create transfer ZIP")
        layout = QVBoxLayout(dialog)
        explanation = QLabel(
            "Full project state includes cached preprocessing, detection, manual corrections, "
            "and measurements. Settings only keeps parameters, labels, exclusions, ROIs, and "
            "specimen comments, but starts analysis from the beginning."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        mode_combo = QComboBox()
        mode_combo.addItem("Full project state", "full")
        mode_combo.addItem("Settings only", "settings_only")
        form.addRow("Transfer contents:", mode_combo)
        include_raw = QCheckBox("Include raw TIFF files")
        include_raw.setChecked(False)
        form.addRow(include_raw)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        base = self.project_path.name[: -len(".synpo.json")]
        suggested = self.project_path.parent / f"{base}{TRANSFER_SUFFIX}"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Synpo transfer ZIP",
            str(suggested),
            "Synpo transfer (*.synpo-transfer.zip)",
        )
        if not selected:
            return
        worker = TransferCreateWorker(
            self.manifest,
            self.project_path,
            Path(selected),
            str(mode_combo.currentData()),
            include_raw.isChecked(),
        )
        worker.completed.connect(self._transfer_created)
        self._start_worker(worker, "transfer_create")

    @Slot(object)
    def _transfer_created(self, archive_path: Path) -> None:
        self.statusBar().showMessage(f"Created transfer ZIP: {archive_path}", 12000)
        QMessageBox.information(
            self,
            "Transfer created",
            f"The verified transfer ZIP was created successfully:\n\n{archive_path}",
        )

    def _open_transfer_archive(self, archive_path: Path) -> None:
        try:
            info = inspect_transfer_archive(archive_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot open transfer", str(exc))
            return
        destination = QFileDialog.getExistingDirectory(
            self,
            "Choose where to create the transferred project folder",
            str(archive_path.parent),
        )
        if not destination:
            return
        raw_directory: Path | None = None
        if not info.include_raw:
            selected_raw = QFileDialog.getExistingDirectory(
                self,
                "Select the folder containing the raw TIFF files",
                str(archive_path.parent),
            )
            if not selected_raw:
                return
            raw_directory = Path(selected_raw)

        folder_name = re.sub(
            r"[^A-Za-z0-9._-]+", "-", info.project_name[: -len(".synpo.json")]
        ).strip(".-") or "Synpo-project"
        existing = Path(destination).resolve() / folder_name
        conflict_policy = "copy"
        if existing.exists():
            answer = QMessageBox.question(
                self,
                "Project folder already exists",
                f"{existing} already exists.\n\n"
                "Choose Yes to replace it, No to create another numbered copy, "
                "or Cancel to stop. Replacing removes the old folder only after the "
                "transfer has passed validation.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            conflict_policy = (
                "replace" if answer == QMessageBox.StandardButton.Yes else "copy"
            )
        self._start_transfer_import(
            archive_path,
            Path(destination),
            raw_directory,
            conflict_policy,
            recover_as_settings_only=False,
        )

    def _start_transfer_import(
        self,
        archive_path: Path,
        destination: Path,
        raw_directory: Path | None,
        conflict_policy: str,
        *,
        recover_as_settings_only: bool,
    ) -> None:
        worker = TransferImportWorker(
            archive_path,
            destination,
            raw_directory,
            conflict_policy,
            recover_as_settings_only,
        )
        request = (archive_path, destination, raw_directory, conflict_policy)
        worker.completed.connect(
            lambda result, saved_request=request: self._transfer_import_completed(
                result, saved_request
            )
        )
        self._start_worker(worker, "transfer_import")

    def _transfer_import_completed(
        self,
        result: TransferImportResult | TransferCacheError,
        request: tuple[Path, Path, Path | None, str],
    ) -> None:
        if isinstance(result, TransferCacheError):
            answer = QMessageBox.question(
                self,
                "Cached results cannot be restored",
                f"{result}\n\nRecover this archive as a settings-only project instead? "
                "Parameters, labels, exclusions, ROIs, and specimen comments will be kept, "
                "but processing and correction state will be reset.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._pending_transfer_recovery = request
            return
        self._activate_project(result.project_path)
        match_note = (
            f" {result.renamed_source_matches} renamed TIFF file(s) were recovered by "
            "size and SHA-256 checksum."
            if result.renamed_source_matches
            else ""
        )
        recovery_note = (
            " The damaged cache was discarded and the project was recovered as settings-only."
            if result.recovered_as_settings_only
            else ""
        )
        QMessageBox.information(
            self,
            "Transfer imported",
            f"The transferred project is ready:\n\n{result.project_path}\n\n"
            f"The original ZIP was not changed.{match_note}{recovery_note}",
        )

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
                renamed = sum(
                    item.get("matched_by") == "size_and_sha256" for item in results
                )
                duplicates = sum(int(item.get("duplicate_matches", 0)) for item in results)
                message = (
                    "All checksums match. Each source file path was relinked and saved; "
                    "subfolders were searched when necessary."
                )
                if renamed:
                    message += f" {renamed} renamed file(s) were identified by size and SHA-256."
                if duplicates:
                    message += (
                        f" {duplicates} additional identical copy/copies were found; "
                        "the first sorted match was used."
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
        self.tabs.setTabEnabled(5, False)
        self.tabs.setTabEnabled(6, False)
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
            self.filter_export_action,
            self.cluster_export_action,
            self.save_action,
            self.verify_action,
            self.relink_action,
            self.create_transfer_action,
        ):
            action.setEnabled(not running)
        if hasattr(self, "run_preprocessing_button"):
            self.run_preprocessing_button.setEnabled(not running and self.manifest is not None)
            self.apply_preprocessing_button.setEnabled(not running and self.manifest is not None)
            self.run_selected_preprocessing_button.setEnabled(
                not running and self.manifest is not None
            )
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
            self.apply_detection_defaults_button.setEnabled(
                not running and self.manifest is not None
            )
            self.run_selected_detection_button.setEnabled(not running and eligible)
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
            self.accept_all_distribution_spines.setEnabled(
                not running
                and self.manifest is not None
                and getattr(self, "_distribution_bulk_eligible", 0) > 0
            )
            self.export_measurements_button.setEnabled(not running and measured)
            self.volume_filter_enabled.setEnabled(not running and self.manifest is not None)
            self.volume_filter_cutoff.setEnabled(not running and self.manifest is not None)
            self.volume_filter_preview_button.setEnabled(not running and measured)
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
        if hasattr(self, "run_morphology_button"):
            measured = bool(
                self.manifest
                and any(
                    specimen["checkpoints"].get("measurements", {}).get("state") == "complete"
                    for specimen in self.manifest["specimens"]
                )
            )
            self.run_morphology_button.setEnabled(not running and measured)
            self.morphology_color_button.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_axes_color_button.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_background_color_button.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_axes_alpha.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_background_alpha.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_show_legend.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_legend_position.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_custom_x.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_custom_y.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_pca_show_points.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_pca_point_alpha.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_correlation_panel.setEnabled(not running and self._active_morphology_run is not None)
            self.advanced_correlation_panel.setEnabled(not running and self._active_advanced_run is not None)
            self.morphology_reviewed_only.setEnabled(not running and measured)
            self.export_morphology_button.setEnabled(not running and self._active_morphology_run is not None)
            self.morphology_geometry_group.setEnabled(not running and measured)
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
        if self._diagnostics is not None:
            self._finish_diagnostics(package=False, reason="application_closed")
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
