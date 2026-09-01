from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from synpo.app import ContextViewerDialog, Volume3DView
from synpo.visualization import ContextVolume, ProjectionData


class VolumeRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_solid_renderer_and_custom_color(self) -> None:
        empty = np.zeros((8, 8), dtype=np.uint32)
        projection = ProjectionData(
            raw=np.zeros((8, 8), dtype=np.uint16),
            dendrites=empty,
            spines=empty,
            clusters=empty,
        )
        volume = ContextVolume(
            xy=projection,
            xz=projection,
            yz=projection,
            points_um=np.asarray(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.1]],
                dtype=np.float32,
            ),
            point_kinds=np.asarray([2, 2, 2], dtype=np.uint8),
            point_object_ids=np.asarray([1, 1, 1], dtype=np.uint32),
            xy_um_per_pixel=0.1,
            z_step_um=0.5,
            source_mode="test",
            corrected=False,
            z_count=2,
            mesh_vertices_um=np.asarray(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ],
                dtype=np.float32,
            ),
            mesh_faces=np.asarray(
                [
                    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
                ],
                dtype=np.int32,
            ),
            mesh_face_kinds=np.full(12, 2, dtype=np.uint8),
        )
        view = Volume3DView()
        view.resize(640, 540)
        view.set_kind_color(2, QColor(255, 20, 20))
        view.set_z_spacing_factor(2.5)
        view.set_volume(volume)
        view.show()
        self.app.processEvents()
        image = view.grab().toImage()
        self.assertFalse(image.isNull())
        self.assertEqual(view.kind_color(2).name(), "#ff1414")
        self.assertEqual(view.z_spacing_factor(), 2.5)
        view.set_kind_opacity(0, 0.85)
        self.app.processEvents()
        self.assertFalse(view.grab().toImage().isNull())
        self.assertEqual(view.kind_opacity(0), 0.85)
        view.close()

        dialog = ContextViewerDialog("3D controls test", volume, 0)
        dialog.volume_z_spacing.setValue(3.0)
        dialog.volume_opacity_spins[0].setValue(90)
        dialog.volume_view.set_rotation(45.0, -20.0, 90.0)
        self.app.processEvents()
        self.assertEqual(dialog.volume_view.z_spacing_factor(), 3.0)
        self.assertEqual(dialog.volume_view.kind_opacity(0), 0.9)
        self.assertEqual(dialog.volume_view.rotation(), (45.0, -20.0, 90.0))
        self.assertEqual(
            [control.value() for control in dialog.volume_rotation_sliders],
            [45, -20, 90],
        )
        dialog.volume_rotation_spins[1].setValue(35)
        self.assertEqual(dialog.volume_view.rotation(), (45.0, 35.0, 90.0))
        dialog.volume_view.reset_rotation()
        self.assertEqual(dialog.volume_view.rotation(), (25.0, 0.0, -35.0))
        dialog.close()


if __name__ == "__main__":
    unittest.main()
