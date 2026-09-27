from __future__ import annotations

import unittest

import numpy as np

from synpo.regions import normalize_rectangles, roi_id_for_mask, roi_mask


class RegionTests(unittest.TestCase):
    def test_rectangles_are_clipped_and_touching_fields_merge(self) -> None:
        rectangles = normalize_rectangles(
            [(-4, 2, 5, 8), (5, 3, 10, 9), (15, 1, 30, 7)],
            (10, 20),
        )

        self.assertEqual(rectangles, [(15, 1, 20, 7), (0, 2, 10, 9)])

    def test_roi_mask_and_assignment_keep_fields_independent(self) -> None:
        rectangles = [(1, 1, 5, 5), (8, 2, 12, 7)]
        selected = roi_mask((8, 14), rectangles)
        self.assertFalse(selected[:, 6].any())
        object_mask = np.zeros_like(selected)
        object_mask[3:6, 9:11] = True

        self.assertEqual(roi_id_for_mask(object_mask, rectangles), 2)


if __name__ == "__main__":
    unittest.main()
