from __future__ import annotations

import unittest

import numpy as np

from synpo.distribution import calculate_spine_distribution


class DistributionBridgeTests(unittest.TestCase):
    def test_two_component_spine_uses_reviewable_virtual_bridge(self) -> None:
        spine = np.zeros((3, 12, 24), dtype=bool)
        spine[1, 5:8, 3:9] = True
        spine[1, 5:8, 11:19] = True
        dendrite = np.zeros_like(spine)
        dendrite[1, 4:9, 1:3] = True
        clusters = np.zeros_like(spine)
        clusters[1, 5:8, 15:18] = True
        guidance = np.zeros(spine.shape, dtype=np.float32)
        guidance[1, 6, 3:19] = 100.0

        result = calculate_spine_distribution(
            spine,
            dendrite,
            clusters,
            sampling_zyx_um=(0.5, 0.1, 0.1),
            guidance_image=guidance,
            maximum_gap_um=1.0,
        )

        self.assertEqual(result.axis_status, "bridged_gap")
        self.assertTrue(result.bridge_used)
        self.assertGreater(result.bridge_length_um, 0.0)
        self.assertLessEqual(result.bridge_length_um, 1.0)
        self.assertTrue(result.bridge_points_zyx)
        self.assertEqual(sum(result.spine_voxels_by_bin), int(spine.sum()))

    def test_gap_over_limit_retains_reviewable_primary_centerline(self) -> None:
        spine = np.zeros((3, 12, 40), dtype=bool)
        spine[1, 5:8, 2:8] = True
        spine[1, 5:8, 28:36] = True
        dendrite = np.zeros_like(spine)
        dendrite[1, 4:9, :2] = True

        result = calculate_spine_distribution(
            spine,
            dendrite,
            np.zeros_like(spine),
            sampling_zyx_um=(0.5, 0.1, 0.1),
            maximum_gap_um=1.0,
        )

        self.assertEqual(result.axis_status, "unbridged_disconnected_part")
        self.assertFalse(result.bridge_used)
        self.assertIn("exceeds", result.axis_note)
        self.assertTrue(result.axis_points_zyx)
        self.assertEqual(sum(result.spine_voxels_by_bin), int(spine.sum()))

    def test_tiny_satellites_do_not_destroy_an_existing_centerline(self) -> None:
        spine = np.zeros((3, 16, 32), dtype=bool)
        spine[1, 6:9, 3:22] = True
        spine[0, 1, 28:30] = True
        spine[2, 14, 26:29] = True
        dendrite = np.zeros_like(spine)
        dendrite[1, 5:11, 1:3] = True

        result = calculate_spine_distribution(
            spine,
            dendrite,
            np.zeros_like(spine),
            sampling_zyx_um=(0.5, 0.1, 0.1),
        )

        self.assertEqual(result.axis_status, "disconnected_fragments_ignored")
        self.assertNotEqual(result.axis_status, "no_usable_path")
        self.assertTrue(result.axis_points_zyx)
        self.assertEqual(sum(result.spine_voxels_by_bin), int(spine.sum()))

    def test_multiple_substantial_components_keep_primary_path_for_manual_review(self) -> None:
        spine = np.zeros((3, 18, 38), dtype=bool)
        spine[1, 7:10, 2:13] = True
        spine[1, 7:10, 17:27] = True
        spine[1, 7:10, 31:37] = True
        dendrite = np.zeros_like(spine)
        dendrite[1, 6:12, :2] = True

        result = calculate_spine_distribution(
            spine,
            dendrite,
            np.zeros_like(spine),
            sampling_zyx_um=(0.5, 0.1, 0.1),
        )

        self.assertEqual(result.axis_status, "multiple_disconnected_components")
        self.assertTrue(result.axis_points_zyx)
        self.assertFalse(result.bridge_used)

    def test_manual_endpoint_selects_distal_component_before_bridge(self) -> None:
        spine = np.zeros((3, 18, 40), dtype=bool)
        spine[1, 7:10, 2:13] = True
        spine[1, 7:10, 15:25] = True
        spine[1, 7:10, 29:37] = True
        dendrite = np.zeros_like(spine)
        dendrite[1, 6:11, :2] = True
        guidance = np.zeros(spine.shape, dtype=np.float32)
        guidance[1, 8, 2:25] = 100.0
        endpoint = (1, 8, 22)

        result = calculate_spine_distribution(
            spine,
            dendrite,
            np.zeros_like(spine),
            sampling_zyx_um=(0.5, 0.1, 0.1),
            endpoint_hint_zyx=endpoint,
            guidance_image=guidance,
            maximum_gap_um=1.0,
        )

        self.assertEqual(result.axis_status, "bridged_gap")
        self.assertTrue(result.bridge_used)
        self.assertEqual(result.endpoint_source, "manual")
        self.assertEqual(result.endpoint_zyx, endpoint)
        self.assertIn("ignored for centerline topology", result.axis_note)


if __name__ == "__main__":
    unittest.main()
