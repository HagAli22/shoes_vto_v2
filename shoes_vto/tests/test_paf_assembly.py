"""
Unit test for OpenPose-inspired PAF-led assembly in keypoint_grouping.py
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.keypoint_grouping import decode_paf_instances_single, load_paf_config
from models.heads.paf_head import generate_pafs_batch
from models.heads.heatmap_head import generate_heatmaps_batch


class TestPAFAssembly(unittest.TestCase):
    def setUp(self):
        self.paf_config = load_paf_config("paf_connections_13kp.yaml")

    def test_01_single_wide_foot_not_split(self):
        """Verify that a single wide foot is assembled into 1 instance, eliminating ghost split."""
        # Simulated wide foot with lateral points at x=15 and medial points at x=45
        kps = [[0.0, 0.0, 0] for _ in range(13)]
        # Keypoints along the foot chain
        kps[10] = [30.0 * 4, 15.0 * 4, 2]  # toe_tip (x=30, y=15 on 64x64)
        kps[0]  = [30.0 * 4, 25.0 * 4, 2]  # toe_ground
        kps[2]  = [30.0 * 4, 50.0 * 4, 2]  # heel_ground
        kps[1]  = [30.0 * 4, 58.0 * 4, 2]  # heel_back
        kps[3]  = [45.0 * 4, 30.0 * 4, 2]  # ball_medial (medial border x=45)
        kps[4]  = [15.0 * 4, 30.0 * 4, 2]  # ball_lateral (lateral border x=15)
        kps[5]  = [30.0 * 4, 35.0 * 4, 2]  # instep_top
        kps[6]  = [42.0 * 4, 45.0 * 4, 2]  # arch_medial (x=42)
        kps[7]  = [18.0 * 4, 45.0 * 4, 2]  # midfoot_lateral (x=18)
        kps[11] = [30.0 * 4, 42.0 * 4, 2]  # throat
        kps[8]  = [38.0 * 4, 52.0 * 4, 2]  # malleolus_medial
        kps[9]  = [22.0 * 4, 52.0 * 4, 2]  # malleolus_lateral
        kps[12] = [30.0 * 4, 55.0 * 4, 2]  # achilles

        inst = {
            'class_id': 0,
            'class_name': 'left_foot',
            'bbox': [0.5, 0.5, 0.6, 0.7],
            'keypoints': kps
        }

        # Generate clean synthetic heatmaps and PAFs
        heatmaps, _ = generate_heatmaps_batch([[inst]], heatmap_size=64, sigma=2.0, device='cpu', num_keypoints=13)
        pafs, _ = generate_pafs_batch([[inst]], heatmap_size=64, paf_width=6, device='cpu', paf_config=self.paf_config, num_limbs=16)

        class_probs = torch.zeros(2, 64, 64)
        class_probs[0] = 0.9  # left_foot
        class_probs[1] = 0.1

        instances = decode_paf_instances_single(
            heatmaps=heatmaps[0],
            pafs=pafs[0],
            class_probs=class_probs,
            peak_threshold=0.10,
            paf_threshold=0.05,
            paf_config=self.paf_config
        )

        self.assertEqual(len(instances), 1, f"Expected exactly 1 foot instance, got {len(instances)} (Ghost split occurred!)")
        inst_0 = instances[0]
        self.assertEqual(inst_0['class_id'], 0)
        self.assertEqual(inst_0['num_keypoints'], 13)
        # Verify both medial and lateral points are in the same instance
        self.assertGreater(inst_0['keypoints'][3][2], 0.0, "ball_medial missing!")
        self.assertGreater(inst_0['keypoints'][4][2], 0.0, "ball_lateral missing!")

    def test_02_two_distinct_feet_both_recovered(self):
        """Verify that two distinct physical feet are assembled into 2 separate instances."""
        # Foot 0 on left (center x ≈ 18)
        kps_0 = [[0.0, 0.0, 0] for _ in range(13)]
        for k in range(13):
            kps_0[k] = [(18.0 + (k % 3) * 2.0) * 4, (20.0 + k * 2.5) * 4, 2]

        # Foot 1 on right (center x ≈ 48)
        kps_1 = [[0.0, 0.0, 0] for _ in range(13)]
        for k in range(13):
            kps_1[k] = [(48.0 + (k % 3) * 2.0) * 4, (20.0 + k * 2.5) * 4, 2]

        batch = [[
            {'class_id': 0, 'class_name': 'left_foot', 'bbox': [18/64, 0.5, 0.25, 0.6], 'keypoints': kps_0},
            {'class_id': 1, 'class_name': 'right_foot', 'bbox': [48/64, 0.5, 0.25, 0.6], 'keypoints': kps_1}
        ]]

        heatmaps, _ = generate_heatmaps_batch(batch, heatmap_size=64, sigma=2.0, device='cpu', num_keypoints=13)
        pafs, _ = generate_pafs_batch(batch, heatmap_size=64, paf_width=6, device='cpu', paf_config=self.paf_config, num_limbs=16)

        class_probs = torch.zeros(2, 64, 64)
        class_probs[0, :, :32] = 0.95  # Left side is left foot
        class_probs[1, :, 32:] = 0.95  # Right side is right foot

        instances = decode_paf_instances_single(
            heatmaps=heatmaps[0],
            pafs=pafs[0],
            class_probs=class_probs,
            peak_threshold=0.10,
            paf_threshold=0.05,
            paf_config=self.paf_config
        )

        self.assertEqual(len(instances), 2, f"Expected exactly 2 foot instances, got {len(instances)}")
        # Verify class assignment
        classes = {inst['class_id'] for inst in instances}
        self.assertEqual(classes, {0, 1}, f"Expected classes {{0, 1}}, got {classes}")


if __name__ == '__main__':
    unittest.main()

