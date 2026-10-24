"""
Synthetic Unit Tests for PAF & Heatmap Supervision and Masking
Verifies:
1. Two non-overlapping feet both contribute to same limb channels (no overwriting).
2. Overlapping limb vectors are averaged consistently.
3. Limbs with unknown endpoints have uncertain foot regions masked to 0 while background is masked to 1.
4. Heatmap channels with all instances unlabeled supervise background as 0 while ignoring foot boxes.
5. Empty negative images supervise all channels as background (target 0, mask 1).
6. Overlapping bboxes cannot erase positive Gaussian heatmap targets.
7. Gradient flow and loss behavior with PAFLoss and AdaptiveWingLoss.
"""

import sys
import unittest
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.heads.paf_head import generate_pafs_batch, load_paf_config
from models.heads.heatmap_head import generate_heatmaps_batch
from losses.paf_loss import PAFLoss
from losses.heatmap_loss import AdaptiveWingLoss


class TestPAFAndHeatmapSupervision(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.paf_config = load_paf_config("paf_connections_13kp.yaml")
        # Limb 0: [10, 0, "toe_tip -> toe_ground"]
        self.limb_0_from = self.paf_config['limb_pairs'][0][0]  # 10
        self.limb_0_to = self.paf_config['limb_pairs'][0][1]    # 0

    def test_01_two_feet_both_contribute_to_paf(self):
        """Test that two non-overlapping feet both contribute to same limb channels without overwrite."""
        # Foot 0 on left (x: 30..80, y: 50..120), Foot 1 on right (x: 170..220, y: 50..120)
        # Limb 0 connects kp 10 to kp 0
        foot_0_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_0_kps[10] = [40.0, 60.0, 2]  # toe_tip
        foot_0_kps[0] = [40.0, 100.0, 2]  # toe_ground
        
        foot_1_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_1_kps[10] = [200.0, 60.0, 2]  # toe_tip
        foot_1_kps[0] = [200.0, 100.0, 2]  # toe_ground

        batch = [[
            {
                'bbox': [60.0 / 256.0, 80.0 / 256.0, 50.0 / 256.0, 60.0 / 256.0],
                'keypoints': foot_0_kps
            },
            {
                'bbox': [200.0 / 256.0, 80.0 / 256.0, 50.0 / 256.0, 60.0 / 256.0],
                'keypoints': foot_1_kps
            }
        ]]

        pafs, masks = generate_pafs_batch(batch, heatmap_size=64, paf_width=4, device=self.device, paf_config=self.paf_config)

        # Check Foot 0 region on 64x64 grid (x ≈ 10, y ≈ 15..25)
        # Vector points down from y=60 to y=100 -> ux=0, uy=1
        foot_0_paf_y = pafs[0, 1, 20, 10].item()
        foot_1_paf_y = pafs[0, 1, 20, 50].item()

        self.assertAlmostEqual(foot_0_paf_y, 1.0, places=2, msg="Foot 0 PAF limb y-vector missing or overwritten!")
        self.assertAlmostEqual(foot_1_paf_y, 1.0, places=2, msg="Foot 1 PAF limb y-vector missing or overwritten!")

        # Both limb regions must be supervised (mask == 1.0)
        self.assertEqual(masks[0, 0, 20, 10].item(), 1.0)
        self.assertEqual(masks[0, 0, 20, 50].item(), 1.0)

        # True background outside feet (e.g. (5, 5)) must have mask == 1.0 and paf == 0.0
        self.assertEqual(masks[0, 0, 5, 5].item(), 1.0, "Background outside feet must be supervised!")
        self.assertEqual(pafs[0, 0, 5, 5].item(), 0.0, "Background PAF x must be 0.0!")
        self.assertEqual(pafs[0, 1, 5, 5].item(), 0.0, "Background PAF y must be 0.0!")

    def test_02_overlapping_limb_vectors_averaged(self):
        """Test that overlapping limb vectors are averaged instead of overwritten."""
        # Foot 0 vector pointing right: ux=1, uy=0
        foot_0_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_0_kps[10] = [100.0, 128.0, 2]
        foot_0_kps[0] = [156.0, 128.0, 2]

        # Foot 1 vector pointing down: ux=0, uy=1, crossing at the middle
        foot_1_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_1_kps[10] = [128.0, 100.0, 2]
        foot_1_kps[0] = [128.0, 156.0, 2]

        batch = [[
            {'bbox': [0.5, 0.5, 0.3, 0.3], 'keypoints': foot_0_kps},
            {'bbox': [0.5, 0.5, 0.3, 0.3], 'keypoints': foot_1_kps}
        ]]

        pafs, masks = generate_pafs_batch(batch, heatmap_size=64, paf_width=4, device=self.device, paf_config=self.paf_config)

        # Center intersection pixel (32, 32)
        cx, cy = 32, 32
        paf_x = pafs[0, 0, cy, cx].item()
        paf_y = pafs[0, 1, cy, cx].item()

        # At the intersection, both limbs contribute: ux=(1+0)/2 = 0.5, uy=(0+1)/2 = 0.5
        self.assertAlmostEqual(paf_x, 0.5, places=2, msg=f"Expected averaged paf_x=0.5, got {paf_x}")
        self.assertAlmostEqual(paf_y, 0.5, places=2, msg=f"Expected averaged paf_y=0.5, got {paf_y}")
        self.assertEqual(masks[0, 0, cy, cx].item(), 1.0)

    def test_03_uncertain_limb_endpoints_masked(self):
        """Test that limbs with unlabeled endpoints have foot bboxes masked to 0 while background remains 1."""
        # Foot 0: Limb 0 labeled
        foot_0_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_0_kps[10] = [40.0, 60.0, 2]
        foot_0_kps[0] = [40.0, 100.0, 2]

        # Foot 1: Limb 0 UNLABELED (v=0 on kp 10)
        foot_1_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_1_kps[10] = [0.0, 0.0, 0]  # Unlabeled
        foot_1_kps[0] = [200.0, 100.0, 2]

        foot_1_bbox = [200.0 / 256.0, 80.0 / 256.0, 50.0 / 256.0, 60.0 / 256.0]

        batch = [[
            {'bbox': [60.0 / 256.0, 80.0 / 256.0, 50.0 / 256.0, 60.0 / 256.0], 'keypoints': foot_0_kps},
            {'bbox': foot_1_bbox, 'keypoints': foot_1_kps}
        ]]

        pafs, masks = generate_pafs_batch(batch, heatmap_size=64, paf_width=4, device=self.device, paf_config=self.paf_config)

        # Foot 0 limb has valid supervision (mask=1.0, paf_y=1.0)
        self.assertEqual(masks[0, 0, 20, 10].item(), 1.0)
        self.assertAlmostEqual(pafs[0, 1, 20, 10].item(), 1.0, places=2)

        # Inside Foot 1 bbox (e.g. y=20, x=50 on 64x64 grid), mask must be 0.0 (ignored)
        self.assertEqual(masks[0, 0, 20, 50].item(), 0.0, "Foot 1 bbox must be masked out when endpoints are unlabeled!")

        # Pure background (5, 5) must remain supervised (mask=1.0)
        self.assertEqual(masks[0, 0, 5, 5].item(), 1.0, "Background outside feet must be supervised as 1.0!")

    def test_04_all_unlabeled_heatmap_channel_supervises_background(self):
        """Test that a heatmap channel with no labeled instances supervises background as 0 while ignoring foot bboxes."""
        # Keypoint 5 (instep_top) is unlabeled (v=0) on both feet
        foot_0_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_0_kps[0] = [40.0, 100.0, 2]  # Only kp 0 labeled
        
        foot_1_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_1_kps[0] = [200.0, 100.0, 2]

        batch = [[
            {'bbox': [60.0 / 256.0, 80.0 / 256.0, 50.0 / 256.0, 60.0 / 256.0], 'keypoints': foot_0_kps},
            {'bbox': [200.0 / 256.0, 80.0 / 256.0, 50.0 / 256.0, 60.0 / 256.0], 'keypoints': foot_1_kps}
        ]]

        heatmaps, masks = generate_heatmaps_batch(batch, heatmap_size=64, sigma=3.5, device=self.device, num_keypoints=13)

        # For kp 5:
        # 1. Target must be 0.0 everywhere
        self.assertEqual(heatmaps[0, 5].sum().item(), 0.0)

        # 2. Inside Foot 0 and Foot 1 bboxes, mask must be 0.0 (ignored)
        self.assertEqual(masks[0, 5, 20, 15].item(), 0.0, "Foot 0 bbox should be masked to 0 for unlabeled kp 5!")
        self.assertEqual(masks[0, 5, 20, 50].item(), 0.0, "Foot 1 bbox should be masked to 0 for unlabeled kp 5!")

        # 3. Outside both bboxes (background, e.g. (5, 5)), mask must be 1.0 (supervised negative)!
        self.assertEqual(masks[0, 5, 5, 5].item(), 1.0, "Background must be supervised with mask=1.0!")

    def test_05_empty_negative_image_supervises_background(self):
        """Test that images with no instances supervise all heatmap and PAF channels as background."""
        batch = [[]]  # Empty image

        heatmaps, hm_masks = generate_heatmaps_batch(batch, heatmap_size=64, sigma=3.5, device=self.device, num_keypoints=13)
        pafs, paf_masks = generate_pafs_batch(batch, heatmap_size=64, paf_width=4, device=self.device, paf_config=self.paf_config, num_limbs=16)

        self.assertEqual(heatmaps.shape, (1, 13, 64, 64))
        self.assertEqual(hm_masks.shape, (1, 13, 64, 64))
        self.assertEqual(pafs.shape, (1, 32, 64, 64))
        self.assertEqual(paf_masks.shape, (1, 16, 64, 64))

        # Targets are 0 everywhere
        self.assertEqual(heatmaps.sum().item(), 0.0)
        self.assertEqual(pafs.sum().item(), 0.0)

        # Masks are 1 everywhere (full background negative supervision)
        self.assertEqual(hm_masks.sum().item(), 1 * 13 * 64 * 64)
        self.assertEqual(paf_masks.sum().item(), 1 * 16 * 64 * 64)

    def test_06_overlapping_bboxes_preserve_positive_targets(self):
        """Test that an overlapping bbox with v=0 cannot erase a valid positive Gaussian peak."""
        # Foot 0 has kp 0 labeled at (100, 100)
        foot_0_kps = [[0.0, 0.0, 0] for _ in range(13)]
        foot_0_kps[0] = [100.0, 100.0, 2]

        # Foot 1 has kp 0 unlabeled (v=0), but its bbox overlaps Foot 0's location!
        foot_1_kps = [[0.0, 0.0, 0] for _ in range(13)]

        batch = [[
            {'bbox': [100.0 / 256.0, 100.0 / 256.0, 0.3, 0.3], 'keypoints': foot_0_kps},
            {'bbox': [100.0 / 256.0, 100.0 / 256.0, 0.3, 0.3], 'keypoints': foot_1_kps}
        ]]

        heatmaps, masks = generate_heatmaps_batch(batch, heatmap_size=64, sigma=3.5, device=self.device, num_keypoints=13)

        # Peak at hm_x = 100*(64/256) = 25, hm_y = 25
        peak_val = heatmaps[0, 0, 25, 25].item()
        mask_val = masks[0, 0, 25, 25].item()

        self.assertAlmostEqual(peak_val, 1.0, places=2, msg="Positive Gaussian peak was erased!")
        self.assertEqual(mask_val, 1.0, msg="Positive Gaussian peak region must have mask=1.0!")

    def test_07_gradient_flow_and_loss(self):
        """Test gradient backpropagation through PAFLoss and AdaptiveWingLoss with new masks."""
        batch = [[
            {
                'bbox': [0.5, 0.5, 0.4, 0.4],
                'keypoints': [[100.0, 100.0, 2], [150.0, 150.0, 0]] + [[0.0, 0.0, 0]] * 11
            }
        ]]

        gt_hm, hm_masks = generate_heatmaps_batch(batch, heatmap_size=64, sigma=3.5, device=self.device, num_keypoints=13)
        gt_paf, paf_masks = generate_pafs_batch(batch, heatmap_size=64, paf_width=4, device=self.device, paf_config=self.paf_config, num_limbs=16)

        pred_hm = torch.zeros(1, 13, 64, 64, requires_grad=True)
        pred_paf = torch.zeros(1, 32, 64, 64, requires_grad=True)

        hm_loss_fn = AdaptiveWingLoss(use_mask=True)
        paf_loss_fn = PAFLoss(use_mask=True)

        hm_loss = hm_loss_fn(pred_hm, gt_hm, masks=hm_masks)
        paf_loss = paf_loss_fn(pred_paf, gt_paf, masks=paf_masks)

        total_loss = hm_loss + paf_loss
        total_loss.backward()

        self.assertFalse(torch.isnan(pred_hm.grad).any(), "NaN found in heatmap gradients!")
        self.assertFalse(torch.isnan(pred_paf.grad).any(), "NaN found in PAF gradients!")

        # Keypoint 0 has positive target -> non-zero grad
        self.assertTrue((pred_hm.grad[0, 0].abs() > 0).any(), "Expected non-zero gradients on supervised keypoint 0!")

        # Keypoint 1 is unlabeled inside bbox -> gradient inside bbox should be zero where masked out
        # Bbox is center 32, size ~25 px
        self.assertEqual(pred_hm.grad[0, 1, 32, 32].item(), 0.0, "Masked out pixel received gradient!")


if __name__ == '__main__':
    unittest.main()

