"""
Unit Tests & Controlled Gradient Verification for Spatial Visibility Masking (Stage 2)

Tests:
1. Two-foot image with mixed visibility:
   - Foot 0: Keypoint 1 (heel_back) is labeled (v=2) at (16, 48), bbox [0.25, 0.75, 0.30, 0.35]
   - Foot 1: Keypoint 1 (heel_back) is unlabeled (v=0), bbox [0.75, 0.75, 0.30, 0.35]
2. Verify unlabeled instance region contributes ZERO loss on that keypoint when model predicts activation there.
3. Verify background outside both feet contributes positive penalty when model predicts false positive there.
4. Verify labeled instance region contributes loss when model fails to predict the target.
5. Verify other channels (e.g. keypoints with v=2 on both feet) maintain full unsuppressed supervision.
6. Verify channels with NO labeled instances for that keypoint across the image have mask = 0.0, contributing 0 loss and 0 gradient.
7. Verify exact gradient flow via loss.backward():
   - grad(pred in Foot 1 bbox) == 0.0
   - grad(pred at Foot 0 target) != 0.0
   - grad(pred in background) != 0.0
8. Verify PAF masking remains unaffected and limbs with v=0 endpoints contribute zero loss.
"""

import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.heads.heatmap_head import generate_heatmaps_batch
from models.heads.paf_head import generate_pafs_batch
from losses.heatmap_loss import AdaptiveWingLoss, HeatmapLoss
from losses.paf_loss import PAFLoss


class TestSpatialVisibilityMasking(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        self.heatmap_size = 64
        self.sigma = 3.5
        self.num_k = 13

        # Construct a synthetic 2-foot sample:
        # Image size: 256x256, Heatmap size: 64x64 (factor = 4)
        # Foot 0 (Left Foot): Bbox [cx=0.25, cy=0.60, w=0.30, h=0.40]
        # Foot 1 (Right Foot): Bbox [cx=0.75, cy=0.60, w=0.30, h=0.40]
        self.foot_0 = {
            'class_id': 0,
            'bbox': [0.25, 0.60, 0.30, 0.40],
            'keypoints': [
                [64.0, 100.0, 2],   # 0: toe_ground (v=2)
                [64.0, 200.0, 2],   # 1: heel_back (v=2, labeled on Foot 0)
                [64.0, 220.0, 2],   # 2: heel_ground (v=2)
                [50.0, 130.0, 2],   # 3: ball_medial (v=2)
                [78.0, 130.0, 2],   # 4: ball_lateral (v=2)
                [64.0, 150.0, 2],   # 5: instep_top (v=2)
                [54.0, 170.0, 2],   # 6: arch_medial (v=2)
                [74.0, 170.0, 2],   # 7: midfoot_lateral (v=2)
                [54.0, 210.0, 2],   # 8: malleolus_medial (v=2)
                [74.0, 210.0, 2],   # 9: malleolus_lateral (v=2)
                [64.0, 80.0, 2],    # 10: toe_tip (v=2)
                [64.0, 180.0, 2],   # 11: throat (v=2)
                [64.0, 215.0, 0],   # 12: achilles (v=0 on Foot 0)
            ]
        }

        self.foot_1 = {
            'class_id': 1,
            'bbox': [0.75, 0.60, 0.30, 0.40],
            'keypoints': [
                [192.0, 100.0, 2],  # 0: toe_ground (v=2)
                [0.0, 0.0, 0],      # 1: heel_back (v=0, UNLABELED on Foot 1)
                [192.0, 220.0, 2],  # 2: heel_ground (v=2)
                [178.0, 130.0, 2],  # 3: ball_medial (v=2)
                [206.0, 130.0, 2],  # 4: ball_lateral (v=2)
                [192.0, 150.0, 2],  # 5: instep_top (v=2)
                [182.0, 170.0, 2],  # 6: arch_medial (v=2)
                [202.0, 170.0, 2],  # 7: midfoot_lateral (v=2)
                [182.0, 210.0, 2],  # 8: malleolus_medial (v=2)
                [202.0, 210.0, 2],  # 9: malleolus_lateral (v=2)
                [192.0, 80.0, 2],   # 10: toe_tip (v=2)
                [192.0, 180.0, 2],  # 11: throat (v=2)
                [0.0, 0.0, 0],      # 12: achilles (v=0 on Foot 1)
            ]
        }

        self.batch = [[self.foot_0, self.foot_1]]

    def test_mixed_visibility_mask_structure(self):
        """Test 1: Verify spatial structure of mask for mixed visibility keypoint."""
        heatmaps, masks = generate_heatmaps_batch(
            self.batch, heatmap_size=self.heatmap_size, sigma=self.sigma,
            device=self.device, num_keypoints=self.num_k
        )

        # Keypoint 1: heel_back (labeled on Foot 0, UNLABELED on Foot 1)
        hm_1 = heatmaps[0, 1]
        m_1 = masks[0, 1]

        # 1. Foot 0 target peak must be present (~1.0) around (16, 50)
        foot0_peak_x = int(64.0 * 64.0 / 256.0)  # 16
        foot0_peak_y = int(200.0 * 64.0 / 256.0) # 50
        self.assertAlmostEqual(hm_1[foot0_peak_y, foot0_peak_x].item(), 1.0, places=2)
        self.assertEqual(m_1[foot0_peak_y, foot0_peak_x].item(), 1.0)

        # 2. Foot 1 region must be ignored (mask == 0.0)
        # Foot 1 bbox: cx=0.75, cy=0.60, w=0.30, h=0.40 -> x in [28, 48], y in [25, 51]
        foot1_center_x = int(0.75 * 64)  # 48
        foot1_center_y = int(0.60 * 64)  # 38
        self.assertEqual(m_1[foot1_center_y, foot1_center_x].item(), 0.0,
                         "Unlabeled instance center must have mask=0.0")

        # 3. Background outside both feet (e.g. top-left corner (5, 5)) must be supervised (mask == 1.0)
        self.assertEqual(m_1[5, 5].item(), 1.0,
                         "Background must remain supervised with mask=1.0")
        self.assertAlmostEqual(hm_1[5, 5].item(), 0.0, places=5,
                               msg="Background target must be 0.0")

    def test_unlabeled_instance_zero_loss_penalty(self):
        """Test 2: Verify that predictions inside unlabeled instance produce ZERO loss."""
        heatmaps, masks = generate_heatmaps_batch(
            self.batch, heatmap_size=self.heatmap_size, sigma=self.sigma,
            device=self.device, num_keypoints=self.num_k
        )
        loss_fn = AdaptiveWingLoss(use_mask=True)

        # Perfect prediction on Foot 0, zero in background
        perfect_preds = heatmaps.clone()

        # Compute loss with perfect predictions
        base_loss = loss_fn(perfect_preds, heatmaps, masks=masks).item()
        self.assertAlmostEqual(base_loss, 0.0, places=4)

        # Model emits a confident detection on Foot 1 (where heel_back is unlabeled)
        corrupted_preds = perfect_preds.clone()
        foot1_center_x = int(0.75 * 64)
        foot1_center_y = int(0.60 * 64)
        corrupted_preds[0, 1, foot1_center_y-2:foot1_center_y+2, foot1_center_x-2:foot1_center_x+2] = 0.95

        loss_with_foot1_pred = loss_fn(corrupted_preds, heatmaps, masks=masks).item()
        self.assertAlmostEqual(
            loss_with_foot1_pred, base_loss, places=5,
            msg="Predictions inside unlabeled instance must NOT be penalized (loss must be identical)"
        )

    def test_background_false_positive_is_penalized(self):
        """Test 3: Verify that false positive in true background IS penalized."""
        heatmaps, masks = generate_heatmaps_batch(
            self.batch, heatmap_size=self.heatmap_size, sigma=self.sigma,
            device=self.device, num_keypoints=self.num_k
        )
        loss_fn = AdaptiveWingLoss(use_mask=True)

        perfect_preds = heatmaps.clone()
        base_loss = loss_fn(perfect_preds, heatmaps, masks=masks).item()

        # Emit false positive in background at (5, 5)
        fp_preds = perfect_preds.clone()
        fp_preds[0, 1, 3:7, 3:7] = 0.95

        fp_loss = loss_fn(fp_preds, heatmaps, masks=masks).item()
        self.assertGreater(
            fp_loss, base_loss + 1e-4,
            "False positive in background must be penalized"
        )

    def test_fully_labeled_channel_remains_intact(self):
        """Test 4: Channel 0 (toe_ground) labeled on both feet must have full supervision."""
        heatmaps, masks = generate_heatmaps_batch(
            self.batch, heatmap_size=self.heatmap_size, sigma=self.sigma,
            device=self.device, num_keypoints=self.num_k
        )
        m_0 = masks[0, 0]
        # Should be 1.0 everywhere (no ignore holes)
        self.assertEqual(m_0.min().item(), 1.0)
        self.assertEqual(m_0.max().item(), 1.0)

    def test_completely_unlabeled_channel_zero_loss_and_grad(self):
        """Test 5: Channel 12 (achilles) has v=0 on BOTH feet. Must produce mask=0 and grad=0."""
        heatmaps, masks = generate_heatmaps_batch(
            self.batch, heatmap_size=self.heatmap_size, sigma=self.sigma,
            device=self.device, num_keypoints=self.num_k
        )
        m_12 = masks[0, 12]
        self.assertEqual(m_12.sum().item(), 0.0, "Channel with no labels must have mask sum=0")

        pred = torch.randn(1, self.num_k, 64, 64, requires_grad=True)
        loss_fn = AdaptiveWingLoss(use_mask=True)
        loss = loss_fn(pred, heatmaps, masks=masks)
        loss.backward()

        self.assertEqual(
            pred.grad[0, 12].abs().sum().item(), 0.0,
            "Channel with no labels must produce exactly zero gradients"
        )

    def test_gradient_flow_isolation(self):
        """Test 6: Verify gradient is zero on Foot 1 bbox and non-zero on Foot 0 / background."""
        heatmaps, masks = generate_heatmaps_batch(
            self.batch, heatmap_size=self.heatmap_size, sigma=self.sigma,
            device=self.device, num_keypoints=self.num_k
        )
        # Uniform prediction of 0.5 everywhere
        pred = torch.full((1, self.num_k, 64, 64), 0.5, requires_grad=True)
        loss_fn = AdaptiveWingLoss(use_mask=True)
        loss = loss_fn(pred, heatmaps, masks=masks)
        loss.backward()

        grad_1 = pred.grad[0, 1]  # Keypoint 1 (heel_back)

        # Gradient inside Foot 1's bounding box must be exactly 0
        foot1_cx = int(0.75 * 64)
        foot1_cy = int(0.60 * 64)
        foot1_grad = grad_1[foot1_cy - 2:foot1_cy + 2, foot1_cx - 2:foot1_cx + 2].abs().max().item()
        self.assertEqual(foot1_grad, 0.0, "Gradient inside unlabeled instance region must be exactly 0.0")

        # Gradient at Foot 0's target must be non-zero
        foot0_px = int(64.0 * 64.0 / 256.0)
        foot0_py = int(200.0 * 64.0 / 256.0)
        foot0_grad = grad_1[foot0_py, foot0_px].abs().item()
        self.assertGreater(foot0_grad, 1e-4, "Gradient at labeled target must be non-zero")

        # Gradient in background must be non-zero (pulling 0.5 down toward 0.0)
        bg_grad = grad_1[5, 5].abs().item()
        self.assertGreater(bg_grad, 1e-4, "Gradient in background must be non-zero")

    def test_paf_masking_unchanged(self):
        """Test 7: Verify PAF masking remains correct when limbs have v=0 endpoints."""
        pafs, paf_masks = generate_pafs_batch(
            self.batch, heatmap_size=64, paf_width=8, device=self.device,
            paf_config="shoes_vto/configs/paf_connections_13kp.yaml"
        )
        # Limb 2: heel_ground (2) -> heel_back (1)
        # On Foot 0: both keypoint 2 and keypoint 1 have v=2 -> limb exists!
        # On Foot 1: keypoint 1 has v=0 -> limb does NOT exist!
        # Check that Limb 2 mask is non-zero on Foot 0, but exactly zero on Foot 1
        limb2_mask = paf_masks[0, 2]
        self.assertGreater(limb2_mask.sum().item(), 0.0)

        foot1_cx = int(0.75 * 64)
        foot1_cy = int(0.60 * 64)
        foot1_limb_mask = limb2_mask[foot1_cy-5:foot1_cy+5, foot1_cx-5:foot1_cx+5].sum().item()
        self.assertEqual(foot1_limb_mask, 0.0, "Missing limb on Foot 1 must have PAF mask=0.0")


if __name__ == '__main__':
    unittest.main()
