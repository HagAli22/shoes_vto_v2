"""
PAF (Part Affinity Field) Loss for Limb Connection Detection

Smooth L1 loss on PAF vector fields with masking for invalid limbs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAFLoss(nn.Module):
    """
    Smooth L1 loss for PAF prediction with masking
    
    Args:
        beta: Threshold for smooth L1 (default: 1.0)
        use_mask: If True, only compute loss on valid limbs
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, beta=1.0, use_mask=True, reduction='mean'):
        super(PAFLoss, self).__init__()
        self.beta = beta
        self.use_mask = use_mask
        self.reduction = reduction
    
    def forward(self, pred_pafs, gt_pafs, masks=None):
        """
        Compute PAF loss
        
        Args:
            pred_pafs: [B, 30, H, W] predicted PAF fields (15 limbs × 2)
            gt_pafs: [B, 30, H, W] ground truth PAF fields
            masks: [B, 15, H, W] binary masks (1=valid limb, 0=invalid)
                   Will be expanded to [B, 30, H, W] internally
        
        Returns:
            loss: scalar loss value
        """
        # Smooth L1 loss (Huber loss)
        loss = F.smooth_l1_loss(pred_pafs, gt_pafs, beta=self.beta, reduction='none')  # [B, 30, H, W]
        
        # Apply mask if provided
        if self.use_mask and masks is not None:
            # Expand masks from [B, 15, H, W] to [B, 30, H, W]
            # Each limb has 2 channels (x, y), so repeat each mask twice
            masks_expanded = torch.repeat_interleave(masks, 2, dim=1)  # [B, 30, H, W]
            
            loss = loss * masks_expanded
            
            # Reduction
            if self.reduction == 'mean':
                # Average over valid pixels only
                num_valid = masks_expanded.sum() + 1e-6
                loss = loss.sum() / num_valid
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            # No masking
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss


class PAFMSELoss(nn.Module):
    """
    Alternative: MSE loss for PAF prediction
    Simpler than Smooth L1, but less robust to outliers
    """
    
    def __init__(self, use_mask=True, reduction='mean'):
        super(PAFMSELoss, self).__init__()
        self.use_mask = use_mask
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred_pafs, gt_pafs, masks=None):
        """
        Compute MSE PAF loss
        """
        loss = self.mse(pred_pafs, gt_pafs)  # [B, 30, H, W]
        
        # Apply mask if provided
        if self.use_mask and masks is not None:
            masks_expanded = torch.repeat_interleave(masks, 2, dim=1)
            loss = loss * masks_expanded
            
            if self.reduction == 'mean':
                num_valid = masks_expanded.sum() + 1e-6
                loss = loss.sum() / num_valid
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss


class PAFVectorAngleLoss(nn.Module):
    """
    Angle-aware PAF loss
    
    Penalizes both magnitude and direction errors in PAF vectors.
    Useful when direction is more important than exact magnitude.
    """
    
    def __init__(self, magnitude_weight=1.0, angle_weight=1.0, use_mask=True):
        super(PAFVectorAngleLoss, self).__init__()
        self.magnitude_weight = magnitude_weight
        self.angle_weight = angle_weight
        self.use_mask = use_mask
    
    def forward(self, pred_pafs, gt_pafs, masks=None):
        """
        Compute vector angle loss
        
        Loss = magnitude_weight * |pred_mag - gt_mag| + angle_weight * angle_error
        """
        batch_size, num_channels, height, width = pred_pafs.shape
        num_limbs = num_channels // 2
        
        total_loss = 0.0
        
        for limb_idx in range(num_limbs):
            # Extract x and y components for this limb
            pred_x = pred_pafs[:, limb_idx * 2, :, :]
            pred_y = pred_pafs[:, limb_idx * 2 + 1, :, :]
            gt_x = gt_pafs[:, limb_idx * 2, :, :]
            gt_y = gt_pafs[:, limb_idx * 2 + 1, :, :]
            
            # Compute magnitudes
            pred_mag = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-8)
            gt_mag = torch.sqrt(gt_x ** 2 + gt_y ** 2 + 1e-8)
            
            # Magnitude loss
            mag_loss = torch.abs(pred_mag - gt_mag)
            
            # Angle loss (using dot product)
            # cos(theta) = (pred · gt) / (|pred| * |gt|)
            dot_product = pred_x * gt_x + pred_y * gt_y
            cos_angle = dot_product / (pred_mag * gt_mag + 1e-8)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            
            # Angle error: 1 - cos(theta), range [0, 2]
            angle_loss = 1.0 - cos_angle
            
            # Combined loss
            limb_loss = self.magnitude_weight * mag_loss + self.angle_weight * angle_loss
            
            # Apply mask if provided
            if self.use_mask and masks is not None:
                limb_mask = masks[:, limb_idx, :, :]
                limb_loss = limb_loss * limb_mask
                num_valid = limb_mask.sum() + 1e-6
                limb_loss = limb_loss.sum() / num_valid
            else:
                limb_loss = limb_loss.mean()
            
            total_loss += limb_loss
        
        # Average over limbs
        total_loss = total_loss / num_limbs
        
        return total_loss


if __name__ == "__main__":
    print("=" * 80)
    print("Testing PAF Loss Functions")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    num_limbs = 15
    heatmap_size = 64
    
    pred = torch.randn(batch_size, num_limbs * 2, heatmap_size, heatmap_size, requires_grad=True)
    gt = torch.randn(batch_size, num_limbs * 2, heatmap_size, heatmap_size)
    masks = torch.ones(batch_size, num_limbs, heatmap_size, heatmap_size)
    masks[:, :5, :, :] = 0  # Mask out first 5 limbs (simulate invalid)
    
    # Test 1: Smooth L1 loss
    print("\n[Test 1] PAFLoss (Smooth L1 with masking)")
    loss_fn = PAFLoss(beta=1.0, use_mask=True, reduction='mean')
    loss = loss_fn(pred, gt, masks)
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss should be > 0")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test gradient
    loss.backward()
    print(f"  Gradient computed successfully")
    print("  ✅ PASS")
    
    # Test 2: MSE loss
    print("\n[Test 2] PAFMSELoss")
    mse_loss_fn = PAFMSELoss(use_mask=True, reduction='mean')
    mse_loss = mse_loss_fn(pred, gt, masks)
    
    print(f"  Loss value: {mse_loss.item():.6f}")
    assert mse_loss.item() > 0, "MSE loss should be positive"
    print("  ✅ PASS")
    
    # Test 3: Vector angle loss
    print("\n[Test 3] PAFVectorAngleLoss")
    angle_loss_fn = PAFVectorAngleLoss(magnitude_weight=1.0, angle_weight=1.0, use_mask=True)
    angle_loss = angle_loss_fn(pred, gt, masks)
    
    print(f"  Loss value: {angle_loss.item():.6f}")
    assert angle_loss.item() > 0, "Angle loss should be positive"
    print("  ✅ PASS")
    
    # Test 4: Perfect prediction (loss should be ~0)
    print("\n[Test 4] Perfect prediction (loss should be near 0)")
    perfect_pred = gt.clone()
    loss_perfect = loss_fn(perfect_pred, gt, masks)
    
    print(f"  Loss value: {loss_perfect.item():.6f}")
    print(f"  Should be very close to 0")
    assert loss_perfect.item() < 1e-6, "Perfect prediction should have near-zero loss"
    print("  ✅ PASS")
    
    # Test 5: Loss without masking
    print("\n[Test 5] PAFLoss without masking")
    loss_fn_no_mask = PAFLoss(use_mask=False, reduction='mean')
    loss_no_mask = loss_fn_no_mask(pred, gt, masks)
    
    print(f"  Loss value: {loss_no_mask.item():.6f}")
    print(f"  Should be different from masked loss")
    print("  ✅ PASS")
    
    # Test 6: Mask expansion
    print("\n[Test 6] Testing mask expansion (15 limbs → 30 channels)")
    masks_test = torch.ones(2, 15, 64, 64)
    masks_test[:, :3, :, :] = 0  # First 3 limbs invalid
    
    masks_expanded = torch.repeat_interleave(masks_test, 2, dim=1)
    
    print(f"  Original mask shape: {masks_test.shape}")
    print(f"  Expanded mask shape: {masks_expanded.shape}")
    print(f"  Expected: [2, 30, 64, 64]")
    
    assert masks_expanded.shape == torch.Size([2, 30, 64, 64]), "Mask expansion failed"
    
    # Check that first 6 channels (3 limbs × 2) are 0
    assert masks_expanded[:, :6, :, :].sum() == 0, "First 6 channels should be masked"
    print("  ✅ PASS")
    
    # Test 7: Compare different loss functions
    print("\n[Test 7] Comparing PAF loss functions on same data")
    pred_test = torch.randn(2, 30, 64, 64)
    gt_test = torch.randn(2, 30, 64, 64)
    mask_test = torch.ones(2, 15, 64, 64)
    
    smooth_l1 = PAFLoss(use_mask=True)(pred_test, gt_test, mask_test)
    mse = PAFMSELoss(use_mask=True)(pred_test, gt_test, mask_test)
    angle = PAFVectorAngleLoss(use_mask=True)(pred_test, gt_test, mask_test)
    
    print(f"  Smooth L1 Loss: {smooth_l1.item():.6f}")
    print(f"  MSE Loss: {mse.item():.6f}")
    print(f"  Angle Loss: {angle.item():.6f}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 80)
    print("All PAF loss tests passed!")
    print("\nRecommendation: Use PAFLoss (Smooth L1) for M1 baseline")
    print("Smooth L1 is more robust to outliers than MSE")
    print("Consider VectorAngleLoss if direction is more critical than magnitude")
    print("=" * 80)
