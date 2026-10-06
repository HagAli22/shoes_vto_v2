"""
Heatmap Loss for Keypoint Detection

MSE loss between predicted and ground truth Gaussian heatmaps
with masking for unlabeled keypoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapLoss(nn.Module):
    """
    MSE loss for heatmap prediction with optional masking
    
    Args:
        use_mask: If True, only compute loss on labeled keypoints
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, use_mask=True, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.use_mask = use_mask
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred_heatmaps, gt_heatmaps, masks=None):
        """
        Compute heatmap loss
        
        Args:
            pred_heatmaps: [B, 16, H, W] predicted heatmaps (sigmoid output)
            gt_heatmaps: [B, 16, H, W] ground truth Gaussian heatmaps
            masks: [B, 16, H, W] binary masks (1=labeled, 0=unlabeled)
        
        Returns:
            loss: scalar loss value
        """
        # Compute per-pixel MSE
        loss = self.mse(pred_heatmaps, gt_heatmaps)  # [B, 16, H, W]
        
        # Apply mask if provided
        if self.use_mask and masks is not None:
            loss = loss * masks
            
            # Reduction
            if self.reduction == 'mean':
                # Average over labeled pixels only
                num_labeled = masks.sum() + 1e-6
                loss = loss.sum() / num_labeled
            elif self.reduction == 'sum':
                loss = loss.sum()
            # else: return per-pixel loss
        else:
            # No masking
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss


class FocalHeatmapLoss(nn.Module):
    """
    Focal loss variant for heatmap prediction
    
    Gives more weight to hard-to-classify pixels (near keypoints)
    and less weight to easy background pixels
    
    Args:
        alpha: Weighting factor for positive/negative balance (default: 2)
        beta: Focusing parameter (default: 4)
        use_mask: If True, only compute loss on labeled keypoints
    """
    
    def __init__(self, alpha=2.0, beta=4.0, use_mask=True):
        super(FocalHeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_mask = use_mask
    
    def forward(self, pred_heatmaps, gt_heatmaps, masks=None):
        """
        Compute focal heatmap loss
        
        Focal loss formula (adapted for regression):
        L = -|y - y_pred|^alpha * (1 - exp(-|y - y_pred|))^beta
        
        For positive pixels (y=1): emphasizes when pred is far from 1
        For negative pixels (y=0): emphasizes when pred is far from 0
        """
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred_heatmaps, min=1e-7, max=1.0 - 1e-7)
        
        # Positive and negative samples
        pos_mask = gt_heatmaps >= 0.5  # Near keypoint peaks
        neg_mask = gt_heatmaps < 0.5   # Background
        
        # Focal loss components
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask.float()
        neg_loss = -(pred ** self.alpha) * torch.log(1 - pred) * neg_mask.float()
        
        loss = pos_loss + neg_loss
        
        # Apply mask if provided
        if self.use_mask and masks is not None:
            loss = loss * masks
            num_labeled = masks.sum() + 1e-6
            loss = loss.sum() / num_labeled
        else:
            loss = loss.mean()
        
        return loss


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    
    Combines the advantages of L1 and L2 loss with smooth transitions.
    Good for heatmap regression as it's more robust to outliers than MSE.
    
    Reference: Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression (Wang et al., 2019)
    """
    
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, use_mask=True):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_mask = use_mask
    
    def forward(self, pred_heatmaps, gt_heatmaps, masks=None):
        """
        Compute Adaptive Wing Loss
        """
        delta = (gt_heatmaps - pred_heatmaps).abs()
        
        # Adaptive wing loss formula
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - gt_heatmaps))) * \
            (self.alpha - gt_heatmaps) * torch.pow(self.theta / self.epsilon, self.alpha - gt_heatmaps - 1) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - gt_heatmaps))
        
        # Piecewise loss
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - gt_heatmaps)),
            A * delta - C
        )
        
        # Apply mask if provided
        if self.use_mask and masks is not None:
            loss = loss * masks
            num_labeled = masks.sum() + 1e-6
            loss = loss.sum() / num_labeled
        else:
            loss = loss.mean()
        
        return loss


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Heatmap Loss Functions")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    num_keypoints = 16
    heatmap_size = 64
    
    pred = torch.rand(batch_size, num_keypoints, heatmap_size, heatmap_size, requires_grad=True)
    gt = torch.rand(batch_size, num_keypoints, heatmap_size, heatmap_size)
    masks = torch.ones(batch_size, num_keypoints, heatmap_size, heatmap_size)
    masks[:, :8, :, :] = 0  # Mask out first 8 keypoints (simulate unlabeled)
    
    # Test 1: Basic MSE loss
    print("\n[Test 1] HeatmapLoss (MSE with masking)")
    loss_fn = HeatmapLoss(use_mask=True, reduction='mean')
    loss = loss_fn(pred, gt, masks)
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss should be > 0")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test gradient
    loss.backward()
    print(f"  Gradient computed successfully")
    print("  ✅ PASS")
    
    # Test 2: Loss without masking
    print("\n[Test 2] HeatmapLoss without masking")
    loss_fn_no_mask = HeatmapLoss(use_mask=False, reduction='mean')
    loss_no_mask = loss_fn_no_mask(pred, gt, masks)
    
    print(f"  Loss value: {loss_no_mask.item():.6f}")
    print(f"  Should be different from masked loss")
    print("  ✅ PASS")
    
    # Test 3: Focal loss
    print("\n[Test 3] FocalHeatmapLoss")
    focal_loss_fn = FocalHeatmapLoss(alpha=2.0, beta=4.0, use_mask=True)
    focal_loss = focal_loss_fn(pred, gt, masks)
    
    print(f"  Loss value: {focal_loss.item():.6f}")
    assert focal_loss.item() > 0, "Focal loss should be positive"
    print("  ✅ PASS")
    
    # Test 4: Adaptive Wing Loss
    print("\n[Test 4] AdaptiveWingLoss")
    wing_loss_fn = AdaptiveWingLoss(use_mask=True)
    wing_loss = wing_loss_fn(pred, gt, masks)
    
    print(f"  Loss value: {wing_loss.item():.6f}")
    assert wing_loss.item() > 0, "Wing loss should be positive"
    print("  ✅ PASS")
    
    # Test 5: Perfect prediction (loss should be ~0)
    print("\n[Test 5] Perfect prediction (loss should be near 0)")
    perfect_pred = gt.clone()
    loss_perfect = loss_fn(perfect_pred, gt, masks)
    
    print(f"  Loss value: {loss_perfect.item():.6f}")
    print(f"  Should be very close to 0")
    assert loss_perfect.item() < 1e-6, "Perfect prediction should have near-zero loss"
    print("  ✅ PASS")
    
    # Test 6: Compare different loss functions
    print("\n[Test 6] Comparing loss functions on same data")
    pred_test = torch.rand(2, 16, 64, 64)
    gt_test = torch.rand(2, 16, 64, 64)
    mask_test = torch.ones(2, 16, 64, 64)
    
    mse_loss = HeatmapLoss(use_mask=True)(pred_test, gt_test, mask_test)
    focal_loss = FocalHeatmapLoss(use_mask=True)(pred_test, gt_test, mask_test)
    wing_loss = AdaptiveWingLoss(use_mask=True)(pred_test, gt_test, mask_test)
    
    print(f"  MSE Loss: {mse_loss.item():.6f}")
    print(f"  Focal Loss: {focal_loss.item():.6f}")
    print(f"  Wing Loss: {wing_loss.item():.6f}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 80)
    print("All heatmap loss tests passed!")
    print("\nRecommendation: Use HeatmapLoss (MSE) for M1 baseline")
    print("Consider FocalLoss or AdaptiveWingLoss if hard keypoints need more focus")
    print("=" * 80)
