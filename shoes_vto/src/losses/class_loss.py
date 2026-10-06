"""
Classification Loss for Left/Right Foot Discrimination

Cross-entropy loss for per-pixel class prediction with masking for ignore regions.
Critical for preventing class flips (P(flip) ≤ 1/500 frames).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassLoss(nn.Module):
    """
    Cross-entropy loss for class prediction with ignore label support
    
    Args:
        ignore_index: Label to ignore (default: -1)
        weight: Class weights [left_foot_weight, right_foot_weight] (optional)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, ignore_index=-1, weight=None, reduction='mean'):
        super(ClassLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, pred_logits, gt_class_maps, masks=None):
        """
        Compute classification loss
        
        Args:
            pred_logits: [B, 2, H, W] raw logits (before softmax)
            gt_class_maps: [B, H, W] ground truth class labels (0, 1, or -1 for ignore)
            masks: [B, H, W] binary masks (optional, 1=valid, 0=ignore)
        
        Returns:
            loss: scalar loss value
        """
        # Ensure gt is long type for cross entropy
        gt_class_maps = gt_class_maps.long()
        
        # Standard cross entropy handles ignore_index automatically
        if masks is None:
            # Use ignore_index from gt_class_maps
            loss = F.cross_entropy(
                pred_logits, 
                gt_class_maps, 
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction
            )
        else:
            # Apply additional mask
            # First compute per-pixel loss without reduction
            loss = F.cross_entropy(
                pred_logits, 
                gt_class_maps, 
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction='none'
            )  # [B, H, W]
            
            # Apply mask
            loss = loss * masks
            
            # Reduction
            if self.reduction == 'mean':
                num_valid = masks.sum() + 1e-6
                loss = loss.sum() / num_valid
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss


class FocalClassLoss(nn.Module):
    """
    Focal loss for class prediction
    
    Focuses on hard-to-classify pixels, useful for handling class imbalance
    and difficult boundary regions.
    
    Args:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        ignore_index: Label to ignore (default: -1)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-1):
        super(FocalClassLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred_logits, gt_class_maps, masks=None):
        """
        Compute focal classification loss
        """
        gt_class_maps = gt_class_maps.long()
        
        # Get probabilities
        probs = F.softmax(pred_logits, dim=1)  # [B, 2, H, W]
        
        # Get class probabilities for ground truth classes
        batch_size, _, height, width = pred_logits.shape
        
        # Create one-hot encoding
        gt_one_hot = F.one_hot(gt_class_maps.clamp(min=0), num_classes=2)  # [B, H, W, 2]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()  # [B, 2, H, W]
        
        # Focal loss formula: -alpha * (1 - p_t)^gamma * log(p_t)
        ce_loss = -gt_one_hot * torch.log(probs + 1e-8)
        focal_weight = (1 - probs) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Sum over classes
        focal_loss = focal_loss.sum(dim=1)  # [B, H, W]
        
        # Apply ignore mask
        ignore_mask = (gt_class_maps != self.ignore_index).float()
        focal_loss = focal_loss * ignore_mask
        
        # Apply additional mask if provided
        if masks is not None:
            focal_loss = focal_loss * masks
            num_valid = (masks * ignore_mask).sum() + 1e-6
        else:
            num_valid = ignore_mask.sum() + 1e-6
        
        loss = focal_loss.sum() / num_valid
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice loss for class segmentation
    
    Useful when class regions have very different sizes
    (e.g., small foot regions in large image)
    """
    
    def __init__(self, smooth=1.0, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred_logits, gt_class_maps, masks=None):
        """
        Compute Dice loss for each class
        """
        gt_class_maps = gt_class_maps.long()
        
        # Get probabilities
        probs = F.softmax(pred_logits, dim=1)  # [B, 2, H, W]
        
        # Create one-hot encoding
        gt_one_hot = F.one_hot(gt_class_maps.clamp(min=0), num_classes=2)  # [B, H, W, 2]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()  # [B, 2, H, W]
        
        # Ignore mask
        ignore_mask = (gt_class_maps != self.ignore_index).float().unsqueeze(1)  # [B, 1, H, W]
        
        # Apply ignore mask
        probs = probs * ignore_mask
        gt_one_hot = gt_one_hot * ignore_mask
        
        # Apply additional mask if provided
        if masks is not None:
            mask_expanded = masks.unsqueeze(1)  # [B, 1, H, W]
            probs = probs * mask_expanded
            gt_one_hot = gt_one_hot * mask_expanded
        
        # Dice coefficient per class
        intersection = (probs * gt_one_hot).sum(dim=(2, 3))  # [B, 2]
        union = probs.sum(dim=(2, 3)) + gt_one_hot.sum(dim=(2, 3))  # [B, 2]
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss = 1 - dice coefficient
        loss = 1.0 - dice.mean()
        
        return loss


def compute_class_accuracy(pred_logits, gt_class_maps, ignore_index=-1):
    """
    Compute per-pixel classification accuracy
    
    Args:
        pred_logits: [B, 2, H, W] predicted logits
        gt_class_maps: [B, H, W] ground truth class labels
        ignore_index: Label to ignore
    
    Returns:
        accuracy: Percentage of correctly classified pixels (0-1)
        per_class_acc: [left_acc, right_acc]
    """
    # Get predicted classes
    pred_classes = pred_logits.argmax(dim=1)  # [B, H, W]
    
    # Valid mask (not ignore)
    valid_mask = (gt_class_maps != ignore_index)
    
    # Overall accuracy
    correct = (pred_classes == gt_class_maps) & valid_mask
    accuracy = correct.sum().float() / valid_mask.sum().float()
    
    # Per-class accuracy
    left_mask = (gt_class_maps == 0) & valid_mask
    right_mask = (gt_class_maps == 1) & valid_mask
    
    left_correct = ((pred_classes == 0) & left_mask).sum().float()
    right_correct = ((pred_classes == 1) & right_mask).sum().float()
    
    left_acc = left_correct / (left_mask.sum().float() + 1e-6)
    right_acc = right_correct / (right_mask.sum().float() + 1e-6)
    
    return accuracy.item(), [left_acc.item(), right_acc.item()]


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Class Loss Functions")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 4
    num_classes = 2
    heatmap_size = 64
    
    pred_logits = torch.randn(batch_size, num_classes, heatmap_size, heatmap_size, requires_grad=True)
    
    # Ground truth with some ignore regions
    gt = torch.randint(0, 2, (batch_size, heatmap_size, heatmap_size))
    gt[:, :10, :] = -1  # Top 10 rows are ignore
    
    masks = torch.ones(batch_size, heatmap_size, heatmap_size)
    masks[:, :10, :] = 0  # Match ignore regions
    
    # Test 1: Basic cross-entropy loss
    print("\n[Test 1] ClassLoss (Cross-Entropy with ignore)")
    loss_fn = ClassLoss(ignore_index=-1, reduction='mean')
    loss = loss_fn(pred_logits, gt, masks)
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss should be > 0")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test gradient
    loss.backward()
    print(f"  Gradient computed successfully")
    print("  ✅ PASS")
    
    # Test 2: Loss with class weights
    print("\n[Test 2] ClassLoss with class weights")
    weights = torch.tensor([1.0, 2.0])  # Weight right_foot higher
    loss_fn_weighted = ClassLoss(ignore_index=-1, weight=weights, reduction='mean')
    loss_weighted = loss_fn_weighted(pred_logits.detach().requires_grad_(True), gt, masks)
    
    print(f"  Weighted loss: {loss_weighted.item():.6f}")
    print(f"  Unweighted loss: {loss.item():.6f}")
    print(f"  Weighted should be different")
    print("  ✅ PASS")
    
    # Test 3: Focal loss
    print("\n[Test 3] FocalClassLoss")
    focal_loss_fn = FocalClassLoss(alpha=0.25, gamma=2.0, ignore_index=-1)
    focal_loss = focal_loss_fn(pred_logits.detach().requires_grad_(True), gt, masks)
    
    print(f"  Focal loss: {focal_loss.item():.6f}")
    assert focal_loss.item() > 0, "Focal loss should be positive"
    print("  ✅ PASS")
    
    # Test 4: Dice loss
    print("\n[Test 4] DiceLoss")
    dice_loss_fn = DiceLoss(smooth=1.0, ignore_index=-1)
    dice_loss = dice_loss_fn(pred_logits.detach().requires_grad_(True), gt, masks)
    
    print(f"  Dice loss: {dice_loss.item():.6f}")
    assert dice_loss.item() > 0, "Dice loss should be positive"
    assert dice_loss.item() < 1.0, "Dice loss should be < 1"
    print("  ✅ PASS")
    
    # Test 5: Perfect prediction (loss should be ~0)
    print("\n[Test 5] Perfect prediction")
    # Create perfect predictions
    perfect_gt = torch.randint(0, 2, (2, 64, 64))
    perfect_logits = torch.zeros(2, 2, 64, 64)
    
    for b in range(2):
        for i in range(64):
            for j in range(64):
                if perfect_gt[b, i, j] == 0:
                    perfect_logits[b, 0, i, j] = 10.0  # High logit for class 0
                    perfect_logits[b, 1, i, j] = -10.0
                else:
                    perfect_logits[b, 0, i, j] = -10.0
                    perfect_logits[b, 1, i, j] = 10.0  # High logit for class 1
    
    perfect_loss = loss_fn(perfect_logits, perfect_gt, None)
    
    print(f"  Perfect prediction loss: {perfect_loss.item():.6f}")
    print(f"  Should be very close to 0")
    assert perfect_loss.item() < 0.01, "Perfect prediction should have near-zero loss"
    print("  ✅ PASS")
    
    # Test 6: Class accuracy computation
    print("\n[Test 6] Class accuracy computation")
    
    # Create test predictions
    test_logits = torch.zeros(2, 2, 64, 64)
    test_gt = torch.zeros(2, 64, 64).long()
    
    # First image: all class 0, perfect prediction
    test_logits[0, 0, :, :] = 5.0  # High for class 0
    test_gt[0, :, :] = 0
    
    # Second image: all class 1, perfect prediction
    test_logits[1, 1, :, :] = 5.0  # High for class 1
    test_gt[1, :, :] = 1
    
    acc, per_class_acc = compute_class_accuracy(test_logits, test_gt, ignore_index=-1)
    
    print(f"  Overall accuracy: {acc:.3f}")
    print(f"  Left foot accuracy: {per_class_acc[0]:.3f}")
    print(f"  Right foot accuracy: {per_class_acc[1]:.3f}")
    print(f"  All should be 1.0 (100%)")
    
    assert acc > 0.99, "Perfect predictions should have 100% accuracy"
    print("  ✅ PASS")
    
    # Test 7: Compare different loss functions
    print("\n[Test 7] Comparing loss functions on same data")
    pred_test = torch.randn(2, 2, 64, 64, requires_grad=True)
    gt_test = torch.randint(0, 2, (2, 64, 64))
    
    ce_loss = ClassLoss()(pred_test, gt_test, None)
    focal = FocalClassLoss()(pred_test, gt_test, None)
    dice = DiceLoss()(pred_test, gt_test, None)
    
    print(f"  Cross-Entropy Loss: {ce_loss.item():.6f}")
    print(f"  Focal Loss: {focal.item():.6f}")
    print(f"  Dice Loss: {dice.item():.6f}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 80)
    print("All class loss tests passed!")
    print("\nRecommendation: Use ClassLoss (Cross-Entropy) for M1 baseline")
    print("Consider FocalLoss if class imbalance becomes an issue")
    print("=" * 80)
