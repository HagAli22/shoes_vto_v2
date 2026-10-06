"""
Class Decoder Head for Left/Right Foot Classification

Generates per-pixel class probability maps for left_foot (class 0) and right_foot (class 1).
Output: [B, 2, 64, 64] with softmax activation

This explicit classification head is critical for preventing class flips,
which are catastrophic in shoe VTO (swaps both shoes on screen).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassHead(nn.Module):
    """
    Class decoder head for left/right foot classification
    
    Architecture:
    - Conv 128→64, 3×3, BN, ReLU
    - Conv 64→2, 1×1, Softmax
    
    Output: [B, 2, 64, 64] class probability maps
    """
    
    def __init__(self, in_channels=128, num_classes=2, intermediate_channels=32):
        super(ClassHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Layer 1: 1x1 conv to compress channels (128 -> 32)
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Layer 2: 3x3 conv for spatial context and feature extraction (32 -> 32)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        
        # Layer 3: 3x3 conv for deeper discrimination (32 -> 32)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)
        
        # Layer 4: 1x1 conv to produce class logits (32 -> 2)
        self.conv4 = nn.Conv2d(intermediate_channels, num_classes, kernel_size=1, bias=True)
        
        # Softmax activation (baked in for ONNX export)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 128, 64, 64] encoder features
        
        Returns:
            class_probs: [B, 2, 64, 64] softmax probabilities
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        logits = self.conv4(x)
        return self.softmax(logits)
    
    def forward_logits(self, x):
        """
        Forward pass returning logits (before softmax)
        Useful for training with CrossEntropyLoss
        
        Args:
            x: [B, 128, 64, 64] encoder features
        
        Returns:
            logits: [B, 2, 64, 64] raw logits (no softmax)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)


def generate_class_map(bbox, class_id, heatmap_size=64, image_size=256, device='cpu'):
    """
    Generate ground truth class map for a single instance
    
    Args:
        bbox: [cx, cy, w, h] normalized bbox (0-1)
        class_id: 0 for left_foot, 1 for right_foot
        heatmap_size: Size of class map (typically 64)
        image_size: Original image size (typically 256)
        device: 'cpu' or 'cuda'
    
    Returns:
        class_map: [H, W] class labels (0 or 1) inside bbox, -1 outside
    """
    # Initialize with -1 (ignore label) on target device
    class_map = torch.ones(heatmap_size, heatmap_size, device=device) * -1
    
    # Convert normalized bbox to heatmap coordinates
    cx = bbox[0] * heatmap_size
    cy = bbox[1] * heatmap_size
    w = bbox[2] * heatmap_size
    h = bbox[3] * heatmap_size
    
    # Get bbox corners
    x1 = int(max(0, cx - w / 2))
    y1 = int(max(0, cy - h / 2))
    x2 = int(min(heatmap_size, cx + w / 2))
    y2 = int(min(heatmap_size, cy + h / 2))
    
    # Fill bbox region with class label
    class_map[y1:y2, x1:x2] = class_id
    
    return class_map


def generate_class_maps_batch(instances_batch, heatmap_size=64, image_size=256, device='cpu'):
    """
    Generate ground truth class maps for a batch - OPTIMIZED FOR GPU
    
    Args:
        instances_batch: List of instances per image in batch
            Each element is a list of instances in that image
            Each instance has 'bbox': [cx, cy, w, h] and 'class_id': 0 or 1
        heatmap_size: Size of class map (64 for stride=4)
        image_size: Original image size (256)
        device: 'cpu' or 'cuda' for GPU acceleration
    
    Returns:
        class_maps: [B, H, W] class labels (0, 1, or -1 for ignore)
        masks: [B, H, W] binary masks (1 where labeled, 0 where ignore)
    """
    batch_size = len(instances_batch)
    
    # Create tensors directly on target device
    class_maps = torch.ones(batch_size, heatmap_size, heatmap_size, device=device) * -1
    masks = torch.zeros(batch_size, heatmap_size, heatmap_size, device=device)
    
    for batch_idx, instances in enumerate(instances_batch):
        for instance in instances:
            bbox = instance['bbox']  # [cx, cy, w, h] normalized
            class_id = instance['class_id']  # 0 or 1
            
            # Generate class map for this instance ON SAME DEVICE
            instance_map = generate_class_map(bbox, class_id, heatmap_size, image_size, device=device)
            
            # Overlay on batch (later instances can overwrite earlier ones)
            valid_mask = instance_map >= 0
            class_maps[batch_idx][valid_mask] = instance_map[valid_mask]
            masks[batch_idx][valid_mask] = 1.0
    
    return class_maps, masks


def extract_class_from_keypoints(keypoints, class_probs, heatmap_size=64):
    """
    Extract class prediction for an instance based on its keypoints
    
    Args:
        keypoints: [[x, y, v], ...] 16 keypoints (in pixel coordinates 0-256)
        class_probs: [2, H, W] class probability map
        heatmap_size: Size of class map
    
    Returns:
        class_id: Predicted class (0 or 1)
        confidence: Confidence of the prediction (0-1)
    """
    # Sample class probabilities at keypoint locations
    votes = []
    
    for kp in keypoints:
        x, y, v = kp
        if v > 0:  # Only use visible keypoints
            # Convert to heatmap coordinates
            hm_x = int(x * (heatmap_size / 256.0))
            hm_y = int(y * (heatmap_size / 256.0))
            
            # Clip to valid range
            hm_x = max(0, min(heatmap_size - 1, hm_x))
            hm_y = max(0, min(heatmap_size - 1, hm_y))
            
            # Get class probabilities at this location
            left_prob = class_probs[0, hm_y, hm_x].item()
            right_prob = class_probs[1, hm_y, hm_x].item()
            
            votes.append((left_prob, right_prob))
    
    if len(votes) == 0:
        # No visible keypoints, default to left
        return 0, 0.5
    
    # Average votes
    avg_left = sum(v[0] for v in votes) / len(votes)
    avg_right = sum(v[1] for v in votes) / len(votes)
    
    # Determine class
    if avg_left > avg_right:
        class_id = 0
        confidence = avg_left
    else:
        class_id = 1
        confidence = avg_right
    
    return class_id, confidence


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Class Head")
    print("=" * 80)
    
    # Test 1: Basic class head forward pass
    print("\n[Test 1] ClassHead forward pass")
    head = ClassHead(in_channels=128, num_classes=2)
    
    # Dummy encoder features
    features = torch.randn(4, 128, 64, 64)
    class_probs = head(features)
    
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {class_probs.shape}")
    print(f"  Output range: [{class_probs.min():.3f}, {class_probs.max():.3f}]")
    
    # Check softmax constraint: sum over class dimension = 1
    sum_over_classes = class_probs.sum(dim=1)
    print(f"  Sum over classes (should be 1.0): {sum_over_classes[0, 0, 0]:.6f}")
    
    assert class_probs.shape == torch.Size([4, 2, 64, 64]), "Shape mismatch"
    assert torch.allclose(sum_over_classes, torch.ones_like(sum_over_classes), atol=1e-5), "Softmax constraint violated"
    print("  ✅ PASS")
    
    # Test 2: Forward logits (for training)
    print("\n[Test 2] ClassHead forward_logits")
    logits = head.forward_logits(features)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Manually apply softmax and compare
    manual_probs = F.softmax(logits, dim=1)
    diff = (manual_probs - class_probs).abs().max()
    print(f"  Difference from forward(): {diff:.6f}")
    
    assert torch.allclose(manual_probs, class_probs, atol=1e-5), "Logits don't match probabilities"
    print("  ✅ PASS")
    
    # Test 3: Class map generation for single instance
    print("\n[Test 3] Single instance class map generation")
    bbox = [0.5, 0.5, 0.3, 0.6]  # Centered bbox
    class_id = 1  # right_foot
    
    class_map = generate_class_map(bbox, class_id, heatmap_size=64)
    
    print(f"  Class map shape: {class_map.shape}")
    print(f"  Unique values: {class_map.unique()}")
    print(f"  Pixels with class 1: {(class_map == 1).sum()}")
    print(f"  Pixels with ignore (-1): {(class_map == -1).sum()}")
    
    assert class_map.shape == torch.Size([64, 64]), "Shape mismatch"
    assert 1 in class_map, "Class label not present in map"
    print("  ✅ PASS")
    
    # Test 4: Batch class map generation
    print("\n[Test 4] Batch class map generation")
    
    # Create dummy batch
    dummy_batch = [
        [  # Image 0 - 2 feet
            {
                'bbox': [0.3, 0.5, 0.2, 0.6],
                'class_id': 0  # left_foot
            },
            {
                'bbox': [0.7, 0.5, 0.2, 0.6],
                'class_id': 1  # right_foot
            }
        ],
        [  # Image 1 - 1 foot
            {
                'bbox': [0.5, 0.5, 0.3, 0.7],
                'class_id': 0  # left_foot
            }
        ]
    ]
    
    class_maps_gt, masks_gt = generate_class_maps_batch(dummy_batch, heatmap_size=64)
    
    print(f"  Class maps shape: {class_maps_gt.shape}")
    print(f"  Masks shape: {masks_gt.shape}")
    print(f"  Image 0 - labeled pixels: {masks_gt[0].sum()}")
    print(f"  Image 1 - labeled pixels: {masks_gt[1].sum()}")
    print(f"  Image 0 - unique classes: {class_maps_gt[0][masks_gt[0] > 0].unique()}")
    
    assert class_maps_gt.shape == torch.Size([2, 64, 64]), "Class maps shape mismatch"
    assert masks_gt.shape == torch.Size([2, 64, 64]), "Masks shape mismatch"
    assert 0 in class_maps_gt[0] and 1 in class_maps_gt[0], "Both classes should be in image 0"
    print("  ✅ PASS")
    
    # Test 5: Extract class from keypoints
    print("\n[Test 5] Extract class from keypoints")
    
    # Create dummy keypoints and class probability map
    keypoints = [
        [100, 100, 2],  # visible keypoint
        [120, 110, 2],
        [110, 120, 1],  # covered
    ] + [[0, 0, 0]] * 13  # rest unlabeled
    
    # Create class probs with higher right_foot probability
    class_probs_test = torch.zeros(2, 64, 64)
    class_probs_test[0, :, :] = 0.3  # left_foot = 30%
    class_probs_test[1, :, :] = 0.7  # right_foot = 70%
    
    pred_class, confidence = extract_class_from_keypoints(keypoints, class_probs_test, heatmap_size=64)
    
    print(f"  Predicted class: {pred_class} ({'left_foot' if pred_class == 0 else 'right_foot'})")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Expected: 1 (right_foot) with ~0.7 confidence")
    
    assert pred_class == 1, "Should predict right_foot"
    assert confidence > 0.6, "Confidence should be high"
    print("  ✅ PASS")
    
    # Test 6: Parameter count
    print("\n[Test 6] Parameter count")
    param_count = sum(p.numel() for p in head.parameters())
    
    print(f"  ClassHead parameters: {param_count:,}")
    print(f"  Target: < 50,000 (lightweight)")
    
    assert param_count < 50_000, "Too many parameters for class head"
    print("  ✅ PASS")
    
    # Test 7: Gradient flow
    print("\n[Test 7] Gradient flow test")
    
    head_test = ClassHead()
    features_test = torch.randn(2, 128, 64, 64, requires_grad=True)
    
    # Forward
    class_probs_test = head_test(features_test)
    
    # Dummy loss
    loss = class_probs_test.mean()
    loss.backward()
    
    has_grads = features_test.grad is not None
    
    print(f"  Gradient on input: {'Yes' if has_grads else 'No'}")
    assert has_grads, "Gradients should flow back to input"
    print("  ✅ PASS")
    
    print("\n" + "=" * 80)
    print("All tests passed! Class head is ready.")
    print("=" * 80)
