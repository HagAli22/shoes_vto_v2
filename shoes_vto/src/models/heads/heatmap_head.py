"""
Heatmap Decoder Head for 16-Keypoint Detection

Generates per-keypoint confidence heatmaps from encoder features.
Output: [B, 16, 64, 64] with sigmoid activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapHead(nn.Module):
    """
    Heatmap decoder head
    
    Architecture:
    - Conv 128→64, 3×3, BN, ReLU
    - Conv 64→16, 1×1, Sigmoid
    
    Output: [B, 16, 64, 64] heatmaps (one per keypoint)
    """
    
    def __init__(self, in_channels=128, num_keypoints=16, intermediate_channels=64):
        super(HeatmapHead, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Intermediate convolution to reduce channels
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Final convolution to produce heatmaps
        self.conv2 = nn.Conv2d(intermediate_channels, num_keypoints, kernel_size=1, bias=True)
        
        # Sigmoid activation (baked in for ONNX export)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 128, 64, 64] encoder features
        
        Returns:
            heatmaps: [B, 16, 64, 64] sigmoid-activated heatmaps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        heatmaps = self.sigmoid(x)  # [B, 16, 64, 64]
        
        return heatmaps


class HeatmapHeadWithPixelShuffle(nn.Module):
    """
    Alternative heatmap head using Pixel Shuffle upsampling
    (Following ARShoe's design philosophy)
    
    This version could produce higher resolution heatmaps if needed,
    but for now we keep 64×64 output to match encoder stride.
    """
    
    def __init__(self, in_channels=128, num_keypoints=16, intermediate_channels=64):
        super(HeatmapHeadWithPixelShuffle, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Intermediate processing
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Pixel shuffle could upsample here if needed (upscale_factor > 1)
        # For now, we use upscale_factor=1 (no upsampling)
        upscale_factor = 1
        self.conv_shuffle = nn.Conv2d(
            intermediate_channels, 
            num_keypoints * (upscale_factor ** 2), 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass with pixel shuffle
        
        Args:
            x: [B, 128, 64, 64] encoder features
        
        Returns:
            heatmaps: [B, 16, 64, 64] sigmoid-activated heatmaps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv_shuffle(x)
        x = self.pixel_shuffle(x)  # With upscale_factor=1, spatial size unchanged
        
        heatmaps = self.sigmoid(x)
        
        return heatmaps


def generate_gaussian_heatmap(height, width, center_x, center_y, sigma=2.0, device='cpu'):
    """
    Generate a single Gaussian heatmap - OPTIMIZED FOR GPU
    
    Args:
        height: Heatmap height (typically 64)
        width: Heatmap width (typically 64)
        center_x: Keypoint x coordinate (0 to width-1)
        center_y: Keypoint y coordinate (0 to height-1)
        sigma: Gaussian standard deviation in pixels
        device: 'cpu' or 'cuda' for GPU acceleration
    
    Returns:
        heatmap: [H, W] Gaussian heatmap with peak at (center_x, center_y)
    """
    # Create coordinate grids directly on target device
    x = torch.arange(0, width, dtype=torch.float32, device=device)
    y = torch.arange(0, height, dtype=torch.float32, device=device)
    
    # Meshgrid
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Gaussian formula: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    heatmap = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    
    return heatmap


def generate_heatmaps_batch(keypoints_batch, heatmap_size=64, sigma=3.5, device='cpu', num_keypoints=None):
    """
    Generate ground truth heatmaps for a batch - OPTIMIZED FOR GPU
    
    Args:
        keypoints_batch: List of instances per image in batch
            Each element is a list of instances in that image
            Each instance has 'keypoints': [[x, y, v], ...]
        heatmap_size: Size of heatmap (64 for stride=4)
        sigma: Gaussian sigma in pixels on heatmap (increased to 3.5 for better learning)
        device: 'cpu' or 'cuda' for GPU acceleration
        num_keypoints: Number of keypoints (e.g. 13 or 16), auto-detected if None
    
    Returns:
        heatmaps: [B, K, H, W] ground truth heatmaps
        masks: [B, K, H, W] binary masks (1 where keypoint is labeled, 0 otherwise)
    """
    batch_size = len(keypoints_batch)
    if num_keypoints is None:
        found_k = 16
        for insts in keypoints_batch:
            if insts and 'keypoints' in insts[0]:
                found_k = len(insts[0]['keypoints'])
                break
        num_keypoints = found_k
    
    # Create tensors directly on target device
    heatmaps = torch.zeros(batch_size, num_keypoints, heatmap_size, heatmap_size, device=device)
    masks = torch.zeros(batch_size, num_keypoints, heatmap_size, heatmap_size, device=device)
    
    for batch_idx, instances in enumerate(keypoints_batch):
        if not instances:
            continue
            
        for kp_idx in range(num_keypoints):
            # Check if any instance in this image has keypoint kp_idx labeled (v > 0)
            has_labeled = False
            for instance in instances:
                kps = instance.get('keypoints', [])
                if kp_idx < len(kps) and kps[kp_idx][2] > 0:
                    has_labeled = True
                    break
                    
            if not has_labeled:
                # No instance has keypoint kp_idx labeled.
                # Leave masks[batch_idx, kp_idx] = 0.0 everywhere (zero loss for this channel).
                continue
                
            # At least one instance has this keypoint labeled.
            # Step 1: Initialize mask to 1.0 everywhere so background is supervised.
            masks[batch_idx, kp_idx] = 1.0
            
            # Step 2: Carve out spatial ignore region (0.0) over instances where keypoint is v=0.
            # Uses reliable instance bounding box rather than placeholder keypoint coordinates.
            for instance in instances:
                kps = instance.get('keypoints', [])
                v = kps[kp_idx][2] if kp_idx < len(kps) else 0
                if v == 0 and 'bbox' in instance:
                    cx, cy, w, h = instance['bbox']
                    x1 = int(max(0, (cx - w / 2.0) * heatmap_size))
                    y1 = int(max(0, (cy - h / 2.0) * heatmap_size))
                    x2 = int(min(heatmap_size, (cx + w / 2.0) * heatmap_size))
                    y2 = int(min(heatmap_size, (cy + h / 2.0) * heatmap_size))
                    if x2 > x1 and y2 > y1:
                        masks[batch_idx, kp_idx, y1:y2, x1:x2] = 0.0
            
            # Step 3: Draw Gaussian peaks for instances with v > 0 and guarantee peak region is unmasked.
            for instance in instances:
                kps = instance.get('keypoints', [])
                if kp_idx < len(kps) and kps[kp_idx][2] > 0:
                    x, y, v = kps[kp_idx]
                    hm_x = x * (heatmap_size / 256.0)
                    hm_y = y * (heatmap_size / 256.0)
                    hm_x = max(0, min(heatmap_size - 1, hm_x))
                    hm_y = max(0, min(heatmap_size - 1, hm_y))
                    
                    gaussian = generate_gaussian_heatmap(heatmap_size, heatmap_size, hm_x, hm_y, sigma, device=device)
                    heatmaps[batch_idx, kp_idx] = torch.maximum(
                        heatmaps[batch_idx, kp_idx], 
                        gaussian
                    )
                    # Protect positive peak region from being masked out if bboxes overlap
                    peak_mask = gaussian > 0.01
                    masks[batch_idx, kp_idx] = torch.where(
                        peak_mask, 
                        torch.ones_like(masks[batch_idx, kp_idx]), 
                        masks[batch_idx, kp_idx]
                    )
    
    return heatmaps, masks


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Heatmap Head")
    print("=" * 80)
    
    # Test 1: Basic heatmap head
    print("\n[Test 1] HeatmapHead forward pass")
    head = HeatmapHead(in_channels=128, num_keypoints=16)
    
    # Dummy encoder features
    features = torch.randn(4, 128, 64, 64)
    heatmaps = head(features)
    
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {heatmaps.shape}")
    print(f"  Output range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")
    print(f"  Expected: [0.0, 1.0] (sigmoid)")
    
    assert heatmaps.shape == torch.Size([4, 16, 64, 64]), "Shape mismatch"
    assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0, "Values not in [0, 1]"
    print("  ✅ PASS")
    
    # Test 2: Pixel shuffle version
    print("\n[Test 2] HeatmapHeadWithPixelShuffle forward pass")
    head_ps = HeatmapHeadWithPixelShuffle(in_channels=128, num_keypoints=16)
    heatmaps_ps = head_ps(features)
    
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {heatmaps_ps.shape}")
    print(f"  Output range: [{heatmaps_ps.min():.3f}, {heatmaps_ps.max():.3f}]")
    
    assert heatmaps_ps.shape == torch.Size([4, 16, 64, 64]), "Shape mismatch"
    print("  ✅ PASS")
    
    # Test 3: Gaussian heatmap generation
    print("\n[Test 3] Gaussian heatmap generation")
    gaussian = generate_gaussian_heatmap(64, 64, center_x=32.0, center_y=32.0, sigma=2.0)
    
    print(f"  Heatmap shape: {gaussian.shape}")
    print(f"  Peak value: {gaussian.max():.3f} (expected: ~1.0)")
    print(f"  Peak location: {torch.where(gaussian == gaussian.max())}")
    
    assert gaussian.shape == torch.Size([64, 64]), "Shape mismatch"
    assert gaussian.max() > 0.99, "Peak not at expected value"
    print("  ✅ PASS")
    
    # Test 4: Batch heatmap generation
    print("\n[Test 4] Batch heatmap generation")
    
    # Create dummy keypoints batch
    dummy_batch = [
        [  # Image 0
            {
                'keypoints': [
                    [128.0, 200.0, 2],  # kp 0: visible
                    [100.0, 50.0, 2],   # kp 1: visible
                    [120.0, 30.0, 2],   # kp 2: visible
                ] + [[0, 0, 0]] * 13  # Rest are unlabeled
            }
        ],
        [  # Image 1
            {
                'keypoints': [
                    [150.0, 180.0, 2],  # kp 0: visible
                    [130.0, 60.0, 1],   # kp 1: covered
                ] + [[0, 0, 0]] * 14
            }
        ]
    ]
    
    heatmaps_gt, masks_gt = generate_heatmaps_batch(dummy_batch, heatmap_size=64, sigma=2.0)
    
    print(f"  Heatmaps shape: {heatmaps_gt.shape}")
    print(f"  Masks shape: {masks_gt.shape}")
    print(f"  Num labeled keypoints (batch 0): {masks_gt[0].sum()}")
    print(f"  Num labeled keypoints (batch 1): {masks_gt[1].sum()}")
    
    assert heatmaps_gt.shape == torch.Size([2, 16, 64, 64]), "Heatmaps shape mismatch"
    assert masks_gt.shape == torch.Size([2, 16, 64, 64]), "Masks shape mismatch"
    print("  ✅ PASS")
    
    # Test 5: Parameter count
    print("\n[Test 5] Parameter count")
    param_count = sum(p.numel() for p in head.parameters())
    param_count_ps = sum(p.numel() for p in head_ps.parameters())
    
    print(f"  HeatmapHead parameters: {param_count:,}")
    print(f"  HeatmapHeadWithPixelShuffle parameters: {param_count_ps:,}")
    print(f"  Difference: {abs(param_count - param_count_ps):,}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 80)
    print("All tests passed! Heatmap head is ready.")
    print("=" * 80)
