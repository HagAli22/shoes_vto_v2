"""
PAF (Part Affinity Field) Decoder Head for Limb Connection Detection

Generates vector fields representing limb connections between keypoints.
Output: [B, 30, 64, 64] (15 limbs × 2 directions, no activation)
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path


class PAFHead(nn.Module):
    """
    PAF decoder head
    
    Architecture:
    - Conv 128→64, 3×3, BN, ReLU
    - Conv 64→30, 1×1 (15 limbs × 2 for x,y vectors)
    
    Output: [B, 30, 64, 64] PAF fields (no activation)
    """
    
    def __init__(self, in_channels=128, num_limbs=15, intermediate_channels=64):
        super(PAFHead, self).__init__()
        
        self.num_limbs = num_limbs
        self.num_channels = num_limbs * 2  # x and y components
        
        # Intermediate convolution
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Final convolution to produce PAF fields
        self.conv2 = nn.Conv2d(intermediate_channels, self.num_channels, kernel_size=1, bias=True)
        
        # No activation - PAFs are raw vector fields
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 128, 64, 64] encoder features
        
        Returns:
            pafs: [B, 30, 64, 64] PAF vector fields
                Channels 0-1: Limb 0 (x, y)
                Channels 2-3: Limb 1 (x, y)
                ...
                Channels 28-29: Limb 14 (x, y)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        pafs = self.conv2(x)  # [B, 30, 64, 64]
        
        return pafs


def load_paf_config(config_name="paf_connections_13kp.yaml"):
    """Load PAF limb connections from config"""
    if isinstance(config_name, (str, Path)) and Path(config_name).is_file():
        with open(config_name, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    fname = Path(config_name).name if config_name else "paf_connections_13kp.yaml"
    possible_paths = [
        Path(__file__).parent.parent.parent / "configs" / fname,  # From src/models/heads/
        Path("shoes_vto/configs") / fname,  # From project root
        Path("configs") / fname,  # From shoes_vto/
        Path(__file__).parent.parent.parent / "configs" / "paf_connections.yaml",
        Path("shoes_vto/configs/paf_connections.yaml"),
        Path("configs/paf_connections.yaml"),
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
    
    raise FileNotFoundError(f"Could not find {fname} in any of: {possible_paths}")


_GRID_CACHE_PAF = {}


def _get_paf_grid(height, width, device):
    key = (height, width, str(device))
    if key not in _GRID_CACHE_PAF:
        xx = torch.arange(width, dtype=torch.float32, device=device)
        yy = torch.arange(height, dtype=torch.float32, device=device)
        yy_grid, xx_grid = torch.meshgrid(yy, xx, indexing='ij')
        _GRID_CACHE_PAF[key] = (xx_grid, yy_grid)
    return _GRID_CACHE_PAF[key]


def generate_paf_field(height, width, kp_from, kp_to, paf_width=8, device='cpu'):
    """
    Generate PAF vector field for a single limb - VECTORIZED FOR SPEED
    
    Args:
        height: PAF field height (typically 64)
        width: PAF field width (typically 64)
        kp_from: (x, y) starting keypoint coordinates
        kp_to: (x, y) ending keypoint coordinates
        paf_width: Width of PAF region along the limb (in pixels)
        device: 'cpu' or 'cuda'
    
    Returns:
        paf_x: [H, W] x-component of PAF vector field
        paf_y: [H, W] y-component of PAF vector field
    """
    paf_x = torch.zeros(height, width, device=device)
    paf_y = torch.zeros(height, width, device=device)
    
    x1, y1 = kp_from
    x2, y2 = kp_to
    
    # Check if keypoints are valid
    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        return paf_x, paf_y
    
    # Vector from kp_from to kp_to
    limb_vec_x = x2 - x1
    limb_vec_y = y2 - y1
    limb_length = (limb_vec_x ** 2 + limb_vec_y ** 2) ** 0.5
    
    if limb_length < 1e-6:
        return paf_x, paf_y
    
    # Unit vector
    limb_unit_x = limb_vec_x / limb_length
    limb_unit_y = limb_vec_y / limb_length
    
    # VECTORIZED: Use pre-cached coordinate grids on device
    xx_grid, yy_grid = _get_paf_grid(height, width, device)
    
    # Vector from kp_from to all pixels (vectorized)
    pixel_vec_x = xx_grid - x1
    pixel_vec_y = yy_grid - y1
    
    # Project onto limb direction (vectorized)
    proj_length = pixel_vec_x * limb_unit_x + pixel_vec_y * limb_unit_y
    
    # Distance from limb line (vectorized)
    dist_from_line = torch.abs(pixel_vec_x * (-limb_unit_y) + pixel_vec_y * limb_unit_x)
    
    # Mask: within limb segment and within width
    valid_mask = (proj_length >= 0) & (proj_length <= limb_length) & (dist_from_line <= paf_width)
    
    # Assign unit vectors where valid
    paf_x[valid_mask] = limb_unit_x
    paf_y[valid_mask] = limb_unit_y
    
    return paf_x, paf_y


def generate_pafs_batch(keypoints_batch, heatmap_size=64, paf_width=8, device='cpu', paf_config=None, num_limbs=None):
    """
    Generate ground truth PAF fields for a batch with multi-instance accumulation
    and robust spatial visibility masking.
    
    Args:
        keypoints_batch: List of instances per image in batch.
            Each instance is a dict with 'keypoints': [[x, y, v], ...] and optional 'bbox': [cx, cy, w, h] (normalized).
        heatmap_size: Size of PAF field (typically 64 for stride=4).
        paf_width: Width/radius of PAF region along limb.
        device: 'cpu' or 'cuda' for GPU acceleration.
        paf_config: Optional PAF config dict, path, or None (defaults to paf_connections_13kp.yaml).
        num_limbs: Optional number of limbs (defaults to len(limb_pairs)).
    
    Returns:
        pafs: [B, 2*L, H, W] ground truth PAF fields (averaged unit vectors on limbs, (0,0) on background)
        masks: [B, L, H, W] binary masks (1 where supervised, 0 where uncertain)
    """
    if paf_config is None:
        config = load_paf_config("paf_connections_13kp.yaml")
    elif isinstance(paf_config, (str, Path)):
        with open(paf_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = paf_config
    limb_pairs = config['limb_pairs']
    
    batch_size = len(keypoints_batch)
    if num_limbs is None:
        num_limbs = len(limb_pairs)
    
    pafs = torch.zeros(batch_size, num_limbs * 2, heatmap_size, heatmap_size, device=device)
    masks = torch.ones(batch_size, num_limbs, heatmap_size, heatmap_size, device=device)
    
    for batch_idx, instances in enumerate(keypoints_batch):
        if not instances:
            # Truly negative image: entire image supervised as (0, 0)
            continue
            
        for limb_idx, (from_idx, to_idx, name) in enumerate(limb_pairs):
            if limb_idx >= num_limbs:
                break
                
            sum_x = torch.zeros(heatmap_size, heatmap_size, device=device)
            sum_y = torch.zeros(heatmap_size, heatmap_size, device=device)
            count = torch.zeros(heatmap_size, heatmap_size, device=device)
            
            # Step 1: Carve out ignore regions inside foot bounding boxes where limb endpoints are unknown
            for inst in instances:
                kps = inst.get('keypoints', [])
                v_from = kps[from_idx][2] if from_idx < len(kps) else 0
                v_to = kps[to_idx][2] if to_idx < len(kps) else 0
                
                if not (v_from > 0 and v_to > 0):
                    # Uncertain limb on this foot
                    if 'bbox' in inst:
                        cx, cy, w, h = inst['bbox']
                        x1 = int(max(0, (cx - w / 2.0) * heatmap_size))
                        y1 = int(max(0, (cy - h / 2.0) * heatmap_size))
                        x2 = int(min(heatmap_size, (cx + w / 2.0) * heatmap_size))
                        y2 = int(min(heatmap_size, (cy + h / 2.0) * heatmap_size))
                        if x2 > x1 and y2 > y1:
                            masks[batch_idx, limb_idx, y1:y2, x1:x2] = 0.0
                            
            # Step 2: Accumulate valid limb unit vectors across all instances
            for inst in instances:
                kps = inst.get('keypoints', [])
                v_from = kps[from_idx][2] if from_idx < len(kps) else 0
                v_to = kps[to_idx][2] if to_idx < len(kps) else 0
                
                if v_from > 0 and v_to > 0:
                    from_x = kps[from_idx][0] * (heatmap_size / 256.0)
                    from_y = kps[from_idx][1] * (heatmap_size / 256.0)
                    to_x = kps[to_idx][0] * (heatmap_size / 256.0)
                    to_y = kps[to_idx][1] * (heatmap_size / 256.0)
                    
                    paf_x, paf_y = generate_paf_field(
                        heatmap_size, heatmap_size,
                        (from_x, from_y), (to_x, to_y),
                        paf_width=paf_width,
                        device=device
                    )
                    valid_strip = (paf_x != 0) | (paf_y != 0)
                    sum_x[valid_strip] += paf_x[valid_strip]
                    sum_y[valid_strip] += paf_y[valid_strip]
                    count[valid_strip] += 1.0
            
            # Step 3: Average non-zero contributing pixels and enforce positive mask
            has_limb = count > 0
            pafs[batch_idx, limb_idx * 2][has_limb] = sum_x[has_limb] / count[has_limb]
            pafs[batch_idx, limb_idx * 2 + 1][has_limb] = sum_y[has_limb] / count[has_limb]
            masks[batch_idx, limb_idx][has_limb] = 1.0  # Positive supervision overrides ignore bbox
            
    return pafs, masks


if __name__ == "__main__":
    print("=" * 80)
    print("Testing PAF Head")
    print("=" * 80)
    
    # Test 1: Basic PAF head
    print("\n[Test 1] PAFHead forward pass")
    head = PAFHead(in_channels=128, num_limbs=15)
    
    # Dummy encoder features
    features = torch.randn(4, 128, 64, 64)
    pafs = head(features)
    
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {pafs.shape}")
    print(f"  Expected: [4, 30, 64, 64] (15 limbs × 2)")
    print(f"  Output range: [{pafs.min():.3f}, {pafs.max():.3f}]")
    
    assert pafs.shape == torch.Size([4, 30, 64, 64]), "Shape mismatch"
    print("  ✅ PASS")
    
    # Test 2: PAF field generation for single limb
    print("\n[Test 2] Single PAF field generation")
    paf_x, paf_y = generate_paf_field(
        64, 64,
        kp_from=(10, 10),
        kp_to=(50, 50),
        paf_width=8
    )
    
    print(f"  PAF X shape: {paf_x.shape}")
    print(f"  PAF Y shape: {paf_y.shape}")
    print(f"  Non-zero pixels: {(paf_x != 0).sum()}")
    print(f"  Vector magnitude at (30, 30): {torch.sqrt(paf_x[30, 30]**2 + paf_y[30, 30]**2):.3f}")
    
    assert paf_x.shape == torch.Size([64, 64]), "Shape mismatch"
    assert paf_y.shape == torch.Size([64, 64]), "Shape mismatch"
    print("  ✅ PASS")
    
    # Test 3: Load PAF config
    print("\n[Test 3] Loading PAF configuration")
    try:
        config = load_paf_config()
        limb_pairs = config['limb_pairs']
        
        print(f"  Number of limbs: {len(limb_pairs)}")
        print(f"  Expected: 15")
        print(f"  PAF width: {config['paf_width']}")
        print(f"  First 3 limbs:")
        for i in range(3):
            from_idx, to_idx, name = limb_pairs[i]
            print(f"    {i}: {name} ({from_idx} → {to_idx})")
        
        assert len(limb_pairs) == 15, "Should have 15 limb pairs"
        print("  ✅ PASS")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
    
    # Test 4: Batch PAF generation
    print("\n[Test 4] Batch PAF generation")
    
    # Create dummy keypoints batch
    dummy_batch = [
        [  # Image 0
            {
                'keypoints': [
                    [30.0, 200.0, 2],   # kp 0: toe_ground
                    [50.0, 50.0, 2],    # kp 1: heel_back
                    [40.0, 230.0, 2],   # kp 2: heel_ground
                ] + [[100.0 + i*10, 150.0, 2] for i in range(13)]  # Rest visible
            }
        ],
        [  # Image 1
            {
                'keypoints': [
                    [150.0, 180.0, 2],
                    [130.0, 60.0, 2],
                ] + [[0, 0, 0]] * 14  # Rest unlabeled
            }
        ]
    ]
    
    try:
        pafs_gt, masks_gt = generate_pafs_batch(dummy_batch, heatmap_size=64, paf_width=8)
        
        print(f"  PAFs shape: {pafs_gt.shape}")
        print(f"  Masks shape: {masks_gt.shape}")
        print(f"  Expected PAFs: [2, 30, 64, 64]")
        print(f"  Expected Masks: [2, 15, 64, 64]")
        print(f"  Valid limbs (batch 0): {masks_gt[0].sum(dim=(1,2)).nonzero().numel()}")
        print(f"  Valid limbs (batch 1): {masks_gt[1].sum(dim=(1,2)).nonzero().numel()}")
        
        assert pafs_gt.shape == torch.Size([2, 30, 64, 64]), "PAFs shape mismatch"
        assert masks_gt.shape == torch.Size([2, 15, 64, 64]), "Masks shape mismatch"
        print("  ✅ PASS")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Parameter count
    print("\n[Test 5] Parameter count")
    param_count = sum(p.numel() for p in head.parameters())
    
    print(f"  PAFHead parameters: {param_count:,}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 80)
    print("PAF head testing complete!")
    print("=" * 80)
