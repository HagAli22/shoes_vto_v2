"""
ARShoe M1_V2 Model - 14-Keypoint Foot Detection (ankle_center and shin_mid removed)

Combines:
- FastSCNN Encoder (24K params)
- HeatmapHead (74.8K params, 14 keypoint channels)
- PAFHead (75.7K params, 14 limbs = 28 PAF channels)
- ClassHead (22.8K params)
Total: ~197K params (15.1% of 1.3M budget)

Outputs:
- Heatmaps: [B, 14, 64, 64] sigmoid
- PAFs: [B, 28, 64, 64] raw vectors
- Class: [B, 2, 64, 64] softmax
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.encoder import FastSCNNEncoderV2
from models.heads.heatmap_head import HeatmapHead
from models.heads.paf_head import PAFHead
from models.heads.class_head import ClassHead


class ARShoeM1V2(nn.Module):
    """
    ARShoe M1_V2: 14-keypoint detection model
    
    Architecture:
        Input [B, 3, 256, 256] 
          → Encoder [B, 128, 64, 64]
          → Heatmap Head [B, 14, 64, 64]
          → PAF Head [B, 28, 64, 64]
          → Class Head [B, 2, 64, 64]
    
    14 Keypoints:
        0: toe_ground, 1: heel_back, 2: heel_ground, 3: ball_medial, 4: ball_lateral,
        5: ball_top, 6: instep_top, 7: arch_medial, 8: midfoot_lateral,
        9: malleolus_medial, 10: malleolus_lateral, 11: toe_tip, 12: throat, 13: achilles
    """
    
    def __init__(self, 
                 encoder_channels=128,
                 num_keypoints=14,
                 num_limbs=14,
                 num_classes=2):
        super(ARShoeM1V2, self).__init__()
        
        self.num_keypoints = num_keypoints
        self.num_limbs = num_limbs
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = FastSCNNEncoderV2(output_channels=encoder_channels)
        
        # Decoder heads
        self.heatmap_head = HeatmapHead(
            in_channels=encoder_channels,
            num_keypoints=num_keypoints
        )
        
        self.paf_head = PAFHead(
            in_channels=encoder_channels,
            num_limbs=num_limbs
        )
        
        self.class_head = ClassHead(
            in_channels=encoder_channels,
            num_classes=num_classes
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [B, 3, 256, 256] input images
        
        Returns:
            dict with:
                - heatmaps: [B, 14, 64, 64] sigmoid
                - pafs: [B, 28, 64, 64] raw vectors
                - class_logits: [B, 2, 64, 64] raw logits
                - class_probs: [B, 2, 64, 64] softmax
        """
        # Encoder
        features = self.encoder(x)  # [B, 128, 64, 64]
        
        # Decoder heads
        heatmaps = self.heatmap_head(features)    # [B, 14, 64, 64] sigmoid
        pafs = self.paf_head(features)            # [B, 28, 64, 64] raw
        class_logits = self.class_head(features)  # [B, 2, 64, 64] raw
        
        # Softmax for class probabilities
        class_probs = torch.softmax(class_logits, dim=1)
        
        return {
            'heatmaps': heatmaps,
            'pafs': pafs,
            'class_logits': class_logits,
            'class_probs': class_probs
        }
    
    def count_parameters(self):
        """Count total trainable parameters"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        breakdown = {
            'encoder': sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            'heatmap_head': sum(p.numel() for p in self.heatmap_head.parameters() if p.requires_grad),
            'paf_head': sum(p.numel() for p in self.paf_head.parameters() if p.requires_grad),
            'class_head': sum(p.numel() for p in self.class_head.parameters() if p.requires_grad),
        }
        
        return total, breakdown


if __name__ == "__main__":
    print("=" * 80)
    print("ARShoe M1_V2 Model Test")
    print("=" * 80)
    
    model = ARShoeM1V2()
    model.eval()
    
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(x)
    
    print("\nForward Pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Heatmaps: {outputs['heatmaps'].shape}")
    print(f"  PAFs: {outputs['pafs'].shape}")
    print(f"  Class logits: {outputs['class_logits'].shape}")
    print(f"  Class probs: {outputs['class_probs'].shape}")
    
    total, breakdown = model.count_parameters()
    print("\nParameter Count:")
    print(f"  Encoder: {breakdown['encoder']:,}")
    print(f"  Heatmap Head: {breakdown['heatmap_head']:,}")
    print(f"  PAF Head: {breakdown['paf_head']:,}")
    print(f"  Class Head: {breakdown['class_head']:,}")
    print(f"  TOTAL: {total:,}")
    print(f"  Budget: {total / 1_300_000 * 100:.1f}% of 1.3M")
    print("\n" + "=" * 80)
    print("✅ ARShoe M1_V2 model ready")
    print("=" * 80)

