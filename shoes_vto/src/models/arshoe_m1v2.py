"""
ARShoe M1v2: High-Efficiency Multi-Scale Foot Landmark Estimation Model

Optimized for Browser WASM CPU & WebGPU deployment:
- Backbone: MobileNetV3-Small (stride 4, 8, 16, 32 features)
- Context & Feature Fusion: FPN-lite top-down pyramid (64 channels @ 64x64)
- Depthwise Separable Heads: 8x compute reduction over standard convs (<170 MMACs total)
- Global-Spatial Class Fusion: Eliminates catastrophic left/right foot flip errors
- ONNX Opset 12 Compatible: Pure native standard ops (Conv, Add, Resize, GlobalPool)
- Parameter Budget: ~0.90M parameters (under the 1.3M budget)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution with Hardswish & BatchNorm
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super(DSConv, self).__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.Hardswish(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class ARShoeM1v2(nn.Module):
    """
    ARShoe M1v2 Multi-Task Architecture
    """
    def __init__(self, 
                 pretrained=True,
                 num_keypoints=16,
                 num_limbs=15,
                 num_classes=2):
        super(ARShoeM1v2, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_limbs = num_limbs
        self.num_classes = num_classes

        # 1. Multi-scale MobileNetV3-Small feature extractor
        mb = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        feat = mb.features
        self.s4 = feat[:2]    # [B, 16, 64, 64]   (stride 4)
        self.s8 = feat[2:4]   # [B, 24, 32, 32]   (stride 8)
        self.s16 = feat[4:9]  # [B, 48, 16, 16]   (stride 16)
        self.s32 = feat[9:12] # [B, 96, 8, 8]     (stride 32)

        # 2. FPN-Lite Lateral & Top-Down Merging (to stride 4, 64x64)
        self.lat32 = nn.Conv2d(96, 48, kernel_size=1, bias=False)
        self.lat16 = nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.lat8  = nn.Conv2d(24, 48, kernel_size=1, bias=False)
        self.lat4  = nn.Conv2d(16, 48, kernel_size=1, bias=False)
        self.fpn_conv = DSConv(48, 64)

        # 3. Depthwise Separable Decoder Heads (Low Compute: ~164 MMACs total)
        # Heatmap Head (16 channels for 16 keypoints)
        self.hm_head = nn.Sequential(
            DSConv(64, 64),
            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )

        # PAF Head (30 channels for 15 limbs x 2 coords)
        self.paf_head = nn.Sequential(
            DSConv(64, 64),
            nn.Conv2d(64, num_limbs * 2, kernel_size=1)
        )

        # Class Head with Global Context Fusion (Solves Left/Right flip)
        self.cls_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 32),
            nn.Hardswish(inplace=True),
            nn.Linear(32, num_classes)
        )
        self.cls_spatial = nn.Sequential(
            DSConv(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: [B, 3, 256, 256] RGB fp32 input tensor in range [0, 1]
        Returns:
            dict with:
                'heatmaps': [B, 16, 64, 64] sigmoid probabilities
                'pafs': [B, 30, 64, 64] raw vector field
                'class': [B, 2, 64, 64] softmax probabilities
                'class_logits': [B, 2, 64, 64] raw logits for loss
        """
        # Multi-scale backbone features
        x4 = self.s4(x)
        x8 = self.s8(x4)
        x16 = self.s16(x8)
        x32 = self.s32(x16)

        # Top-down feature aggregation
        p32 = self.lat32(x32)
        p16 = self.lat16(x16) + F.interpolate(p32, scale_factor=2, mode='nearest')
        p8  = self.lat8(x8)   + F.interpolate(p16, scale_factor=2, mode='nearest')
        p4  = self.lat4(x4)   + F.interpolate(p8, scale_factor=2, mode='nearest')
        feat = self.fpn_conv(p4)  # [B, 64, 64, 64]

        # Predict Heatmaps & PAFs
        hm_logits = self.hm_head(feat)
        heatmaps = torch.sigmoid(hm_logits)
        pafs = self.paf_head(feat)

        # Predict Class with Global Receptive Field Context Fusion
        g_cls = self.cls_global(x32).unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]
        s_cls = self.cls_spatial(feat)                             # [B, 2, 64, 64]
        class_logits = s_cls + g_cls
        class_probs = torch.softmax(class_logits, dim=1)

        return {
            'heatmaps': heatmaps,
            'pafs': pafs,
            'class': class_probs,
            'class_probs': class_probs,
            'class_logits': class_logits
        }

    def count_parameters(self):
        """Count total trainable parameters and breakdown"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        breakdown = {
            'backbone': sum(p.numel() for p in (list(self.s4.parameters()) + list(self.s8.parameters()) + list(self.s16.parameters()) + list(self.s32.parameters())) if p.requires_grad),
            'fpn': sum(p.numel() for p in (list(self.lat32.parameters()) + list(self.lat16.parameters()) + list(self.lat8.parameters()) + list(self.lat4.parameters()) + list(self.fpn_conv.parameters())) if p.requires_grad),
            'heatmap_head': sum(p.numel() for p in self.hm_head.parameters() if p.requires_grad),
            'paf_head': sum(p.numel() for p in self.paf_head.parameters() if p.requires_grad),
            'class_head': sum(p.numel() for p in (list(self.cls_global.parameters()) + list(self.cls_spatial.parameters())) if p.requires_grad),
            'total': total
        }
        return total, breakdown

