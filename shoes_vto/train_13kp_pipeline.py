"""
👟 ARShoe M1v2 13-Keypoint High-Speed Training Pipeline
======================================================
Self-contained, production-grade training pipeline for real-time 13-keypoint foot pose estimation.

Key Optimizations:
1. Target Grid Pre-Caching: Pre-allocated coordinate meshgrids on GPU eliminate >37,000 tensor allocations per epoch.
2. Fast DataLoader: batch_size=64, pin_memory=True, num_workers=2, persistent_workers=True (tuned for Colab 2 vCPUs).
3. Mixed Precision: Native PyTorch AMP (FP16) with GradScaler.
4. Corrected Multi-Instance PAF Accumulation & Heatmap Visibility Masking.
5. Live every-10-epoch test visualizations via OpenPose PAF-led assembly.

Usage:
    python shoes_vto/train_13kp_pipeline.py \
        --dataset_root /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1_13kp \
        --output_dir shoes_vto/outputs/m1v2_13kp_stage2_paf_fix \
        --epochs 100 \
        --batch_size 64 \
        --lr 0.001 \
        --vis_interval 10
"""

import os
import sys
import time
import json
import math
import random
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from scipy.ndimage import maximum_filter, label as nd_label
from tqdm import tqdm


# =============================================================================
# 1. Anatomical Schema & Constants (13 Keypoints, 16 Limbs)
# =============================================================================

KEYPOINT_NAMES_13 = [
    "toe_ground",        # 0
    "heel_back",         # 1
    "heel_ground",       # 2
    "ball_medial",       # 3
    "ball_lateral",      # 4
    "instep_top",        # 5
    "arch_medial",       # 6
    "midfoot_lateral",   # 7
    "malleolus_medial",  # 8
    "malleolus_lateral", # 9
    "toe_tip",           # 10
    "throat",            # 11
    "achilles"           # 12
]

# 16 Anatomical Limbs: (from_idx, to_idx, name)
PAF_LIMBS_13 = [
    (10, 0,  "toe_tip->toe_ground"),
    (0,  2,  "toe_ground->heel_ground"),
    (2,  1,  "heel_ground->heel_back"),
    (3,  4,  "ball_medial->ball_lateral"),
    (3,  5,  "ball_medial->instep_top"),
    (4,  5,  "ball_lateral->instep_top"),
    (2,  6,  "heel_ground->arch_medial"),
    (2,  7,  "heel_ground->midfoot_lateral"),
    (6,  5,  "arch_medial->instep_top"),
    (7,  5,  "midfoot_lateral->instep_top"),
    (5,  11, "instep_top->throat"),
    (11, 8,  "throat->malleolus_medial"),
    (11, 9,  "throat->malleolus_lateral"),
    (8,  12, "malleolus_medial->achilles"),
    (9,  12, "malleolus_lateral->achilles"),
    (12, 1,  "achilles->heel_back")
]

# Horizontal flip index swap (left/right symmetric keypoint pairs)
FLIP_IDX_13 = [0, 1, 2, 4, 3, 5, 7, 6, 9, 8, 10, 11, 12]

# BGR Colors for 13 Keypoints
KP_COLORS_13 = [
    (0, 0, 255),     # 0: toe_ground (Red)
    (0, 128, 255),   # 1: heel_back (Orange)
    (0, 255, 255),   # 2: heel_ground (Yellow)
    (0, 255, 0),     # 3: ball_medial (Green)
    (255, 255, 0),   # 4: ball_lateral (Cyan)
    (255, 0, 0),     # 5: instep_top (Blue)
    (255, 0, 128),   # 6: arch_medial (Purple)
    (128, 0, 255),   # 7: midfoot_lateral (Violet)
    (0, 165, 255),   # 8: malleolus_medial (Amber)
    (147, 20, 255),  # 9: malleolus_lateral (Pink)
    (50, 205, 50),   # 10: toe_tip (Lime)
    (208, 224, 64),  # 11: throat (Teal)
    (180, 105, 255)  # 12: achilles (Rose)
]

LIMB_COLORS_16 = [
    (0, 255, 0), (0, 230, 80), (0, 200, 140), (255, 200, 0),
    (255, 140, 0), (255, 80, 0), (0, 160, 255), (0, 100, 255),
    (140, 0, 255), (200, 0, 255), (255, 0, 200), (255, 0, 100),
    (255, 50, 50), (180, 180, 0), (100, 220, 100), (0, 255, 180)
]


# =============================================================================
# 2. Pre-Cached Coordinate Grids & GPU Target Generators
# =============================================================================

_GRID_CACHE_HM = {}
_GRID_CACHE_PAF = {}


def _get_hm_grid(height, width, device):
    """Retrieve pre-cached coordinate grid for Gaussian heatmap generation"""
    key = (height, width, str(device))
    if key not in _GRID_CACHE_HM:
        x = torch.arange(0, width, dtype=torch.float32, device=device)
        y = torch.arange(0, height, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        _GRID_CACHE_HM[key] = (xx, yy)
    return _GRID_CACHE_HM[key]


def _get_paf_grid(height, width, device):
    """Retrieve pre-cached coordinate grid for PAF vector field generation"""
    key = (height, width, str(device))
    if key not in _GRID_CACHE_PAF:
        x = torch.arange(0, width, dtype=torch.float32, device=device)
        y = torch.arange(0, height, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        _GRID_CACHE_PAF[key] = (xx, yy)
    return _GRID_CACHE_PAF[key]


def generate_gaussian_heatmap(height, width, center_x, center_y, sigma=2.0, device='cpu'):
    """Generate a single Gaussian peak using pre-cached coordinate grids"""
    xx, yy = _get_hm_grid(height, width, device)
    return torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))


def generate_paf_field(height, width, kp_from, kp_to, paf_width=8, device='cpu'):
    """Generate a unit-vector limb affinity field using pre-cached coordinate grids"""
    paf_x = torch.zeros(height, width, device=device)
    paf_y = torch.zeros(height, width, device=device)
    x1, y1 = kp_from
    x2, y2 = kp_to
    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        return paf_x, paf_y
    limb_vec_x = x2 - x1
    limb_vec_y = y2 - y1
    limb_length = (limb_vec_x ** 2 + limb_vec_y ** 2) ** 0.5
    if limb_length < 1e-6:
        return paf_x, paf_y
    limb_unit_x = limb_vec_x / limb_length
    limb_unit_y = limb_vec_y / limb_length
    xx_grid, yy_grid = _get_paf_grid(height, width, device)
    pixel_vec_x = xx_grid - x1
    pixel_vec_y = yy_grid - y1
    proj_length = pixel_vec_x * limb_unit_x + pixel_vec_y * limb_unit_y
    dist_from_line = torch.abs(pixel_vec_x * (-limb_unit_y) + pixel_vec_y * limb_unit_x)
    valid_mask = (proj_length >= 0) & (proj_length <= limb_length) & (dist_from_line <= paf_width)
    paf_x[valid_mask] = limb_unit_x
    paf_y[valid_mask] = limb_unit_y
    return paf_x, paf_y


def generate_heatmaps_batch(keypoints_batch, heatmap_size=64, sigma=3.5, device='cpu', num_keypoints=13):
    """
    Generate ground truth heatmaps with visibility masking and background supervision.
    - Regions inside feet where keypoint is unlabeled (v=0) are ignored (mask=0.0).
    - Positive Gaussian peaks are supervised (mask=1.0).
    - Background regions outside feet are supervised as zero (mask=1.0, target=0.0).
    """
    batch_size = len(keypoints_batch)
    heatmaps = torch.zeros(batch_size, num_keypoints, heatmap_size, heatmap_size, device=device)
    masks = torch.ones(batch_size, num_keypoints, heatmap_size, heatmap_size, device=device)

    for b_idx, instances in enumerate(keypoints_batch):
        if not instances:
            continue
        for kp_idx in range(num_keypoints):
            # 1. Carve out ignore mask over feet where keypoint is missing (v=0)
            for inst in instances:
                kps = inst.get('keypoints', [])
                v = kps[kp_idx][2] if kp_idx < len(kps) else 0
                if v == 0 and 'bbox' in inst:
                    cx, cy, w, h = inst['bbox']
                    x1 = int(max(0, (cx - w / 2.0) * heatmap_size))
                    y1 = int(max(0, (cy - h / 2.0) * heatmap_size))
                    x2 = int(min(heatmap_size, (cx + w / 2.0) * heatmap_size))
                    y2 = int(min(heatmap_size, (cy + h / 2.0) * heatmap_size))
                    if x2 > x1 and y2 > y1:
                        masks[b_idx, kp_idx, y1:y2, x1:x2] = 0.0

            # 2. Draw Gaussian peaks for visible keypoints (v > 0)
            for inst in instances:
                kps = inst.get('keypoints', [])
                if kp_idx < len(kps) and kps[kp_idx][2] > 0:
                    x, y, v = kps[kp_idx]
                    hm_x = max(0.0, min(float(heatmap_size - 1), x * (heatmap_size / 256.0)))
                    hm_y = max(0.0, min(float(heatmap_size - 1), y * (heatmap_size / 256.0)))
                    gaussian = generate_gaussian_heatmap(heatmap_size, heatmap_size, hm_x, hm_y, sigma, device=device)
                    heatmaps[b_idx, kp_idx] = torch.maximum(heatmaps[b_idx, kp_idx], gaussian)
                    masks[b_idx, kp_idx][gaussian > 0.01] = 1.0

    return heatmaps, masks


def generate_pafs_batch(keypoints_batch, heatmap_size=64, paf_width=8, device='cpu', num_limbs=16):
    """
    Generate ground truth PAF fields with multi-instance accumulation and non-zero averaging.
    - Multiple instances contributing to the same limb are summed and averaged.
    - Foot bboxes with missing endpoints (v=0) are masked out (mask=0.0).
    - Background outside feet is supervised as zero vectors (0, 0) with mask=1.0.
    """
    batch_size = len(keypoints_batch)
    pafs = torch.zeros(batch_size, num_limbs * 2, heatmap_size, heatmap_size, device=device)
    masks = torch.ones(batch_size, num_limbs, heatmap_size, heatmap_size, device=device)

    for b_idx, instances in enumerate(keypoints_batch):
        if not instances:
            continue
        for l_idx, (u, v, _) in enumerate(PAF_LIMBS_13[:num_limbs]):
            sum_x = torch.zeros(heatmap_size, heatmap_size, device=device)
            sum_y = torch.zeros(heatmap_size, heatmap_size, device=device)
            count = torch.zeros(heatmap_size, heatmap_size, device=device)

            # 1. Carve out ignore mask over feet with uncertain limb endpoints
            for inst in instances:
                kps = inst.get('keypoints', [])
                v_u = kps[u][2] if u < len(kps) else 0
                v_v = kps[v][2] if v < len(kps) else 0
                if not (v_u > 0 and v_v > 0) and 'bbox' in inst:
                    cx, cy, w, h = inst['bbox']
                    x1 = int(max(0, (cx - w / 2.0) * heatmap_size))
                    y1 = int(max(0, (cy - h / 2.0) * heatmap_size))
                    x2 = int(min(heatmap_size, (cx + w / 2.0) * heatmap_size))
                    y2 = int(min(heatmap_size, (cy + h / 2.0) * heatmap_size))
                    if x2 > x1 and y2 > y1:
                        masks[b_idx, l_idx, y1:y2, x1:x2] = 0.0

            # 2. Accumulate valid limb unit vectors across all feet
            for inst in instances:
                kps = inst.get('keypoints', [])
                v_u = kps[u][2] if u < len(kps) else 0
                v_v = kps[v][2] if v < len(kps) else 0
                if v_u > 0 and v_v > 0:
                    ux = kps[u][0] * (heatmap_size / 256.0)
                    uy = kps[u][1] * (heatmap_size / 256.0)
                    vx = kps[v][0] * (heatmap_size / 256.0)
                    vy = kps[v][1] * (heatmap_size / 256.0)
                    px, py = generate_paf_field(heatmap_size, heatmap_size, (ux, uy), (vx, vy), paf_width=paf_width, device=device)
                    valid_strip = (px != 0) | (py != 0)
                    sum_x[valid_strip] += px[valid_strip]
                    sum_y[valid_strip] += py[valid_strip]
                    count[valid_strip] += 1.0

            # 3. Average non-zero contributing pixels and enforce positive supervision
            has_limb = count > 0
            pafs[b_idx, l_idx * 2][has_limb] = sum_x[has_limb] / count[has_limb]
            pafs[b_idx, l_idx * 2 + 1][has_limb] = sum_y[has_limb] / count[has_limb]
            masks[b_idx, l_idx][has_limb] = 1.0

    return pafs, masks


def generate_class_maps_batch(instances_batch, heatmap_size=64, device='cpu'):
    """Generate ground truth left/right foot classification maps (-1 for background)"""
    batch_size = len(instances_batch)
    class_maps = torch.full((batch_size, heatmap_size, heatmap_size), -1, dtype=torch.long, device=device)
    for b_idx, instances in enumerate(instances_batch):
        for inst in instances:
            cls_id = int(inst.get('class_id', 0))
            if 'bbox' in inst:
                cx, cy, w, h = inst['bbox']
                x1 = int(max(0, (cx - w / 2.0) * heatmap_size))
                y1 = int(max(0, (cy - h / 2.0) * heatmap_size))
                x2 = int(min(heatmap_size, (cx + w / 2.0) * heatmap_size))
                y2 = int(min(heatmap_size, (cy + h / 2.0) * heatmap_size))
                if x2 > x1 and y2 > y1:
                    class_maps[b_idx, y1:y2, x1:x2] = cls_id
    return class_maps


# =============================================================================
# 3. YOLO 13-Keypoint Dataset & Fast DataLoader
# =============================================================================

class Dataset13KP(Dataset):
    """
    Optimized YOLO Pose Dataset for 13 Foot Keypoints.
    Parses `<class> <cx> <cy> <w> <h> <x0> <y0> <v0> ... <x12> <y12> <v12>`.
    """
    def __init__(self, images_dir, labels_dir, img_size=256, augment=True, flip_prob=0.5):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.flip_prob = flip_prob

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.image_files = sorted([
            f for f in self.images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts
        ])
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No valid images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        orig_h, orig_w = img_bgr.shape[:2]

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        instances = []
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = [float(p) for p in parts[1:5]]
                    raw_kps = [float(p) for p in parts[5:]]
                    kps = []
                    for k in range(0, min(len(raw_kps), 13 * 3), 3):
                        kx = raw_kps[k] * orig_w
                        ky = raw_kps[k + 1] * orig_h
                        kv = int(raw_kps[k + 2]) if k + 2 < len(raw_kps) else 0
                        kps.append([kx, ky, kv])
                    while len(kps) < 13:
                        kps.append([0.0, 0.0, 0])
                    instances.append({
                        'class_id': cls_id,
                        'bbox': [cx, cy, w, h],
                        'keypoints': kps
                    })

        # Resize to square 256x256
        scale_x = self.img_size / float(orig_w)
        scale_y = self.img_size / float(orig_h)
        img_resized = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        for inst in instances:
            for kp in inst['keypoints']:
                if kp[2] > 0:
                    kp[0] *= scale_x
                    kp[1] *= scale_y

        # Data augmentation: Random horizontal flip
        if self.augment and random.random() < self.flip_prob:
            img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))
            for inst in instances:
                inst['bbox'][0] = 1.0 - inst['bbox'][0]
                inst['class_id'] = 1 - inst['class_id']
                orig_kps = [list(kp) for kp in inst['keypoints']]
                flipped_kps = [None] * 13
                for orig_idx, sym_idx in enumerate(FLIP_IDX_13):
                    kp = orig_kps[orig_idx]
                    flipped_x = float(self.img_size) - 1.0 - kp[0] if kp[2] > 0 else 0.0
                    flipped_kps[sym_idx] = [flipped_x, kp[1], kp[2]]
                inst['keypoints'] = flipped_kps

        # Convert image to [3, 256, 256] float tensor normalized to [0, 1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        return {
            'image': img_tensor,
            'instances': instances,
            'image_path': str(img_path)
        }


def collate_fn_13kp(batch):
    images = torch.stack([b['image'] for b in batch])
    instances = [b['instances'] for b in batch]
    image_paths = [b['image_path'] for b in batch]
    return {
        'image': images,
        'instances': instances,
        'image_paths': image_paths
    }


# =============================================================================
# 4. ARShoe M1v2 Architecture (MobileNetV3 + FPN-Lite + 3 DSConv Heads)
# =============================================================================

class DSConv(nn.Module):
    """Depthwise Separable Convolution with Hardswish & BatchNorm"""
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
    ARShoe M1v2: Lightweight MobileNetV3 + FPN-lite architecture.
    Budget: ~902K parameters, ~164 MMACs.
    """
    def __init__(self, num_keypoints=13, num_limbs=16, num_classes=2, pretrained_backbone=True):
        super(ARShoeM1v2, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_limbs = num_limbs
        self.num_classes = num_classes

        # 1. MobileNetV3-Small Backbone
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        mobilenet = models.mobilenet_v3_small(weights=weights)
        features = mobilenet.features

        self.s4 = features[:2]    # [B, 16, 64, 64]   (stride 4)
        self.s8 = features[2:4]   # [B, 24, 32, 32]   (stride 8)
        self.s16 = features[4:9]  # [B, 48, 16, 16]   (stride 16)
        self.s32 = features[9:12] # [B, 96, 8, 8]     (stride 32)

        # 2. FPN-Lite Neck
        self.lat32 = nn.Conv2d(96, 48, kernel_size=1, bias=False)
        self.lat16 = nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.lat8  = nn.Conv2d(24, 48, kernel_size=1, bias=False)
        self.lat4  = nn.Conv2d(16, 48, kernel_size=1, bias=False)
        self.fpn_conv = DSConv(48, 64)

        # 3. Depthwise Separable Heads
        self.hm_head = nn.Sequential(
            DSConv(64, 64),
            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )
        self.paf_head = nn.Sequential(
            DSConv(64, 64),
            nn.Conv2d(64, num_limbs * 2, kernel_size=1)
        )
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
        x4 = self.s4(x)
        x8 = self.s8(x4)
        x16 = self.s16(x8)
        x32 = self.s32(x16)

        p32 = self.lat32(x32)
        p16 = self.lat16(x16) + F.interpolate(p32, scale_factor=2, mode='nearest')
        p8  = self.lat8(x8)   + F.interpolate(p16, scale_factor=2, mode='nearest')
        p4  = self.lat4(x4)   + F.interpolate(p8, scale_factor=2, mode='nearest')
        feat = self.fpn_conv(p4)

        hm_logits = self.hm_head(feat)
        heatmaps = torch.sigmoid(hm_logits)
        pafs = self.paf_head(feat)

        g_cls = self.cls_global(x32).unsqueeze(-1).unsqueeze(-1)
        s_cls = self.cls_spatial(feat)
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
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total


# =============================================================================
# 5. Loss Functions
# =============================================================================

class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss with spatial visibility masking for heatmap regression"""
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1, use_mask=True):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_mask = use_mask

    def forward(self, pred, gt, masks=None):
        delta = (gt - pred).abs()
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - gt))) * \
            (self.alpha - gt) * torch.pow(self.theta / self.epsilon, self.alpha - gt - 1) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - gt))
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - gt)),
            A * delta - C
        )
        if self.use_mask and masks is not None:
            loss = loss * masks
            return loss.sum() / (masks.sum() + 1e-6)
        return loss.mean()


class PAFLoss(nn.Module):
    """Smooth L1 Loss on vector fields with limb masking"""
    def __init__(self, beta=1.0, use_mask=True):
        super(PAFLoss, self).__init__()
        self.beta = beta
        self.use_mask = use_mask

    def forward(self, pred, gt, masks=None):
        loss = F.smooth_l1_loss(pred, gt, beta=self.beta, reduction='none')
        if self.use_mask and masks is not None:
            masks_expanded = torch.repeat_interleave(masks, 2, dim=1)
            loss = loss * masks_expanded
            return loss.sum() / (masks_expanded.sum() + 1e-6)
        return loss.mean()


class ClassLoss(nn.Module):
    """Cross-entropy foot classification loss (ignoring background index -1)"""
    def __init__(self, ignore_index=-1):
        super(ClassLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, pred_logits, gt_class_maps):
        return self.ce(pred_logits, gt_class_maps)


# =============================================================================
# 6. Post-Processing & OpenPose PAF-Led Assembly (For Visualizations)
# =============================================================================

def decode_paf_instances_single(heatmaps, class_probs, pafs, peak_threshold=0.20, nms_window=5, max_limb_dist=48.0):
    """
    OpenPose-inspired PAF-led assembly decoder for 13 foot keypoints.
    """
    num_k = heatmaps.shape[0]
    H, W = heatmaps.shape[1], heatmaps.shape[2]

    # 1. Peak detection with subpixel Taylor interpolation
    all_peaks = {}
    for k in range(num_k):
        hm = heatmaps[k]
        local_max = maximum_filter(hm, size=nms_window)
        peaks_binary = (hm == local_max) & (hm >= peak_threshold)
        labeled_mask, num_features = nd_label(peaks_binary)
        k_peaks = []
        if num_features > 0:
            for feat_idx in range(1, num_features + 1):
                ys, xs = np.where(labeled_mask == feat_idx)
                best_i = np.argmax(hm[ys, xs])
                y, x = int(ys[best_i]), int(xs[best_i])
                conf = float(hm[y, x])
                sub_x, sub_y = float(x), float(y)
                if 1 <= x < W - 1 and 1 <= y < H - 1:
                    dx = 0.5 * (hm[y, x + 1] - hm[y, x - 1])
                    dxx = hm[y, x + 1] - 2.0 * hm[y, x] + hm[y, x - 1]
                    if abs(dxx) > 1e-5:
                        sub_x += float(np.clip(-dx / dxx, -0.5, 0.5))
                    dy = 0.5 * (hm[y + 1, x] - hm[y - 1, x])
                    dyy = hm[y + 1, x] - 2.0 * hm[y, x] + hm[y - 1, x]
                    if abs(dyy) > 1e-5:
                        sub_y += float(np.clip(-dy / dyy, -0.5, 0.5))
                k_peaks.append((sub_x, sub_y, conf))
        k_peaks.sort(key=lambda p: p[2], reverse=True)
        all_peaks[k] = k_peaks[:6]

    # 2. Score candidate pairs along 16 limbs
    all_matched = []
    for l_idx, (u, v, _) in enumerate(PAF_LIMBS_13):
        if l_idx * 2 + 1 >= pafs.shape[0]:
            continue
        peaks_u = all_peaks.get(u, [])
        peaks_v = all_peaks.get(v, [])
        if not peaks_u or not peaks_v:
            continue
        paf_x = pafs[l_idx * 2]
        paf_y = pafs[l_idx * 2 + 1]
        limb_conns = []
        for i, p_u in enumerate(peaks_u):
            for j, p_v in enumerate(peaks_v):
                dx = p_v[0] - p_u[0]
                dy = p_v[1] - p_u[1]
                dist = math.hypot(dx, dy)
                if dist > max_limb_dist or dist < 0.5:
                    continue
                ux, uy = dx / dist, dy / dist
                t_vals = np.linspace(0.0, 1.0, 10, dtype=np.float32)
                sample_xs = np.clip(np.round(p_u[0] + t_vals * dx).astype(int), 0, W - 1)
                sample_ys = np.clip(np.round(p_u[1] + t_vals * dy).astype(int), 0, H - 1)
                dot_prods = paf_x[sample_ys, sample_xs] * ux + paf_y[sample_ys, sample_xs] * uy
                if np.sum(dot_prods > 0.05) >= 4 and np.mean(dot_prods) > 0.05:
                    comb_score = float(np.mean(dot_prods)) + 0.5 * (p_u[2] + p_v[2])
                    limb_conns.append((comb_score, i, j, p_u, p_v))
        limb_conns.sort(key=lambda item: item[0], reverse=True)
        matched_u, matched_v = set(), set()
        for score, i, j, p_u, p_v in limb_conns:
            if i not in matched_u and j not in matched_v:
                matched_u.add(i)
                matched_v.add(j)
                all_matched.append({'score': score, 'u': u, 'v': v, 'cand_u': i, 'cand_v': j, 'p_u': p_u, 'p_v': p_v})

    all_matched.sort(key=lambda item: item['score'], reverse=True)

    # 3. Assemble connected limbs into feet
    assembled = []
    for conn in all_matched:
        u, v = conn['u'], conn['v']
        cu, cv = conn['cand_u'], conn['cand_v']
        pu, pv = conn['p_u'], conn['p_v']
        score = conn['score']
        iu_idx, iv_idx = None, None
        for idx, inst in enumerate(assembled):
            if u in inst['kps'] and inst['kps'][u][3] == cu:
                iu_idx = idx
            if v in inst['kps'] and inst['kps'][v][3] == cv:
                iv_idx = idx
        if iu_idx is None and iv_idx is None:
            assembled.append({'kps': {u: (pu[0], pu[1], pu[2], cu), v: (pv[0], pv[1], pv[2], cv)}, 'score': score, 'edges': 1})
        elif iu_idx is not None and iv_idx is None:
            inst = assembled[iu_idx]
            if v not in inst['kps']:
                inst['kps'][v] = (pv[0], pv[1], pv[2], cv)
                inst['score'] += score
                inst['edges'] += 1
        elif iu_idx is None and iv_idx is not None:
            inst = assembled[iv_idx]
            if u not in inst['kps']:
                inst['kps'][u] = (pu[0], pu[1], pu[2], cu)
                inst['score'] += score
                inst['edges'] += 1
        elif iu_idx != iv_idx:
            inst_u, inst_v = assembled[iu_idx], assembled[iv_idx]
            conflict = any(k in inst_u['kps'] and inst_u['kps'][k][3] != inst_v['kps'][k][3] for k in inst_v['kps'])
            if not conflict:
                inst_u['kps'].update(inst_v['kps'])
                inst_u['score'] += inst_v['score'] + score
                inst_u['edges'] += inst_v['edges'] + 1
                assembled.pop(iv_idx)

    # 4. Filter and format final instances
    valid_instances = [inst for inst in assembled if len(inst['kps']) >= 3]
    valid_instances.sort(key=lambda inst: inst['score'], reverse=True)
    final_preds = []
    scale = 256.0 / float(H)

    for inst in valid_instances[:2]:
        kps_out = []
        inst_points = []
        for k_idx in range(num_k):
            if k_idx in inst['kps']:
                x, y, conf, _ = inst['kps'][k_idx]
                kps_out.append([float(x * scale), float(y * scale), float(conf)])
                inst_points.append((x, y))
            else:
                kps_out.append([0.0, 0.0, 0.0])

        # Class determination from class_probs map
        left_votes, right_votes = 0.0, 0.0
        for px, py in inst_points:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < W and 0 <= iy < H:
                left_votes += float(class_probs[0, iy, ix])
                right_votes += float(class_probs[1, iy, ix])
        pred_cls = 0 if left_votes >= right_votes else 1
        final_preds.append({
            'class_id': pred_cls,
            'class_name': "left_foot" if pred_cls == 0 else "right_foot",
            'keypoints': kps_out,
            'score': float(inst['score'])
        })

    return final_preds


# =============================================================================
# 7. High-Speed Training Engine
# =============================================================================

class Fast13KPTrainer:
    """End-to-end high throughput trainer with AMP, grid caching, and live visualizations"""
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss functions
        self.hm_loss = AdaptiveWingLoss(use_mask=True)
        self.paf_loss = PAFLoss(use_mask=True)
        self.cls_loss = ClassLoss(ignore_index=-1)
        self.loss_weights = {'heatmap': 4.0, 'paf': 2.0, 'class': 1.5}

        # Optimizer with differential learning rates
        backbone_params = (
            list(self.model.s4.parameters()) + list(self.model.s8.parameters()) +
            list(self.model.s16.parameters()) + list(self.model.s32.parameters())
        )
        head_params = (
            list(self.model.lat32.parameters()) + list(self.model.lat16.parameters()) +
            list(self.model.lat8.parameters()) + list(self.model.lat4.parameters()) +
            list(self.model.fpn_conv.parameters()) + list(self.model.hm_head.parameters()) +
            list(self.model.paf_head.parameters()) + list(self.model.cls_global.parameters()) +
            list(self.model.cls_spatial.parameters())
        )
        lr = config['lr']
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': lr * 0.5},
            {'params': head_params, 'lr': lr}
        ], weight_decay=config.get('weight_decay', 1e-4))

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config.get('eta_min', 1e-6)
        )

        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.output_dir / 'epoch_visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.history = []

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:3d}/{self.config['epochs']}")
        epoch_losses = {'total': 0.0, 'hm': 0.0, 'paf': 0.0, 'cls': 0.0}
        n_batches = len(self.train_loader)

        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            instances = batch['instances']

            # Target generation on GPU
            gt_hm, hm_masks = generate_heatmaps_batch(instances, heatmap_size=64, device=self.device)
            gt_paf, paf_masks = generate_pafs_batch(instances, heatmap_size=64, device=self.device)
            gt_cls = generate_class_maps_batch(instances, heatmap_size=64, device=self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                l_hm = self.hm_loss(outputs['heatmaps'], gt_hm, masks=hm_masks)
                l_paf = self.paf_loss(outputs['pafs'], gt_paf, masks=paf_masks)
                l_cls = self.cls_loss(outputs['class_logits'], gt_cls)
                total_loss = (
                    self.loss_weights['heatmap'] * l_hm +
                    self.loss_weights['paf'] * l_paf +
                    self.loss_weights['class'] * l_cls
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['hm'] += l_hm.item()
            epoch_losses['paf'] += l_paf.item()
            epoch_losses['cls'] += l_cls.item()

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'hm': f"{l_hm.item():.4f}",
                'paf': f"{l_paf.item():.4f}",
                'cls': f"{l_cls.item():.4f}"
            })

        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)
        return epoch_losses

    def validate(self):
        self.model.eval()
        val_losses = {'total': 0.0, 'hm': 0.0, 'paf': 0.0, 'cls': 0.0}
        n_batches = len(self.val_loader)
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                instances = batch['instances']
                gt_hm, hm_masks = generate_heatmaps_batch(instances, heatmap_size=64, device=self.device)
                gt_paf, paf_masks = generate_pafs_batch(instances, heatmap_size=64, device=self.device)
                gt_cls = generate_class_maps_batch(instances, heatmap_size=64, device=self.device)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(images)
                    l_hm = self.hm_loss(outputs['heatmaps'], gt_hm, masks=hm_masks)
                    l_paf = self.paf_loss(outputs['pafs'], gt_paf, masks=paf_masks)
                    l_cls = self.cls_loss(outputs['class_logits'], gt_cls)
                    total_loss = (
                        self.loss_weights['heatmap'] * l_hm +
                        self.loss_weights['paf'] * l_paf +
                        self.loss_weights['class'] * l_cls
                    )
                val_losses['total'] += total_loss.item()
                val_losses['hm'] += l_hm.item()
                val_losses['paf'] += l_paf.item()
                val_losses['cls'] += l_cls.item()

        for k in val_losses:
            val_losses[k] /= max(1, n_batches)
        return val_losses

    def visualize_epoch(self, epoch):
        """Render side-by-side test visualizations comparing GT and OpenPose PAF predictions"""
        self.model.eval()
        panel_w, panel_h = 280, 280
        all_panels = []

        with torch.no_grad():
            for i in range(min(2, len(self.val_loader.dataset))):
                sample = self.val_loader.dataset[i]
                img_t = sample['image'].unsqueeze(0).to(self.device)
                outputs = self.model(img_t)
                hm_np = outputs['heatmaps'][0].cpu().numpy()
                cls_np = outputs['class_probs'][0].cpu().numpy()
                paf_np = outputs['pafs'][0].cpu().numpy()

                preds = decode_paf_instances_single(hm_np, cls_np, paf_np)

                img_np = (sample['image'].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Ground truth panel
                p_gt = cv2.resize(img_bgr, (panel_w, panel_h))
                sx = panel_w / 256.0
                sy = panel_h / 256.0
                for inst in sample['instances']:
                    kps = inst['keypoints']
                    for u, v, _ in PAF_LIMBS_13:
                        if kps[u][2] > 0 and kps[v][2] > 0:
                            cv2.line(p_gt, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 255, 0), 2, cv2.LINE_AA)
                    for k_idx in range(13):
                        if kps[k_idx][2] > 0:
                            cv2.circle(p_gt, (int(kps[k_idx][0]*sx), int(kps[k_idx][1]*sy)), 4, KP_COLORS_13[k_idx], -1, cv2.LINE_AA)
                cv2.rectangle(p_gt, (0, 0), (panel_w, 24), (30, 30, 30), -1)
                cv2.putText(p_gt, f"Sample {i+1}: Ground Truth", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                # Prediction panel
                p_pred = cv2.resize(img_bgr, (panel_w, panel_h))
                for pred in preds:
                    kps = pred['keypoints']
                    for u, v, _ in PAF_LIMBS_13:
                        if kps[u][2] > 0.1 and kps[v][2] > 0.1:
                            cv2.line(p_pred, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 200, 255), 2, cv2.LINE_AA)
                    for k_idx in range(13):
                        if kps[k_idx][2] > 0.1:
                            cv2.circle(p_pred, (int(kps[k_idx][0]*sx), int(kps[k_idx][1]*sy)), 4, KP_COLORS_13[k_idx], -1, cv2.LINE_AA)
                cv2.rectangle(p_pred, (0, 0), (panel_w, 24), (30, 30, 30), -1)
                pred_label = f"Epoch {epoch:03d} Prediction ({len(preds)} feet)"
                cv2.putText(p_pred, pred_label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

                sample_combined = np.hstack([p_gt, p_pred])
                all_panels.append(sample_combined)

        if all_panels:
            final_vis = np.vstack(all_panels)
            vis_file = self.vis_dir / f"epoch_{epoch:03d}_vis.jpg"
            cv2.imwrite(str(vis_file), final_vis)
            print(f"  🖼️ [Epoch {epoch:03d}] Live visualization saved -> {vis_file}")

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        latest_file = self.checkpoint_dir / 'latest.pth'
        torch.save(ckpt, latest_file)
        if is_best:
            best_file = self.checkpoint_dir / 'best.pth'
            torch.save(ckpt, best_file)
            print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")

    def run(self):
        print("=" * 80)
        print(f"👟 Training ARShoe M1v2 (13-KP) for {self.config['epochs']} epochs")
        print(f"  Device           : {self.device} (AMP: {self.use_amp})")
        print(f"  Batch size       : {self.config['batch_size']}")
        print(f"  Learning rate    : {self.config['lr']} (Cosine decay -> {self.config['eta_min']})")
        print(f"  Train samples    : {len(self.train_loader.dataset)} ({len(self.train_loader)} batches/epoch)")
        print(f"  Val samples      : {len(self.val_loader.dataset)}")
        print("=" * 80)

        for epoch in range(1, self.config['epochs'] + 1):
            t0 = time.time()
            train_l = self.train_epoch(epoch)
            self.scheduler.step()
            epoch_time = time.time() - t0

            # Validate every 2 epochs
            if epoch % 2 == 0 or epoch == self.config['epochs']:
                val_l = self.validate()
                is_best = val_l['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_l['total']
                self.save_checkpoint(epoch, val_l['total'], is_best=is_best)
                print(f"Epoch {epoch:3d}/{self.config['epochs']} ({epoch_time:.1f}s) | "
                      f"Train Loss: {train_l['total']:.4f} (HM: {train_l['hm']:.4f}, PAF: {train_l['paf']:.4f}) | "
                      f"Val Loss: {val_l['total']:.4f} (HM: {val_l['hm']:.4f}, PAF: {val_l['paf']:.4f}) | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Epoch {epoch:3d}/{self.config['epochs']} ({epoch_time:.1f}s) | "
                      f"Train Loss: {train_l['total']:.4f} (HM: {train_l['hm']:.4f}, PAF: {train_l['paf']:.4f}) | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Live visualization callback
            vis_interval = self.config.get('vis_interval', 10)
            if vis_interval > 0 and (epoch % vis_interval == 0 or epoch == 1 or epoch == self.config['epochs']):
                self.visualize_epoch(epoch)

        print("\n" + "=" * 80)
        print(f"✅ Training completed! Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 80)


# =============================================================================
# 8. Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="High-Speed 13-Keypoint ARShoe M1v2 Training Pipeline")
    parser.add_argument('--dataset_root', type=str, default='dataset/shuffled_v4_1_13kp', help='Path to 13-KP dataset root')
    parser.add_argument('--output_dir', type=str, default='shoes_vto/outputs/m1v2_13kp_stage2_paf_fix', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64 for high GPU utilization)')
    parser.add_argument('--lr', type=float, default=0.001, help='Peak learning rate')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers (default: 2 for Colab)')
    parser.add_argument('--vis_interval', type=int, default=10, help='Epoch interval for live test visualizations')
    parser.add_argument('--no_amp', action='store_true', help='Disable Automatic Mixed Precision')
    parser.add_argument('--dry_run', action='store_true', help='Run a fast 1-epoch synthetic sanity test')
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("👟 Initializing ARShoe M1v2 13-KP Pipeline...")
    print("=" * 80)

    # 1. Model
    model = ARShoeM1v2(num_keypoints=13, num_limbs=16, num_classes=2, pretrained_backbone=not args.dry_run)
    total_params = model.count_parameters()
    print(f"  Model Parameters: {total_params:,} (~{total_params/1_300_000*100:.1f}% of 1.3M mobile budget)")

    # 2. Dry run synthetic mode
    if args.dry_run:
        print("\n🧪 Running fast synthetic dry run (1 batch, forward + backward test)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        dummy_img = torch.randn(4, 3, 256, 256, device=device)
        dummy_instances = [
            [{'class_id': 0, 'bbox': [0.5, 0.5, 0.4, 0.4], 'keypoints': [[128, 128, 2] for _ in range(13)]}]
            for _ in range(4)
        ]
        gt_hm, hm_m = generate_heatmaps_batch(dummy_instances, heatmap_size=64, device=device)
        gt_paf, paf_m = generate_pafs_batch(dummy_instances, heatmap_size=64, device=device)
        gt_cls = generate_class_maps_batch(dummy_instances, heatmap_size=64, device=device)

        out = model(dummy_img)
        hm_l = AdaptiveWingLoss()(out['heatmaps'], gt_hm, masks=hm_m)
        paf_l = PAFLoss()(out['pafs'], gt_paf, masks=paf_m)
        cls_l = ClassLoss()(out['class_logits'], gt_cls)
        loss = 4.0 * hm_l + 2.0 * paf_l + 1.5 * cls_l
        loss.backward()
        print(f"  ✅ Synthetic dry-run passed cleanly! Total Loss: {loss.item():.4f}")
        return

    # 3. Load Datasets
    ds_root = Path(args.dataset_root)
    train_img_dir = ds_root / "train" / "images"
    train_lbl_dir = ds_root / "train" / "labels"
    val_img_dir = ds_root / "valid" / "images"
    val_lbl_dir = ds_root / "valid" / "labels"

    if not train_img_dir.exists():
        raise FileNotFoundError(f"Training directory not found at {train_img_dir}")
    if not val_img_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at {val_img_dir}")

    train_ds = Dataset13KP(train_img_dir, train_lbl_dir, img_size=256, augment=True)
    val_ds = Dataset13KP(val_img_dir, val_lbl_dir, img_size=256, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_13kp,
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_13kp,
        persistent_workers=(args.num_workers > 0)
    )

    # 4. Launch Trainer
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'eta_min': args.eta_min,
        'output_dir': args.output_dir,
        'use_amp': not args.no_amp,
        'vis_interval': args.vis_interval
    }

    trainer = Fast13KPTrainer(model, train_loader, val_loader, config)
    trainer.run()


if __name__ == '__main__':
    main()
