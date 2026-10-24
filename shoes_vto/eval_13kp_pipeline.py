"""
👟 ARShoe M1v2 13-Keypoint High-Precision Evaluation & Inference Pipeline
========================================================================
Self-contained, production-grade evaluation pipeline for 13-keypoint foot pose estimation.

Features:
1. OpenPose-Inspired PAF-Led Assembly Decoder:
   - Subpixel Taylor quadratic peak refinement.
   - 16-limb line integral candidate scoring.
   - One-to-one bipartite limb matching.
   - Graph tree assembly with strict contradiction rejection.
   - Left/right foot classification assignment.
2. Multi-Threshold PCK Metrics:
   - PCK@0.20, PCK@0.10, PCK@0.05 normalized by foot bounding box diagonal.
   - Per-keypoint accuracy breakdown across all 13 anatomical landmarks.
   - Instance matching: Hungarian bipartite matching between predicted and GT feet.
   - Instance Recall, Precision, and F1-score (matched / missed / ghost feet).
3. Rich Visualizations:
   - Side-by-side 3-panel comparisons: [Original | Ground Truth | M1v2 Prediction].

Usage:
    # Full Validation Benchmark:
    python shoes_vto/eval_13kp_pipeline.py \
        --checkpoint shoes_vto/outputs/m1v2_13kp_training/checkpoints/best.pth \
        --valid_dir dataset/shuffled_v4_1_13kp/valid

    # Single Image Inference:
    python shoes_vto/eval_13kp_pipeline.py \
        --checkpoint shoes_vto/outputs/m1v2_13kp_training/checkpoints/best.pth \
        --image path/to/image.jpg
"""

import os
import sys
import json
import math
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
import torchvision.models as models
from scipy.ndimage import maximum_filter, label as nd_label
from scipy.optimize import linear_sum_assignment
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


# =============================================================================
# 2. ARShoe M1v2 Architecture
# =============================================================================

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super(DSConv, self).__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.Hardswish(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.act2(self.bn2(self.pw(self.act1(self.bn1(self.dw(x))))))


class ARShoeM1v2(nn.Module):
    def __init__(self, num_keypoints=13, num_limbs=16, num_classes=2):
        super(ARShoeM1v2, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_limbs = num_limbs
        self.num_classes = num_classes

        # MobileNetV3-Small Backbone
        mobilenet = models.mobilenet_v3_small(weights=None)
        features = mobilenet.features

        self.s4 = features[:2]    # [B, 16, 64, 64]   (stride 4)
        self.s8 = features[2:4]   # [B, 24, 32, 32]   (stride 8)
        self.s16 = features[4:9]  # [B, 48, 16, 16]   (stride 16)
        self.s32 = features[9:12] # [B, 96, 8, 8]     (stride 32)

        # FPN-Lite Neck
        self.lat32 = nn.Conv2d(96, 48, kernel_size=1, bias=False)
        self.lat16 = nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.lat8  = nn.Conv2d(24, 48, kernel_size=1, bias=False)
        self.lat4  = nn.Conv2d(16, 48, kernel_size=1, bias=False)
        self.fpn_conv = DSConv(48, 64)

        # Heads
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


def load_model(checkpoint_path, device='cpu'):
    """Load ARShoeM1v2 weights from file"""
    model = ARShoeM1v2(num_keypoints=13, num_limbs=16, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Clean potential _orig_mod prefix from torch.compile()
    cleaned = {}
    for k, v in state_dict.items():
        k_clean = k.replace('_orig_mod.', '').replace('module.', '')
        cleaned[k_clean] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# 3. OpenPose PAF-Led Assembly Decoder
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
# 4. Metrics & Evaluation Engine
# =============================================================================

def compute_box_diag(kps):
    """Compute bounding box diagonal from visible keypoints"""
    valid = [k for k in kps if k[2] > 0]
    if len(valid) < 2:
        return 100.0
    xs = [k[0] for k in valid]
    ys = [k[1] for k in valid]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return max(10.0, math.hypot(w, h))


def evaluate_dataset(model, valid_dir, device='cpu', max_samples=None, save_vis=True, output_dir=None):
    """
    Run full validation benchmark: PCK@0.20, PCK@0.10, PCK@0.05, per-keypoint accuracy,
    and foot instance recall metrics.
    """
    valid_path = Path(valid_dir)
    img_dir = valid_path / "images"
    lbl_dir = valid_path / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"Images folder not found at {img_dir}")

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_files = sorted([f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    if max_samples:
        img_files = img_files[:max_samples]

    if output_dir:
        out_path = Path(output_dir)
        vis_dir = out_path / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        vis_dir = None

    pck_thresholds = [0.20, 0.10, 0.05]
    kp_correct = {t: np.zeros(13, dtype=int) for t in pck_thresholds}
    kp_total = np.zeros(13, dtype=int)
    kp_pixel_errors = {k: [] for k in range(13)}

    total_gt_feet = 0
    total_pred_feet = 0
    matched_feet = 0

    print("=" * 80)
    print(f"📊 Running ARShoe M1v2 13-KP Validation Benchmark ({len(img_files)} images)...")
    print("=" * 80)

    for s_idx, img_file in enumerate(tqdm(img_files, desc="Evaluating")):
        img_bgr = cv2.imread(str(img_file))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        # Parse ground truth
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        gt_instances = []
        if lbl_file.exists():
            with open(lbl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    raw_kps = [float(p) for p in parts[5:]]
                    kps = []
                    for k in range(0, min(len(raw_kps), 13 * 3), 3):
                        kx = raw_kps[k] * 256.0
                        ky = raw_kps[k + 1] * 256.0
                        kv = int(raw_kps[k + 2]) if k + 2 < len(raw_kps) else 0
                        kps.append([kx, ky, kv])
                    while len(kps) < 13:
                        kps.append([0.0, 0.0, 0])
                    gt_instances.append({'class_id': cls_id, 'keypoints': kps})

        total_gt_feet += len(gt_instances)

        # Preprocess image
        img_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            hm_np = outputs['heatmaps'][0].cpu().numpy()
            cls_np = outputs['class_probs'][0].cpu().numpy()
            paf_np = outputs['pafs'][0].cpu().numpy()

        preds = decode_paf_instances_single(hm_np, cls_np, paf_np)
        total_pred_feet += len(preds)

        # Instance Hungarian matching between GT and Pred feet
        if len(gt_instances) > 0 and len(preds) > 0:
            cost_matrix = np.zeros((len(gt_instances), len(preds)), dtype=np.float32)
            for g_i, gt in enumerate(gt_instances):
                gt_kps = gt['keypoints']
                diag = compute_box_diag(gt_kps)
                for p_i, pred in enumerate(preds):
                    pr_kps = pred['keypoints']
                    dists = []
                    for k in range(13):
                        if gt_kps[k][2] > 0 and pr_kps[k][2] > 0.05:
                            dists.append(math.hypot(gt_kps[k][0] - pr_kps[k][0], gt_kps[k][1] - pr_kps[k][1]))
                    cost_matrix[g_i, p_i] = np.mean(dists) if dists else 999.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_pairs = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 120.0:
                    matched_pairs.append((r, c))
                    matched_feet += 1

            for g_idx, p_idx in matched_pairs:
                gt_kps = gt_instances[g_idx]['keypoints']
                pr_kps = preds[p_idx]['keypoints']
                diag = compute_box_diag(gt_kps)

                for k in range(13):
                    if gt_kps[k][2] > 0:
                        kp_total[k] += 1
                        if pr_kps[k][2] > 0.05:
                            err = math.hypot(gt_kps[k][0] - pr_kps[k][0], gt_kps[k][1] - pr_kps[k][1])
                            kp_pixel_errors[k].append(err)
                            for t in pck_thresholds:
                                if err <= t * diag:
                                    kp_correct[t][k] += 1
        elif len(gt_instances) > 0:
            for gt in gt_instances:
                for k in range(13):
                    if gt['keypoints'][k][2] > 0:
                        kp_total[k] += 1

        # Save side-by-side visualization if requested
        if vis_dir and s_idx < 20:
            panel_w, panel_h = 280, 280
            p_orig = cv2.resize(img_bgr, (panel_w, panel_h))
            cv2.rectangle(p_orig, (0, 0), (panel_w, 24), (30, 30, 30), -1)
            cv2.putText(p_orig, f"{img_file.name[:18]}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            p_gt = p_orig.copy()
            sx, sy = panel_w / 256.0, panel_h / 256.0
            for gt in gt_instances:
                kps = gt['keypoints']
                for u, v, _ in PAF_LIMBS_13:
                    if kps[u][2] > 0 and kps[v][2] > 0:
                        cv2.line(p_gt, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 255, 0), 2, cv2.LINE_AA)
                for k in range(13):
                    if kps[k][2] > 0:
                        cv2.circle(p_gt, (int(kps[k][0]*sx), int(kps[k][1]*sy)), 4, KP_COLORS_13[k], -1, cv2.LINE_AA)
            cv2.rectangle(p_gt, (0, 0), (panel_w, 24), (30, 30, 30), -1)
            cv2.putText(p_gt, f"Ground Truth ({len(gt_instances)} feet)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            p_pred = p_orig.copy()
            for pred in preds:
                kps = pred['keypoints']
                for u, v, _ in PAF_LIMBS_13:
                    if kps[u][2] > 0.1 and kps[v][2] > 0.1:
                        cv2.line(p_pred, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 200, 255), 2, cv2.LINE_AA)
                for k in range(13):
                    if kps[k][2] > 0.1:
                        cv2.circle(p_pred, (int(kps[k][0]*sx), int(kps[k][1]*sy)), 4, KP_COLORS_13[k], -1, cv2.LINE_AA)
            cv2.rectangle(p_pred, (0, 0), (panel_w, 24), (30, 30, 30), -1)
            cv2.putText(p_pred, f"M1v2 OpenPose PAF ({len(preds)} feet)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

            vis_triple = np.hstack([p_orig, p_gt, p_pred])
            cv2.imwrite(str(vis_dir / f"eval_{s_idx:03d}_{img_file.stem[:12]}.jpg"), vis_triple)

    # Compute Final Summary Metrics
    total_valid_kps = int(np.sum(kp_total))
    overall_pck_20 = float(np.sum(kp_correct[0.20]) / max(1, total_valid_kps)) * 100.0
    overall_pck_10 = float(np.sum(kp_correct[0.10]) / max(1, total_valid_kps)) * 100.0
    overall_pck_05 = float(np.sum(kp_correct[0.05]) / max(1, total_valid_kps)) * 100.0

    recall = float(matched_feet / max(1, total_gt_feet)) * 100.0
    precision = float(matched_feet / max(1, total_pred_feet)) * 100.0
    f1 = float(2 * precision * recall / max(1e-5, precision + recall))

    # Print Clean Formatted Markdown Table
    print("\n" + "=" * 80)
    print("📈 ARSHOE M1V2 13-KEYPOINT VALIDATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Total Images Evaluated : {len(img_files)}")
    print(f"  Ground Truth Feet      : {total_gt_feet}")
    print(f"  Predicted Feet         : {total_pred_feet} (Matched: {matched_feet}, Missed: {total_gt_feet - matched_feet})")
    print(f"  Foot Instance Recall   : {recall:.2f}% | Precision: {precision:.2f}% | F1: {f1:.2f}%")
    print("-" * 80)
    print(f"  Overall PCK@0.20       : {overall_pck_20:.2f}%")
    print(f"  Overall PCK@0.10       : {overall_pck_10:.2f}%")
    print(f"  Overall PCK@0.05       : {overall_pck_05:.2f}%")
    print("=" * 80)

    print("\n| Keypoint Name | Total KPs | PCK@0.20 | PCK@0.10 | PCK@0.05 | Mean Error (px) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    per_kp_results = {}
    for k in range(13):
        tot = int(kp_total[k])
        p20 = float(kp_correct[0.20][k] / max(1, tot)) * 100.0
        p10 = float(kp_correct[0.10][k] / max(1, tot)) * 100.0
        p05 = float(kp_correct[0.05][k] / max(1, tot)) * 100.0
        err = float(np.mean(kp_pixel_errors[k])) if kp_pixel_errors[k] else 0.0
        per_kp_results[KEYPOINT_NAMES_13[k]] = {
            'total': tot, 'pck@0.20': p20, 'pck@0.10': p10, 'pck@0.05': p05, 'mean_error_px': err
        }
        print(f"| `{KEYPOINT_NAMES_13[k]:<18}` | {tot:5d} | {p20:6.2f}% | {p10:6.2f}% | {p05:6.2f}% | {err:6.2f} px |")
    print("=" * 80)

    results_dict = {
        'total_images': len(img_files),
        'gt_feet': total_gt_feet,
        'pred_feet': total_pred_feet,
        'matched_feet': matched_feet,
        'foot_recall': recall,
        'foot_precision': precision,
        'foot_f1': f1,
        'overall_pck@0.20': overall_pck_20,
        'overall_pck@0.10': overall_pck_10,
        'overall_pck@0.05': overall_pck_05,
        'per_keypoint': per_kp_results
    }

    if output_dir:
        json_file = Path(output_dir) / "benchmark_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\n💾 Benchmark results saved -> {json_file}")

    return results_dict


# =============================================================================
# 5. Single Image Inference
# =============================================================================

def run_single_image_inference(model, image_path, device='cpu', output_dir='shoes_vto/outputs/single_image_infer'):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load image from {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        hm_np = outputs['heatmaps'][0].cpu().numpy()
        cls_np = outputs['class_probs'][0].cpu().numpy()
        paf_np = outputs['pafs'][0].cpu().numpy()

    preds = decode_paf_instances_single(hm_np, cls_np, paf_np)

    vis_img = img_bgr.copy()
    sx = orig_w / 256.0
    sy = orig_h / 256.0

    print(f"\n👟 Detected {len(preds)} foot instance(s) in {image_path}:")
    for idx, pred in enumerate(preds):
        print(f"  Foot {idx+1}: {pred['class_name']} (score: {pred['score']:.2f})")
        kps = pred['keypoints']
        for u, v, limb_name in PAF_LIMBS_13:
            if kps[u][2] > 0.1 and kps[v][2] > 0.1:
                pt1 = (int(kps[u][0] * sx), int(kps[u][1] * sy))
                pt2 = (int(kps[v][0] * sx), int(kps[v][1] * sy))
                cv2.line(vis_img, pt1, pt2, (0, 200, 255), 3, cv2.LINE_AA)
        for k in range(13):
            if kps[k][2] > 0.1:
                pt = (int(kps[k][0] * sx), int(kps[k][1] * sy))
                cv2.circle(vis_img, pt, 5, KP_COLORS_13[k], -1, cv2.LINE_AA)
                cv2.circle(vis_img, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

    out_file = out_dir / f"pred_{Path(image_path).name}"
    cv2.imwrite(str(out_file), vis_img)
    print(f"  🖼️ Rendered prediction saved -> {out_file}")


# =============================================================================
# 6. Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="High-Precision 13-Keypoint ARShoe M1v2 Evaluation Pipeline")
    parser.add_argument('--checkpoint', type=str, default='shoes_vto/outputs/m1v2_13kp_training/checkpoints/best.pth', help='Path to checkpoint')
    parser.add_argument('--valid_dir', type=str, default='dataset/shuffled_v4_1_13kp/valid', help='Path to valid split')
    parser.add_argument('--image', type=str, default=None, help='Single image path for inference test')
    parser.add_argument('--output_dir', type=str, default='shoes_vto/outputs/eval_13kp_results', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum images to evaluate')
    parser.add_argument('--dry_run', action='store_true', help='Synthetic validation test')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dry_run:
        print("🧪 Running dry run test for eval_13kp_pipeline...")
        model = ARShoeM1v2(num_keypoints=13, num_limbs=16, num_classes=2)
        model.to(device).eval()
        dummy_in = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out = model(dummy_in)
        preds = decode_paf_instances_single(out['heatmaps'][0].cpu().numpy(), out['class_probs'][0].cpu().numpy(), out['pafs'][0].cpu().numpy())
        print(f"  ✅ Synthetic eval pass successful! Decoded instances: {len(preds)}")
        return

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Fallback check
        alt_paths = [
            Path("shoes_vto/outputs/m1v2_13kp_stage2_paf_fix/checkpoints/best.pth"),
            Path("outputs/m1v2_13kp_training/checkpoints/best.pth")
        ]
        found = False
        for alt in alt_paths:
            if alt.exists():
                ckpt_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print(f"Loading weights from {ckpt_path}...")
    model = load_model(str(ckpt_path), device=device)

    if args.image:
        run_single_image_inference(model, args.image, device=device, output_dir=args.output_dir)
    else:
        evaluate_dataset(model, args.valid_dir, device=device, max_samples=args.max_samples, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
