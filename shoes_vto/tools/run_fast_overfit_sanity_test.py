"""
Fast Overfit Sanity Test for ARShoe M1v2 (13 Keypoints)
Validates the training, loss (spatial visibility masking), and inference pipeline
on a fixed 8-image subset from the training split.
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from models.heads.heatmap_head import generate_heatmaps_batch
from models.heads.paf_head import generate_pafs_batch
from models.heads.class_head import generate_class_maps_batch
from losses.heatmap_loss import AdaptiveWingLoss
from losses.paf_loss import PAFLoss
from losses.class_loss import ClassLoss
from utils.keypoint_grouping import (
    detect_peaks,
    decode_spatial_instances_single
)

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
    "achilles",          # 12
]

SKELETON_LIMBS_13 = [
    (10, 0),  # toe_tip -> toe_ground
    (0, 2),   # toe_ground -> heel_ground
    (2, 1),   # heel_ground -> heel_back
    (3, 4),   # ball_medial -> ball_lateral
    (3, 5),   # ball_medial -> instep_top
    (4, 5),   # ball_lateral -> instep_top
    (2, 6),   # heel_ground -> arch_medial
    (2, 7),   # heel_ground -> midfoot_lateral
    (6, 5),   # arch_medial -> instep_top
    (7, 5),   # midfoot_lateral -> instep_top
    (5, 11),  # instep_top -> throat
    (11, 8),  # throat -> malleolus_medial
    (11, 9),  # throat -> malleolus_lateral
    (8, 12),  # malleolus_medial -> achilles
    (9, 12),  # malleolus_lateral -> achilles
    (12, 1),  # achilles -> heel_back
]

# 8 Representative images from the training split
SELECTED_STEMS = [
    # 4 Two-foot images with asymmetric visibility (keypoint visible on one foot, v=0 on other)
    '-4_jpg.rf.98277f4406f91f57d139a8bf7589cb7c',
    '-6_jpg.rf.0a5d49b0343b6e6a0d6e086b5e1a3707',
    '-_jpg.rf.b7e532c737ff3a14ebb16c7a208e941c',
    '-attachedteen-on-discord-tiktok_jpg.rf.91be9c55bf305fc60dc8fca200174b57',
    # 4 Single-foot images with mixed visibility (v=2, v=1, v=0)
    '-3_jpg.rf.5062bfbecadec3871c37cde866374226',
    '-peace_jpg.rf.f84bd928736ce2a1c7341f5e4506fc08',
    '1095571046880362857_jpg.rf.23ea8296be773a47af8675ab984aae08',
    '1114148395287951061_jpg.rf.56ba51c3c8704158190c4fb0c2db0201'
]


def safe_path(p):
    p_str = str(Path(p).resolve())
    if sys.platform == 'win32' and len(p_str) > 240 and not p_str.startswith('\\\\?\\'):
        return '\\\\?\\' + p_str
    return p_str


def parse_label_file(label_path, img_size=256):
    instances = []
    p_safe = safe_path(label_path)
    if not os.path.exists(p_safe):
        return instances
    with open(p_safe, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 44:
                continue
            cls_id = int(tokens[0])
            cx, cy, w, h = [float(x) for x in tokens[1:5]]
            kps = []
            for i in range(13):
                kx = float(tokens[5 + i * 3]) * img_size
                ky = float(tokens[5 + i * 3 + 1]) * img_size
                kv = int(float(tokens[5 + i * 3 + 2]))
                kps.append([kx, ky, kv])
            instances.append({
                'class_id': cls_id,
                'class_name': 'left_foot' if cls_id == 0 else 'right_foot',
                'bbox': [cx, cy, w, h],  # normalized 0-1
                'keypoints': kps          # in 0-256 pixel space
            })
    return instances


def load_dataset(dataset_root, stems, img_size=256):
    img_dir = dataset_root / "train" / "images"
    lbl_dir = dataset_root / "train" / "labels"
    
    samples = []
    for stem in stems:
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"
        
        safe_img_p = safe_path(img_path)
        img_raw = Image.open(safe_img_p).convert("RGB")
        orig_w, orig_h = img_raw.size
        
        img_resized = img_raw.resize((img_size, img_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
        
        instances = parse_label_file(lbl_path, img_size=img_size)
        
        samples.append({
            'stem': stem,
            'image': img_tensor,
            'instances': instances,
            'orig_size': (orig_w, orig_h),
            'img_path': str(img_path)
        })
    return samples


def match_instances(preds, gts):
    matches = {}
    for g_i, gt in enumerate(gts):
        best_p_i = None
        best_dist = 1e9
        for p_i, pred in enumerate(preds):
            if p_i in matches.values():
                continue
            if pred.get('class_id') == gt['class_id'] or len(preds) == 1:
                d_sum = 0.0
                d_cnt = 0
                for k in range(13):
                    if gt['keypoints'][k][2] > 0:
                        gx, gy, _ = gt['keypoints'][k]
                        px, py = pred['keypoints'][k][:2]
                        d_sum += np.hypot(px - gx, py - gy)
                        d_cnt += 1
                avg_d = d_sum / max(1, d_cnt)
                if avg_d < best_dist:
                    best_dist = avg_d
                    best_p_i = p_i
        if best_p_i is not None:
            matches[g_i] = best_p_i
        elif len(preds) > 0:
            # Fallback to nearest
            matches[g_i] = 0
    return matches


def evaluate_batch(model, samples, device):
    model.eval()
    thresholds = [0.20, 0.10, 0.05]
    
    stats_raw = {th: {k: {'correct': 0, 'total': 0} for k in range(13)} for th in thresholds}
    stats_dec = {th: {k: {'correct': 0, 'total': 0} for k in range(13)} for th in thresholds}
    
    kp_pixel_errors_raw = {k: [] for k in range(13)}
    kp_pixel_errors_dec = {k: [] for k in range(13)}
    
    instance_matches_info = []
    
    with torch.no_grad():
        for s_idx, sample in enumerate(samples):
            img_tensor = sample['image'].unsqueeze(0).to(device)
            gt_instances = sample['instances']
            stem = sample['stem']
            
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()  # [13, 64, 64]
            pafs = outputs['pafs'][0].cpu().numpy()          # [32, 64, 64]
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
            
            # 1. Raw heatmap peak detection
            scale = 256.0 / 64.0
            raw_peaks_per_kp = {}
            for k in range(13):
                pks = detect_peaks(heatmaps[k], threshold=0.08, nms_window=3, subpixel=True)
                if not pks:
                    py, px = np.unravel_index(np.argmax(heatmaps[k]), heatmaps[k].shape)
                    pks = [(float(px), float(py), float(heatmaps[k][py, px]))]
                raw_peaks_per_kp[k] = [(p[0] * scale, p[1] * scale, p[2]) for p in pks]
            
            # Raw peak per GT instance (nearest peak)
            raw_kps_per_gt = {}
            for g_i, gt in enumerate(gt_instances):
                raw_kps = []
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    pks = raw_peaks_per_kp[k]
                    if gv > 0:
                        best_pk = min(pks, key=lambda p: np.hypot(p[0] - gx, p[1] - gy))
                        raw_kps.append(best_pk)
                    else:
                        raw_kps.append(pks[0])
                raw_kps_per_gt[g_i] = raw_kps
            
            # 2. Production decoder
            raw_preds = decode_spatial_instances_single(heatmaps, cls_probs, pafs=pafs)
            preds_decoded = []
            for inst in raw_preds:
                c_inst = copy.deepcopy(inst)
                c_inst['keypoints'] = [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
                preds_decoded.append(c_inst)
            
            matches = match_instances(preds_decoded, gt_instances)
            instance_matches_info.append({
                'stem': stem,
                'gt_count': len(gt_instances),
                'pred_count': len(preds_decoded),
                'matches': {f"gt_{g}": f"pred_{p}" for g, p in matches.items()},
                'gt_classes': [gt['class_id'] for gt in gt_instances],
                'pred_classes': [p['class_id'] for p in preds_decoded]
            })
            
            # 3. Compute PCK and errors per GT instance
            for g_i, gt in enumerate(gt_instances):
                # Normalized bbox diagonal in 256 space
                w_px = gt['bbox'][2] * 256.0
                h_px = gt['bbox'][3] * 256.0
                bbox_diag = np.sqrt(w_px**2 + h_px**2)
                if bbox_diag < 1e-4:
                    continue
                
                matched_p_idx = matches.get(g_i, None)
                pred_inst = preds_decoded[matched_p_idx] if matched_p_idx is not None and matched_p_idx < len(preds_decoded) else None
                
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv == 0:
                        # Exclude unlabeled keypoints
                        continue
                    
                    # Raw peak
                    pk_raw = raw_kps_per_gt[g_i][k]
                    dist_raw = np.hypot(pk_raw[0] - gx, pk_raw[1] - gy)
                    kp_pixel_errors_raw[k].append(dist_raw)
                    
                    for th in thresholds:
                        limit = th * bbox_diag
                        stats_raw[th][k]['total'] += 1
                        if dist_raw < limit:
                            stats_raw[th][k]['correct'] += 1
                    
                    # Decoded
                    if pred_inst is not None:
                        pk_dec = pred_inst['keypoints'][k]
                        dist_dec = np.hypot(pk_dec[0] - gx, pk_dec[1] - gy)
                        kp_pixel_errors_dec[k].append(dist_dec)
                        
                        for th in thresholds:
                            limit = th * bbox_diag
                            stats_dec[th][k]['total'] += 1
                            if dist_dec < limit:
                                stats_dec[th][k]['correct'] += 1
                    else:
                        for th in thresholds:
                            stats_dec[th][k]['total'] += 1
    
    results = {}
    for th in thresholds:
        corr_r = sum(stats_raw[th][k]['correct'] for k in range(13))
        tot_r = sum(stats_raw[th][k]['total'] for k in range(13))
        corr_d = sum(stats_dec[th][k]['correct'] for k in range(13))
        tot_d = sum(stats_dec[th][k]['total'] for k in range(13))
        
        pck_raw = (corr_r / max(1, tot_r)) * 100.0
        pck_dec = (corr_d / max(1, tot_d)) * 100.0
        results[f'PCK@{th:.2f}'] = {'raw': pck_raw, 'decoded': pck_dec}
    
    per_kp = {}
    for k in range(13):
        name = KEYPOINT_NAMES_13[k]
        cnt = stats_dec[0.20][k]['total']
        pck20_r = (stats_raw[0.20][k]['correct'] / max(1, cnt)) * 100.0
        pck20_d = (stats_dec[0.20][k]['correct'] / max(1, cnt)) * 100.0
        pck10_d = (stats_dec[0.10][k]['correct'] / max(1, cnt)) * 100.0
        pck05_d = (stats_dec[0.05][k]['correct'] / max(1, cnt)) * 100.0
        mean_err_r = np.mean(kp_pixel_errors_raw[k]) if kp_pixel_errors_raw[k] else -1.0
        mean_err_d = np.mean(kp_pixel_errors_dec[k]) if kp_pixel_errors_dec[k] else -1.0
        per_kp[name] = {
            'count': cnt,
            'pck20_raw': pck20_r,
            'pck20_dec': pck20_d,
            'pck10_dec': pck10_d,
            'pck05_dec': pck05_d,
            'mean_px_err_raw': mean_err_r,
            'mean_px_err_dec': mean_err_d
        }
    
    return results, per_kp, instance_matches_info


def render_visualizations(model, samples, device, output_dir):
    model.eval()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    scale = 256.0 / 64.0
    vis_h, vis_w = 512, 512
    
    def draw_skeleton(img_rgb, kps_list, title, color_theme):
        canvas = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (vis_w, vis_h))
        sx = vis_w / 256.0
        sy = vis_h / 256.0
        
        for (u, v) in SKELETON_LIMBS_13:
            for kps in kps_list:
                p1 = kps[u]
                p2 = kps[v]
                v1 = p1[2] if len(p1) > 2 else 1.0
                v2 = p2[2] if len(p2) > 2 else 1.0
                if v1 > 0 and v2 > 0:
                    pt1 = (int(round(p1[0] * sx)), int(round(p1[1] * sy)))
                    pt2 = (int(round(p2[0] * sx)), int(round(p2[1] * sy)))
                    cv2.line(canvas, pt1, pt2, color_theme['limb'], 2, cv2.LINE_AA)
        
        for kps in kps_list:
            for k_i, p in enumerate(kps):
                v_flg = p[2] if len(p) > 2 else 1.0
                if v_flg > 0:
                    pt = (int(round(p[0] * sx)), int(round(p[1] * sy)))
                    if k_i in [3, 4, 6, 7]:
                        cv2.circle(canvas, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, pt, 4, color_theme['point'], -1, cv2.LINE_AA)
                        cv2.circle(canvas, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)
                        
        cv2.rectangle(canvas, (0, 0), (vis_w, 32), (30, 30, 30), -1)
        cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    with torch.no_grad():
        for s_idx, sample in enumerate(samples):
            img_tensor = sample['image'].unsqueeze(0).to(device)
            gt_instances = sample['instances']
            stem = sample['stem']
            
            # Original RGB image resized to 256
            img_np = (sample['image'].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
            
            # Panel 1: GT
            gt_kps = [gt['keypoints'] for gt in gt_instances]
            p1 = draw_skeleton(img_np, gt_kps, f"1. Ground Truth (Inst: {len(gt_instances)})", {'limb': (0, 255, 0), 'point': (0, 220, 0)})
            
            # Panel 2: Raw Peaks
            raw_peaks_per_kp = {}
            for k in range(13):
                pks = detect_peaks(heatmaps[k], threshold=0.08, nms_window=3, subpixel=True)
                if not pks:
                    py, px = np.unravel_index(np.argmax(heatmaps[k]), heatmaps[k].shape)
                    pks = [(float(px), float(py), float(heatmaps[k][py, px]))]
                raw_peaks_per_kp[k] = [(p[0] * scale, p[1] * scale, p[2]) for p in pks]
            
            raw_kps_list = []
            for g_i, gt in enumerate(gt_instances):
                raw_kps = []
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    pks = raw_peaks_per_kp[k]
                    if gv > 0:
                        best_pk = min(pks, key=lambda p: np.hypot(p[0] - gx, p[1] - gy))
                        raw_kps.append(best_pk)
                    else:
                        raw_kps.append(pks[0])
                raw_kps_list.append(raw_kps)
            p2 = draw_skeleton(img_np, raw_kps_list, "2. Raw Heatmap Peaks", {'limb': (255, 120, 0), 'point': (255, 180, 0)})
            
            # Panel 3: Decoded
            raw_preds = decode_spatial_instances_single(heatmaps, cls_probs, pafs=pafs)
            dec_kps = []
            for inst in raw_preds:
                kps = [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
                dec_kps.append(kps)
            p3 = draw_skeleton(img_np, dec_kps, f"3. Decoded (Inst: {len(dec_kps)})", {'limb': (0, 100, 255), 'point': (0, 150, 255)})
            
            # Combine side-by-side
            combined = np.hstack([p1, p2, p3])
            out_file = vis_dir / f"eval_sample_{s_idx}_{stem[:25]}.jpg"
            cv2.imwrite(str(out_file), combined)
    print(f"🖼️  Visualizations saved to: {vis_dir}")


def run_overfit_sanity_test():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("👟 FAST OVERFIT SANITY TEST — 13-KP M1v2 PIPELINE VALIDATION")
    print("=" * 80)
    print(f"  Device              : {device}")
    if device.type == 'cuda':
        print(f"  GPU Name            : {torch.cuda.get_device_name(0)}")
    
    dataset_root = PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp"
    if not dataset_root.exists():
        dataset_root = BASE_DIR / "dataset" / "merged_v3_v4_13kp"
    output_dir = BASE_DIR / "outputs" / "overfit_sanity_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Dataset Root        : {dataset_root}")
    print(f"  Output Directory    : {output_dir}")
    print(f"  Existing Checkpoints: UNTOUCHED")
    print("=" * 80)
    
    # 1. Load Fixed 8 Images
    samples = load_dataset(dataset_root, SELECTED_STEMS, img_size=256)
    print(f"\n📁 Loaded {len(samples)} representative images for overfit test:")
    total_inst = 0
    total_v2 = 0
    total_v1 = 0
    total_v0 = 0
    
    for i, s in enumerate(samples):
        n_inst = len(s['instances'])
        total_inst += n_inst
        classes = [inst['class_id'] for inst in s['instances']]
        v_counts = [(sum(1 for kp in inst['keypoints'] if kp[2]==2),
                     sum(1 for kp in inst['keypoints'] if kp[2]==1),
                     sum(1 for kp in inst['keypoints'] if kp[2]==0)) for inst in s['instances']]
        for vc in v_counts:
            total_v2 += vc[0]
            total_v1 += vc[1]
            total_v0 += vc[2]
        print(f"  [{i+1}/8] {s['stem'][:35]} | Feet: {n_inst} | Cls: {classes} | (v=2, v=1, v=0): {v_counts}")
    
    print(f"\n  Total Feet Instances : {total_inst}")
    print(f"  Total Keypoints      : Visible(v=2): {total_v2}, Occluded(v=1): {total_v1}, Unlabeled(v=0): {total_v0}")
    print(f"  Keypoints Supervised : {total_v2 + total_v1} labeled ({total_v0} masked with spatial ignore)")
    
    # 2. Build Batched Tensors on Device
    images_batch = torch.stack([s['image'] for s in samples]).to(device)
    instances_batch = [s['instances'] for s in samples]
    
    paf_config_path = BASE_DIR / "configs" / "paf_connections_13kp.yaml"
    
    # Pre-generate ground truth targets directly on GPU
    gt_heatmaps, hm_masks = generate_heatmaps_batch(
        instances_batch,
        heatmap_size=64,
        sigma=6.0,
        device=device
    )
    gt_pafs, paf_masks = generate_pafs_batch(
        instances_batch,
        heatmap_size=64,
        paf_width=8,
        device=device,
        paf_config=str(paf_config_path)
    )
    gt_class_maps, _ = generate_class_maps_batch(
        instances_batch,
        heatmap_size=64,
        image_size=256,
        device=device
    )
    
    print(f"\n🎯 Ground Truth Targets Generated:")
    print(f"  Heatmaps : {gt_heatmaps.shape}, Mask Active Pixels: {int(hm_masks.sum().item()):,}")
    print(f"  PAFs     : {gt_pafs.shape}, Mask Active Pixels: {int(paf_masks.sum().item()):,}")
    print(f"  ClassMaps: {gt_class_maps.shape}")
    
    # 3. Model & Losses
    model = ARShoeM1v2(
        pretrained=True,
        num_keypoints=13,
        num_limbs=16,
        num_classes=2
    ).to(device)
    
    heatmap_loss_fn = AdaptiveWingLoss(use_mask=True)
    paf_loss_fn = PAFLoss(use_mask=True)
    class_loss_fn = ClassLoss()
    
    loss_weights = {'heatmap': 4.0, 'paf': 2.0, 'class': 1.5}
    
    backbone_params = (
        list(model.s4.parameters()) + 
        list(model.s8.parameters()) + 
        list(model.s16.parameters()) + 
        list(model.s32.parameters())
    )
    head_params = (
        list(model.lat32.parameters()) + 
        list(model.lat16.parameters()) + 
        list(model.lat8.parameters()) + 
        list(model.lat4.parameters()) + 
        list(model.fpn_conv.parameters()) + 
        list(model.hm_head.parameters()) + 
        list(model.paf_head.parameters()) + 
        list(model.cls_global.parameters()) + 
        list(model.cls_spatial.parameters())
    )
    
    # Differential learning rate
    lr = 0.001
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr * 0.5},
        {'params': head_params, 'lr': lr}
    ], lr=lr, weight_decay=1e-4)
    
    # 4. Step 0 Initial Evaluation
    print("\n" + "=" * 80)
    print("📈 Step 0 (Untrained Initialization) Evaluation:")
    res_0, per_kp_0, matches_0 = evaluate_batch(model, samples, device)
    print(f"  Raw Peaks PCK@0.20: {res_0['PCK@0.20']['raw']:.2f}% | PCK@0.10: {res_0['PCK@0.10']['raw']:.2f}% | PCK@0.05: {res_0['PCK@0.05']['raw']:.2f}%")
    print(f"  Decoded   PCK@0.20: {res_0['PCK@0.20']['decoded']:.2f}% | PCK@0.10: {res_0['PCK@0.10']['decoded']:.2f}% | PCK@0.05: {res_0['PCK@0.05']['decoded']:.2f}%")
    
    # 5. Overfitting Training Loop (500 steps)
    eval_steps = {0, 50, 100, 200, 500}
    history = []
    
    print("\n" + "=" * 80)
    print("🚀 Training on Fixed 8 Images (Max 500 steps)...")
    print("=" * 80)
    print(f" {'Step':<6} | {'Total Loss':<11} | {'Heatmap Loss':<13} | {'PAF Loss':<10} | {'Class Loss':<11} | {'Time':<8}")
    print("-" * 75)
    
    t_start = time.time()
    
    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(images_batch)
        
        hm_loss = heatmap_loss_fn(outputs['heatmaps'], gt_heatmaps, masks=hm_masks)
        paf_loss = paf_loss_fn(outputs['pafs'], gt_pafs, masks=paf_masks)
        cls_loss = class_loss_fn(outputs['class_logits'], gt_class_maps)
        
        total_loss = (
            loss_weights['heatmap'] * hm_loss +
            loss_weights['paf'] * paf_loss +
            loss_weights['class'] * cls_loss
        )
        
        total_loss.backward()
        optimizer.step()
        
        if step in eval_steps or step % 25 == 0:
            elapsed = time.time() - t_start
            print(f" {step:<6} | {total_loss.item():<11.4f} | {hm_loss.item():<13.4f} | {paf_loss.item():<10.4f} | {cls_loss.item():<11.4f} | {elapsed:<6.1f}s")
            history.append({
                'step': step,
                'total_loss': total_loss.item(),
                'hm_loss': hm_loss.item(),
                'paf_loss': paf_loss.item(),
                'cls_loss': cls_loss.item()
            })
            
        if step in eval_steps and step > 0:
            res_s, _, _ = evaluate_batch(model, samples, device)
            print(f"   ↳ [Step {step} Eval] Raw Peaks: PCK@0.20={res_s['PCK@0.20']['raw']:.1f}%, PCK@0.10={res_s['PCK@0.10']['raw']:.1f}%, PCK@0.05={res_s['PCK@0.05']['raw']:.1f}%")
            print(f"   ↳ [Step {step} Eval] Decoded  : PCK@0.20={res_s['PCK@0.20']['decoded']:.1f}%, PCK@0.10={res_s['PCK@0.10']['decoded']:.1f}%, PCK@0.05={res_s['PCK@0.05']['decoded']:.1f}%")
            
            # Check early stop if clear overfit reached
            if res_s['PCK@0.20']['raw'] >= 99.0 and res_s['PCK@0.20']['decoded'] >= 95.0:
                print(f"\n🎉 Model has successfully overfitted the 8-sample subset at step {step}!")
                break
    
    total_train_time = time.time() - t_start
    print(f"\n⏱️ Training finished in {total_train_time:.2f}s ({step} steps)")
    
    # 6. Save Overfitted Checkpoint to Isolated Directory
    ckpt_path = output_dir / "overfit_checkpoint.pth"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': total_loss.item()
    }, ckpt_path)
    print(f"💾 Checkpoint saved to: {ckpt_path}")
    
    # 7. Final Comprehensive Evaluation
    print("\n" + "=" * 80)
    print("📊 Final Post-Training Evaluation on the 8 Fixed Images (Without Augmentation)")
    print("=" * 80)
    final_res, final_per_kp, final_matches = evaluate_batch(model, samples, device)
    
    print("\n📈 Overall PCK Metrics:")
    print(f"  PCK@0.20 -> Raw Peaks: {final_res['PCK@0.20']['raw']:>6.2f}% | Decoded: {final_res['PCK@0.20']['decoded']:>6.2f}%")
    print(f"  PCK@0.10 -> Raw Peaks: {final_res['PCK@0.10']['raw']:>6.2f}% | Decoded: {final_res['PCK@0.10']['decoded']:>6.2f}%")
    print(f"  PCK@0.05 -> Raw Peaks: {final_res['PCK@0.05']['raw']:>6.2f}% | Decoded: {final_res['PCK@0.05']['decoded']:>6.2f}%")
    
    print("\n🔬 Per-Keypoint Metrics & Pixel Distances (256x256 resolution):")
    hdr = f" {'#':<3} | {'Keypoint Name':<18} | {'Count':<5} | {'Raw PCK@20':<11} | {'Dec PCK@20':<11} | {'Dec PCK@10':<11} | {'Dec PCK@05':<11} | {'Raw Err(px)':<11} | {'Dec Err(px)':<11}"
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for k in range(13):
        name = KEYPOINT_NAMES_13[k]
        d = final_per_kp[name]
        print(f" {k:<3} | {name:<18} | {d['count']:<5} | {d['pck20_raw']:>6.1f}%     | {d['pck20_dec']:>6.1f}%     | {d['pck10_dec']:>6.1f}%     | {d['pck05_dec']:>6.1f}%     | {d['mean_px_err_raw']:>6.2f} px    | {d['mean_px_err_dec']:>6.2f} px")
    print("-" * len(hdr))
    
    print("\n👟 Instance Matching & Foot Localization:")
    for m in final_matches:
        match_str = ", ".join([f"{k}->{v}" for k, v in m['matches'].items()])
        print(f"  {m['stem'][:30]} | GT: {m['gt_count']} (cls: {m['gt_classes']}) | Pred: {m['pred_count']} (cls: {m['pred_classes']}) | Matches: {match_str}")
        
    # 8. Render Visualizations
    render_visualizations(model, samples, device, output_dir)
    
    # 9. Save JSON Summary
    summary = {
        'steps_run': step,
        'training_time_sec': total_train_time,
        'initial_eval': res_0,
        'final_eval': final_res,
        'per_keypoint': final_per_kp,
        'instance_matches': final_matches,
        'loss_history': history
    }
    with open(output_dir / "sanity_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Full report saved to: {output_dir / 'sanity_test_results.json'}")


if __name__ == '__main__':
    run_overfit_sanity_test()
