"""
Corrected Fast Overfit Sanity Test for ARShoe M1v2 (13 Keypoints)
Validates the corrected supervision (PAF accumulation + Heatmap background masking)
and compares PAF-Led Assembly vs Spatial K-Means on the fixed 8-image subset.
Uses strictly PREDICTION-ONLY Hungarian matching (no GT assistance).
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
from scipy.optimize import linear_sum_assignment
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
from models.heads.paf_head import generate_pafs_batch, load_paf_config
from models.heads.class_head import generate_class_maps_batch
from losses.heatmap_loss import AdaptiveWingLoss
from losses.paf_loss import PAFLoss
from losses.class_loss import ClassLoss
from utils.keypoint_grouping import (
    detect_peaks,
    decode_spatial_instances_single,
    decode_paf_instances_single
)

KEYPOINT_NAMES_13 = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "instep_top", "arch_medial", "midfoot_lateral", "malleolus_medial",
    "malleolus_lateral", "toe_tip", "throat", "achilles"
]

SKELETON_LIMBS_13 = [
    (10, 0), (0, 2), (2, 1), (3, 4), (3, 5), (4, 5), (2, 6), (2, 7),
    (6, 5), (7, 5), (5, 11), (11, 8), (11, 9), (8, 12), (9, 12), (12, 1)
]

SELECTED_STEMS = [
    '-4_jpg.rf.98277f4406f91f57d139a8bf7589cb7c',
    '-6_jpg.rf.0a5d49b0343b6e6a0d6e086b5e1a3707',
    '-_jpg.rf.b7e532c737ff3a14ebb16c7a208e941c',
    '-attachedteen-on-discord-tiktok_jpg.rf.91be9c55bf305fc60dc8fca200174b57',
    '-3_jpg.rf.5062bfbecadec3871c37cde866374226',
    '-peace_jpg.rf.f84bd928736ce2a1c7341f5e4506fc08',
    '1095571046880362857_jpg.rf.23ea8296be773a47af8675ab984aae08',
    '1114148395287951061_jpg.rf.56ba51c3c8704158190c4fb0c2db0201'
]


def load_dataset(dataset_root, stems, img_size=256):
    img_dir = dataset_root / "train" / "images"
    lbl_dir = dataset_root / "train" / "labels"
    samples = []
    for s in stems:
        img_path = img_dir / f"{s}.jpg"
        lbl_path = lbl_dir / f"{s}.txt"
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_res = img.resize((img_size, img_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_res)).float().permute(2, 0, 1) / 255.0
        
        instances = []
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 44:
                    continue
                cls_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                kps = []
                for i in range(13):
                    kx = float(parts[5 + i * 3]) * img_size
                    ky = float(parts[5 + i * 3 + 1]) * img_size
                    kv = int(float(parts[5 + i * 3 + 2]))
                    kps.append([kx, ky, kv])
                instances.append({
                    'class_id': cls_id,
                    'class_name': 'left_foot' if cls_id == 0 else 'right_foot',
                    'bbox': bbox,
                    'keypoints': kps
                })
        samples.append({
            'stem': s,
            'image': img_tensor,
            'instances': instances,
            'orig_size': (orig_w, orig_h)
        })
    return samples


def match_instances_hungarian(pred_instances, gt_instances):
    if len(pred_instances) == 0 or len(gt_instances) == 0:
        return {}
    num_gt = len(gt_instances)
    num_pred = len(pred_instances)
    cost_matrix = np.full((num_gt, num_pred), 1e6, dtype=np.float32)
    for g_i, gt in enumerate(gt_instances):
        for p_i, pred in enumerate(pred_instances):
            dists = []
            for k in range(13):
                gx, gy, gv = gt['keypoints'][k]
                if gv > 0 and pred['keypoints'][k][2] > 0:
                    px, py = pred['keypoints'][k][:2]
                    dists.append(np.hypot(px - gx, py - gy))
            if dists:
                cost_matrix[g_i, p_i] = np.mean(dists)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 150.0:
            matches[r] = c
    return matches


def evaluate_batch(model, samples, device, paf_config):
    model.eval()
    thresholds = [0.20, 0.10, 0.05]
    scale = 256.0 / 64.0

    eval_results = {
        'spatial': {th: {'correct': 0, 'total': 0} for th in thresholds},
        'paf_led': {th: {'correct': 0, 'total': 0} for th in thresholds}
    }
    pixel_errors = {
        'spatial': [],
        'paf_led': []
    }

    with torch.no_grad():
        for sample in samples:
            img_tensor = sample['image'].unsqueeze(0).to(device)
            gt_instances = sample['instances']

            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()

            # Decoder 1: Spatial K-Means
            raw_spatial = decode_spatial_instances_single(heatmaps, cls_probs, pafs=pafs, paf_config=paf_config)
            preds_spatial = []
            for inst in raw_spatial:
                preds_spatial.append({
                    'class_id': inst['class_id'],
                    'keypoints': [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
                })

            # Decoder 2: PAF-Led Assembly (OpenPose-inspired)
            raw_paf = decode_paf_instances_single(heatmaps, pafs, cls_probs, paf_config=paf_config)
            preds_paf = []
            for inst in raw_paf:
                preds_paf.append({
                    'class_id': inst['class_id'],
                    'keypoints': [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
                })

            for dec_name, preds in [('spatial', preds_spatial), ('paf_led', preds_paf)]:
                matches = match_instances_hungarian(preds, gt_instances)
                for g_i, gt in enumerate(gt_instances):
                    w_px = gt['bbox'][2] * 256.0
                    h_px = gt['bbox'][3] * 256.0
                    diag = np.sqrt(w_px**2 + h_px**2)
                    p_idx = matches.get(g_i, None)
                    pred = preds[p_idx] if p_idx is not None else None

                    for k in range(13):
                        gx, gy, gv = gt['keypoints'][k]
                        if gv == 0:
                            continue

                        for th in thresholds:
                            eval_results[dec_name][th]['total'] += 1

                        if pred is not None and pred['keypoints'][k][2] > 0:
                            px, py = pred['keypoints'][k][:2]
                            d = np.hypot(px - gx, py - gy)
                            pixel_errors[dec_name].append(d)
                            for th in thresholds:
                                if d < th * diag:
                                    eval_results[dec_name][th]['correct'] += 1

    summary = {}
    for dec_name in ['spatial', 'paf_led']:
        summary[dec_name] = {}
        for th in thresholds:
            corr = eval_results[dec_name][th]['correct']
            tot = eval_results[dec_name][th]['total']
            summary[dec_name][f'PCK@{th:.2f}'] = (corr / max(1, tot)) * 100.0
        summary[dec_name]['mean_error_px'] = float(np.mean(pixel_errors[dec_name])) if pixel_errors[dec_name] else -1.0
    return summary


def run_test():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("👟 FAST OVERFIT SANITY TEST WITH CORRECTED SUPERVISION & DECODER")
    print("=" * 80)
    print(f"  Device              : {device}")

    dataset_root = PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp"
    output_dir = BASE_DIR / "outputs" / "overfit_sanity_test_corrected"
    output_dir.mkdir(parents=True, exist_ok=True)
    paf_config_path = BASE_DIR / "configs" / "paf_connections_13kp.yaml"
    paf_config = load_paf_config(paf_config_path)

    samples = load_dataset(dataset_root, SELECTED_STEMS, img_size=256)
    print(f"  Loaded {len(samples)} images for overfit test.")

    # Generate GT targets with corrected supervision
    batch_instances = [s['instances'] for s in samples]
    gt_heatmaps, hm_masks = generate_heatmaps_batch(batch_instances, heatmap_size=64, sigma=4.0, device=device, num_keypoints=13)
    gt_pafs, paf_masks = generate_pafs_batch(batch_instances, heatmap_size=64, paf_width=6, device=device, paf_config=paf_config, num_limbs=16)
    gt_class_maps, _ = generate_class_maps_batch(batch_instances, heatmap_size=64, image_size=256, device=device)

    print(f"  GT Heatmaps Active Mask Pixels: {int(hm_masks.sum().item()):,}")
    print(f"  GT PAFs Active Mask Pixels    : {int(paf_masks.sum().item()):,}")

    model = ARShoeM1v2(pretrained=True, num_keypoints=13, num_limbs=16, num_classes=2).to(device)
    hm_loss_fn = AdaptiveWingLoss(use_mask=True)
    paf_loss_fn = PAFLoss(use_mask=True)
    class_loss_fn = ClassLoss()

    backbone_params = (
        list(model.s4.parameters()) + list(model.s8.parameters()) +
        list(model.s16.parameters()) + list(model.s32.parameters())
    )
    head_params = (
        list(model.lat32.parameters()) + list(model.lat16.parameters()) +
        list(model.lat8.parameters()) + list(model.lat4.parameters()) +
        list(model.fpn_conv.parameters()) + list(model.hm_head.parameters()) +
        list(model.paf_head.parameters()) + list(model.cls_global.parameters()) +
        list(model.cls_spatial.parameters())
    )

    lr = 0.001
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr * 0.5},
        {'params': head_params, 'lr': lr}
    ], lr=lr, weight_decay=1e-4)

    images_batch = torch.stack([s['image'] for s in samples]).to(device)

    # Initial Step 0 Eval
    print("\n[Step 0 Evaluation]")
    res_0 = evaluate_batch(model, samples, device, paf_config)
    print(f"  Spatial Decoder : PCK@0.20: {res_0['spatial']['PCK@0.20']:.2f}% | Mean Err: {res_0['spatial']['mean_error_px']:.1f} px")
    print(f"  PAF-Led Assembly: PCK@0.20: {res_0['paf_led']['PCK@0.20']:.2f}% | Mean Err: {res_0['paf_led']['mean_error_px']:.1f} px")

    loss_history = []
    eval_checkpoints = [50, 100, 200, 500]

    print("\n🚀 Training on fixed 8 images for up to 500 steps...")
    t0 = time.time()

    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()

        outputs = model(images_batch)
        hm_loss = hm_loss_fn(outputs['heatmaps'], gt_heatmaps, masks=hm_masks)
        paf_loss = paf_loss_fn(outputs['pafs'], gt_pafs, masks=paf_masks)
        class_loss = class_loss_fn(outputs['class_logits'], gt_class_maps)

        total_loss = 4.0 * hm_loss + 2.0 * paf_loss + 1.5 * class_loss
        total_loss.backward()
        optimizer.step()

        loss_entry = {
            'step': step,
            'total': total_loss.item(),
            'hm': hm_loss.item(),
            'paf': paf_loss.item(),
            'class': class_loss.item()
        }
        loss_history.append(loss_entry)

        if step in eval_checkpoints:
            res = evaluate_batch(model, samples, device, paf_config)
            print(f"  Step {step:3d} | Loss: {total_loss.item():.4f} (HM: {hm_loss.item():.4f}, PAF: {paf_loss.item():.4f}, Cls: {class_loss.item():.4f})")
            print(f"     Spatial : PCK@0.20 = {res['spatial']['PCK@0.20']:.2f}%, PCK@0.10 = {res['spatial']['PCK@0.10']:.2f}%, PCK@0.05 = {res['spatial']['PCK@0.05']:.2f}%, Mean Err = {res['spatial']['mean_error_px']:.1f} px")
            print(f"     PAF-Led : PCK@0.20 = {res['paf_led']['PCK@0.20']:.2f}%, PCK@0.10 = {res['paf_led']['PCK@0.10']:.2f}%, PCK@0.05 = {res['paf_led']['PCK@0.05']:.2f}%, Mean Err = {res['paf_led']['mean_error_px']:.1f} px")

    t1 = time.time()
    print(f"\nTraining completed in {t1 - t0:.1f}s.")

    # Save checkpoint
    ckpt_path = output_dir / "overfit_corrected_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")

    # Final summary report
    final_res = evaluate_batch(model, samples, device, paf_config)
    report = {
        'initial_step_0': res_0,
        'final_step_500': final_res,
        'loss_history': loss_history[::10]
    }
    with open(output_dir / "overfit_summary_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL 8-IMAGE OVERFIT COMPARISON (Step 500):")
    print(f"  Spatial Decoder : PCK@0.20 = {final_res['spatial']['PCK@0.20']:.2f}%, PCK@0.10 = {final_res['spatial']['PCK@0.10']:.2f}%, PCK@0.05 = {final_res['spatial']['PCK@0.05']:.2f}%, Mean Err = {final_res['spatial']['mean_error_px']:.1f} px")
    print(f"  PAF-Led Assembly: PCK@0.20 = {final_res['paf_led']['PCK@0.20']:.2f}%, PCK@0.10 = {final_res['paf_led']['PCK@0.10']:.2f}%, PCK@0.05 = {final_res['paf_led']['PCK@0.05']:.2f}%, Mean Err = {final_res['paf_led']['mean_error_px']:.1f} px")
    print("=" * 80)


if __name__ == '__main__':
    run_test()

