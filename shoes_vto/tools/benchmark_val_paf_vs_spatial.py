"""
Comprehensive Benchmark: PAF-Led Assembly vs Original Spatial Decoder
on Full Validation Split (253 images, 449 feet) using canonical best.pth
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from utils.keypoint_grouping import (
    decode_spatial_instances_single,
    decode_paf_instances_single,
    load_paf_config
)
from tools.evaluate_decoders_on_validation import load_all_validation_samples

VALID_DIR = PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp" / "valid"
CKPT_PATH = BASE_DIR / "outputs" / "m1v2_13kp_training" / "checkpoints" / "best.pth"
paf_config = load_paf_config("paf_connections_13kp.yaml")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running evaluation on device: {device}")

model = ARShoeM1v2(num_keypoints=13, num_limbs=16, num_classes=2).to(device)
sd = torch.load(CKPT_PATH, map_location=device)
if 'model_state_dict' in sd:
    sd = sd['model_state_dict']
sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()

samples = load_all_validation_samples(VALID_DIR, img_size=256)
print(f"Loaded {len(samples)} validation images.")

thresholds = [0.20, 0.10, 0.05]
scale = 256.0 / 64.0

trackers = {
    'spatial': {
        'total_feet': 0, 'matched_feet': 0, 'extra_instances': 0, 'missing_instances': 0,
        'stats': {th: {'correct': 0, 'total': 0} for th in thresholds},
        'errors': []
    },
    'paf_led': {
        'total_feet': 0, 'matched_feet': 0, 'extra_instances': 0, 'missing_instances': 0,
        'stats': {th: {'correct': 0, 'total': 0} for th in thresholds},
        'errors': []
    }
}

with torch.no_grad():
    for s_idx, sample in enumerate(samples):
        if (s_idx + 1) % 50 == 0 or s_idx == len(samples) - 1:
            print(f"  Processed {s_idx + 1}/{len(samples)} images...")

        img_tensor = sample['image'].unsqueeze(0).to(device)
        gt_instances = sample['instances']
        num_gt = len(gt_instances)

        outputs = model(img_tensor)
        heatmaps = outputs['heatmaps'][0].cpu().numpy()
        pafs = outputs['pafs'][0].cpu().numpy()
        cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()

        # 1. Original Spatial Decoder
        raw_sp = decode_spatial_instances_single(heatmaps, cls_probs, pafs=pafs, paf_config=paf_config)
        preds_sp = []
        for inst in raw_sp:
            preds_sp.append({
                'class_id': inst['class_id'],
                'keypoints': [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
            })

        # 2. PAF-Led Assembly Decoder
        raw_paf = decode_paf_instances_single(heatmaps, pafs, cls_probs, paf_config=paf_config)
        preds_paf = []
        for inst in raw_paf:
            preds_paf.append({
                'class_id': inst['class_id'],
                'keypoints': [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
            })

        for dec_name, preds in [('spatial', preds_sp), ('paf_led', preds_paf)]:
            t = trackers[dec_name]
            num_pred = len(preds)
            t['total_feet'] += num_gt
            if num_pred > num_gt:
                t['extra_instances'] += (num_pred - num_gt)
            elif num_pred < num_gt:
                t['missing_instances'] += (num_gt - num_pred)

            # Hungarian Matching
            cost_matrix = np.full((num_gt, num_pred), 1e6)
            for g_i, gt in enumerate(gt_instances):
                for p_i, pred in enumerate(preds):
                    dists = []
                    for k in range(13):
                        gx, gy, gv = gt['keypoints'][k]
                        if gv > 0 and pred['keypoints'][k][2] > 0:
                            px, py = pred['keypoints'][k][:2]
                            dists.append(np.hypot(px - gx, py - gy))
                    if dists:
                        cost_matrix[g_i, p_i] = np.mean(dists)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = {r: c for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < 120.0}
            t['matched_feet'] += len(matches)

            # PCK Scoring per GT instance
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
                        t['stats'][th]['total'] += 1

                    if pred is not None and pred['keypoints'][k][2] > 0:
                        px, py = pred['keypoints'][k][:2]
                        d = np.hypot(px - gx, py - gy)
                        t['errors'].append(d)
                        for th in thresholds:
                            if d < th * diag:
                                t['stats'][th]['correct'] += 1

print("\n" + "=" * 80)
print("CANONICAL VALIDATION BENCHMARK RESULTS (253 IMAGES, 449 FEET):")
print("=" * 80)

for dec_name, label in [('spatial', 'Original Spatial Decoder'), ('paf_led', 'OpenPose PAF-Led Assembly')]:
    t = trackers[dec_name]
    pck20 = (t['stats'][0.20]['correct'] / max(1, t['stats'][0.20]['total'])) * 100.0
    pck10 = (t['stats'][0.10]['correct'] / max(1, t['stats'][0.10]['total'])) * 100.0
    pck05 = (t['stats'][0.05]['correct'] / max(1, t['stats'][0.05]['total'])) * 100.0
    mean_err = np.mean(t['errors']) if t['errors'] else 0.0

    print(f"\n👉 {label}:")
    print(f"  PCK@0.20           : {pck20:.2f}%")
    print(f"  PCK@0.10           : {pck10:.2f}%")
    print(f"  PCK@0.05           : {pck05:.2f}%")
    print(f"  Mean Pixel Error   : {mean_err:.2f} px")
    print(f"  Matched Feet       : {t['matched_feet']}/{t['total_feet']} ({t['matched_feet']/t['total_feet']*100:.1f}%)")
    print(f"  Extra (Ghost) Feet : {t['extra_instances']}")
    print(f"  Missing Feet       : {t['missing_instances']}")

print("\n" + "=" * 80)

