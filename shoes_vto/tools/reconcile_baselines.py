"""
Baseline Reconciliation Script:
Investigates the discrepancy between:
- Stage 1 experiment baseline (70.89% / 70.67%)
- Full-validation evaluation baseline (75.22%)

Compares:
1. Coordinate space: 256x256 network space vs original image pixel space
2. Instance matching: Class-conditional greedy matching vs Hungarian class-independent assignment
3. Decoder settings: max_radius (20 vs 22), LEG_KPS ({12} vs set())
"""

import os
import sys
import json
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from utils.keypoint_grouping import (
    detect_peaks,
    load_paf_config,
    load_skeleton_priors,
    compute_paf_score,
    decode_spatial_instances_single
)
from tools.experiment_stage1_decoder import (
    decode_spatial_instances_configurable,
    parse_ground_truth
)

VALID_DIR = PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp" / "valid"
CKPT_PATH = BASE_DIR / "outputs" / "m1v2_13kp_training" / "checkpoints" / "best.pth"


def safe_path(p):
    p_str = str(Path(p).resolve())
    if sys.platform == 'win32' and len(p_str) > 240 and not p_str.startswith('\\\\?\\'):
        return '\\\\?\\' + p_str
    return p_str


def parse_yolo_labels_256(label_path, img_size=256):
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
                'bbox': [cx, cy, w, h],
                'bbox_diag': np.sqrt((w * img_size) ** 2 + (h * img_size) ** 2),
                'keypoints': kps
            })
    return instances


def evaluate_all_variants():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint on {device}: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    model = ARShoeM1v2(num_keypoints=13, num_classes=2, num_limbs=16, pretrained=False)
    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device)
    model.eval()

    img_dir = VALID_DIR / "images"
    lbl_dir = VALID_DIR / "labels"
    img_files = sorted(list(img_dir.glob("*.jpg")))
    print(f"Loaded {len(img_files)} validation images.")

    # We test 4 combinations:
    # 1. OrigStage1_Protocol: Original image space + Class-conditional greedy matching + max_radius=20
    # 2. Stage1_Hungarian: Original image space + Hungarian class-independent matching + max_radius=20
    # 3. Space256_GreedyClass: 256x256 space + Class-conditional greedy matching + max_radius=22
    # 4. Canonical_256_Hungarian: 256x256 space + Hungarian class-independent matching + max_radius=22 (evaluate_decoders_on_validation.py)

    results = {
        '1_Stage1_OrigSpace_ClassGreedy': {'correct': 0, 'total': 0},
        '2_Stage1_OrigSpace_Hungarian': {'correct': 0, 'total': 0},
        '3_Space256_ClassGreedy': {'correct': 0, 'total': 0},
        '4_Canonical_256_Hungarian': {'correct': 0, 'total': 0},
    }

    with torch.no_grad():
        for idx, img_p in enumerate(img_files):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            img_bgr = cv2.imread(safe_path(img_p))
            if img_bgr is None:
                continue
            orig_h, orig_w = img_bgr.shape[:2]

            gt_orig = parse_ground_truth(lbl_p, orig_w, orig_h)
            gt_256 = parse_yolo_labels_256(lbl_p, img_size=256)
            if not gt_orig or not gt_256:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()

            # Decoder 1: Configurable (Stage 1 settings: max_radius=20, LEG_KPS=set())
            preds_stage1_raw = decode_spatial_instances_configurable(
                heatmaps, cls_probs, pafs=pafs,
                peripheral_kps={0, 1, 2, 8, 9, 10},
                leg_kps=set(),
                interior_penalty=0.015,
                max_radius=20.0
            )

            # Decoder 2: Production keypoint_grouping (Canonical: max_radius=22, LEG_KPS=set())
            preds_prod_raw = decode_spatial_instances_single(heatmaps, cls_probs, pafs=pafs)

            # --- Eval Mode 1: Stage 1 Original (Orig space, greedy class-conditional) ---
            preds_orig_space = []
            for inst in preds_stage1_raw:
                p = copy.deepcopy(inst)
                p['keypoints'] = [[kp[0] * (orig_w / 64.0), kp[1] * (orig_h / 64.0), kp[2]] for kp in inst['keypoints']]
                preds_orig_space.append(p)

            # Greedy class-conditional matching
            matches_1 = {}
            for g_i, gt in enumerate(gt_orig):
                best_p_i = None
                best_dist = 1e9
                for p_i, pred in enumerate(preds_orig_space):
                    if p_i in matches_1.values():
                        continue
                    if pred['class_id'] == gt['class_id'] or len(preds_orig_space) == 1:
                        d_sum = sum(np.hypot(pred['keypoints'][k][0] - gt['keypoints'][k][0],
                                             pred['keypoints'][k][1] - gt['keypoints'][k][1])
                                    for k in range(13) if gt['keypoints'][k][2] > 0)
                        d_cnt = sum(1 for k in range(13) if gt['keypoints'][k][2] > 0)
                        avg_d = d_sum / max(1, d_cnt)
                        if avg_d < best_dist:
                            best_dist = avg_d
                            best_p_i = p_i
                if best_p_i is not None:
                    matches_1[g_i] = best_p_i
                elif len(preds_orig_space) > 0:
                    matches_1[g_i] = 0

            for g_i, gt in enumerate(gt_orig):
                diag = gt['bbox_diag']
                p_idx = matches_1.get(g_i, None)
                pred = preds_orig_space[p_idx] if p_idx is not None and p_idx < len(preds_orig_space) else None
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv == 0:
                        continue
                    results['1_Stage1_OrigSpace_ClassGreedy']['total'] += 1
                    if pred and pred['keypoints'][k][2] > 0.15:
                        d = np.hypot(pred['keypoints'][k][0] - gx, pred['keypoints'][k][1] - gy)
                        if d < 0.20 * diag:
                            results['1_Stage1_OrigSpace_ClassGreedy']['correct'] += 1

            # --- Eval Mode 2: Orig Space + Hungarian Class-Independent ---
            num_gt = len(gt_orig)
            num_p = len(preds_orig_space)
            cost_mat = np.full((num_gt, num_p), 1e6)
            for g_i, gt in enumerate(gt_orig):
                for p_i, pred in enumerate(preds_orig_space):
                    dists = [np.hypot(pred['keypoints'][k][0] - gt['keypoints'][k][0],
                                      pred['keypoints'][k][1] - gt['keypoints'][k][1])
                             for k in range(13) if gt['keypoints'][k][2] > 0 and pred['keypoints'][k][2] > 0]
                    if dists:
                        cost_mat[g_i, p_i] = np.mean(dists)
            r_idx, c_idx = linear_sum_assignment(cost_mat)
            matches_2 = {r: c for r, c in zip(r_idx, c_idx) if cost_mat[r, c] < 120.0 * (orig_w / 256.0)}

            for g_i, gt in enumerate(gt_orig):
                diag = gt['bbox_diag']
                p_idx = matches_2.get(g_i, None)
                pred = preds_orig_space[p_idx] if p_idx is not None else None
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv == 0:
                        continue
                    results['2_Stage1_OrigSpace_Hungarian']['total'] += 1
                    if pred and pred['keypoints'][k][2] > 0.15:
                        d = np.hypot(pred['keypoints'][k][0] - gx, pred['keypoints'][k][1] - gy)
                        if d < 0.20 * diag:
                            results['2_Stage1_OrigSpace_Hungarian']['correct'] += 1

            # --- Eval Mode 3 & 4: 256x256 Space (with scale 4.0 from 64x64) ---
            preds_256 = []
            for inst in preds_prod_raw:
                p = copy.deepcopy(inst)
                p['keypoints'] = [[kp[0] * 4.0, kp[1] * 4.0, kp[2]] for kp in inst['keypoints']]
                preds_256.append(p)

            # Mode 3: 256 space + Class-conditional greedy
            matches_3 = {}
            for g_i, gt in enumerate(gt_256):
                best_p_i = None
                best_dist = 1e9
                for p_i, pred in enumerate(preds_256):
                    if p_i in matches_3.values():
                        continue
                    if pred['class_id'] == gt['class_id'] or len(preds_256) == 1:
                        d_sum = sum(np.hypot(pred['keypoints'][k][0] - gt['keypoints'][k][0],
                                             pred['keypoints'][k][1] - gt['keypoints'][k][1])
                                    for k in range(13) if gt['keypoints'][k][2] > 0)
                        d_cnt = sum(1 for k in range(13) if gt['keypoints'][k][2] > 0)
                        avg_d = d_sum / max(1, d_cnt)
                        if avg_d < best_dist:
                            best_dist = avg_d
                            best_p_i = p_i
                if best_p_i is not None:
                    matches_3[g_i] = best_p_i
                elif len(preds_256) > 0:
                    matches_3[g_i] = 0

            for g_i, gt in enumerate(gt_256):
                diag = gt['bbox_diag']
                p_idx = matches_3.get(g_i, None)
                pred = preds_256[p_idx] if p_idx is not None and p_idx < len(preds_256) else None
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv == 0:
                        continue
                    results['3_Space256_ClassGreedy']['total'] += 1
                    if pred and pred['keypoints'][k][2] > 0.15:
                        d = np.hypot(pred['keypoints'][k][0] - gx, pred['keypoints'][k][1] - gy)
                        if d < 0.20 * diag:
                            results['3_Space256_ClassGreedy']['correct'] += 1

            # Mode 4: Canonical (256 space + Hungarian class-independent)
            cost_mat_4 = np.full((num_gt, len(preds_256)), 1e6)
            for g_i, gt in enumerate(gt_256):
                for p_i, pred in enumerate(preds_256):
                    dists = [np.hypot(pred['keypoints'][k][0] - gt['keypoints'][k][0],
                                      pred['keypoints'][k][1] - gt['keypoints'][k][1])
                             for k in range(13) if gt['keypoints'][k][2] > 0 and pred['keypoints'][k][2] > 0]
                    if dists:
                        cost_mat_4[g_i, p_i] = np.mean(dists)
            r_idx, c_idx = linear_sum_assignment(cost_mat_4)
            matches_4 = {r: c for r, c in zip(r_idx, c_idx) if cost_mat_4[r, c] < 120.0}

            for g_i, gt in enumerate(gt_256):
                diag = gt['bbox_diag']
                p_idx = matches_4.get(g_i, None)
                pred = preds_256[p_idx] if p_idx is not None else None
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv == 0:
                        continue
                    results['4_Canonical_256_Hungarian']['total'] += 1
                    if pred and pred['keypoints'][k][2] > 0:
                        d = np.hypot(pred['keypoints'][k][0] - gx, pred['keypoints'][k][1] - gy)
                        if d < 0.20 * diag:
                            results['4_Canonical_256_Hungarian']['correct'] += 1

    print("\n" + "=" * 80)
    print("🔬 RECONCILIATION RESULTS ACROSS PROTOCOLS")
    print("=" * 80)
    for name, data in results.items():
        pck = (data['correct'] / max(1, data['total'])) * 100.0
        print(f"  {name:<35}: {pck:6.2f}% ({data['correct']}/{data['total']})")
    print("=" * 80)

    # Save to json
    out_file = BASE_DIR / "outputs" / "baseline_reconciliation.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({k: {'pck20': (v['correct']/v['total'])*100, 'correct': v['correct'], 'total': v['total']}
                   for k, v in results.items()}, f, indent=2)
    print(f"Saved reconciliation to {out_file}")


if __name__ == '__main__':
    evaluate_all_variants()
