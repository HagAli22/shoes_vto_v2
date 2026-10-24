"""
Comprehensive Evaluation of Original vs. Fixed Decoders on Full Validation Split
Validates ARShoe M1v2 on dataset/merged_v3_v4_13kp/valid (253 images, 449 foot instances)
Using checkpoint: shoes_vto/outputs/m1v2_13kp_training/checkpoints/best.pth
"""

import os
import sys
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
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
from utils.keypoint_grouping import (
    detect_peaks,
    decode_spatial_instances_single,
    decode_paf_instances_single,
    load_paf_config
)
from tools.test_fixed_decoder import (
    decode_spatial_instances_fixed,
    KP_COLORS,
    KEYPOINT_NAMES_13,
    SKELETON_LIMBS_13
)

VALID_DIR = PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp" / "valid"
CKPT_PATH = BASE_DIR / "outputs" / "m1v2_13kp_training" / "checkpoints" / "best.pth"
OUT_DIR = BASE_DIR / "outputs" / "validation_decoder_comparison"
VIS_DIR = OUT_DIR / "vis"
ARTIFACT_DIR = Path(r"C:\Users\Matrix Store\.gemini\antigravity\brain\729e2265-efb4-4e0a-8890-a75ed53005c3")

OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


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
                'keypoints': kps          # 0-256 pixel coordinates
            })
    return instances


def load_all_validation_samples(valid_dir, img_size=256):
    img_dir = valid_dir / "images"
    lbl_dir = valid_dir / "labels"
    img_files = sorted(list(img_dir.glob("*.*")))
    
    samples = []
    for img_p in img_files:
        stem = img_p.stem
        lbl_p = lbl_dir / f"{stem}.txt"
        
        safe_img_p = safe_path(img_p)
        try:
            img_raw = Image.open(safe_img_p).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_p}: {e}")
            continue
            
        orig_w, orig_h = img_raw.size
        img_resized = img_raw.resize((img_size, img_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
        instances = parse_label_file(lbl_p, img_size=img_size)
        
        samples.append({
            'stem': stem,
            'image': img_tensor,
            'instances': instances,
            'orig_size': (orig_w, orig_h),
            'img_path': str(img_p)
        })
    return samples


def evaluate_decoders(model, samples, device):
    thresholds = [0.20, 0.10, 0.05]
    
    def init_metric_tracker():
        return {
            'stats': {th: {'correct': 0, 'total': 0} for th in thresholds},
            'per_kp': {k: {th: {'correct': 0, 'total': 0} for th in thresholds} for k in range(13)},
            'single_foot_stats': {th: {'correct': 0, 'total': 0} for th in thresholds},
            'two_foot_stats': {th: {'correct': 0, 'total': 0} for th in thresholds},
            'errs': [],
            'single_foot_errs': [],
            'two_foot_errs': [],
            'matched_feet': 0,
            'total_feet': 0,
            'extra_instances': 0,
            'missing_instances': 0,
            'single_foot_extra': 0,
            'single_foot_missing': 0,
            'two_foot_extra': 0,
            'two_foot_missing': 0,
            'per_image': []
        }

    tracker_orig = init_metric_tracker()
    tracker_fixed = init_metric_tracker()
    
    # Audit for >15 peaks filter
    filter_audit = {
        'total_channels': 0,
        'suppressed_channels': 0,
        'suppressed_gt_present': 0, # FALSE POSITIVES (bad!)
        'suppressed_gt_absent': 0,  # TRUE POSITIVES (good noise filtering!)
        'false_positive_details': [],
        'peak_counts_gt_present': [],
        'peak_counts_gt_absent': []
    }
    
    detailed_outputs = []

    print(f"Running inference and decoding over {len(samples)} validation images...")
    with torch.no_grad():
        for s_idx, s in enumerate(samples):
            if (s_idx + 1) % 50 == 0 or s_idx == len(samples) - 1:
                print(f"  Processed {s_idx + 1}/{len(samples)} images...")
                
            img_tensor = s['image'].unsqueeze(0).to(device)
            gt_insts = s['instances']
            num_gt = len(gt_insts)
            is_single = (num_gt == 1)
            
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            class_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            
            # --- Audit Filter Behavior ---
            for k in range(13):
                pks = detect_peaks(heatmaps[k], threshold=0.10, nms_window=3, subpixel=True)
                pks_020 = [p for p in pks if p[2] >= 0.20]
                num_pks_020 = len(pks_020)
                
                gt_has_k = any(gt['keypoints'][k][2] > 0 for gt in gt_insts)
                
                filter_audit['total_channels'] += 1
                if gt_has_k:
                    filter_audit['peak_counts_gt_present'].append(num_pks_020)
                else:
                    filter_audit['peak_counts_gt_absent'].append(num_pks_020)
                    
                if num_pks_020 > 15:
                    filter_audit['suppressed_channels'] += 1
                    if gt_has_k:
                        filter_audit['suppressed_gt_present'] += 1
                        filter_audit['false_positive_details'].append({
                            'stem': s['stem'],
                            'kp_idx': k,
                            'kp_name': KEYPOINT_NAMES_13[k],
                            'peak_count': num_pks_020
                        })
                    else:
                        filter_audit['suppressed_gt_absent'] += 1

            # --- Decode using Original Decoder ---
            preds_orig = decode_spatial_instances_single(heatmaps, class_probs, pafs=pafs)
            # Scale coordinates from 64x64 to 256x256
            scale = 256.0 / 64.0
            for p in preds_orig:
                p['keypoints'] = [[kp[0] * scale, kp[1] * scale, kp[2]] for kp in p['keypoints']]
                
            # --- Decode using OpenPose PAF-Led Assembly ---
            paf_cfg = load_paf_config("paf_connections_13kp.yaml")
            raw_paf = decode_paf_instances_single(heatmaps, pafs, class_probs, paf_config=paf_cfg)
            preds_paf = []
            for inst in raw_paf:
                preds_paf.append({
                    'class_id': inst['class_id'],
                    'keypoints': [[kp[0] * scale, kp[1] * scale, kp[2]] for kp in inst['keypoints']]
                })

            # --- Decode using Fixed Decoder ---
            preds_fixed = decode_spatial_instances_fixed(heatmaps, class_probs, pafs=pafs)
            # preds_fixed already returns keypoints in 256x256 space

            # Store for visualization selection
            detailed_outputs.append({
                'sample': s,
                'preds_orig': preds_orig,
                'preds_fixed': preds_fixed,
                'num_gt': num_gt
            })

            # Evaluate both decoders
            for preds, tracker, dec_name in [
                (preds_orig, tracker_orig, "original"),
                (preds_fixed, tracker_fixed, "fixed")
            ]:
                num_pred = len(preds)
                tracker['total_feet'] += num_gt
                
                if num_pred > num_gt:
                    tracker['extra_instances'] += (num_pred - num_gt)
                    if is_single:
                        tracker['single_foot_extra'] += (num_pred - num_gt)
                    else:
                        tracker['two_foot_extra'] += (num_pred - num_gt)
                elif num_pred < num_gt:
                    tracker['missing_instances'] += (num_gt - num_pred)
                    if is_single:
                        tracker['single_foot_missing'] += (num_gt - num_pred)
                    else:
                        tracker['two_foot_missing'] += (num_gt - num_pred)
                        
                # Hungarian assignment (class-independent)
                cost_matrix = np.full((num_gt, num_pred), 1e6)
                for g_i, gt in enumerate(gt_insts):
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
                tracker['matched_feet'] += len(matches)
                
                # PCK Scoring per GT instance
                for g_i, gt in enumerate(gt_insts):
                    w_px = gt['bbox'][2] * 256.0
                    h_px = gt['bbox'][3] * 256.0
                    diag = np.sqrt(w_px**2 + h_px**2)
                    p_idx = matches.get(g_i, None)
                    pred = preds[p_idx] if p_idx is not None else None
                    
                    for k in range(13):
                        gx, gy, gv = gt['keypoints'][k]
                        if gv == 0:
                            continue  # Strictly exclude unlabeled keypoints
                            
                        for th in thresholds:
                            tracker['stats'][th]['total'] += 1
                            tracker['per_kp'][k][th]['total'] += 1
                            if is_single:
                                tracker['single_foot_stats'][th]['total'] += 1
                            else:
                                tracker['two_foot_stats'][th]['total'] += 1
                                
                        if pred is not None and pred['keypoints'][k][2] > 0:
                            px, py = pred['keypoints'][k][:2]
                            d = np.hypot(px - gx, py - gy)
                            tracker['errs'].append(d)
                            if is_single:
                                tracker['single_foot_errs'].append(d)
                            else:
                                tracker['two_foot_errs'].append(d)
                                
                            for th in thresholds:
                                if d < th * diag:
                                    tracker['stats'][th]['correct'] += 1
                                    tracker['per_kp'][k][th]['correct'] += 1
                                    if is_single:
                                        tracker['single_foot_stats'][th]['correct'] += 1
                                    else:
                                        tracker['two_foot_stats'][th]['correct'] += 1

                tracker['per_image'].append({
                    'stem': s['stem'],
                    'num_gt': num_gt,
                    'num_pred': num_pred,
                    'matched': len(matches)
                })

    def format_summary(tracker, name):
        pck = {f"pck_{int(th*100):02d}": (tracker['stats'][th]['correct'] / max(1, tracker['stats'][th]['total'])) * 100.0 for th in thresholds}
        pck_single = {f"pck_{int(th*100):02d}": (tracker['single_foot_stats'][th]['correct'] / max(1, tracker['single_foot_stats'][th]['total'])) * 100.0 for th in thresholds}
        pck_two = {f"pck_{int(th*100):02d}": (tracker['two_foot_stats'][th]['correct'] / max(1, tracker['two_foot_stats'][th]['total'])) * 100.0 for th in thresholds}
        
        per_kp_dict = {}
        for k in range(13):
            k_name = KEYPOINT_NAMES_13[k]
            tot = tracker['per_kp'][k][0.20]['total']
            c20 = tracker['per_kp'][k][0.20]['correct']
            c10 = tracker['per_kp'][k][0.10]['correct']
            c05 = tracker['per_kp'][k][0.05]['correct']
            per_kp_dict[k_name] = {
                'pck20': (c20 / max(1, tot)) * 100.0,
                'pck10': (c10 / max(1, tot)) * 100.0,
                'pck05': (c05 / max(1, tot)) * 100.0,
                'count': tot
            }

        return {
            'decoder_name': name,
            'total_evaluated_keypoints': tracker['stats'][0.20]['total'],
            'total_feet': tracker['total_feet'],
            'matched_feet': tracker['matched_feet'],
            'missing_instances': tracker['missing_instances'],
            'extra_instances': tracker['extra_instances'],
            'mean_error_px': float(np.mean(tracker['errs'])) if tracker['errs'] else -1.0,
            'overall_pck': pck,
            'single_foot': {
                'total_feet': 57,
                'mean_error_px': float(np.mean(tracker['single_foot_errs'])) if tracker['single_foot_errs'] else -1.0,
                'extra_instances': tracker['single_foot_extra'],
                'missing_instances': tracker['single_foot_missing'],
                'pck': pck_single
            },
            'two_foot': {
                'total_feet': 392,
                'mean_error_px': float(np.mean(tracker['two_foot_errs'])) if tracker['two_foot_errs'] else -1.0,
                'extra_instances': tracker['two_foot_extra'],
                'missing_instances': tracker['two_foot_missing'],
                'pck': pck_two
            },
            'per_keypoint': per_kp_dict
        }

    summary_orig = format_summary(tracker_orig, "Production Decoder (Before Fix)")
    summary_fixed = format_summary(tracker_fixed, "Fixed Decoder (After Fix)")
    
    return summary_orig, summary_fixed, filter_audit, detailed_outputs


def generate_visualizations(detailed_outputs, num_vis=12):
    print("\nSelecting 12 diverse validation images for 3-panel visualization...")
    
    # Categories:
    # 1. Single-foot where original created a ghost (pred=2, gt=1)
    single_ghost = [d for d in detailed_outputs if d['num_gt'] == 1 and len(d['preds_orig']) > 1]
    # 2. Single-foot where both got 1 foot
    single_clean = [d for d in detailed_outputs if d['num_gt'] == 1 and len(d['preds_orig']) == 1]
    # 3. Two-foot images where fixed improved matching / removed spurious predictions
    two_foot_diff = [d for d in detailed_outputs if d['num_gt'] == 2 and len(d['preds_orig']) != len(d['preds_fixed'])]
    # 4. Two-foot standard / close-proximity
    two_foot_close = [d for d in detailed_outputs if d['num_gt'] == 2 and len(d['preds_orig']) == 2 and len(d['preds_fixed']) == 2]

    selected = []
    # Pick 4 single foot (preferring ghost cases to demonstrate the fix)
    selected.extend(single_ghost[:3])
    if len(selected) < 4 and single_clean:
        selected.extend(single_clean[:4 - len(selected)])
        
    # Pick 4 two-foot cases
    if two_foot_diff:
        selected.extend(two_foot_diff[:3])
    remaining_two = 8 - len(selected)
    if remaining_two > 0 and two_foot_close:
        selected.extend(two_foot_close[:remaining_two])
        
    # Pick 4 challenging cases (remaining diverse ones)
    stems_chosen = set(d['sample']['stem'] for d in selected)
    for d in detailed_outputs:
        if d['sample']['stem'] not in stems_chosen:
            selected.append(d)
            stems_chosen.add(d['sample']['stem'])
            if len(selected) >= num_vis:
                break
                
    selected = selected[:num_vis]
    print(f"Selected {len(selected)} samples for visualization.")

    vis_w, vis_h = 320, 320
    saved_paths = []

    for idx, item in enumerate(selected):
        s = item['sample']
        stem = s['stem']
        gt_insts = s['instances']
        preds_orig = item['preds_orig']
        preds_fixed = item['preds_fixed']
        
        img_np = (s['image'].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        sx = vis_w / 256.0
        sy = vis_h / 256.0
        
        # Panel 1: Ground Truth
        p1 = cv2.resize(img_bgr, (vis_w, vis_h))
        for g_i, gt in enumerate(gt_insts):
            for u, v in SKELETON_LIMBS_13:
                p_u = gt['keypoints'][u]
                p_v = gt['keypoints'][v]
                if p_u[2] > 0 and p_v[2] > 0:
                    cv2.line(p1, (int(p_u[0]*sx), int(p_u[1]*sy)), (int(p_v[0]*sx), int(p_v[1]*sy)), (0, 255, 0), 2, cv2.LINE_AA)
            for k in range(13):
                gx, gy, gv = gt['keypoints'][k]
                if gv > 0:
                    pt = (int(gx * sx), int(gy * sy))
                    cv2.circle(p1, pt, 5, KP_COLORS[k], -1, cv2.LINE_AA)
                    cv2.putText(p1, f"F{g_i}:k{k}", (pt[0]+3, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(p1, (0, 0), (vis_w, 26), (30, 30, 30), -1)
        cv2.putText(p1, f"1. GT ({len(gt_insts)} feet)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Panel 2: Original Production Decoder
        p2 = cv2.resize(img_bgr, (vis_w, vis_h))
        for p_i, pred in enumerate(preds_orig):
            kps = pred['keypoints']
            for u, v in SKELETON_LIMBS_13:
                if kps[u][2] > 0.10 and kps[v][2] > 0.10:
                    cv2.line(p2, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 100, 255), 2, cv2.LINE_AA)
            for k in range(13):
                pk = kps[k]
                if pk[2] > 0.10:
                    pt = (int(pk[0]*sx), int(pk[1]*sy))
                    cv2.circle(p2, pt, 5, KP_COLORS[k], -1, cv2.LINE_AA)
                    cv2.putText(p2, f"P{p_i}:k{k}", (pt[0]+3, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(p2, (0, 0), (vis_w, 26), (30, 30, 30), -1)
        cv2.putText(p2, f"2. Original ({len(preds_orig)} insts)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Panel 3: Fixed Decoder
        p3 = cv2.resize(img_bgr, (vis_w, vis_h))
        for p_i, pred in enumerate(preds_fixed):
            kps = pred['keypoints']
            for u, v in SKELETON_LIMBS_13:
                if kps[u][2] > 0.10 and kps[v][2] > 0.10:
                    cv2.line(p3, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (255, 200, 0), 2, cv2.LINE_AA)
            for k in range(13):
                pk = kps[k]
                if pk[2] > 0.10:
                    pt = (int(pk[0]*sx), int(pk[1]*sy))
                    cv2.circle(p3, pt, 5, KP_COLORS[k], -1, cv2.LINE_AA)
                    cv2.putText(p3, f"P{p_i}:k{k}", (pt[0]+3, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(p3, (0, 0), (vis_w, 26), (30, 30, 30), -1)
        cv2.putText(p3, f"3. Fixed ({len(preds_fixed)} insts)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        combined = np.hstack([p1, p2, p3])
        fname = f"val_comp_{idx:02d}_{stem[:22]}.jpg"
        out_vis_path = VIS_DIR / fname
        cv2.imwrite(str(out_vis_path), combined)
        
        # Copy to artifact dir for report if artifact dir exists
        if ARTIFACT_DIR.exists():
            art_vis_path = ARTIFACT_DIR / fname
            shutil.copy2(out_vis_path, art_vis_path)
        saved_paths.append(fname)
        print(f"  Saved [{idx+1}/{num_vis}]: {fname}")

    return saved_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate 13-KP decoders on validation split")
    parser.add_argument('--checkpoint', type=str, default=str(CKPT_PATH), help="Model checkpoint path")
    parser.add_argument('--valid_dir', type=str, default=str(VALID_DIR), help="Path to valid directory")
    parser.add_argument('--out_dir', type=str, default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    valid_dir = Path(args.valid_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"🚀 RUNNING FULL VALIDATION DECODER COMPARISON ON {device}")
    print("=" * 80)

    # 1. Load model
    print("Loading model checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    
    model = ARShoeM1v2(num_keypoints=13, num_classes=2, num_limbs=16, pretrained=False)
    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device)
    model.eval()
    print("Model loaded with strict=True.")

    # 2. Load dataset
    samples = load_all_validation_samples(valid_dir, img_size=256)
    print(f"Loaded {len(samples)} validation images.")

    # 3. Evaluate both decoders
    summary_orig, summary_fixed, filter_audit, detailed_outputs = evaluate_decoders(model, samples, device)

    # 4. Generate visualizations
    vis_files = generate_visualizations(detailed_outputs, num_vis=12)

    # 5. Save report
    final_report = {
        'checkpoint': str(ckpt_path),
        'dataset': str(valid_dir),
        'total_images': len(samples),
        'summary_original': summary_orig,
        'summary_fixed': summary_fixed,
        'filter_audit': {
            'total_channels_evaluated': filter_audit['total_channels'],
            'total_times_suppressed': filter_audit['suppressed_channels'],
            'suppressed_gt_present_count': filter_audit['suppressed_gt_present'],
            'suppressed_gt_absent_count': filter_audit['suppressed_gt_absent'],
            'false_positive_details': filter_audit['false_positive_details'],
            'gt_present_peaks_summary': {
                'mean': float(np.mean(filter_audit['peak_counts_gt_present'])),
                'median': float(np.median(filter_audit['peak_counts_gt_present'])),
                'max': int(np.max(filter_audit['peak_counts_gt_present'])),
                'min': int(np.min(filter_audit['peak_counts_gt_present']))
            },
            'gt_absent_peaks_summary': {
                'mean': float(np.mean(filter_audit['peak_counts_gt_absent'])),
                'median': float(np.median(filter_audit['peak_counts_gt_absent'])),
                'max': int(np.max(filter_audit['peak_counts_gt_absent'])),
                'min': int(np.min(filter_audit['peak_counts_gt_absent']))
            }
        },
        'visualization_files': vis_files
    }

    report_path = out_dir / "validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved comprehensive report to: {report_path}")

    # Copy report to artifact dir if exists
    if ARTIFACT_DIR.exists():
        shutil.copy2(report_path, ARTIFACT_DIR / "validation_report.json")

    # Print summary tables to stdout
    print("\n" + "=" * 80)
    print("📊 FULL VALIDATION RESULTS SUMMARY (253 IMAGES, 449 FEET)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Original Decoder':<20} | {'Fixed Decoder':<20} | {'Delta':<15}")
    print("-" * 95)
    
    pck20_o = summary_orig['overall_pck']['pck_20']
    pck20_f = summary_fixed['overall_pck']['pck_20']
    pck10_o = summary_orig['overall_pck']['pck_10']
    pck10_f = summary_fixed['overall_pck']['pck_10']
    pck05_o = summary_orig['overall_pck']['pck_05']
    pck05_f = summary_fixed['overall_pck']['pck_05']
    err_o = summary_orig['mean_error_px']
    err_f = summary_fixed['mean_error_px']
    
    print(f"{'PCK@0.20':<35} | {pck20_o:6.2f}%              | {pck20_f:6.2f}%              | {pck20_f - pck20_o:+6.2f}%")
    print(f"{'PCK@0.10':<35} | {pck10_o:6.2f}%              | {pck10_f:6.2f}%              | {pck10_f - pck10_o:+6.2f}%")
    print(f"{'PCK@0.05':<35} | {pck05_o:6.2f}%              | {pck05_f:6.2f}%              | {pck05_f - pck05_o:+6.2f}%")
    print(f"{'Mean Pixel Error (px)':<35} | {err_o:6.2f} px            | {err_f:6.2f} px            | {err_f - err_o:+6.2f} px")
    print(f"{'Matched Feet (out of 449)':<35} | {summary_orig['matched_feet']:<20} | {summary_fixed['matched_feet']:<20} | {summary_fixed['matched_feet'] - summary_orig['matched_feet']:+d}")
    print(f"{'Missed Feet':<35} | {summary_orig['missing_instances']:<20} | {summary_fixed['missing_instances']:<20} | {summary_fixed['missing_instances'] - summary_orig['missing_instances']:+d}")
    print(f"{'Extra (Ghost) Feet':<35} | {summary_orig['extra_instances']:<20} | {summary_fixed['extra_instances']:<20} | {summary_fixed['extra_instances'] - summary_orig['extra_instances']:+d}")
    
    print("-" * 95)
    print("Subgroup: Single-Foot Images (57 images, 57 feet)")
    sf_o = summary_orig['single_foot']
    sf_f = summary_fixed['single_foot']
    print(f"  PCK@0.20: Original = {sf_o['pck']['pck_20']:.2f}% vs Fixed = {sf_f['pck']['pck_20']:.2f}% ({sf_f['pck']['pck_20'] - sf_o['pck']['pck_20']:+.2f}%)")
    print(f"  Ghost instances: Original = {sf_o['extra_instances']} vs Fixed = {sf_f['extra_instances']} ({sf_f['extra_instances'] - sf_o['extra_instances']:+d})")
    
    print("Subgroup: Two-Foot Images (196 images, 392 feet)")
    tf_o = summary_orig['two_foot']
    tf_f = summary_fixed['two_foot']
    print(f"  PCK@0.20: Original = {tf_o['pck']['pck_20']:.2f}% vs Fixed = {tf_f['pck']['pck_20']:.2f}% ({tf_f['pck']['pck_20'] - tf_o['pck']['pck_20']:+.2f}%)")
    print(f"  Ghost instances: Original = {tf_o['extra_instances']} vs Fixed = {tf_f['extra_instances']} ({tf_f['extra_instances'] - tf_o['extra_instances']:+d})")
    print(f"  Missed instances: Original = {tf_o['missing_instances']} vs Fixed = {tf_f['missing_instances']} ({tf_f['missing_instances'] - tf_o['missing_instances']:+d})")

    print("-" * 95)
    print("Channel Filter Audit (>15 peaks at 0.20):")
    print(f"  Total channels evaluated: {filter_audit['total_channels']}")
    print(f"  Total times triggered: {filter_audit['suppressed_channels']}")
    print(f"  Triggered on valid GT channels (False Positives): {filter_audit['suppressed_gt_present']}")
    print(f"  Triggered on absent channels (True Noise Rejection): {filter_audit['suppressed_gt_absent']}")
    if filter_audit['false_positive_details']:
        print("  False positive details:", filter_audit['false_positive_details'][:5])
    print("=" * 80)


if __name__ == '__main__':
    main()

