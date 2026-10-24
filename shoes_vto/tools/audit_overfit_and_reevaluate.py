"""
Audit of Overfit Evaluation & Multi-Instance Pipeline
Performs rigorous re-evaluation comparing:
A. Prediction-Only Raw Peaks (NO GT coordinates used during candidate extraction)
B. Existing Production Decoder
C. Diagnostic Oracle / GT-Assisted (Explicitly labeled as diagnostic upper bound)
Generates audit visualizations showing GT, Raw Candidate Peaks, and Decoded Instances with explicit IDs.
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
from utils.keypoint_grouping import detect_peaks, decode_spatial_instances_single
from tools.run_fast_overfit_sanity_test import (
    load_dataset,
    SELECTED_STEMS,
    KEYPOINT_NAMES_13,
    SKELETON_LIMBS_13
)

# Colors for keypoints (13 distinct BGR colors)
KP_COLORS = [
    (0, 0, 255),     # 0: toe_ground (Red)
    (0, 128, 255),   # 1: heel_back (Orange)
    (0, 255, 255),   # 2: heel_ground (Yellow)
    (0, 255, 0),     # 3: ball_medial (Green)
    (255, 255, 0),   # 4: ball_lateral (Cyan)
    (255, 0, 0),     # 5: instep_top (Blue)
    (255, 0, 128),   # 6: arch_medial (Purple)
    (128, 0, 255),   # 7: midfoot_lateral (Violet)
    (0, 165, 255),   # 8: malleolus_medial (Amber)
    (147, 20, 255),  # 9: malleolus_lateral (Deep Pink)
    (50, 205, 50),   # 10: toe_tip (Lime)
    (208, 224, 64),  # 11: throat (Teal)
    (180, 105, 255), # 12: achilles (Hot Pink)
]


def match_instances_hungarian(pred_instances, gt_instances):
    """
    Globally match predicted instances to ground-truth instances using Hungarian algorithm.
    Cost is average distance between corresponding visible keypoints.
    Independent of predicted class labels.
    """
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
                if gv > 0:
                    px, py = pred['keypoints'][k][:2]
                    dists.append(np.hypot(px - gx, py - gy))
            if dists:
                cost_matrix[g_i, p_i] = np.mean(dists)
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {}
    for r, c in zip(row_ind, col_ind):
        # Keep match if average error < 120 pixels
        if cost_matrix[r, c] < 150.0:
            matches[r] = c
    return matches


def extract_prediction_only_raw_peaks(heatmaps, num_instances=1):
    """
    Method A: Extract raw peak predictions WITHOUT ANY GT INFORMATION.
    1. Detects high-confidence peaks (threshold >= 0.15).
    2. Clusters peaks into num_instances spatial centers using K-Means on peaks (no GT).
    3. For each instance and each keypoint channel k:
       Selects the highest-confidence peak within the instance's cluster.
    """
    scale = 256.0 / 64.0
    all_pks = {}
    strong_pts = []
    
    for k in range(13):
        pks = detect_peaks(heatmaps[k], threshold=0.15, nms_window=3, subpixel=True)
        all_pks[k] = [(p[0] * scale, p[1] * scale, p[2]) for p in pks]
        for p in all_pks[k]:
            if p[2] > 0.35:
                strong_pts.append([p[0], p[1]])
                
    if not strong_pts:
        # Fallback to lower threshold
        for k in range(13):
            for p in all_pks[k]:
                if p[2] > 0.15:
                    strong_pts.append([p[0], p[1]])
                    
    if not strong_pts:
        return []
        
    pts_arr = np.array(strong_pts, dtype=np.float32)
    
    # Cluster into centers purely from predicted peaks
    if num_instances >= 2 and len(pts_arr) >= 6:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _, labels, centers = cv2.kmeans(pts_arr, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = sorted(centers, key=lambda c: c[0])  # Sort by x
    else:
        centers = [np.mean(pts_arr, axis=0)]
        
    instances = []
    for c_i, (cx, cy) in enumerate(centers):
        other_centers = [other for other in centers if not (np.isclose(other[0], cx) and np.isclose(other[1], cy))]
        inst_kps = []
        for k in range(13):
            cands = all_pks[k]
            valid = []
            for p in cands:
                d_this = np.hypot(p[0] - cx, p[1] - cy)
                if other_centers:
                    d_other = min(np.hypot(p[0] - oc[0], p[1] - oc[1]) for oc in other_centers)
                    if d_this >= d_other:
                        continue
                if d_this < 90.0:  # Within foot radius in 256 space
                    valid.append((p, d_this))
                    
            if valid:
                # Pick highest confidence peak in this foot region
                valid.sort(key=lambda x: x[0][2], reverse=True)
                inst_kps.append(valid[0][0])
            else:
                # If no peak in region, use global top peak if within radius
                if cands:
                    inst_kps.append(cands[0])
                else:
                    inst_kps.append((0.0, 0.0, 0.0))
        instances.append({'instance_id': c_i, 'keypoints': inst_kps})
    return instances


def evaluate_three_methods(model, samples, device):
    thresholds = [0.20, 0.10, 0.05]
    
    stats_a = {th: {'correct': 0, 'total': 0} for th in thresholds}
    stats_b = {th: {'correct': 0, 'total': 0} for th in thresholds}
    stats_c = {th: {'correct': 0, 'total': 0} for th in thresholds}
    
    errs_a = []
    errs_b = []
    errs_c = []
    
    matched_feet_a = 0
    matched_feet_b = 0
    total_feet = 0
    total_kps_evaluated = 0
    
    scale = 256.0 / 64.0
    
    per_image_details = []
    
    with torch.no_grad():
        for s_idx, sample in enumerate(samples):
            img_tensor = sample['image'].unsqueeze(0).to(device)
            gt_instances = sample['instances']
            stem = sample['stem']
            num_gt = len(gt_instances)
            total_feet += num_gt
            
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
            
            # --- Method A: Prediction-Only Raw Peaks ---
            preds_a = extract_prediction_only_raw_peaks(heatmaps, num_instances=num_gt)
            matches_a = match_instances_hungarian(preds_a, gt_instances)
            matched_feet_a += len(matches_a)
            
            # --- Method B: Production Decoder ---
            raw_b = decode_spatial_instances_single(heatmaps, cls_probs, pafs=pafs)
            preds_b = []
            for inst in raw_b:
                c_inst = copy.deepcopy(inst)
                c_inst['keypoints'] = [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
                preds_b.append(c_inst)
            matches_b = match_instances_hungarian(preds_b, gt_instances)
            matched_feet_b += len(matches_b)
            
            # --- Method C: Diagnostic Oracle / GT-Assisted ---
            # Collect all detected peaks per channel
            all_raw_pks = {}
            for k in range(13):
                pks = detect_peaks(heatmaps[k], threshold=0.08, nms_window=3, subpixel=True)
                if not pks:
                    py, px = np.unravel_index(np.argmax(heatmaps[k]), heatmaps[k].shape)
                    pks = [(float(px), float(py), float(heatmaps[k][py, px]))]
                all_raw_pks[k] = [(p[0] * scale, p[1] * scale, p[2]) for p in pks]
                
            img_details = {
                'stem': stem,
                'num_gt': num_gt,
                'matches_a': {str(k): int(v) for k, v in matches_a.items()},
                'matches_b': {str(k): int(v) for k, v in matches_b.items()},
                'kps': []
            }
            
            # Compute metrics for each GT instance
            for g_i, gt in enumerate(gt_instances):
                w_px = gt['bbox'][2] * 256.0
                h_px = gt['bbox'][3] * 256.0
                diag = np.sqrt(w_px**2 + h_px**2)
                if diag < 1e-4:
                    continue
                    
                p_idx_a = matches_a.get(g_i, None)
                inst_a = preds_a[p_idx_a] if p_idx_a is not None and p_idx_a < len(preds_a) else None
                
                p_idx_b = matches_b.get(g_i, None)
                inst_b = preds_b[p_idx_b] if p_idx_b is not None and p_idx_b < len(preds_b) else None
                
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv == 0:
                        continue  # EXCLUDE UNLABELED KEYPOINTS
                    total_kps_evaluated += 1
                    
                    # Method A
                    if inst_a is not None:
                        pa = inst_a['keypoints'][k]
                        da = np.hypot(pa[0] - gx, pa[1] - gy)
                        errs_a.append(da)
                        for th in thresholds:
                            stats_a[th]['total'] += 1
                            if da < th * diag:
                                stats_a[th]['correct'] += 1
                    else:
                        for th in thresholds:
                            stats_a[th]['total'] += 1
                            
                    # Method B
                    if inst_b is not None:
                        pb = inst_b['keypoints'][k]
                        db = np.hypot(pb[0] - gx, pb[1] - gy)
                        errs_b.append(db)
                        for th in thresholds:
                            stats_b[th]['total'] += 1
                            if db < th * diag:
                                stats_b[th]['correct'] += 1
                    else:
                        for th in thresholds:
                            stats_b[th]['total'] += 1
                            
                    # Method C (Oracle)
                    pks_c = all_raw_pks[k]
                    best_pk_c = min(pks_c, key=lambda p: np.hypot(p[0] - gx, p[1] - gy))
                    dc = np.hypot(best_pk_c[0] - gx, best_pk_c[1] - gy)
                    errs_c.append(dc)
                    for th in thresholds:
                        stats_c[th]['total'] += 1
                        if dc < th * diag:
                            stats_c[th]['correct'] += 1
                            
            per_image_details.append(img_details)
            
    res_a = {str(th): (stats_a[th]['correct'] / max(1, stats_a[th]['total'])) * 100 for th in thresholds}
    res_b = {str(th): (stats_b[th]['correct'] / max(1, stats_b[th]['total'])) * 100 for th in thresholds}
    res_c = {str(th): (stats_c[th]['correct'] / max(1, stats_c[th]['total'])) * 100 for th in thresholds}
    
    return {
        'total_feet': total_feet,
        'total_kps_evaluated': total_kps_evaluated,
        'matched_feet_a': matched_feet_a,
        'matched_feet_b': matched_feet_b,
        'method_a': {'pck': res_a, 'mean_err': float(np.mean(errs_a)) if errs_a else -1.0},
        'method_b': {'pck': res_b, 'mean_err': float(np.mean(errs_b)) if errs_b else -1.0},
        'method_c_oracle': {'pck': res_c, 'mean_err': float(np.mean(errs_c)) if errs_c else -1.0},
        'details': per_image_details
    }


def render_audit_visualizations(model, samples, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 256.0 / 64.0
    vis_h, vis_w = 512, 512
    
    # We will visualize Sample 0 (two-foot) and Sample 6 (single-foot)
    sample_indices = [0, 6]
    
    with torch.no_grad():
        for s_idx in sample_indices:
            sample = samples[s_idx]
            stem = sample['stem']
            gt_instances = sample['instances']
            img_tensor = sample['image'].unsqueeze(0).to(device)
            img_rgb = (sample['image'].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
            
            # --- Panel 1: Ground Truth ---
            p1 = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (vis_w, vis_h))
            sx = vis_w / 256.0
            sy = vis_h / 256.0
            for g_i, gt in enumerate(gt_instances):
                # Limbs
                for u, v in SKELETON_LIMBS_13:
                    p_u = gt['keypoints'][u]
                    p_v = gt['keypoints'][v]
                    if p_u[2] > 0 and p_v[2] > 0:
                        pt_u = (int(p_u[0] * sx), int(p_u[1] * sy))
                        pt_v = (int(p_v[0] * sx), int(p_v[1] * sy))
                        cv2.line(p1, pt_u, pt_v, (0, 255, 0), 2, cv2.LINE_AA)
                # Points with ID & Color
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv > 0:
                        pt = (int(gx * sx), int(gy * sy))
                        col = KP_COLORS[k]
                        cv2.circle(p1, pt, 6, col, -1, cv2.LINE_AA)
                        cv2.circle(p1, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(p1, f"F{g_i}:k{k}", (pt[0] + 5, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(p1, (0, 0), (vis_w, 30), (30, 30, 30), -1)
            cv2.putText(p1, f"1. GT Instances ({len(gt_instances)} feet)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            # --- Panel 2: Raw Heatmap Candidate Peaks (Prediction-Only) ---
            # Shows all candidate peaks without lines/PAF to reveal true raw detections
            p2 = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (vis_w, vis_h))
            total_cand_peaks = 0
            for k in range(13):
                pks = detect_peaks(heatmaps[k], threshold=0.15, nms_window=3, subpixel=True)
                for pk in pks:
                    total_cand_peaks += 1
                    pt = (int(pk[0] * scale * sx), int(pk[1] * scale * sy))
                    col = KP_COLORS[k]
                    # Radius proportional to confidence
                    r = max(3, int(pk[2] * 8))
                    cv2.circle(p2, pt, r, col, -1, cv2.LINE_AA)
                    cv2.circle(p2, pt, r + 1, (0, 0, 0), 1, cv2.LINE_AA)
                    if pk[2] > 0.40:
                        cv2.putText(p2, f"k{k}:{pk[2]:.2f}", (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(p2, (0, 0), (vis_w, 30), (30, 30, 30), -1)
            cv2.putText(p2, f"2. Raw Candidate Peaks (thres>=0.15, N={total_cand_peaks})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            # --- Panel 3: Decoded Predictions (Method B) ---
            p3 = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (vis_w, vis_h))
            raw_b = decode_spatial_instances_single(heatmaps, cls_probs, pafs=pafs)
            preds_b = []
            for inst in raw_b:
                c_inst = copy.deepcopy(inst)
                c_inst['keypoints'] = [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in inst['keypoints']]
                preds_b.append(c_inst)
            for p_i, pred in enumerate(preds_b):
                kps = pred['keypoints']
                # Limbs
                for u, v in SKELETON_LIMBS_13:
                    pt_u = (int(kps[u][0] * sx), int(kps[u][1] * sy))
                    pt_v = (int(kps[v][0] * sx), int(kps[v][1] * sy))
                    if kps[u][2] > 0.05 and kps[v][2] > 0.05:
                        cv2.line(p3, pt_u, pt_v, (255, 120, 0), 2, cv2.LINE_AA)
                for k in range(13):
                    pk = kps[k]
                    if pk[2] > 0.05:
                        pt = (int(pk[0] * sx), int(pk[1] * sy))
                        col = KP_COLORS[k]
                        cv2.circle(p3, pt, 6, col, -1, cv2.LINE_AA)
                        cv2.circle(p3, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(p3, f"P{p_i}:k{k}", (pt[0] + 5, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(p3, (0, 0), (vis_w, 30), (30, 30, 30), -1)
            cv2.putText(p3, f"3. Production Decoder ({len(preds_b)} instances)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            combined = np.hstack([p1, p2, p3])
            out_path = out_dir / f"audit_vis_sample_{s_idx}_{stem[:20]}.jpg"
            cv2.imwrite(str(out_path), combined)
            print(f"  🖼️ Saved audit visualization: {out_path}")


def run_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("🔬 FOCUSED AUDIT OF OVERFIT EVALUATION & PIPELINE")
    print("=" * 80)
    
    ckpt_path = BASE_DIR / "outputs" / "overfit_sanity_test" / "overfit_checkpoint.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ARShoeM1v2(pretrained=False, num_keypoints=13, num_limbs=16, num_classes=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    dataset_root = PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp"
    samples = load_dataset(dataset_root, SELECTED_STEMS, img_size=256)
    
    # 1. Re-evaluate using 3 separate methods
    print("\n📊 Running 3-Method Evaluation across 8 Overfit Images...")
    results = evaluate_three_methods(model, samples, device)
    
    print("\n" + "=" * 80)
    print("🏆 THREE-WAY RE-EVALUATION RESULTS (Total Labeled Keypoints = 102 across 12 instances)")
    print("=" * 80)
    print(f"Total Ground-Truth Feet Instances: {results['total_feet']}")
    print(f"Total Keypoints Evaluated (v > 0): {results['total_kps_evaluated']}")
    print(f"Explicit PCK Normalization Denominator: Instance Bounding-Box Diagonal (sqrt(W_bbox^2 + H_bbox^2))")
    print("-" * 80)
    
    mA = results['method_a']
    mB = results['method_b']
    mC = results['method_c_oracle']
    
    print(f"Method A: Prediction-Only Raw Peaks (NO GT used during extraction):")
    print(f"  Matched Feet: {results['matched_feet_a']} / {results['total_feet']}")
    print(f"  PCK@0.20    : {mA['pck']['0.2']:>6.2f}%")
    print(f"  PCK@0.10    : {mA['pck']['0.1']:>6.2f}%")
    print(f"  PCK@0.05    : {mA['pck']['0.05']:>6.2f}%")
    print(f"  Mean Error  : {mA['mean_err']:.2f} px")
    print("-" * 80)
    
    print(f"Method B: Production Decoder (decode_spatial_instances_single):")
    print(f"  Matched Feet: {results['matched_feet_b']} / {results['total_feet']}")
    print(f"  PCK@0.20    : {mB['pck']['0.2']:>6.2f}%")
    print(f"  PCK@0.10    : {mB['pck']['0.1']:>6.2f}%")
    print(f"  PCK@0.05    : {mB['pck']['0.05']:>6.2f}%")
    print(f"  Mean Error  : {mB['mean_err']:.2f} px")
    print("-" * 80)
    
    print(f"Method C: Diagnostic Oracle / GT-Assisted Upper Bound (DIAGNOSTIC ONLY):")
    print(f"  Matched Feet: N/A (GT queried directly per keypoint)")
    print(f"  PCK@0.20    : {mC['pck']['0.2']:>6.2f}% (Matches the previously reported 100%!)")
    print(f"  PCK@0.10    : {mC['pck']['0.1']:>6.2f}%")
    print(f"  PCK@0.05    : {mC['pck']['0.05']:>6.2f}%")
    print(f"  Mean Error  : {mC['mean_err']:.2f} px")
    print("=" * 80)
    
    # 2. Render Audit Visualizations
    out_vis_dir = BASE_DIR / "outputs" / "overfit_sanity_test" / "audit_visualizations"
    print("\n🖼️ Rendering side-by-side audit panels with identical colors & explicit IDs...")
    render_audit_visualizations(model, samples, device, out_vis_dir)
    
    # Save audit report json
    out_json = BASE_DIR / "outputs" / "overfit_sanity_test" / "audit_evaluation_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Audit results saved to: {out_json}")


if __name__ == '__main__':
    run_audit()
