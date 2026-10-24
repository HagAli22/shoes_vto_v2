"""
Verification & Regression Test Script for Fixed Decoder
Compares:
1. Production Decoder (Before Fix)
2. Minimal Fixed Decoder (After Fix)
Includes 4 formal regression tests and side-by-side visualizations.
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
    load_paf_config,
    load_skeleton_priors,
    compute_paf_score,
    decode_spatial_instances_single
)
from tools.run_fast_overfit_sanity_test import (
    load_dataset,
    SELECTED_STEMS,
    KEYPOINT_NAMES_13,
    SKELETON_LIMBS_13
)

KP_COLORS = [
    (0, 0, 255),     # 0: toe_ground
    (0, 128, 255),   # 1: heel_back
    (0, 255, 255),   # 2: heel_ground
    (0, 255, 0),     # 3: ball_medial
    (255, 255, 0),   # 4: ball_lateral
    (255, 0, 0),     # 5: instep_top
    (255, 0, 128),   # 6: arch_medial
    (128, 0, 255),   # 7: midfoot_lateral
    (0, 165, 255),   # 8: malleolus_medial
    (147, 20, 255),  # 9: malleolus_lateral
    (50, 205, 50),   # 10: toe_tip
    (208, 224, 64),  # 11: throat
    (180, 105, 255), # 12: achilles
]


def decode_spatial_instances_fixed(
    heatmaps,
    class_probs,
    pafs=None,
    peak_threshold=0.10,
    nms_window=3,
    min_keypoints=4,
    max_instances=2,
    subpixel=True,
    max_radius=22.0,
    w_conf=1.0,
    w_paf=0.5,
    w_geom=0.20,
    max_iter=3,
    use_skeleton=True,
    paf_config=None,
    noise_peak_threshold=15
):
    """
    Principled, minimal decoder fix for ARShoe M1v2.
    1. Channel entropy / noise filter: ignores channels saturated with background noise (> 15 peaks >= 0.20).
    2. Evidence-based instance counting: requires keypoint duplication or dual-class spatial presence to split into 2 feet.
    3. Strict spatial radius & missing keypoint preservation: rejects fabricated background noise.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    if isinstance(class_probs, torch.Tensor):
        class_probs = class_probs.detach().cpu().numpy()
    if isinstance(pafs, torch.Tensor):
        pafs = pafs.detach().cpu().numpy()

    H, W = heatmaps.shape[1], heatmaps.shape[2]
    num_k = heatmaps.shape[0]

    # Step 1: Detect peaks and identify clean vs noisy channels purely from predictions
    all_peaks = {}
    clean_kps = []
    for kp_idx in range(num_k):
        pks = detect_peaks(heatmaps[kp_idx], threshold=peak_threshold, nms_window=nms_window, subpixel=subpixel)
        all_peaks[kp_idx] = pks
        pks_020 = [p for p in pks if p[2] >= 0.20]
        if len(pks_020) <= noise_peak_threshold:
            clean_kps.append(kp_idx)

    # Step 2: Collect high-confidence anchor peaks only from clean channels
    preferred_anchors = [k for k in [1, 2, 3, 4, 8, 12, 5, 10] if k in clean_kps]
    anchor_pts = []
    for kp_idx in preferred_anchors:
        for p in all_peaks.get(kp_idx, []):
            if p[2] > 0.35:
                anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 4:
        for kp_idx in clean_kps:
            for p in all_peaks.get(kp_idx, []):
                if p[2] > 0.25:
                    anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 3:
        return []

    pts_xy = np.array([[p[0], p[1]] for p in anchor_pts], dtype=np.float32)

    # Step 3: Determine number of instances (1 or 2) with independent evidence
    # Check class head spatial confirmation
    has_dual_class = (class_probs[0].max() >= 0.35 and class_probs[1].max() >= 0.35 and
                      np.sum(class_probs[0] > 0.35) >= 200 and np.sum(class_probs[1] > 0.35) >= 200)

    centers = [np.mean(pts_xy, axis=0)]
    if max_instances >= 2 and len(pts_xy) >= 6:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _, labels, c_kmeans = cv2.kmeans(pts_xy, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        dist_between = np.linalg.norm(c_kmeans[0] - c_kmeans[1])

        kps_c0 = set(anchor_pts[i][2] for i in range(len(labels)) if labels[i] == 0)
        kps_c1 = set(anchor_pts[i][2] for i in range(len(labels)) if labels[i] == 1)
        common_kps = kps_c0.intersection(kps_c1)

        # Independent evidence rule:
        # A second instance is created ONLY IF:
        # (1) Distance >= 14.0 AND both clusters contain duplicate keypoint types (len(common_kps) >= 2), OR
        # (2) Class probability map confirms both Left Foot and Right Foot presence.
        if dist_between >= 14.0 and (len(common_kps) >= 2 or has_dual_class):
            centers = sorted(c_kmeans, key=lambda c: c[0])
        elif dist_between >= 7.0 and has_dual_class and len(kps_c0) >= 3 and len(kps_c1) >= 3:
            # Adjacent feet confirmed by dual class presence
            centers = sorted(c_kmeans, key=lambda c: c[0])

    PERIPHERAL_KPS = {0, 1, 2, 8, 9, 10} if num_k == 13 else {0, 1, 2, 9, 10, 11}
    LEG_KPS = set()

    # Step 4: Gather candidates and assemble instances
    instances = []
    scale = 256.0 / 64.0

    for c_idx, (cx, cy) in enumerate(centers):
        other_centers = [oc for oc in centers if not (np.isclose(oc[0], cx) and np.isclose(oc[1], cy))]
        candidates_per_kp = {}

        for kp_idx in range(num_k):
            if kp_idx not in clean_kps:
                # Unsupervised noise channel -> leave missing!
                candidates_per_kp[kp_idx] = []
                continue

            c_peaks = all_peaks.get(kp_idx, [])
            valid_cands = []
            for p in c_peaks:
                dist = np.hypot(p[0] - cx, p[1] - cy)
                if other_centers:
                    dist_other = min(np.hypot(p[0] - oc[0], p[1] - oc[1]) for oc in other_centers)
                    if dist >= dist_other:
                        continue
                if dist <= max_radius and p[2] >= peak_threshold:
                    valid_cands.append(p)

            valid_cands.sort(key=lambda item: item[2], reverse=True)
            candidates_per_kp[kp_idx] = valid_cands[:3]

        # Assemble keypoints (take top candidate within radius, or missing)
        inst_kps = []
        scores = []
        for k in range(num_k):
            cands = candidates_per_kp[k]
            if cands:
                top_p = cands[0]
                inst_kps.append([float(top_p[0] * scale), float(top_p[1] * scale), float(top_p[2])])
                scores.append(float(top_p[2]))
            else:
                inst_kps.append([0.0, 0.0, 0.0])

        valid_kps = [k for k in inst_kps if k[2] > 0.10]
        if len(valid_kps) < min_keypoints:
            continue

        # Class assignment from class_probs map
        icx, icy = int(np.clip(round(cx), 0, W - 1)), int(np.clip(round(cy), 0, H - 1))
        class_id = 0 if class_probs[0, icy, icx] >= class_probs[1, icy, icx] else 1

        instances.append({
            'instance_id': c_idx,
            'class_id': class_id,
            'class_name': 'left_foot' if class_id == 0 else 'right_foot',
            'keypoints': inst_kps,
            'score': float(np.mean(scores)) if scores else 0.0,
            'valid_count': len(valid_kps)
        })

    return instances


def evaluate_decoder(model, samples, device, decoder_fn, decoder_name):
    thresholds = [0.20, 0.10, 0.05]
    stats = {th: {'correct': 0, 'total': 0} for th in thresholds}
    per_kp_stats = {k: {th: {'correct': 0, 'total': 0} for th in thresholds} for k in range(13)}
    errs = []
    
    matched_feet = 0
    total_feet = 0
    total_extra_instances = 0
    total_missing_instances = 0
    
    per_image_results = []

    with torch.no_grad():
        for s_idx, s in enumerate(samples):
            img_tensor = s['image'].unsqueeze(0).to(device)
            gt_insts = s['instances']
            num_gt = len(gt_insts)
            total_feet += num_gt
            
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            class_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            
            preds = decoder_fn(heatmaps, class_probs, pafs=pafs)
            num_pred = len(preds)
            
            if num_pred > num_gt:
                total_extra_instances += (num_pred - num_gt)
            elif num_pred < num_gt:
                total_missing_instances += (num_gt - num_pred)
                
            # Hungarian matching (independent of class labels)
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
            matched_feet += len(matches)
            
            # Keypoint evaluation
            for g_i, gt in enumerate(gt_insts):
                w_px = gt['bbox'][2] * 256.0
                h_px = gt['bbox'][3] * 256.0
                diag = np.sqrt(w_px**2 + h_px**2)
                p_idx = matches.get(g_i, None)
                pred = preds[p_idx] if p_idx is not None else None
                
                for k in range(13):
                    gx, gy, gv = gt['keypoints'][k]
                    if gv == 0:
                        continue  # EXCLUDE UNLABELED KEYPOINTS
                        
                    for th in thresholds:
                        stats[th]['total'] += 1
                        per_kp_stats[k][th]['total'] += 1
                        
                    if pred is not None and pred['keypoints'][k][2] > 0:
                        px, py = pred['keypoints'][k][:2]
                        d = np.hypot(px - gx, py - gy)
                        errs.append(d)
                        for th in thresholds:
                            if d < th * diag:
                                stats[th]['correct'] += 1
                                per_kp_stats[k][th]['correct'] += 1
                                
            per_image_results.append({
                'stem': s['stem'],
                'num_gt': num_gt,
                'num_pred': num_pred,
                'matched': len(matches)
            })

    summary = {
        'decoder_name': decoder_name,
        'matched_feet': matched_feet,
        'total_feet': total_feet,
        'extra_instances': total_extra_instances,
        'missing_instances': total_missing_instances,
        'mean_error_px': float(np.mean(errs)) if errs else -1.0,
        'pck': {th: (stats[th]['correct'] / max(1, stats[th]['total'])) * 100.0 for th in thresholds},
        'per_kp': {
            KEYPOINT_NAMES_13[k]: {
                'pck20': (per_kp_stats[k][0.20]['correct'] / max(1, per_kp_stats[k][0.20]['total'])) * 100.0,
                'pck10': (per_kp_stats[k][0.10]['correct'] / max(1, per_kp_stats[k][0.10]['total'])) * 100.0,
                'pck05': (per_kp_stats[k][0.05]['correct'] / max(1, per_kp_stats[k][0.05]['total'])) * 100.0,
                'count': per_kp_stats[k][0.20]['total']
            }
            for k in range(13)
        },
        'per_image': per_image_results
    }
    return summary


def render_before_after_visualizations(model, samples, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = 256.0 / 64.0
    vis_h, vis_w = 512, 512
    
    # Render Sample 0 (two-foot) and Sample 6 (single-foot)
    for s_idx in [0, 6]:
        sample = samples[s_idx]
        stem = sample['stem']
        gt_insts = sample['instances']
        img_np = (sample['image'].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img_tensor = sample['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            class_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            
        preds_before = decode_spatial_instances_single(heatmaps, class_probs, pafs=pafs)
        preds_after = decode_spatial_instances_fixed(heatmaps, class_probs, pafs=pafs)
        
        # Scale preds_before keypoints to 256
        for p in preds_before:
            p['keypoints'] = [(kp[0] * scale, kp[1] * scale, kp[2]) for kp in p['keypoints']]
            
        sx = vis_w / 256.0
        sy = vis_h / 256.0
        
        # Panel 1: GT
        p1 = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), (vis_w, vis_h))
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
                    cv2.circle(p1, pt, 6, KP_COLORS[k], -1, cv2.LINE_AA)
                    cv2.putText(p1, f"F{g_i}:k{k}", (pt[0]+4, pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(p1, (0, 0), (vis_w, 30), (30, 30, 30), -1)
        cv2.putText(p1, f"1. GT Instances ({len(gt_insts)} feet)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Panel 2: Production Decoder (Before)
        p2 = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), (vis_w, vis_h))
        for p_i, pred in enumerate(preds_before):
            kps = pred['keypoints']
            for u, v in SKELETON_LIMBS_13:
                if kps[u][2] > 0.05 and kps[v][2] > 0.05:
                    cv2.line(p2, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 100, 255), 2, cv2.LINE_AA)
            for k in range(13):
                pk = kps[k]
                if pk[2] > 0.05:
                    pt = (int(pk[0]*sx), int(pk[1]*sy))
                    cv2.circle(p2, pt, 6, KP_COLORS[k], -1, cv2.LINE_AA)
                    cv2.putText(p2, f"P{p_i}:k{k}", (pt[0]+4, pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(p2, (0, 0), (vis_w, 30), (30, 30, 30), -1)
        cv2.putText(p2, f"2. Production Decoder (Before) - {len(preds_before)} insts", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Panel 3: Fixed Decoder (After)
        p3 = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), (vis_w, vis_h))
        for p_i, pred in enumerate(preds_after):
            kps = pred['keypoints']
            for u, v in SKELETON_LIMBS_13:
                if kps[u][2] > 0.05 and kps[v][2] > 0.05:
                    cv2.line(p3, (int(kps[u][0]*sx), int(kps[u][1]*sy)), (int(kps[v][0]*sx), int(kps[v][1]*sy)), (255, 200, 0), 2, cv2.LINE_AA)
            for k in range(13):
                pk = kps[k]
                if pk[2] > 0.05:
                    pt = (int(pk[0]*sx), int(pk[1]*sy))
                    cv2.circle(p3, pt, 6, KP_COLORS[k], -1, cv2.LINE_AA)
                    cv2.putText(p3, f"P{p_i}:k{k}", (pt[0]+4, pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(p3, (0, 0), (vis_w, 30), (30, 30, 30), -1)
        cv2.putText(p3, f"3. Fixed Decoder (After) - {len(preds_after)} insts", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        combined = np.hstack([p1, p2, p3])
        out_file = out_dir / f"fix_comparison_sample_{s_idx}_{stem[:20]}.jpg"
        cv2.imwrite(str(out_file), combined)
        print(f"  🖼️ Saved visualization: {out_file}")


def run_regression_tests():
    print("=" * 80)
    print("🧪 FORMAL REGRESSION TESTS FOR DECODER FIX")
    print("=" * 80)
    
    # Regression Test 1: Single-foot synthetic heatmap with 1 foot cluster
    # Must NOT produce 2 instances
    hm_single = np.zeros((13, 64, 64), dtype=np.float32)
    # Put 6 keypoints around (32, 32)
    for k in range(6):
        hm_single[k, 30 + k % 3, 30 + k // 3] = 0.90
    cls_single = np.zeros((2, 64, 64), dtype=np.float32)
    cls_single[0] = 0.95 # Only left foot active
    
    insts_single = decode_spatial_instances_fixed(hm_single, cls_single)
    test1_pass = (len(insts_single) == 1)
    print(f"  Test 1: Single-foot must produce exactly 1 instance -> {'✅ PASS' if test1_pass else '❌ FAIL'} (got {len(insts_single)})")
    
    # Regression Test 2: Two-foot synthetic heatmap with 2 distinct feet
    # Must retain both feet
    hm_two = np.zeros((13, 64, 64), dtype=np.float32)
    # Foot 1 at (15, 15), Foot 2 at (48, 48)
    for k in range(6):
        hm_two[k, 15, 15] = 0.90
        hm_two[k, 48, 48] = 0.90
    cls_two = np.zeros((2, 64, 64), dtype=np.float32)
    cls_two[0, :32, :32] = 0.90 # Left foot
    cls_two[1, 32:, 32:] = 0.90 # Right foot
    
    insts_two = decode_spatial_instances_fixed(hm_two, cls_two)
    test2_pass = (len(insts_two) == 2)
    print(f"  Test 2: Two-foot must retain both feet -> {'✅ PASS' if test2_pass else '❌ FAIL'} (got {len(insts_two)})")
    
    # Regression Test 3: Unsupervised channel with high noise (e.g. 50 peaks)
    # Must NOT create an independent instance
    hm_noise = copy.deepcopy(hm_single)
    # Corrupt channel 5 with 40 random peaks
    np.random.seed(42)
    for _ in range(40):
        rx, ry = np.random.randint(5, 60, size=2)
        hm_noise[5, ry, rx] = 0.95
        
    insts_noise = decode_spatial_instances_fixed(hm_noise, cls_single)
    test3_pass = (len(insts_noise) == 1 and insts_noise[0]['keypoints'][5][2] == 0.0)
    print(f"  Test 3: Unsupervised noisy anchor channel must be rejected -> {'✅ PASS' if test3_pass else '❌ FAIL'} (got {len(insts_noise)} insts, KP5 conf={insts_noise[0]['keypoints'][5][2]})")
    
    # Regression Test 4: Missing keypoints must remain missing [0, 0, 0]
    # In hm_single, keypoints 6..12 are zero
    kps_res = insts_single[0]['keypoints']
    test4_pass = all(kps_res[k][2] == 0.0 for k in range(6, 13))
    print(f"  Test 4: Missing keypoints must remain missing [0, 0, 0] -> {'✅ PASS' if test4_pass else '❌ FAIL'}")
    
    return test1_pass and test2_pass and test3_pass and test4_pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("🔍 EVALUATING PRODUCTION DECODER VS. FIXED DECODER ON 8 OVERFIT IMAGES")
    print("=" * 80)
    
    ckpt_path = BASE_DIR / "outputs" / "overfit_sanity_test" / "overfit_checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ARShoeM1v2(pretrained=False, num_keypoints=13, num_limbs=16, num_classes=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    dataset_root = PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp"
    samples = load_dataset(dataset_root, SELECTED_STEMS, img_size=256)
    
    # 1. Run Regression Tests
    tests_ok = run_regression_tests()
    
    # 2. Evaluate Production Decoder (Before)
    scale = 256.0 / 64.0
    def prod_wrapper(hm, cls_p, pafs=None):
        raw = decode_spatial_instances_single(hm, cls_p, pafs=pafs)
        res = []
        for inst in raw:
            c = copy.deepcopy(inst)
            c['keypoints'] = [(kp[0]*scale, kp[1]*scale, kp[2]) for kp in inst['keypoints']]
            res.append(c)
        return res
        
    print("\n⏳ Evaluating Production Decoder (Before Fix)...")
    res_before = evaluate_decoder(model, samples, device, prod_wrapper, "Production Decoder (Before)")
    
    # 3. Evaluate Fixed Decoder (After)
    print("⏳ Evaluating Fixed Decoder (After Fix)...")
    res_after = evaluate_decoder(model, samples, device, decode_spatial_instances_fixed, "Fixed Decoder (After)")
    
    # 4. Report Comparison
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE RESULTS: BEFORE VS. AFTER FIX (102 Labeled Keypoints, 12 GT Feet)")
    print("=" * 80)
    print(f" {'Metric':<25} | {'Production Decoder (Before)':<27} | {'Fixed Decoder (After)':<23} | {'Delta':<10}")
    print("-" * 90)
    delta_feet = res_after['matched_feet'] - res_before['matched_feet']
    delta_feet_str = f"+{delta_feet}" if delta_feet >= 0 else f"{delta_feet}"
    print(f" {'Matched Feet':<25} | {res_before['matched_feet']:>2} / {res_before['total_feet']} ({res_before['matched_feet']/res_before['total_feet']*100:.1f}%)               | {res_after['matched_feet']:>2} / {res_after['total_feet']} ({res_after['matched_feet']/res_after['total_feet']*100:.1f}%)             | {delta_feet_str:<10}")
    print(f" {'Extra Ghost Instances':<25} | {res_before['extra_instances']:<27} | {res_after['extra_instances']:<23} | {res_after['extra_instances'] - res_before['extra_instances']:<10}")
    print(f" {'Missing Feet':<25} | {res_before['missing_instances']:<27} | {res_after['missing_instances']:<23} | {res_after['missing_instances'] - res_before['missing_instances']:<10}")
    print(f" {'Mean Pixel Error':<25} | {res_before['mean_error_px']:>6.2f} px                   | {res_after['mean_error_px']:>6.2f} px                 | {res_after['mean_error_px'] - res_before['mean_error_px']:>+6.2f} px")
    print("-" * 90)
    print(f" {'PCK@0.20':<25} | {res_before['pck'][0.20]:>6.2f}%                     | {res_after['pck'][0.20]:>6.2f}%                   | {res_after['pck'][0.20] - res_before['pck'][0.20]:>+6.2f}%")
    print(f" {'PCK@0.10':<25} | {res_before['pck'][0.10]:>6.2f}%                     | {res_after['pck'][0.10]:>6.2f}%                   | {res_after['pck'][0.10] - res_before['pck'][0.10]:>+6.2f}%")
    print(f" {'PCK@0.05':<25} | {res_before['pck'][0.05]:>6.2f}%                     | {res_after['pck'][0.05]:>6.2f}%                   | {res_after['pck'][0.05] - res_before['pck'][0.05]:>+6.2f}%")
    print("=" * 90)
    
    print("\n🔬 Per-Keypoint PCK@0.20 Breakdown:")
    hdr = f" {'#':<3} | {'Keypoint':<18} | {'Count':<5} | {'Before Fix':<12} | {'After Fix':<12} | {'Delta':<10}"
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for k in range(13):
        name = KEYPOINT_NAMES_13[k]
        b_pck = res_before['per_kp'][name]['pck20']
        a_pck = res_after['per_kp'][name]['pck20']
        cnt = res_before['per_kp'][name]['count']
        diff = a_pck - b_pck
        print(f" {k:<3} | {name:<18} | {cnt:<5} | {b_pck:>6.1f}%      | {a_pck:>6.1f}%      | {diff:>+6.1f}%")
    print("-" * len(hdr))
    
    # 5. Render Visualizations
    out_vis_dir = BASE_DIR / "outputs" / "overfit_sanity_test" / "fix_visualizations"
    print("\n🖼️ Rendering Before vs. After side-by-side visualizations...")
    render_before_after_visualizations(model, samples, device, out_vis_dir)
    
    # Save Report
    full_report = {
        'regression_tests_passed': tests_ok,
        'results_before': res_before,
        'results_after': res_after
    }
    out_json = BASE_DIR / "outputs" / "overfit_sanity_test" / "decoder_fix_audit_report.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        # Convert float keys in pck dict to str
        for r in [res_before, res_after]:
            r['pck'] = {str(k): v for k, v in r['pck'].items()}
        json.dump(full_report, f, indent=2)
    print(f"\n✅ Comparative report saved to: {out_json}")


if __name__ == '__main__':
    main()
