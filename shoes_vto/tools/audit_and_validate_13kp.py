"""
Audit & Validation Comparison Script for 13-Keypoint Model
Evaluates ONLY the validation split (dataset/merged_v3_v4_13kp/valid)

Compares:
A. Raw Heatmap Argmax / Peak locations
B. Current Production Decoder
C. Decoder with Centroid Penalties disabled only for verified medial/lateral keypoint indices

Excludes all v=0 keypoints from evaluation.
Reports per-keypoint PCK@0.20 / 0.10 / 0.05 and sample counts.
Saves 4-panel visual comparisons on validation samples.
"""

import os
import sys
import json
import copy
from pathlib import Path
import cv2
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
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
    decode_spatial_instances_single,
    compute_paf_score,
    compute_foot_box
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


def load_model(checkpoint_path, device='cpu'):
    model = ARShoeM1v2(pretrained=False, num_keypoints=13, num_limbs=16, num_classes=2)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device)
    model.eval()
    return model


def parse_ground_truth(label_path, img_w, img_h):
    instances = []
    if not label_path.exists():
        return instances
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 44:
                continue
            cls_id = int(tokens[0])
            cx, cy, w, h = [float(x) for x in tokens[1:5]]
            kps = []
            for i in range(13):
                kx = float(tokens[5 + i * 3]) * img_w
                ky = float(tokens[5 + i * 3 + 1]) * img_h
                kv = int(float(tokens[5 + i * 3 + 2]))
                kps.append((kx, ky, kv))
            instances.append({
                'class_id': cls_id,
                'class_name': 'left_foot' if cls_id == 0 else 'right_foot',
                'bbox': (cx * img_w, cy * img_h, w * img_w, h * img_h),
                'bbox_diag': np.sqrt((w * img_w) ** 2 + (h * img_h) ** 2),
                'keypoints': kps
            })
    return instances


from collections import defaultdict


def decode_spatial_instances_custom(
    heatmaps,
    class_probs,
    pafs=None,
    peak_threshold=0.10,
    nms_window=3,
    min_keypoints=5,
    max_instances=2,
    subpixel=True,
    max_radius=20.0,
    w_conf=1.0,
    w_paf=0.5,
    w_geom=0.20,
    max_iter=3,
    use_skeleton=True,
    paf_config=None,
    disable_centroid_penalty_indices=None
):
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    if isinstance(class_probs, torch.Tensor):
        class_probs = class_probs.detach().cpu().numpy()
    if isinstance(pafs, torch.Tensor):
        pafs = pafs.detach().cpu().numpy()

    H, W = heatmaps.shape[1], heatmaps.shape[2]
    num_k = heatmaps.shape[0]

    all_peaks = {}
    for kp_idx in range(num_k):
        all_peaks[kp_idx] = detect_peaks(heatmaps[kp_idx], threshold=peak_threshold, nms_window=nms_window, subpixel=subpixel)

    anchor_indices = [5, 10, 1, 3] if num_k == 13 else [5, 6, 12, 1]
    anchor_pts = []
    for kp_idx in anchor_indices:
        for p in all_peaks.get(kp_idx, []):
            if p[2] > 0.35:
                anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 4:
        for kp_idx in range(num_k):
            if num_k == 16 and kp_idx == 15:
                continue
            for p in all_peaks.get(kp_idx, []):
                if p[2] > 0.35:
                    anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 4:
        for kp_idx in range(num_k):
            if num_k == 16 and kp_idx == 15:
                continue
            for p in all_peaks.get(kp_idx, []):
                if p[2] > 0.15:
                    anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 3:
        return []

    pts_xy = np.array([[p[0], p[1]] for p in anchor_pts], dtype=np.float32)
    
    if max_instances >= 2 and len(pts_xy) >= 6:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _, labels, centers = cv2.kmeans(pts_xy, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        dist_between_centers = np.linalg.norm(centers[0] - centers[1])
        c0_count = np.sum(labels == 0)
        c1_count = np.sum(labels == 1)
        
        if dist_between_centers < 7.0 or c0_count < 3 or c1_count < 3:
            centers = [np.mean(pts_xy, axis=0)]
        else:
            centers = sorted(centers, key=lambda c: c[0])
    else:
        centers = [np.mean(pts_xy, axis=0)]

    if disable_centroid_penalty_indices is not None:
        PERIPHERAL_KPS = set(disable_centroid_penalty_indices)
        LEG_KPS = set()
    else:
        PERIPHERAL_KPS = {0, 1, 2, 8, 9, 10} if num_k == 13 else {0, 1, 2, 9, 10, 11}
        LEG_KPS = {12} if num_k == 13 else {12, 15}

    yy, xx = np.indices((H, W), dtype=np.float32)
    voronoi_masks = []
    for i, (cx, cy) in enumerate(centers):
        dist_this = (xx - cx)**2 + (yy - cy)**2
        mask = np.ones((H, W), dtype=bool)
        for j, (ocx, ocy) in enumerate(centers):
            if i != j:
                dist_other = (xx - ocx)**2 + (yy - ocy)**2
                mask &= (dist_this < dist_other)
        voronoi_masks.append(mask)

    if paf_config is None:
        paf_cfg = load_paf_config("paf_connections_13kp.yaml" if num_k == 13 else "paf_connections.yaml")
    else:
        paf_cfg = paf_config
    limb_pairs = paf_cfg.get("limb_pairs", [])

    skeleton_priors = load_skeleton_priors("skeleton_priors_train_13kp.json" if num_k == 13 else "skeleton_priors_train.json")

    neighbors = defaultdict(list)
    for l_idx, (u, v, _) in enumerate(limb_pairs):
        neighbors[u].append((v, l_idx, True))
        neighbors[v].append((u, l_idx, False))

    extra_edges = [(2, 5), (0, 5), (10, 5)] if num_k == 13 else [(2, 5), (0, 5), (11, 5)]
    for u, v in extra_edges:
        neighbors[u].append((v, -1, True))
        neighbors[v].append((u, -1, False))

    instances = []
    for c_idx, (cx, cy) in enumerate(centers):
        other_centers = [other for other in centers if not (np.isclose(other[0], cx) and np.isclose(other[1], cy))]
        foot_mask = voronoi_masks[c_idx]

        candidates_per_kp = {}
        for kp_idx in range(num_k):
            c_peaks = all_peaks.get(kp_idx, [])
            valid_cands = []
            max_r = 22.0 if (num_k == 16 and kp_idx == 15) else max_radius

            for p in c_peaks:
                dist = np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2)
                if other_centers:
                    dist_other = min(np.sqrt((p[0] - oc[0])**2 + (p[1] - oc[1])**2) for oc in other_centers)
                    if dist >= dist_other:
                        continue
                if dist <= max_r and p[2] >= peak_threshold:
                    valid_cands.append(p)

            valid_cands.sort(key=lambda item: item[2], reverse=True)
            valid_cands = valid_cands[:3]

            if not valid_cands:
                fallback_peak = None
                best_fb_conf = 0.0
                max_fb_r = 25.0 if kp_idx == 15 else 22.0
                for p in c_peaks:
                    dist = np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2)
                    if other_centers:
                        dist_other = min(np.sqrt((p[0] - oc[0])**2 + (p[1] - oc[1])**2) for oc in other_centers)
                        if dist >= dist_other:
                            continue
                    if dist <= max_fb_r and p[2] > best_fb_conf:
                        best_fb_conf = p[2]
                        fallback_peak = p

                if fallback_peak is not None and best_fb_conf > 0.12:
                    valid_cands = [fallback_peak]
                else:
                    dist_grid_sq = (xx - cx)**2 + (yy - cy)**2
                    foot_bounded_mask = foot_mask & (dist_grid_sq <= max_fb_r**2)
                    masked_patch = np.where(foot_bounded_mask, heatmaps[kp_idx], 0.0)
                    local_max = float(masked_patch.max()) if masked_patch.size > 0 else 0.0
                    if local_max > 0.12:
                        py_l, px_l = np.unravel_index(np.argmax(masked_patch), masked_patch.shape)
                        valid_cands.append((float(px_l), float(py_l), local_max))

            candidates_per_kp[kp_idx] = valid_cands

        scale_anchors = [5, 0, 10, 2] if num_k == 13 else [5, 6, 0, 11, 2]
        anchor_coords = [candidates_per_kp[k][0][:2] for k in scale_anchors if candidates_per_kp[k]]
        if len(anchor_coords) >= 2:
            pts_arr = np.array(anchor_coords)
            foot_scale = float(max(10.0, np.hypot(pts_arr[:, 0].max() - pts_arr[:, 0].min(),
                                                 pts_arr[:, 1].max() - pts_arr[:, 1].min()) * 1.3))
        else:
            foot_scale = 22.0

        current_selection = {k: (0 if candidates_per_kp[k] else -1) for k in range(num_k)}

        if pafs is not None and use_skeleton:
            for _ in range(max_iter):
                changed = False
                for k in range(num_k):
                    cands = candidates_per_kp[k]
                    if len(cands) <= 1:
                        continue

                    best_idx = current_selection[k]
                    best_score = -1e9

                    for cand_idx, p_cand in enumerate(cands):
                        conf = p_cand[2]
                        dist_to_c = np.hypot(p_cand[0] - cx, p_cand[1] - cy)
                        if k in PERIPHERAL_KPS:
                            unary = w_conf * conf
                        elif k in LEG_KPS:
                            unary = w_conf * conf - 0.020 * dist_to_c
                        else:
                            unary = w_conf * conf - 0.015 * dist_to_c

                        pairwise_score = 0.0
                        for nbr, limb_idx, is_from in neighbors[k]:
                            nbr_sel_idx = current_selection[nbr]
                            if nbr_sel_idx < 0 or not candidates_per_kp[nbr]:
                                continue
                            p_nbr = candidates_per_kp[nbr][nbr_sel_idx]
                            if p_nbr[2] < 0.30:
                                continue

                            paf_score = 0.0
                            if limb_idx >= 0:
                                paf_x = pafs[limb_idx * 2]
                                paf_y = pafs[limb_idx * 2 + 1]
                                pt1 = (p_cand[0], p_cand[1]) if is_from else (p_nbr[0], p_nbr[1])
                                pt2 = (p_nbr[0], p_nbr[1]) if is_from else (p_cand[0], p_cand[1])
                                paf_val = compute_paf_score(paf_x, paf_y, pt1, pt2, num_samples=6)
                                paf_score = w_paf * paf_val

                            geom_pen = 0.0
                            prior_key = f"{k}_{nbr}" if is_from else f"{nbr}_{k}"
                            if prior_key in skeleton_priors:
                                prior = skeleton_priors[prior_key]
                                mu_d = prior["mean_dist"]
                                std_d = max(0.08, prior["std_dist"])
                                actual_dist = np.hypot(p_cand[0] - p_nbr[0], p_cand[1] - p_nbr[1])
                                actual_norm = actual_dist / foot_scale
                                z_score = abs(actual_norm - mu_d) / std_d
                                if z_score > 1.5:
                                    geom_pen = min(2.0, ((z_score - 1.5) ** 2) * 0.5)

                            eff_w_geom = 0.05 if k in (1, 10) else w_geom
                            nbr_weight = min(1.0, p_nbr[2])
                            pairwise_score += nbr_weight * (paf_score - eff_w_geom * geom_pen)

                        total_cand_score = unary + pairwise_score
                        if total_cand_score > best_score:
                            best_score = total_cand_score
                            best_idx = cand_idx

                    if best_idx != current_selection[k]:
                        current_selection[k] = best_idx
                        changed = True

                if not changed:
                    break

        inst_kps = []
        scores = []
        for k in range(num_k):
            sel_idx = current_selection[k]
            if sel_idx >= 0 and candidates_per_kp[k]:
                p = candidates_per_kp[k][sel_idx]
                inst_kps.append([float(p[0]), float(p[1]), float(p[2])])
                scores.append(float(p[2]))
            else:
                inst_kps.append([0.0, 0.0, 0.0])

        valid_kps = [k for k in inst_kps if k[2] > 0.15]
        if len(valid_kps) < min_keypoints:
            continue

        left_votes = []
        right_votes = []
        for kx, ky, kv in valid_kps:
            px, py = int(np.clip(round(kx), 0, W - 1)), int(np.clip(round(ky), 0, H - 1))
            left_votes.append(class_probs[0, py, px])
            right_votes.append(class_probs[1, py, px])

        icx, icy = int(np.clip(round(cx), 0, W - 1)), int(np.clip(round(cy), 0, H - 1))
        r_c = 4
        center_patch_left = class_probs[0, max(0, icy-r_c):min(H, icy+r_c+1), max(0, icx-r_c):min(W, icx+r_c+1)]
        center_patch_right = class_probs[1, max(0, icy-r_c):min(H, icy+r_c+1), max(0, icx-r_c):min(W, icx+r_c+1)]
        if center_patch_left.size > 0:
            left_votes.append(float(np.mean(center_patch_left)))
            right_votes.append(float(np.mean(center_patch_right)))

        mean_left = np.mean(left_votes) if left_votes else 0.5
        mean_right = np.mean(right_votes) if right_votes else 0.5

        pred_cls = 0 if mean_left > mean_right else 1
        conf = float(max(mean_left, mean_right))

        instances.append({
            'class_id': pred_cls,
            'class_confidence': conf,
            'keypoints': inst_kps,
            'score': float(np.mean(scores)) if scores else 0.0,
            'center': (float(cx), float(cy)),
            'margin': float(mean_left - mean_right)
        })

    if len(instances) == 2 and instances[0]['class_id'] == instances[1]['class_id']:
        if instances[0]['margin'] > instances[1]['margin']:
            instances[0]['class_id'] = 0
            instances[1]['class_id'] = 1
        else:
            instances[0]['class_id'] = 1
            instances[1]['class_id'] = 0

    instances = sorted(instances, key=lambda i: i['score'], reverse=True)
    return instances[:max_instances]


def evaluate_validation_split():
    base_dir = Path(__file__).resolve().parent.parent
    val_img_dir = Path("dataset/merged_v3_v4_13kp/valid/images")
    val_lbl_dir = Path("dataset/merged_v3_v4_13kp/valid/labels")
    ckpt_path = base_dir / "outputs" / "m1v2_13kp_training" / "checkpoints" / "best.pth"
    out_dir = base_dir / "outputs" / "validation_13kp_audit_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 13-Keypoint Validation Split Rigorous Audit & Comparison")
    print("=" * 80)
    print(f"  Images Directory : {val_img_dir}")
    print(f"  Labels Directory : {val_lbl_dir}")
    print(f"  Checkpoint Path  : {ckpt_path}")
    print(f"  Output Directory : {out_dir}")
    print("=" * 80)

    device = torch.device('cpu')
    model = load_model(ckpt_path, device=device)

    img_files = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"Total Validation Images: {len(img_files)}")

    # Metric accumulators: per keypoint (13), thresholds 0.20, 0.10, 0.05
    # For Mode A (Raw peaks), Mode B (Prod Decoder), Mode C (Centroid penalty disabled on medial/lat)
    modes = ['A_raw_peaks', 'B_prod_decoder', 'C_relaxed_decoder']
    thresholds = [0.20, 0.10, 0.05]

    stats = {
        m: {
            th: {k: {'correct': 0, 'total': 0} for k in range(13)}
            for th in thresholds
        }
        for m in modes
    }

    # Verified medial/lateral indices to exclude from centroid penalty in Mode C:
    # 0, 1, 2, 3 (ball_med), 4 (ball_lat), 6 (arch_med), 7 (midfoot_lat), 8 (mall_med), 9 (mall_lat), 10 (toe_tip)
    relaxed_indices = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10}

    vis_samples_saved = 0
    max_vis = 4

    for idx, img_path in enumerate(img_files):
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        gt_instances = parse_ground_truth(lbl_path, orig_w, orig_h)
        if len(gt_instances) == 0:
            continue

        # Forward
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            outputs = model(img_tensor)
            heatmaps = outputs['heatmaps'][0].cpu().numpy()  # [13, 64, 64]
            pafs = outputs['pafs'][0].cpu().numpy()          # [32, 64, 64]
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()

        # Generate predictions for all 3 modes:
        scale_x = orig_w / 64.0
        scale_y = orig_h / 64.0

        # Mode B: Production Decoder
        raw_preds_b = decode_spatial_instances_single(
            heatmaps, cls_probs, pafs=pafs
        )
        preds_b = []
        for inst in raw_preds_b:
            inst_copy = copy.deepcopy(inst)
            inst_copy['keypoints'] = [
                (kp[0] * scale_x, kp[1] * scale_y, kp[2]) for kp in inst['keypoints']
            ]
            preds_b.append(inst_copy)

        # Mode C: Relaxed Decoder
        raw_preds_c = decode_spatial_instances_custom(
            heatmaps, cls_probs, pafs=pafs,
            disable_centroid_penalty_indices=relaxed_indices
        )
        preds_c = []
        for inst in raw_preds_c:
            inst_copy = copy.deepcopy(inst)
            inst_copy['keypoints'] = [
                (kp[0] * scale_x, kp[1] * scale_y, kp[2]) for kp in inst['keypoints']
            ]
            preds_c.append(inst_copy)

        # Mode A: Raw Heatmap Peaks
        # For each GT instance, find the closest raw peak on 64x64 mapped to original image
        scale_x = orig_w / 64.0
        scale_y = orig_h / 64.0

        raw_peaks_per_kp = {}
        for k in range(13):
            pks = detect_peaks(heatmaps[k], threshold=0.08, nms_window=3, subpixel=True)
            if not pks:
                py, px = np.unravel_index(np.argmax(heatmaps[k]), heatmaps[k].shape)
                pks = [(float(px), float(py), float(heatmaps[k][py, px]))]
            raw_peaks_per_kp[k] = [(p[0] * scale_x, p[1] * scale_y, p[2]) for p in pks]

        # Match predictions to GT instances
        # For Mode B and Mode C: match by class or centroid distance
        def match_instances(preds, gts):
            matches = {}
            for g_i, gt in enumerate(gts):
                best_p_i = None
                best_dist = 1e9
                for p_i, pred in enumerate(preds):
                    if p_i in matches.values():
                        continue
                    # Match by class first if possible
                    if pred['class_id'] == gt['class_id'] or len(preds) == 1:
                        # Compute distance between visible keypoints
                        d_sum = 0.0
                        d_cnt = 0
                        for k in range(13):
                            if gt['keypoints'][k][2] > 0:
                                d_sum += np.hypot(
                                    pred['keypoints'][k][0] - gt['keypoints'][k][0],
                                    pred['keypoints'][k][1] - gt['keypoints'][k][1]
                                )
                                d_cnt += 1
                        avg_d = d_sum / max(1, d_cnt)
                        if avg_d < best_dist:
                            best_dist = avg_d
                            best_p_i = p_i
                if best_p_i is not None:
                    matches[g_i] = best_p_i
                elif len(preds) > 0:
                    # fallback to closest available
                    matches[g_i] = 0
            return matches

        matches_b = match_instances(preds_b, gt_instances)
        matches_c = match_instances(preds_c, gt_instances)

        # Mode A: For each GT instance, select the raw peak closest to the GT keypoint
        # (This establishes the upper bound accuracy of the raw heatmap representation)
        raw_pred_per_gt = {}
        for g_i, gt in enumerate(gt_instances):
            raw_kps = []
            for k in range(13):
                pks = raw_peaks_per_kp[k]
                gx, gy, gv = gt['keypoints'][k]
                if gv > 0:
                    best_pk = min(pks, key=lambda p: np.hypot(p[0] - gx, p[1] - gy))
                    raw_kps.append(best_pk)
                else:
                    raw_kps.append(pks[0])
            raw_pred_per_gt[g_i] = raw_kps

        # Accumulate PCK metrics per GT instance
        for g_i, gt in enumerate(gt_instances):
            bbox_diag = gt['bbox_diag']
            if bbox_diag < 1e-4:
                continue

            for k in range(13):
                gx, gy, gv = gt['keypoints'][k]
                if gv == 0:
                    # STRICT REQUIREMENT: EXCLUDE v=0 KEYPOINTS
                    continue

                for th in thresholds:
                    norm_limit = th * bbox_diag

                    # Mode A: Raw peaks
                    pk_a = raw_pred_per_gt[g_i][k]
                    dist_a = np.hypot(pk_a[0] - gx, pk_a[1] - gy)
                    stats['A_raw_peaks'][th][k]['total'] += 1
                    if dist_a < norm_limit:
                        stats['A_raw_peaks'][th][k]['correct'] += 1

                    # Mode B: Production Decoder
                    if g_i in matches_b and matches_b[g_i] < len(preds_b):
                        pk_b = preds_b[matches_b[g_i]]['keypoints'][k]
                        dist_b = np.hypot(pk_b[0] - gx, pk_b[1] - gy)
                        stats['B_prod_decoder'][th][k]['total'] += 1
                        if dist_b < norm_limit:
                            stats['B_prod_decoder'][th][k]['correct'] += 1
                    else:
                        stats['B_prod_decoder'][th][k]['total'] += 1

                    # Mode C: Relaxed Decoder
                    if g_i in matches_c and matches_c[g_i] < len(preds_c):
                        pk_c = preds_c[matches_c[g_i]]['keypoints'][k]
                        dist_c = np.hypot(pk_c[0] - gx, pk_c[1] - gy)
                        stats['C_relaxed_decoder'][th][k]['total'] += 1
                        if dist_c < norm_limit:
                            stats['C_relaxed_decoder'][th][k]['correct'] += 1
                    else:
                        stats['C_relaxed_decoder'][th][k]['total'] += 1

        # Visualization for the first 4 samples
        if vis_samples_saved < max_vis:
            vis_samples_saved += 1
            render_comparison_panel(
                img_bgr, gt_instances, raw_pred_per_gt, preds_b, preds_c,
                matches_b, matches_c, out_dir / f"val_sample_{vis_samples_saved}_comparison.jpg"
            )

        if (idx + 1) % 50 == 0 or idx == len(img_files) - 1:
            print(f"  Processed {idx + 1}/{len(img_files)} validation images...")

    # Compute and format final tables
    report = {
        'total_images': len(img_files),
        'metrics': {}
    }

    print("\n" + "=" * 110)
    print("📊 VALIDATION SPLIT EVALUATION (EXCLUDING v=0 KEYPOINTS)")
    print("=" * 110)

    for th in thresholds:
        print(f"\n--- PCK @ {th:.2f} ---")
        header = f" {'KP':<3} | {'Keypoint Name':<18} | {'Count (v>0)':<12} | {'A: Raw Peaks':<15} | {'B: Prod Decoder':<16} | {'C: Relaxed (No Penalty)':<24} | {'Δ (C - B)':<10}"
        print(header)
        print("-" * len(header))

        tot_samples = 0
        tot_corr_a = 0
        tot_corr_b = 0
        tot_corr_c = 0

        for k in range(13):
            name = KEYPOINT_NAMES_13[k]
            cnt = stats['A_raw_peaks'][th][k]['total']
            tot_samples += cnt

            c_a = stats['A_raw_peaks'][th][k]['correct']
            c_b = stats['B_prod_decoder'][th][k]['correct']
            c_c = stats['C_relaxed_decoder'][th][k]['correct']

            tot_corr_a += c_a
            tot_corr_b += c_b
            tot_corr_c += c_c

            pck_a = (c_a / cnt * 100) if cnt > 0 else 0.0
            pck_b = (c_b / cnt * 100) if cnt > 0 else 0.0
            pck_c = (c_c / cnt * 100) if cnt > 0 else 0.0
            diff = pck_c - pck_b

            diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
            print(f" {k:<3} | {name:<18} | {cnt:<12} | {pck_a:>6.2f}%         | {pck_b:>6.2f}%          | {pck_c:>6.2f}%                 | {diff_str:<10}")

        ov_a = (tot_corr_a / tot_samples * 100) if tot_samples > 0 else 0.0
        ov_b = (tot_corr_b / tot_samples * 100) if tot_samples > 0 else 0.0
        ov_c = (tot_corr_c / tot_samples * 100) if tot_samples > 0 else 0.0
        diff_ov = ov_c - ov_b
        diff_ov_str = f"+{diff_ov:.2f}%" if diff_ov >= 0 else f"{diff_ov:.2f}%"

        print("-" * len(header))
        print(f" {'ALL':<3} | {'OVERALL PCK':<18} | {tot_samples:<12} | {ov_a:>6.2f}%         | {ov_b:>6.2f}%          | {ov_c:>6.2f}%                 | {diff_ov_str:<10}")

        report['metrics'][f'PCK_{th}'] = {
            'overall': {'A': ov_a, 'B': ov_b, 'C': ov_c, 'diff': diff_ov, 'total_points': tot_samples},
            'per_keypoint': {
                KEYPOINT_NAMES_13[k]: {
                    'count': stats['A_raw_peaks'][th][k]['total'],
                    'PCK_A': (stats['A_raw_peaks'][th][k]['correct'] / max(1, stats['A_raw_peaks'][th][k]['total'])) * 100,
                    'PCK_B': (stats['B_prod_decoder'][th][k]['correct'] / max(1, stats['B_prod_decoder'][th][k]['total'])) * 100,
                    'PCK_C': (stats['C_relaxed_decoder'][th][k]['correct'] / max(1, stats['C_relaxed_decoder'][th][k]['total'])) * 100,
                }
                for k in range(13)
            }
        }

    with open(out_dir / "audit_comparison_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n🎉 Validation audit report saved to: {out_dir / 'audit_comparison_report.json'}")
    print(f"🖼️  Visual comparison panels saved to: {out_dir}")


def render_comparison_panel(img_bgr, gt_instances, raw_pred_per_gt, preds_b, preds_c, matches_b, matches_c, out_path):
    H, W = img_bgr.shape[:2]
    vis_h, vis_w = 512, 512

    def draw_skeleton(base_img, kps_list, title, color_theme):
        canvas = cv2.resize(base_img.copy(), (vis_w, vis_h))
        sx = vis_w / float(W)
        sy = vis_h / float(H)

        for (u, v) in SKELETON_LIMBS_13:
            for kps in kps_list:
                p1 = kps[u]
                p2 = kps[v]
                # Check visibility if present
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
                    # Highlight medial/lateral points (3, 4, 6, 7) in yellow/cyan
                    if k_i in [3, 4, 6, 7]:
                        cv2.circle(canvas, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, pt, 4, color_theme['point'], -1, cv2.LINE_AA)
                        cv2.circle(canvas, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)

        # Header banner
        cv2.rectangle(canvas, (0, 0), (vis_w, 32), (30, 30, 30), -1)
        cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    # 1. Ground Truth (only v > 0)
    gt_kps = [gt['keypoints'] for gt in gt_instances]
    panel_gt = draw_skeleton(img_bgr, gt_kps, "1. Ground Truth (v > 0)", {'limb': (0, 255, 0), 'point': (0, 220, 0)})

    # 2. Raw Heatmap Peaks
    raw_kps = [raw_pred_per_gt[g_i] for g_i in range(len(gt_instances))]
    panel_raw = draw_skeleton(img_bgr, raw_kps, "2. Raw Heatmap Peaks", {'limb': (255, 100, 0), 'point': (255, 150, 0)})

    # 3. Production Decoder (B)
    b_kps = [p['keypoints'] for p in preds_b]
    panel_b = draw_skeleton(img_bgr, b_kps, "3. Current Decoder B (Centroid Penalty)", {'limb': (0, 140, 255), 'point': (0, 165, 255)})

    # 4. Relaxed Decoder (C)
    c_kps = [p['keypoints'] for p in preds_c]
    panel_c = draw_skeleton(img_bgr, c_kps, "4. Relaxed Decoder C (No Med/Lat Penalty)", {'limb': (255, 0, 200), 'point': (255, 50, 220)})

    # 2x2 grid
    top_row = np.hstack([panel_gt, panel_raw])
    bot_row = np.hstack([panel_b, panel_c])
    combined = np.vstack([top_row, bot_row])

    cv2.imwrite(str(out_path), combined)


if __name__ == "__main__":
    evaluate_validation_split()
