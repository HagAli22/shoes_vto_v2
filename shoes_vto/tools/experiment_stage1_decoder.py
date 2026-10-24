"""
Stage 1: Decoder-Only Experiment on Validation Split (253 images, 4,023 valid GT keypoints)

Tests:
1. Baseline (Existing production decoder):
   PERIPHERAL_KPS = {0, 1, 2, 8, 9, 10}, LEG_KPS = {12}
2. Variant A (Correct LEG_KPS only):
   LEG_KPS = set(), achilles (12) moved to standard 0.015 penalty
   PERIPHERAL_KPS = {0, 1, 2, 8, 9, 10}
3. Variant B (Anatomical Medial/Lateral Perimeter + LEG_KPS=set()):
   PERIPHERAL_KPS = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10}, LEG_KPS = set()
4. Variant C (Full Perimeter including Achilles + LEG_KPS=set()):
   PERIPHERAL_KPS = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12}, LEG_KPS = set()
5. Variant D (Variant B + gentle interior penalty 0.005 instead of 0.015):
   Interior penalty on dorsal points reduced from 0.015 to 0.005

Computes exact per-keypoint PCK@0.20, PCK@0.10, and PCK@0.05, strictly excluding v=0.
Saves side-by-side visualization panels.
"""

import sys
import json
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
from collections import defaultdict

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
    compute_paf_score,
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
    (10, 0), (0, 2), (2, 1), (3, 4), (3, 5), (4, 5),
    (2, 6), (2, 7), (6, 5), (7, 5), (5, 11), (11, 8),
    (11, 9), (8, 12), (9, 12), (12, 1)
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


def decode_spatial_instances_configurable(
    heatmaps,
    class_probs,
    pafs=None,
    peripheral_kps={0, 1, 2, 8, 9, 10},
    leg_kps={12},
    interior_penalty=0.015,
    paf_config=None,
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
    use_skeleton=True
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
            for p in all_peaks.get(kp_idx, []):
                if p[2] > 0.35:
                    anchor_pts.append((p[0], p[1], kp_idx, p[2]))

    if len(anchor_pts) < 4:
        for kp_idx in range(num_k):
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
            max_r = max_radius

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
                max_fb_r = 22.0
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
                        if k in peripheral_kps:
                            unary = w_conf * conf
                        elif k in leg_kps:
                            unary = w_conf * conf - 0.020 * dist_to_c
                        else:
                            unary = w_conf * conf - interior_penalty * dist_to_c

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


def run_stage1_experiments():
    base_dir = Path(__file__).resolve().parent.parent
    val_img_dir = Path("dataset/merged_v3_v4_13kp/valid/images")
    val_lbl_dir = Path("dataset/merged_v3_v4_13kp/valid/labels")
    ckpt_path = base_dir / "outputs" / "m1v2_13kp_training" / "checkpoints" / "best.pth"
    out_dir = base_dir / "outputs" / "stage1_decoder_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("👟 STAGE 1 DECODER-ONLY EXPERIMENT (VALIDATION SPLIT ONLY)")
    print("=" * 90)
    print(f"  Validation Images : {val_img_dir}")
    print(f"  Checkpoint        : {ckpt_path}")
    print(f"  Output Directory  : {out_dir}")
    print("=" * 90)

    device = torch.device('cpu')
    model = load_model(ckpt_path, device=device)

    img_files = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"Total Validation Images to Process: {len(img_files)}")

    # Configurations to test:
    configs = {
        'Baseline': {
            'desc': 'Current: PERIPHERAL={0,1,2,8,9,10}, LEG={12}',
            'peripheral': {0, 1, 2, 8, 9, 10},
            'leg': {12},
            'interior_pen': 0.015
        },
        'Var_A_FixLegOnly': {
            'desc': 'Fix LEG={}: achilles(12) receives 0.015 penalty, PERIPHERAL={0,1,2,8,9,10}',
            'peripheral': {0, 1, 2, 8, 9, 10},
            'leg': set(),
            'interior_pen': 0.015
        },
        'Var_B_AnatomicalPerimeter': {
            'desc': 'PERIPHERAL={0,1,2,3,4,6,7,8,9,10} (Medial/Lateral Perimeter), LEG={}',
            'peripheral': {0, 1, 2, 3, 4, 6, 7, 8, 9, 10},
            'leg': set(),
            'interior_pen': 0.015
        },
        'Var_C_FullPerimeterWithAchilles': {
            'desc': 'PERIPHERAL={0,1,2,3,4,6,7,8,9,10,12} (+Achilles), LEG={}',
            'peripheral': {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12},
            'leg': set(),
            'interior_pen': 0.015
        },
        'Var_D_PerimeterAndGentleInterior': {
            'desc': 'Var_B + Gentle Interior Penalty 0.005 (on instep:5, throat:11, achilles:12)',
            'peripheral': {0, 1, 2, 3, 4, 6, 7, 8, 9, 10},
            'leg': set(),
            'interior_pen': 0.005
        }
    }

    thresholds = [0.20, 0.10, 0.05]
    stats = {
        cfg_name: {
            th: {k: {'correct': 0, 'total': 0} for k in range(13)}
            for th in thresholds
        }
        for cfg_name in configs
    }

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
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            pafs = outputs['pafs'][0].cpu().numpy()
            cls_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()

        scale_x = orig_w / 64.0
        scale_y = orig_h / 64.0

        # Decode for all configs
        decoded_per_cfg = {}
        for cfg_name, cfg in configs.items():
            raw_insts = decode_spatial_instances_configurable(
                heatmaps, cls_probs, pafs=pafs,
                peripheral_kps=cfg['peripheral'],
                leg_kps=cfg['leg'],
                interior_penalty=cfg['interior_pen']
            )
            scaled_insts = []
            for inst in raw_insts:
                inst_c = copy.deepcopy(inst)
                inst_c['keypoints'] = [
                    (kp[0] * scale_x, kp[1] * scale_y, kp[2]) for kp in inst['keypoints']
                ]
                scaled_insts.append(inst_c)
            decoded_per_cfg[cfg_name] = scaled_insts

        # Match to GT instances
        matches_per_cfg = {}
        for cfg_name, preds in decoded_per_cfg.items():
            matches = {}
            for g_i, gt in enumerate(gt_instances):
                best_p_i = None
                best_dist = 1e9
                for p_i, pred in enumerate(preds):
                    if p_i in matches.values():
                        continue
                    if pred['class_id'] == gt['class_id'] or len(preds) == 1:
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
                    matches[g_i] = 0
            matches_per_cfg[cfg_name] = matches

        # Accumulate PCK per GT instance (EXCLUDING v=0)
        for g_i, gt in enumerate(gt_instances):
            bbox_diag = gt['bbox_diag']
            if bbox_diag < 1e-4:
                continue

            for k in range(13):
                gx, gy, gv = gt['keypoints'][k]
                if gv == 0:
                    # STRICT REQUIREMENT: EXCLUDE v=0
                    continue

                for th in thresholds:
                    norm_limit = th * bbox_diag

                    for cfg_name, preds in decoded_per_cfg.items():
                        stats[cfg_name][th][k]['total'] += 1
                        m_map = matches_per_cfg[cfg_name]
                        if g_i in m_map and m_map[g_i] < len(preds):
                            pk = preds[m_map[g_i]]['keypoints'][k]
                            dist = np.hypot(pk[0] - gx, pk[1] - gy)
                            if dist < norm_limit:
                                stats[cfg_name][th][k]['correct'] += 1

        # Save side-by-side visualization panels for 4 samples
        if vis_samples_saved < max_vis:
            vis_samples_saved += 1
            render_stage1_vis(
                img_bgr, gt_instances, decoded_per_cfg,
                out_dir / f"stage1_val_sample_{vis_samples_saved}.jpg"
            )

        if (idx + 1) % 50 == 0 or idx == len(img_files) - 1:
            print(f"  Processed {idx + 1}/{len(img_files)} validation images...")

    # Format comparison summary
    results_json = {
        'total_images': len(img_files),
        'metrics': {}
    }

    print("\n" + "=" * 125)
    print("🏆 STAGE 1: DECODER EXPERIMENT SUMMARY TABLE (VALIDATION SPLIT, STRICTLY v > 0)")
    print("=" * 125)

    for th in thresholds:
        print(f"\n==================== PCK @ {th:.2f} ====================")
        header = f" {'KP':<3} | {'Keypoint Name':<18} | {'Count':<6} | {'Baseline':<10} | {'Var A (Leg)':<12} | {'Var B (Perim)':<14} | {'Var C (+Ach)':<14} | {'Var D (Soft)':<14} | {'Best Δ':<8}"
        print(header)
        print("-" * len(header))

        tot_samples = 0
        tot_corr = {c: 0 for c in configs}

        for k in range(13):
            name = KEYPOINT_NAMES_13[k]
            cnt = stats['Baseline'][th][k]['total']
            tot_samples += cnt

            row_pcks = {}
            for c in configs:
                c_val = stats[c][th][k]['correct']
                tot_corr[c] += c_val
                pck_val = (c_val / cnt * 100) if cnt > 0 else 0.0
                row_pcks[c] = pck_val

            base_pck = row_pcks['Baseline']
            best_c = max(configs.keys(), key=lambda c: row_pcks[c])
            best_diff = row_pcks[best_c] - base_pck
            diff_str = f"+{best_diff:.2f}%" if best_diff >= 0 else f"{best_diff:.2f}%"

            print(f" {k:<3} | {name:<18} | {cnt:<6} | {row_pcks['Baseline']:>6.2f}%   | {row_pcks['Var_A_FixLegOnly']:>6.2f}%     | {row_pcks['Var_B_AnatomicalPerimeter']:>6.2f}%       | {row_pcks['Var_C_FullPerimeterWithAchilles']:>6.2f}%       | {row_pcks['Var_D_PerimeterAndGentleInterior']:>6.2f}%       | {diff_str:<8}")

        print("-" * len(header))
        ov_pcks = {}
        for c in configs:
            ov = (tot_corr[c] / tot_samples * 100) if tot_samples > 0 else 0.0
            ov_pcks[c] = ov
        best_ov_c = max(configs.keys(), key=lambda c: ov_pcks[c])
        best_ov_diff = ov_pcks[best_ov_c] - ov_pcks['Baseline']
        diff_ov_str = f"+{best_ov_diff:.2f}%" if best_ov_diff >= 0 else f"{best_ov_diff:.2f}%"

        print(f" {'ALL':<3} | {'OVERALL PCK':<18} | {tot_samples:<6} | {ov_pcks['Baseline']:>6.2f}%   | {ov_pcks['Var_A_FixLegOnly']:>6.2f}%     | {ov_pcks['Var_B_AnatomicalPerimeter']:>6.2f}%       | {ov_pcks['Var_C_FullPerimeterWithAchilles']:>6.2f}%       | {ov_pcks['Var_D_PerimeterAndGentleInterior']:>6.2f}%       | {diff_ov_str:<8}")

        results_json['metrics'][f'PCK_{th}'] = {
            'overall': {c: ov_pcks[c] for c in configs},
            'best_config': best_ov_c,
            'best_improvement': best_ov_diff,
            'per_keypoint': {
                KEYPOINT_NAMES_13[k]: {
                    c: (stats[c][th][k]['correct'] / max(1, stats[c][th][k]['total'])) * 100
                    for c in configs
                }
                for k in range(13)
            }
        }

    with open(out_dir / "stage1_experiment_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n🎉 Stage 1 Experiment Metrics saved to: {out_dir / 'stage1_experiment_metrics.json'}")
    print(f"🖼️  Stage 1 Side-by-side Visualizations saved to: {out_dir}")
    return results_json


def render_stage1_vis(img_bgr, gt_instances, decoded_per_cfg, out_path):
    H, W = img_bgr.shape[:2]
    vis_h, vis_w = 400, 400

    def draw_panel(base_img, kps_list, title, color_theme):
        canvas = cv2.resize(base_img.copy(), (vis_w, vis_h))
        sx = vis_w / float(W)
        sy = vis_h / float(H)

        for (u, v) in SKELETON_LIMBS_13:
            for kps in kps_list:
                p1, p2 = kps[u], kps[v]
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
                    # Highlight medial/lateral points (3, 4, 6, 7) in yellow
                    if k_i in [3, 4, 6, 7]:
                        cv2.circle(canvas, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)
                    elif k_i in [1, 2, 12]:
                        cv2.circle(canvas, pt, 5, (255, 0, 255), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, pt, 4, color_theme['point'], -1, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (vis_w, 28), (35, 35, 35), -1)
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    gt_kps = [gt['keypoints'] for gt in gt_instances]
    p_gt = draw_panel(img_bgr, gt_kps, "GT (v > 0)", {'limb': (0, 255, 0), 'point': (0, 220, 0)})

    base_kps = [p['keypoints'] for p in decoded_per_cfg['Baseline']]
    p_base = draw_panel(img_bgr, base_kps, "Baseline (LEG={12}, Narrow)", {'limb': (0, 140, 255), 'point': (0, 165, 255)})

    var_b_kps = [p['keypoints'] for p in decoded_per_cfg['Var_B_AnatomicalPerimeter']]
    p_var_b = draw_panel(img_bgr, var_b_kps, "Var B: Med/Lat Perimeter", {'limb': (255, 0, 200), 'point': (255, 50, 220)})

    var_d_kps = [p['keypoints'] for p in decoded_per_cfg['Var_D_PerimeterAndGentleInterior']]
    p_var_d = draw_panel(img_bgr, var_d_kps, "Var D: Soft Interior Penalty", {'limb': (50, 220, 255), 'point': (0, 200, 255)})

    row = np.hstack([p_gt, p_base, p_var_b, p_var_d])
    cv2.imwrite(str(out_path), row)


if __name__ == '__main__':
    run_stage1_experiments()

