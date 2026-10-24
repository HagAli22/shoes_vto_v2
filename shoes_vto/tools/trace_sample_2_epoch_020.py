"""
Complete Step-by-Step Inference Trace for Sample 2 of epoch_020_vis.jpg
Image: -tattoo-feettattoo_jpg.rf.3c9a035c4e9db19f13c1b07001e089be.jpg
Checkpoint: shoes_vto/shoes_vto/outputs/m1v2_13kp_shuffled_v4_1_training/checkpoints/best.pth
"""

import os
import sys
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
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
from datasets.yolo_dataset import YOLOFootDataset

CKPT_PATH = BASE_DIR / "shoes_vto" / "outputs" / "m1v2_13kp_shuffled_v4_1_training" / "checkpoints" / "best.pth"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "shuffled_v4_1_13kp"
TRACE_OUT_DIR = BASE_DIR / "outputs" / "trace_epoch_020_sample2"
TRACE_OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\Matrix Store\.gemini\antigravity\brain\729e2265-efb4-4e0a-8890-a75ed53005c3")

KEYPOINT_NAMES_13 = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "instep_top", "arch_medial", "midfoot_lateral", "malleolus_medial",
    "malleolus_lateral", "toe_tip", "throat", "achilles"
]

SKELETON_LIMBS_13 = [
    (10, 0), (0, 2), (2, 1), (3, 4), (3, 5), (4, 5),
    (2, 6), (2, 7), (6, 5), (7, 5), (5, 11), (11, 8),
    (11, 9), (8, 12), (9, 12), (12, 1)
]

KP_COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (255, 0, 0), (255, 0, 128), (128, 0, 255), (0, 165, 255), (147, 20, 255),
    (50, 205, 50), (208, 224, 64), (180, 105, 255)
]


def run_complete_trace_sample2():
    print("=" * 80)
    print("🔬 RUNNING COMPLETE INFERENCE TRACE FOR SAMPLE 2 IN epoch_020_vis.jpg")
    print("=" * 80)

    val_dataset = YOLOFootDataset(
        images_dir=DATASET_ROOT / "valid" / "images",
        labels_dir=DATASET_ROOT / "valid" / "labels",
        img_size=256,
        augment=False,
        num_keypoints=13
    )
    sample_2 = val_dataset[2]
    img_name = val_dataset.image_files[2].name
    print(f"Sample 2 Image Name: {img_name}")

    raw_img_path = DATASET_ROOT / "valid" / "images" / img_name
    raw_bgr = cv2.imread(str(raw_img_path))
    orig_h, orig_w = raw_bgr.shape[:2]
    print(f"Original Resolution: {orig_w}x{orig_h}")

    # Stage 1: Original and preprocessed
    preprocessed_np = (sample_2['image'].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    preprocessed_bgr = cv2.cvtColor(preprocessed_np, cv2.COLOR_RGB2BGR)

    p_stage1_orig = TRACE_OUT_DIR / "s2_stage1_original_image.jpg"
    p_stage1_prep = TRACE_OUT_DIR / "s2_stage1_preprocessed_256.jpg"
    cv2.imwrite(str(p_stage1_orig), raw_bgr)
    cv2.imwrite(str(p_stage1_prep), preprocessed_bgr)

    # Load Model
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    model = ARShoeM1v2(num_keypoints=13, num_classes=2, num_limbs=16, pretrained=False)
    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()

    # Forward pass
    img_tensor = sample_2['image'].unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        heatmaps = outputs['heatmaps'][0].numpy()
        pafs = outputs['pafs'][0].numpy()
        class_probs = outputs.get('class_probs', outputs.get('class'))[0].numpy()

    print(f"Heatmaps range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    print(f"PAFs range: [{pafs.min():.4f}, {pafs.max():.4f}]")
    print(f"Class probs range: [{class_probs.min():.4f}, {class_probs.max():.4f}]")

    # Stage 2: Individual heatmaps and composite overlay
    for k in range(13):
        hm_norm = np.clip(heatmaps[k] * 255.0, 0, 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        hm_color_large = cv2.resize(hm_color, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.putText(hm_color_large, f"{k}: {KEYPOINT_NAMES_13[k]} (max={heatmaps[k].max():.2f})", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(TRACE_OUT_DIR / f"s2_stage2_heatmap_{k:02d}_{KEYPOINT_NAMES_13[k]}.jpg"), hm_color_large)

    hm_max_all = np.max(heatmaps, axis=0)
    hm_overlay_norm = np.clip(hm_max_all * 255.0, 0, 255).astype(np.uint8)
    hm_overlay_color = cv2.applyColorMap(hm_overlay_norm, cv2.COLORMAP_JET)
    hm_overlay_large = cv2.resize(hm_overlay_color, (256, 256), interpolation=cv2.INTER_LINEAR)
    blended_hm = cv2.addWeighted(preprocessed_bgr, 0.5, hm_overlay_large, 0.5, 0)
    p_hm_overlay = TRACE_OUT_DIR / "s2_stage2_heatmap_overlay_composite.jpg"
    cv2.imwrite(str(p_hm_overlay), blended_hm)

    # Stage 3: PAF visualization
    paf_mags = [np.hypot(pafs[l * 2], pafs[l * 2 + 1]) for l in range(16)]
    paf_max_mag = np.max(paf_mags, axis=0)
    paf_norm = np.clip((paf_max_mag / (paf_max_mag.max() + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    paf_color = cv2.applyColorMap(paf_norm, cv2.COLORMAP_VIRIDIS)
    paf_large = cv2.resize(paf_color, (256, 256), interpolation=cv2.INTER_LINEAR)
    blended_paf = cv2.addWeighted(preprocessed_bgr, 0.5, paf_large, 0.5, 0)
    p_paf_vis = TRACE_OUT_DIR / "s2_stage3_paf_magnitude_overlay.jpg"
    cv2.imwrite(str(p_paf_vis), blended_paf)

    # Stage 4: Class-map visualization
    cls_left = cv2.resize(class_probs[0], (256, 256), interpolation=cv2.INTER_LINEAR)
    cls_right = cv2.resize(class_probs[1], (256, 256), interpolation=cv2.INTER_LINEAR)
    cls_vis = np.zeros((256, 256, 3), dtype=np.uint8)
    cls_vis[:, :, 0] = np.clip(cls_left * 255.0, 0, 255).astype(np.uint8)
    cls_vis[:, :, 2] = np.clip(cls_right * 255.0, 0, 255).astype(np.uint8)
    blended_cls = cv2.addWeighted(preprocessed_bgr, 0.5, cls_vis, 0.5, 0)
    cv2.putText(blended_cls, f"Left Foot max={cls_left.max():.2f} (Blue) | Right Foot max={cls_right.max():.2f} (Red)",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    p_cls_vis = TRACE_OUT_DIR / "s2_stage4_class_map_overlay.jpg"
    cv2.imwrite(str(p_cls_vis), blended_cls)

    # Decoder Step-by-Step Trace
    H, W = 64, 64
    num_k = 13
    peak_threshold = 0.10
    nms_window = 3
    subpixel = True

    # Step A: Peak Detection
    all_peaks = {}
    candidate_peaks_table = []
    cand_counter = 0

    for kp_idx in range(num_k):
        pks = detect_peaks(heatmaps[kp_idx], threshold=peak_threshold, nms_window=nms_window, subpixel=subpixel)
        all_peaks[kp_idx] = pks
        for p in pks:
            candidate_peaks_table.append({
                'candidate_id': f"C_{cand_counter:03d}",
                'kp_idx': kp_idx,
                'kp_name': KEYPOINT_NAMES_13[kp_idx],
                'raw_heatmap_x': float(p[0]),
                'raw_heatmap_y': float(p[1]),
                'confidence': float(p[2])
            })
            cand_counter += 1

    print(f"Detected {len(candidate_peaks_table)} total candidate peaks.")

    # Step B: Anchor Selection
    preferred_anchors = [5, 10, 1, 3]
    anchor_pts = []
    anchor_decisions = []

    for c in candidate_peaks_table:
        kp_idx = c['kp_idx']
        conf = c['confidence']
        is_preferred = kp_idx in preferred_anchors
        if is_preferred and conf > 0.35:
            accepted = True
            reason = f"Preferred anchor channel ({c['kp_name']}) with confidence {conf:.3f} > 0.35 threshold"
            anchor_pts.append((c['raw_heatmap_x'], c['raw_heatmap_y'], kp_idx, conf, c['candidate_id']))
        elif conf <= 0.15:
            accepted = False
            reason = f"Confidence {conf:.3f} below minimum anchor threshold (0.15)"
        elif not is_preferred:
            accepted = False
            reason = f"Channel {c['kp_name']} not in preferred anchor set [5, 10, 1, 3] (fallback condition not needed)"
        else:
            accepted = False
            reason = f"Confidence {conf:.3f} <= 0.35 threshold for preferred channel {c['kp_name']}"

        anchor_decisions.append({
            'candidate_id': c['candidate_id'],
            'kp_idx': kp_idx,
            'kp_name': c['kp_name'],
            'raw_xy': (c['raw_heatmap_x'], c['raw_heatmap_y']),
            'confidence': conf,
            'accepted_as_anchor': accepted,
            'reason': reason
        })

    print(f"Selected {len(anchor_pts)} anchor points for clustering.")

    # Step C: K-Means Clustering & Instance Count Decision
    pts_xy = np.array([[p[0], p[1]] for p in anchor_pts], dtype=np.float32)
    clustering_trace = {}

    if len(pts_xy) >= 6:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _, labels, centers = cv2.kmeans(pts_xy, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        dist_between_centers = float(np.linalg.norm(centers[0] - centers[1]))
        c0_count = int(np.sum(labels == 0))
        c1_count = int(np.sum(labels == 1))

        clustering_trace['kmeans_ran'] = True
        clustering_trace['cluster_0_center'] = [float(centers[0][0]), float(centers[0][1])]
        clustering_trace['cluster_1_center'] = [float(centers[1][0]), float(centers[1][1])]
        clustering_trace['dist_between_centers'] = dist_between_centers
        clustering_trace['c0_count'] = c0_count
        clustering_trace['c1_count'] = c1_count

        if dist_between_centers < 7.0 or c0_count < 3 or c1_count < 3:
            centers = [np.mean(pts_xy, axis=0)]
            clustering_trace['decision'] = "Rejected 2nd instance: distance < 7.0 or cluster count < 3 -> Single Instance"
        else:
            centers = sorted(centers, key=lambda c: c[0])
            clustering_trace['decision'] = f"Accepted 2 instances: distance {dist_between_centers:.2f} >= 7.0 and counts (c0={c0_count}, c1={c1_count}) >= 3"
    else:
        centers = [np.mean(pts_xy, axis=0)]
        clustering_trace['kmeans_ran'] = False
        clustering_trace['decision'] = f"Single instance: fewer than 6 anchor points ({len(pts_xy)} pts)"

    num_instances = len(centers)
    clustering_trace['final_instance_count'] = num_instances
    clustering_trace['final_centers'] = [[float(c[0]), float(c[1])] for c in centers]
    print(f"Instance clustering result: {clustering_trace['decision']}")

    # Step D: Voronoi Partition & Candidate Assignment per Instance
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

    max_radius = 22.0
    instances_trace = []
    candidate_assignment_trace = []

    paf_cfg = load_paf_config("paf_connections_13kp.yaml")
    limb_pairs = paf_cfg.get("limb_pairs", [])
    skeleton_priors = load_skeleton_priors("skeleton_priors_train_13kp.json")

    from collections import defaultdict
    neighbors = defaultdict(list)
    for l_idx, (u, v, _) in enumerate(limb_pairs):
        neighbors[u].append((v, l_idx, True))
        neighbors[v].append((u, l_idx, False))

    PERIPHERAL_KPS = {0, 1, 2, 8, 9, 10}
    LEG_KPS = set()

    for c_idx, (cx, cy) in enumerate(centers):
        candidates_per_kp = {}
        for kp_idx in range(num_k):
            valid_cands = []
            c_peaks = all_peaks.get(kp_idx, [])
            for p in c_peaks:
                px_grid, py_grid = int(np.clip(round(p[0]), 0, W - 1)), int(np.clip(round(p[1]), 0, H - 1))
                in_voronoi = bool(voronoi_masks[c_idx][py_grid, px_grid])
                dist_to_center = float(np.hypot(p[0] - cx, p[1] - cy))
                in_radius = dist_to_center <= max_radius
                conf_ok = p[2] >= peak_threshold

                cid = None
                for c in candidate_peaks_table:
                    if c['kp_idx'] == kp_idx and np.isclose(c['raw_heatmap_x'], p[0]) and np.isclose(c['raw_heatmap_y'], p[1]):
                        cid = c['candidate_id']
                        break

                accepted_for_inst = in_voronoi and in_radius and conf_ok
                rejection_reason = "Accepted"
                if not in_voronoi:
                    rejection_reason = f"Rejected: Peak outside instance {c_idx} Voronoi cell"
                elif not in_radius:
                    rejection_reason = f"Rejected: Distance to center ({dist_to_center:.1f}px) exceeds max_radius (22.0px)"
                elif not conf_ok:
                    rejection_reason = f"Rejected: Confidence ({p[2]:.3f}) below peak_threshold (0.10)"

                candidate_assignment_trace.append({
                    'candidate_id': cid,
                    'instance_id': c_idx,
                    'kp_idx': kp_idx,
                    'kp_name': KEYPOINT_NAMES_13[kp_idx],
                    'raw_xy': [float(p[0]), float(p[1])],
                    'confidence': float(p[2]),
                    'dist_to_center': dist_to_center,
                    'in_voronoi': in_voronoi,
                    'in_radius': in_radius,
                    'accepted': accepted_for_inst,
                    'rejection_reason': rejection_reason
                })

                if accepted_for_inst:
                    valid_cands.append(p)

            valid_cands.sort(key=lambda item: item[2], reverse=True)
            candidates_per_kp[kp_idx] = valid_cands[:3]

        # Iterative PAF & Unary refinement
        current_selection = {k: (0 if candidates_per_kp[k] else -1) for k in range(num_k)}
        for _ in range(3):
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
                        unary = 1.0 * conf
                    else:
                        unary = 1.0 * conf - 0.015 * dist_to_c

                    pairwise = 0.0
                    for nbr, limb_idx, is_from in neighbors[k]:
                        nbr_sel = current_selection[nbr]
                        if nbr_sel < 0 or not candidates_per_kp[nbr]:
                            continue
                        p_nbr = candidates_per_kp[nbr][nbr_sel]
                        if p_nbr[2] < 0.30:
                            continue
                        paf_x = pafs[limb_idx * 2]
                        paf_y = pafs[limb_idx * 2 + 1]
                        pt1 = (p_cand[0], p_cand[1]) if is_from else (p_nbr[0], p_nbr[1])
                        pt2 = (p_nbr[0], p_nbr[1]) if is_from else (p_cand[0], p_cand[1])
                        paf_val = compute_paf_score(paf_x, paf_y, pt1, pt2, num_samples=6)
                        pairwise += min(1.0, p_nbr[2]) * 0.5 * paf_val

                    total_score = unary + pairwise
                    if total_score > best_score:
                        best_score = total_score
                        best_idx = cand_idx
                if best_idx != current_selection[k]:
                    current_selection[k] = best_idx
                    changed = True
            if not changed:
                break

        inst_kps_64 = []
        inst_kps_256 = []
        inst_kps_disp = []
        scores = []
        scale = 256.0 / 64.0
        disp_scale = 280.0 / 256.0

        for k in range(num_k):
            sel_idx = current_selection[k]
            if sel_idx >= 0 and candidates_per_kp[k]:
                p = candidates_per_kp[k][sel_idx]
                inst_kps_64.append([float(p[0]), float(p[1]), float(p[2])])
                inst_kps_256.append([float(p[0] * scale), float(p[1] * scale), float(p[2])])
                inst_kps_disp.append([float(p[0] * scale * disp_scale), float(p[1] * scale * disp_scale), float(p[2])])
                scores.append(float(p[2]))
            else:
                inst_kps_64.append([0.0, 0.0, 0.0])
                inst_kps_256.append([0.0, 0.0, 0.0])
                inst_kps_disp.append([0.0, 0.0, 0.0])

        icx, icy = int(np.clip(round(cx), 0, W - 1)), int(np.clip(round(cy), 0, H - 1))
        class_id = 0 if class_probs[0, icy, icx] >= class_probs[1, icy, icx] else 1
        class_name = 'left_foot' if class_id == 0 else 'right_foot'

        instances_trace.append({
            'instance_id': c_idx,
            'center_64': [float(cx), float(cy)],
            'class_id': class_id,
            'class_name': class_name,
            'keypoints_64': inst_kps_64,
            'keypoints_256': inst_kps_256,
            'keypoints_display': inst_kps_disp,
            'mean_score': float(np.mean(scores)) if scores else 0.0
        })

    # Stage 10: Final rendered prediction compared with ground truth
    panel_w, panel_h = 280, 280
    sx = panel_w / 256.0
    sy = panel_h / 256.0
    gt_instances = sample_2.get('instances', [])

    p_gt = cv2.resize(preprocessed_bgr, (panel_w, panel_h))
    for g_i, gt in enumerate(gt_instances):
        kps = gt['keypoints']
        for u, v in SKELETON_LIMBS_13:
            if u < len(kps) and v < len(kps) and kps[u][2] > 0 and kps[v][2] > 0:
                cv2.line(p_gt, (int(kps[u][0]*sx), int(kps[u][1]*sy)),
                               (int(kps[v][0]*sx), int(kps[v][1]*sy)), (0, 255, 0), 2, cv2.LINE_AA)
        for k_i in range(min(num_k, len(kps))):
            gx, gy, gv = kps[k_i]
            if gv > 0:
                cv2.circle(p_gt, (int(gx*sx), int(gy*sy)), 4, KP_COLORS[k_i], -1, cv2.LINE_AA)
                cv2.putText(p_gt, f"F{g_i}:k{k_i}", (int(gx*sx)+3, int(gy*sy)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(p_gt, (0, 0), (panel_w, 24), (30, 30, 30), -1)
    cv2.putText(p_gt, f"Ground Truth ({len(gt_instances)} feet)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    p_pred = cv2.resize(preprocessed_bgr, (panel_w, panel_h))
    for p_i, inst in enumerate(instances_trace):
        kps_disp = inst['keypoints_display']
        for u, v in SKELETON_LIMBS_13:
            if kps_disp[u][2] > 0.10 and kps_disp[v][2] > 0.10:
                cv2.line(p_pred, (int(kps_disp[u][0]), int(kps_disp[u][1])),
                                 (int(kps_disp[v][0]), int(kps_disp[v][1])), (0, 140, 255), 2, cv2.LINE_AA)
        for k_i in range(num_k):
            pk = kps_disp[k_i]
            if pk[2] > 0.10:
                cv2.circle(p_pred, (int(pk[0]), int(pk[1])), 4, KP_COLORS[k_i], -1, cv2.LINE_AA)
                cv2.putText(p_pred, f"P{p_i}:k{k_i}", (int(pk[0])+3, int(pk[1])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(p_pred, (0, 0), (panel_w, 24), (30, 30, 30), -1)
    cv2.putText(p_pred, f"Prediction ({len(instances_trace)} feet)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    side_by_side = np.hstack([p_gt, p_pred])
    p_final_comp = TRACE_OUT_DIR / "s2_stage10_final_prediction_vs_gt.jpg"
    cv2.imwrite(str(p_final_comp), side_by_side)

    # Copy visualizations to artifact directory
    vis_artifacts = [
        p_stage1_orig, p_stage1_prep, p_hm_overlay, p_paf_vis, p_cls_vis, p_final_comp
    ]
    for p in vis_artifacts:
        shutil.copy2(p, ARTIFACT_DIR / p.name)

    full_trace_data = {
        'image_name': img_name,
        'image_path': str(raw_img_path),
        'original_dimensions': [orig_w, orig_h],
        'candidate_peaks': candidate_peaks_table,
        'anchor_decisions': anchor_decisions,
        'clustering_trace': clustering_trace,
        'candidate_assignment_trace': candidate_assignment_trace,
        'instances_trace': instances_trace,
        'ground_truth_instances': gt_instances
    }

    trace_json_path = TRACE_OUT_DIR / "trace_data_sample2.json"
    with open(trace_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_trace_data, f, indent=2)
    shutil.copy2(trace_json_path, ARTIFACT_DIR / "trace_data_sample2.json")

    print("✅ Complete trace for Sample 2 finished successfully!")


if __name__ == '__main__':
    run_complete_trace_sample2()

