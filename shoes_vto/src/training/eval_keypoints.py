"""
Keypoint Evaluation Metrics for ARShoe M1
Computes PCK@threshold and keypoint decoding from heatmaps
"""

import torch
import numpy as np


def decode_heatmaps_to_keypoints(heatmaps, image_size=256):
    """
    Decode heatmap peaks to pixel coordinates.
    
    Args:
        heatmaps: [B, 16, H, W] predicted heatmaps (tensor or numpy)
        image_size: image dimension (default: 256)
        
    Returns:
        keypoints: [B, 16, 2] in pixel coords (0..image_size)
        confidences: [B, 16] peak confidence values
    """
    if not isinstance(heatmaps, torch.Tensor):
        heatmaps = torch.from_numpy(heatmaps)
        
    B, K, H, W = heatmaps.shape
    scale = float(image_size) / float(H)
    
    device = heatmaps.device
    flat = heatmaps.view(B, K, -1)
    max_vals, max_idx = flat.max(dim=2)
    
    y_hm = (max_idx // W).float()
    x_hm = (max_idx % W).float()
    
    # Fully vectorized sub-pixel refinement on GPU (no Python loops)
    pad_hm = torch.nn.functional.pad(heatmaps, (1, 1, 1, 1), mode='replicate')
    b_idx = torch.arange(B, device=device).unsqueeze(1)
    k_idx = torch.arange(K, device=device).unsqueeze(0)
    cy = (y_hm.long() + 1).clamp(1, H)
    cx = (x_hm.long() + 1).clamp(1, W)
    
    dx = pad_hm[b_idx, k_idx, cy, cx + 1] - pad_hm[b_idx, k_idx, cy, cx - 1]
    dy = pad_hm[b_idx, k_idx, cy + 1, cx] - pad_hm[b_idx, k_idx, cy - 1, cx]
    
    x_hm = x_hm + torch.clamp(0.25 * torch.sign(dx), -0.5, 0.5)
    y_hm = y_hm + torch.clamp(0.25 * torch.sign(dy), -0.5, 0.5)
    
    x_img = (x_hm + 0.5) * scale
    y_img = (y_hm + 0.5) * scale
    
    keypoints = torch.stack([x_img, y_img], dim=-1)
    return keypoints, max_vals


def compute_pck(pred_keypoints, gt_keypoints, gt_visibility, bbox_diags, threshold=0.2):
    """
    Percentage of Correct Keypoints (PCK)
    
    A predicted keypoint is correct if:
        dist(pred, gt) < threshold * bbox_diagonal
        
    Args:
        pred_keypoints: [N, 16, 2] predicted (x, y) in pixel coords
        gt_keypoints:   [N, 16, 2] ground truth (x, y) in pixel coords
        gt_visibility:  [N, 16] visibility flags (v > 0 evaluated)
        bbox_diags:     [N] bounding box diagonal in pixels
        threshold:      PCK threshold (default 0.2 = 20% of bbox diagonal)
        
    Returns:
        overall_pck: float (0.0 to 1.0)
        per_kp_pck: list of 16 floats
    """
    if len(pred_keypoints) == 0:
        return 0.0, [0.0] * 16
        
    if not isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = torch.tensor(np.asarray(pred_keypoints), dtype=torch.float32)
    if not isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = torch.tensor(np.asarray(gt_keypoints), dtype=torch.float32)
    if not isinstance(gt_visibility, torch.Tensor):
        gt_visibility = torch.tensor(np.asarray(gt_visibility), dtype=torch.float32)
    if not isinstance(bbox_diags, torch.Tensor):
        bbox_diags = torch.tensor(np.asarray(bbox_diags), dtype=torch.float32)
        
    dist = torch.norm(pred_keypoints - gt_keypoints, dim=-1)  # [N, 16]
    norm_dist = dist / (bbox_diags.unsqueeze(1) + 1e-6)
    
    valid = gt_visibility > 0
    correct = (norm_dist < threshold) & valid
    
    total_valid = valid.sum().float()
    if total_valid > 0:
        overall_pck = (correct.sum().float() / total_valid).item()
    else:
        overall_pck = 0.0
        
    per_kp_pck = []
    for kp_idx in range(16):
        kp_v = valid[:, kp_idx]
        if kp_v.sum() > 0:
            pck_k = (correct[:, kp_idx].sum().float() / kp_v.sum().float()).item()
        else:
            pck_k = 0.0
        per_kp_pck.append(pck_k)
        
    return overall_pck, per_kp_pck


# Official frozen 16 keypoint names (from dataset/keypoint-order-audit.md)
KEYPOINT_NAMES = [
    "toe_ground",        # 0
    "heel_back",         # 1
    "heel_ground",       # 2
    "ball_medial",       # 3
    "ball_lateral",      # 4
    "ball_top",          # 5
    "instep_top",        # 6
    "arch_medial",       # 7
    "midfoot_lateral",   # 8
    "malleolus_medial",  # 9
    "malleolus_lateral", # 10
    "toe_tip",           # 11
    "ankle_center",      # 12
    "throat",            # 13
    "achilles",          # 14
    "shin_mid",          # 15
]

# 15 limb connections (from configs/paf_connections.yaml)
SKELETON_LIMBS = [
    (11, 0),   # toe_tip -> toe_ground
    (0, 2),    # toe_ground -> heel_ground
    (2, 1),    # heel_ground -> heel_back
    (3, 5),    # ball_medial -> ball_top
    (5, 4),    # ball_top -> ball_lateral
    (2, 7),    # heel_ground -> arch_medial
    (2, 8),    # heel_ground -> midfoot_lateral
    (7, 6),    # arch_medial -> instep_top
    (8, 6),    # midfoot_lateral -> instep_top
    (9, 12),   # malleolus_medial -> ankle_center
    (12, 10),  # ankle_center -> malleolus_lateral
    (12, 13),  # ankle_center -> throat
    (12, 14),  # ankle_center -> achilles
    (12, 15),  # ankle_center -> shin_mid
    (5, 13),   # ball_top -> throat
]


def compute_multi_threshold_pck(pred_keypoints, gt_keypoints, gt_visibility, bbox_diags, thresholds=(0.2, 0.1, 0.05)):
    """
    Compute PCK at multiple distance thresholds (e.g. 0.2, 0.1, 0.05).
    
    Returns:
        results: dict containing:
            - 'overall': {thresh: score}
            - 'per_keypoint': {kp_idx: {'name': str, thresh: score, 'count': int}}
            - 'total_evaluated': int
            - 'poor_keypoints_0.2': list of (name, pck) with pck < 0.40
    """
    if len(pred_keypoints) == 0:
        return {'overall': {t: 0.0 for t in thresholds}, 'per_keypoint': {}, 'total_evaluated': 0}
        
    if not isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = torch.tensor(np.asarray(pred_keypoints), dtype=torch.float32)
    if not isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = torch.tensor(np.asarray(gt_keypoints), dtype=torch.float32)
    if not isinstance(gt_visibility, torch.Tensor):
        gt_visibility = torch.tensor(np.asarray(gt_visibility), dtype=torch.float32)
    if not isinstance(bbox_diags, torch.Tensor):
        bbox_diags = torch.tensor(np.asarray(bbox_diags), dtype=torch.float32)
        
    dist = torch.norm(pred_keypoints - gt_keypoints, dim=-1)  # [N, 16]
    norm_dist = dist / (bbox_diags.unsqueeze(1) + 1e-6)       # [N, 16]
    valid = gt_visibility > 0                                 # [N, 16]
    
    total_valid = int(valid.sum().item())
    
    overall_results = {}
    for t in thresholds:
        correct_t = (norm_dist < t) & valid
        overall_results[t] = (correct_t.sum().float() / (total_valid + 1e-6)).item()
        
    # Also compute 14 foot keypoints (excluding ankle_center:12 and shin_mid:15)
    # and 2 leg keypoints (ankle_center:12 and shin_mid:15)
    foot_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    leg_indices = [12, 15]
    
    foot_results = {}
    leg_results = {}
    for t in thresholds:
        foot_v = valid[:, foot_indices]
        if foot_v.sum() > 0:
            foot_c = ((norm_dist[:, foot_indices] < t) & foot_v).sum().float()
            foot_results[t] = (foot_c / foot_v.sum().float()).item()
        else:
            foot_results[t] = 0.0
            
        leg_v = valid[:, leg_indices]
        if leg_v.sum() > 0:
            leg_c = ((norm_dist[:, leg_indices] < t) & leg_v).sum().float()
            leg_results[t] = (leg_c / leg_v.sum().float()).item()
        else:
            leg_results[t] = 0.0
        
    per_kp_results = {}
    for kp_idx in range(16):
        kp_name = KEYPOINT_NAMES[kp_idx]
        v_k = valid[:, kp_idx]
        count_k = int(v_k.sum().item())
        kp_dict = {'name': kp_name, 'valid_count': count_k}
        for t in thresholds:
            if count_k > 0:
                c_kt = ((norm_dist[:, kp_idx] < t) & v_k).sum().float()
                score = (c_kt / count_k).item()
            else:
                score = 0.0
            kp_dict[f'pck@{t}'] = score
        per_kp_results[kp_idx] = kp_dict
        
    # Rank keypoints by PCK@0.2 (lowest to highest)
    ranked_by_pck20 = sorted(
        [(v['name'], v['pck@0.2'], v['valid_count']) for v in per_kp_results.values()],
        key=lambda x: x[1]
    )
    
    return {
        'overall': overall_results,
        'foot_14': foot_results,
        'leg_2': leg_results,
        'per_keypoint': per_kp_results,
        'ranked_by_pck@0.2': ranked_by_pck20,
        'total_instances': len(pred_keypoints),
        'total_keypoints_evaluated': total_valid
    }


def draw_keypoints_on_image(image_rgb, gt_kps=None, gt_vis=None, pred_kps=None, pred_conf=None, bbox=None):
    """
    Draw Ground Truth (Green) and Predictions (Red/Cyan) on image using OpenCV.
    
    Returns:
        annotated_image: RGB numpy uint8 array
    """
    import cv2
    img = image_rgb.copy()
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
        
    H, W = img.shape[:2]
    
    # Draw Bounding Box if present (thin yellow)
    if bbox is not None:
        cx, cy, bw, bh = bbox
        x1 = int(max(0, (cx - bw / 2.0) * W))
        y1 = int(max(0, (cy - bh / 2.0) * H))
        x2 = int(min(W, (cx + bw / 2.0) * W))
        y2 = int(min(H, (cy + bh / 2.0) * H))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 230, 0), 1, cv2.LINE_AA)
        
    # Draw GT Skeleton Limbs (Thin Green)
    if gt_kps is not None and gt_vis is not None:
        for p1, p2 in SKELETON_LIMBS:
            if gt_vis[p1] > 0 and gt_vis[p2] > 0:
                pt1 = (int(round(gt_kps[p1][0])), int(round(gt_kps[p1][1])))
                pt2 = (int(round(gt_kps[p2][0])), int(round(gt_kps[p2][1])))
                cv2.line(img, pt1, pt2, (0, 180, 0), 1, cv2.LINE_AA)
                
    # Draw Predicted Skeleton Limbs (Thin Coral/Magenta)
    if pred_kps is not None:
        for p1, p2 in SKELETON_LIMBS:
            pt1 = (int(round(pred_kps[p1][0])), int(round(pred_kps[p1][1])))
            pt2 = (int(round(pred_kps[p2][0])), int(round(pred_kps[p2][1])))
            cv2.line(img, pt1, pt2, (255, 80, 80), 1, cv2.LINE_AA)
            
    # Draw Displacement Error Vectors (Yellow line from GT to Pred)
    if gt_kps is not None and gt_vis is not None and pred_kps is not None:
        for idx in range(16):
            if gt_vis[idx] > 0:
                gt_pt = (int(round(gt_kps[idx][0])), int(round(gt_kps[idx][1])))
                pr_pt = (int(round(pred_kps[idx][0])), int(round(pred_kps[idx][1])))
                cv2.line(img, gt_pt, pr_pt, (255, 255, 0), 1, cv2.LINE_AA)
                
    # Draw GT Keypoints (Bright Green circles with dark outline)
    if gt_kps is not None and gt_vis is not None:
        for idx in range(16):
            if gt_vis[idx] > 0:
                pt = (int(round(gt_kps[idx][0])), int(round(gt_kps[idx][1])))
                cv2.circle(img, pt, 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(img, str(idx), (pt[0] + 4, pt[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                
    # Draw Predicted Keypoints (Bright Red circles with white outline)
    if pred_kps is not None:
        for idx in range(16):
            pt = (int(round(pred_kps[idx][0])), int(round(pred_kps[idx][1])))
            cv2.circle(img, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, 3, (255, 20, 20), -1, cv2.LINE_AA)
            cv2.putText(img, str(idx), (pt[0] - 12, pt[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 60, 60), 1, cv2.LINE_AA)
            
    # Add Top Legend Header Banner
    cv2.rectangle(img, (0, 0), (W, 22), (20, 20, 20), -1)
    cv2.circle(img, (12, 11), 4, (0, 255, 0), -1)
    cv2.putText(img, "GT", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(img, (52, 11), 4, (255, 20, 20), -1)
    cv2.putText(img, "Pred", (60, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 80, 80), 1, cv2.LINE_AA)
    cv2.line(img, (98, 11), (115, 11), (255, 255, 0), 1)
    cv2.putText(img, "Error", (120, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    
    return img


def create_side_by_side_visualization(image_rgb, gt_kps, gt_vis, pred_kps, pred_confs=None, bbox=None, img_name=""):
    """
    Creates a 3-panel visualization:
    [Left: Ground Truth] | [Center: Prediction] | [Right: Overlay with Error Vectors]
    """
    import cv2
    base = image_rgb.copy()
    if base.max() <= 1.0:
        base = (base * 255.0).astype(np.uint8)
    else:
        base = base.astype(np.uint8)
        
    H, W = base.shape[:2]
    
    # 1. Left Panel: Ground Truth only
    panel_gt = base.copy()
    cv2.rectangle(panel_gt, (0, 0), (W, 22), (20, 20, 20), -1)
    cv2.putText(panel_gt, f"Ground Truth: {img_name}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
    if gt_kps is not None and gt_vis is not None:
        for p1, p2 in SKELETON_LIMBS:
            if gt_vis[p1] > 0 and gt_vis[p2] > 0:
                cv2.line(panel_gt, (int(round(gt_kps[p1][0])), int(round(gt_kps[p1][1]))),
                                  (int(round(gt_kps[p2][0])), int(round(gt_kps[p2][1]))), (0, 180, 0), 1, cv2.LINE_AA)
        for idx in range(16):
            if gt_vis[idx] > 0:
                pt = (int(round(gt_kps[idx][0])), int(round(gt_kps[idx][1])))
                cv2.circle(panel_gt, pt, 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(panel_gt, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(panel_gt, str(idx), (pt[0] + 4, pt[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                
    # 2. Center Panel: Predictions only
    panel_pred = base.copy()
    cv2.rectangle(panel_pred, (0, 0), (W, 22), (20, 20, 20), -1)
    cv2.putText(panel_pred, "M1 Model Prediction", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 80, 80), 1, cv2.LINE_AA)
    if pred_kps is not None:
        for p1, p2 in SKELETON_LIMBS:
            cv2.line(panel_pred, (int(round(pred_kps[p1][0])), int(round(pred_kps[p1][1]))),
                                 (int(round(pred_kps[p2][0])), int(round(pred_kps[p2][1]))), (255, 80, 80), 1, cv2.LINE_AA)
        for idx in range(16):
            pt = (int(round(pred_kps[idx][0])), int(round(pred_kps[idx][1])))
            cv2.circle(panel_pred, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel_pred, pt, 3, (255, 20, 20), -1, cv2.LINE_AA)
            cv2.putText(panel_pred, str(idx), (pt[0] + 4, pt[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 60, 60), 1, cv2.LINE_AA)
            
    # 3. Right Panel: Overlay with error lines
    panel_overlay = draw_keypoints_on_image(base, gt_kps, gt_vis, pred_kps, pred_confs, bbox)
    
    # Concatenate horizontally
    combined = np.hstack([panel_gt, panel_pred, panel_overlay])
    return combined
