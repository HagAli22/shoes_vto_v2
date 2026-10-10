"""
Complete validation evaluation of M1v2:
Compares OLD post-processing vs NEW post-processing on the entire VALIDATION split.
Reports:
1. Box IoU (Mean, Median, % >= 0.50, % >= 0.70)
2. Foot-keypoint PCK@0.20, PCK@0.10, PCK@0.05
3. Class Accuracy (Left vs Right Foot)
4. Keypoint spread vs GT spread (Width and Height ratio)

IMPORTANT: The test split is left completely untouched.
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch

BASE_DIR = Path("shoes_vto")
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from utils.keypoint_grouping import (
    load_paf_config,
    detect_peaks,
    load_box_calibration,
    compute_foot_box
)

def safe_open(p, mode="r", encoding="utf-8"):
    p_str = os.path.abspath(str(p))
    if os.name == "nt" and not p_str.startswith("\\\\?\\"):
        p_str = "\\\\?\\" + p_str
    return open(p_str, mode, encoding=encoding)

def compute_box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def decode_spatial_old(heatmaps, class_probs, peak_threshold=0.10):
    H, W = heatmaps.shape[1], heatmaps.shape[2]
    all_peaks = {}
    for kp_idx in range(16):
        all_peaks[kp_idx] = detect_peaks(heatmaps[kp_idx], threshold=peak_threshold, nms_window=3, subpixel=True)

    strong_pts = []
    for kp_idx, pts in all_peaks.items():
        for p in pts:
            if p[2] > 0.35:
                strong_pts.append((p[0], p[1], kp_idx, p[2]))
    if len(strong_pts) < 4:
        for kp_idx, pts in all_peaks.items():
            for p in pts:
                if p[2] > 0.15:
                    strong_pts.append((p[0], p[1], kp_idx, p[2]))
    if len(strong_pts) < 3:
        return []

    pts_xy = np.array([[p[0], p[1]] for p in strong_pts], dtype=np.float32)
    if len(pts_xy) >= 6:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _, labels, centers = cv2.kmeans(pts_xy, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        if np.linalg.norm(centers[0] - centers[1]) < 7.0:
            centers = [np.mean(pts_xy, axis=0)]
        else:
            centers = sorted(centers, key=lambda c: c[0])
    else:
        centers = [np.mean(pts_xy, axis=0)]

    instances = []
    for cx, cy in centers:
        inst_kps = []
        for kp_idx in range(16):
            c_peaks = all_peaks.get(kp_idx, [])
            best_p = None
            min_d = 999.0
            for p in c_peaks:
                d = np.sqrt((p[0] - cx)**2 + (p[1] - cy)**2)
                if d <= 20.0 and d < min_d:
                    min_d = d
                    best_p = p
            if best_p is not None and best_p[2] >= peak_threshold:
                inst_kps.append([best_p[0], best_p[1], best_p[2]])
            else:
                inst_kps.append([0.0, 0.0, 0.0])

        valid = [k for k in inst_kps if k[2] > 0.15]
        if len(valid) < 5:
            continue
        left_v = [class_probs[0, int(np.clip(round(k[1]), 0, H-1)), int(np.clip(round(k[0]), 0, W-1))] for k in valid]
        right_v = [class_probs[1, int(np.clip(round(k[1]), 0, H-1)), int(np.clip(round(k[0]), 0, W-1))] for k in valid]
        p_cls = 0 if np.mean(left_v) > np.mean(right_v) else 1
        instances.append({'class_id': p_cls, 'keypoints': inst_kps, 'score': np.mean([k[2] for k in valid])})
    return instances

def evaluate_on_validation():
    ckpt_path = BASE_DIR / "outputs/m1v2_training/checkpoints/best.pth"
    model = ARShoeM1v2()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict({k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()})
    model.eval()

    val_img_dir = Path("dataset/merged_v3_v4/valid/images")
    val_lbl_dir = Path("dataset/merged_v3_v4/valid/labels")
    lbl_files = sorted(list(val_lbl_dir.glob("*.txt")))
    print(f"Loaded {len(lbl_files)} validation ground truth files.")

    FOOT_KPS = [i for i in range(16) if i not in (12, 15)]

    from utils.keypoint_grouping import decode_spatial_instances_single

    # Tracking metrics
    ious_old = []
    ious_new = []
    pck20_old = []
    pck20_new = []
    pck10_old = []
    pck10_new = []
    pck05_old = []
    pck05_new = []
    cls_correct_old = []
    cls_correct_new = []
    w_ratio_old, h_ratio_old = [], []
    w_ratio_new, h_ratio_new = [], []

    calib = load_box_calibration()

    for idx, lf in enumerate(lbl_files, 1):
        stem = lf.stem
        img_p = val_img_dir / f"{stem}.jpg"
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]

        t = torch.from_numpy(cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            out = model(t)
            hm = out['heatmaps'][0].numpy()
            cls_p = out['class'][0].numpy()

        preds_old = decode_spatial_old(hm, cls_p)
        preds_new = decode_spatial_instances_single(hm, cls_p)

        # Parse ground truth instances
        gt_instances = []
        with safe_open(lf, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 53:
                    continue
                cls_gt = int(tokens[0])
                cx, cy, bw, bh = map(float, tokens[1:5])
                gx1 = int(round((cx - bw / 2.0) * W))
                gy1 = int(round((cy - bh / 2.0) * H))
                gx2 = int(round((cx + bw / 2.0) * W))
                gy2 = int(round((cy + bh / 2.0) * H))
                kps = np.array([[float(tokens[5+i*3]) * W, float(tokens[6+i*3]) * H, float(tokens[7+i*3])] for i in range(16)])
                gt_instances.append({
                    'class_id': cls_gt,
                    'box': (gx1, gy1, gx2, gy2),
                    'kps': kps
                })

        for gt in gt_instances:
            gt_box = gt['box']
            gt_kps = gt['kps'][FOOT_KPS]
            vis_mask = gt_kps[:, 2] > 0
            if vis_mask.sum() < 5:
                continue

            diag = np.hypot(gt_kps[vis_mask, 0].max() - gt_kps[vis_mask, 0].min(),
                            gt_kps[vis_mask, 1].max() - gt_kps[vis_mask, 1].min())
            gw = max(1.0, gt_kps[vis_mask, 0].max() - gt_kps[vis_mask, 0].min())
            gh = max(1.0, gt_kps[vis_mask, 1].max() - gt_kps[vis_mask, 1].min())

            gcx = (gt_box[0] + gt_box[2]) / 2.0
            gcy = (gt_box[1] + gt_box[3]) / 2.0

            # Match OLD prediction
            best_old = None
            min_d_old = 1e9
            for p in preds_old:
                pk = np.array(p['keypoints'])
                pk[:, 0] *= (W / 64.0)
                pk[:, 1] *= (H / 64.0)
                pvis = pk[:, 2] > 0.15
                if pvis.sum() < 4:
                    continue
                d = np.hypot(pk[pvis, 0].mean() - gcx, pk[pvis, 1].mean() - gcy)
                if d < min_d_old and d < 0.35 * max(W, H):
                    min_d_old = d
                    best_old = (p, pk)

            if best_old is not None:
                p, pk = best_old
                # Old box logic: min/max of visible points +- 15 px
                pvis = pk[:, 2] > 0.15
                xs, ys = pk[pvis, 0], pk[pvis, 1]
                b_old = (max(0, min(xs) - 15), max(0, min(ys) - 15), min(W - 1, max(xs) + 15), min(H - 1, max(ys) + 15))
                iou_val = compute_box_iou(b_old, gt_box)
                ious_old.append(iou_val)

                cls_correct_old.append(1 if p['class_id'] == gt['class_id'] else 0)

                # Keypoint evaluation
                f_pk = pk[FOOT_KPS]
                f_vis = f_pk[:, 2] > 0.15
                w_ratio_old.append((f_pk[f_vis, 0].max() - f_pk[f_vis, 0].min()) / gw if f_vis.sum() > 1 else 0.0)
                h_ratio_old.append((f_pk[f_vis, 1].max() - f_pk[f_vis, 1].min()) / gh if f_vis.sum() > 1 else 0.0)

                mut = vis_mask & f_vis
                if mut.sum() > 0:
                    dists = np.linalg.norm(f_pk[mut, :2] - gt_kps[mut, :2], axis=1)
                    pck20_old.append((dists <= 0.20 * diag).mean())
                    pck10_old.append((dists <= 0.10 * diag).mean())
                    pck05_old.append((dists <= 0.05 * diag).mean())

            # Match NEW prediction
            best_new = None
            min_d_new = 1e9
            for p in preds_new:
                pk = np.array(p['keypoints'])
                pk[:, 0] *= (W / 64.0)
                pk[:, 1] *= (H / 64.0)
                pvis = pk[:, 2] > 0.15
                if pvis.sum() < 4:
                    continue
                d = np.hypot(pk[pvis, 0].mean() - gcx, pk[pvis, 1].mean() - gcy)
                if d < min_d_new and d < 0.35 * max(W, H):
                    min_d_new = d
                    best_new = (p, pk)

            if best_new is not None:
                p, pk = best_new
                # New box logic: scale-aware calibrated box from 14 foot kps
                b_new = compute_foot_box(pk, W, H, scale_x=1.0, scale_y=1.0, calib=calib)
                if b_new is not None:
                    iou_val = compute_box_iou(b_new, gt_box)
                    ious_new.append(iou_val)

                cls_correct_new.append(1 if p['class_id'] == gt['class_id'] else 0)

                f_pk = pk[FOOT_KPS]
                f_vis = f_pk[:, 2] > 0.15
                w_ratio_new.append((f_pk[f_vis, 0].max() - f_pk[f_vis, 0].min()) / gw if f_vis.sum() > 1 else 0.0)
                h_ratio_new.append((f_pk[f_vis, 1].max() - f_pk[f_vis, 1].min()) / gh if f_vis.sum() > 1 else 0.0)

                mut = vis_mask & f_vis
                if mut.sum() > 0:
                    dists = np.linalg.norm(f_pk[mut, :2] - gt_kps[mut, :2], axis=1)
                    pck20_new.append((dists <= 0.20 * diag).mean())
                    pck10_new.append((dists <= 0.10 * diag).mean())
                    pck05_new.append((dists <= 0.05 * diag).mean())

        if idx % 50 == 0:
            print(f"Evaluated {idx}/{len(lbl_files)} validation images...")

    print("=" * 80)
    print("📊 VALIDATION EVALUATION RESULTS: BEFORE vs AFTER")
    print("=" * 80)
    print(f"Total evaluated foot instances: {len(ious_new)} (New) / {len(ious_old)} (Old)")
    print("\n1. Bounding Box IoU vs Ground Truth:")
    print(f"  Mean IoU:         {np.mean(ious_old):.3f} (Old)  -->  {np.mean(ious_new):.3f} (New)  [+{(np.mean(ious_new)-np.mean(ious_old))*100:.1f}%]")
    print(f"  Median IoU:       {np.median(ious_old):.3f} (Old)  -->  {np.median(ious_new):.3f} (New)  [+{(np.median(ious_new)-np.median(ious_old))*100:.1f}%]")
    print(f"  Box IoU >= 0.50:  {(np.array(ious_old) >= 0.50).mean()*100:.1f}% (Old)  -->  {(np.array(ious_new) >= 0.50).mean()*100:.1f}% (New)")
    print(f"  Box IoU >= 0.70:  {(np.array(ious_old) >= 0.70).mean()*100:.1f}% (Old)  -->  {(np.array(ious_new) >= 0.70).mean()*100:.1f}% (New)")

    print("\n2. Keypoint Accuracy (PCK):")
    print(f"  PCK@0.20:         {np.mean(pck20_old)*100:.1f}% (Old)  -->  {np.mean(pck20_new)*100:.1f}% (New)")
    print(f"  PCK@0.10:         {np.mean(pck10_old)*100:.1f}% (Old)  -->  {np.mean(pck10_new)*100:.1f}% (New)")
    print(f"  PCK@0.05:         {np.mean(pck05_old)*100:.1f}% (Old)  -->  {np.mean(pck05_new)*100:.1f}% (New)")

    print("\n3. Foot Classification (Left vs Right):")
    print(f"  Accuracy:         {np.mean(cls_correct_old)*100:.1f}% (Old)  -->  {np.mean(cls_correct_new)*100:.1f}% (New)")

    print("\n4. Foot Keypoint Spread vs Ground Truth (Median):")
    print(f"  Width Ratio:      {np.median(w_ratio_old):.2f}x (Old)  -->  {np.median(w_ratio_new):.2f}x (New)")
    print(f"  Height Ratio:     {np.median(h_ratio_old):.2f}x (Old)  -->  {np.median(h_ratio_new):.2f}x (New)")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_on_validation()

