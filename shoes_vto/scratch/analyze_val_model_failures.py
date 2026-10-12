import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from utils.keypoint_grouping import group_keypoints, detect_peaks, compute_foot_box, load_box_calibration

def safe_open(p, mode="r", encoding="utf-8"):
    p_str = os.path.abspath(str(p))
    if os.name == "nt" and not p_str.startswith("\\\\?\\"):
        p_str = "\\\\?\\" + p_str
    return open(p_str, mode, encoding=encoding)

def compute_box_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return float(inter_area) / float(area1 + area2 - inter_area)

VAL_IMG = Path("dataset/merged_v3_v4/valid/images")
VAL_LBL = Path("dataset/merged_v3_v4/valid/labels")
OUT_DIR = BASE_DIR / "outputs" / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KP_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "ball_top", "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "ankle_center",
    "throat", "achilles", "shin_mid"
]

def run_val_failure_analysis():
    print("=" * 80)
    print("PHASE 1: VALIDATION FAILURE DECOMPOSITION & MULTI-AXIS STRATIFICATION")
    print("=" * 80)

    # 1. Load Model
    ckpt_path = BASE_DIR / "outputs/m1v2_training/checkpoints/best.pth"
    model = ARShoeM1v2()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict({k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()})
    model.eval()

    val_lbl_files = sorted(VAL_LBL.glob("*.txt"))
    calib = load_box_calibration()
    print(f"Total validation label files: {len(val_lbl_files)}")

    # Containers for analysis
    all_kps_data = []
    feet_data = []

    for lf in tqdm(val_lbl_files, desc="Analyzing Validation Set"):
        ip = VAL_IMG / f"{lf.stem}.jpg"
        if not ip.exists():
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Calculate image illumination / contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        if mean_lum < 85.0:
            lum_cat = "Low-Light / Dark (<85)"
        elif mean_lum > 175.0:
            lum_cat = "High-Exposure / Glare (>175)"
        else:
            lum_cat = "Normal Illumination (85-175)"

        # Model Inference
        inp = torch.from_numpy(cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            out = model(inp)
            hm = out['heatmaps']
            paf = out['pafs']
            cp = out['class']

        # Production Decoder Predictions
        batch_insts = group_keypoints(heatmaps=hm, pafs=paf, class_probs=cp, method='spatial')
        preds = batch_insts[0]

        # Raw Heatmaps in numpy [16, 64, 64]
        hm_np = hm[0].cpu().numpy()
        raw_peaks = {k: detect_peaks(hm_np[k], threshold=0.10, nms_window=3, subpixel=True) for k in range(16)}

        # Parse Ground Truth
        with safe_open(lf, "r") as f:
            lines = f.readlines()

        gt_instances = []
        for line in lines:
            t = line.strip().split()
            if len(t) < 53:
                continue
            cls_id = int(t[0])
            cx, cy, w, h = [float(x) for x in t[1:5]]
            diag = float(np.hypot(w * orig_w, h * orig_h))
            diag64 = float(np.hypot(w * 64.0, h * 64.0))
            gx1, gy1 = int(round((cx - w / 2) * orig_w)), int(round((cy - h / 2) * orig_h))
            gx2, gy2 = int(round((cx + w / 2) * orig_w)), int(round((cy + h / 2) * orig_h))

            kps = []
            for i in range(16):
                kx = float(t[5 + i * 3]) * orig_w
                ky = float(t[6 + i * 3]) * orig_h
                kv = float(t[7 + i * 3])
                kps.append([kx, ky, kv])

            # Foot orientation angle (heel_ground -> toe)
            p_heel = kps[2]
            p_toe = kps[11] if kps[11][2] > 0 else kps[0]
            if p_heel[2] > 0 and p_toe[2] > 0:
                dx = p_toe[0] - p_heel[0]
                dy = p_toe[1] - p_heel[1]
                angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
                angle_dev = min(angle_deg, abs(180.0 - angle_deg))
            else:
                angle_dev = 0.0

            if angle_dev < 25.0:
                angle_cat = "Flat / Horizontal (0-25 deg)"
            elif angle_dev < 55.0:
                angle_cat = "Oblique (25-55 deg)"
            else:
                angle_cat = "Steep / Vertical (>55 deg)"

            # Foot scale category
            if diag64 < 22.0:
                scale_cat = "Small (<22 px)"
            elif diag64 < 32.0:
                scale_cat = "Medium (22-32 px)"
            else:
                scale_cat = "Large (>32 px)"

            num_occl = sum(1 for k in kps if k[2] == 0)
            if num_occl <= 1:
                occl_cat = "Mild (0-1 occluded)"
            elif num_occl <= 4:
                occl_cat = "Moderate (2-4 occluded)"
            else:
                occl_cat = "Severe (>=5 occluded)"

            gt_instances.append({
                'class_id': cls_id, 'bbox': (gx1, gy1, gx2, gy2),
                'diag': diag, 'diag64': diag64, 'keypoints': kps,
                'center': ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0),
                'scale_cat': scale_cat,
                'angle_cat': angle_cat,
                'occl_cat': occl_cat,
                'num_occl': num_occl,
                'angle_dev': angle_dev
            })

        # Overlapping / clutter check
        is_overlapping = False
        if len(gt_instances) >= 2:
            b1 = gt_instances[0]['bbox']
            b2 = gt_instances[1]['bbox']
            inter_iou = compute_box_iou(b1, b2)
            c1 = gt_instances[0]['center']
            c2 = gt_instances[1]['center']
            center_dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
            if inter_iou > 0.0 or center_dist < orig_w * 0.22:
                is_overlapping = True
        clutter_cat = "Overlapping / Adjacent Feet" if is_overlapping else "Isolated Single Foot"

        # Match Predictions to GT Feet
        matched_preds = set()
        for gt in gt_instances:
            best_pred = None
            best_d = float('inf')
            best_pidx = -1

            for pidx, p in enumerate(preds):
                if pidx in matched_preds:
                    continue
                pk_sc = [[k[0] * (orig_w / 64.0), k[1] * (orig_h / 64.0), k[2]] for k in p['keypoints']]
                vis = [k[:2] for k in pk_sc if k[2] > 0.15]
                if not vis:
                    continue
                c = np.mean(vis, axis=0)
                d = np.hypot(c[0] - gt['center'][0], c[1] - gt['center'][1])
                if d < best_d and d < orig_w * 0.40:
                    best_d = d
                    best_pred = p
                    best_pidx = pidx

            matched = False
            pred_kps_sc = None
            box_iou = 0.0
            class_correct = False

            if best_pred is not None:
                matched_preds.add(best_pidx)
                matched = True
                pred_kps_sc = [[k[0] * (orig_w / 64.0), k[1] * (orig_h / 64.0), k[2]] for k in best_pred['keypoints']]
                pbox = compute_foot_box(pred_kps_sc, orig_w, orig_h, scale_x=1.0, scale_y=1.0, calib=calib)
                box_iou = compute_box_iou(pbox, gt['bbox'])
                class_correct = (best_pred['class_id'] == gt['class_id'])

            feet_data.append({
                'stem': lf.stem, 'class_id': gt['class_id'], 'matched': matched,
                'class_correct': class_correct, 'box_iou': box_iou,
                'scale_cat': gt['scale_cat'], 'angle_cat': gt['angle_cat'],
                'occl_cat': gt['occl_cat'], 'lum_cat': lum_cat, 'clutter_cat': clutter_cat
            })

            # Evaluate each keypoint
            diag = max(1.0, gt['diag'])
            for k in range(16):
                gx, gy, gv = gt['keypoints'][k]
                if gv <= 0:
                    continue  # Visible only! Exclude occluded/unlabelled from error diagnosis

                gx64 = gx * (64.0 / orig_w)
                gy64 = gy * (64.0 / orig_h)

                # Raw heatmap peaks near GT
                c_pks = raw_peaks[k]
                near_peaks = sorted(c_pks, key=lambda p: np.hypot(p[0] - gx64, p[1] - gy64))
                best_near_peak = near_peaks[0] if near_peaks else None
                near_dist64 = np.hypot(best_near_peak[0] - gx64, best_near_peak[1] - gy64) if best_near_peak else float('inf')
                near_conf = best_near_peak[2] if best_near_peak else 0.0

                # Decoded keypoint
                dec_px = pred_kps_sc[k] if (matched and pred_kps_sc is not None) else [0.0, 0.0, 0.0]
                has_dec = (dec_px[2] > 0.15)
                dec_dist_px = np.hypot(dec_px[0] - gx, dec_px[1] - gy) if has_dec else float('inf')
                norm_dist = dec_dist_px / diag if has_dec else float('inf')

                is_pck20 = (norm_dist < 0.20)
                is_pck10 = (norm_dist < 0.10)
                is_pck05 = (norm_dist < 0.05)

                # Detailed Error Attribution
                if is_pck20:
                    category = "ACCURATE (Passed PCK@0.20)"
                elif not matched:
                    category = "GROUPING_MISS (Foot Not Detected)"
                elif near_dist64 <= 2.0 and near_conf >= 0.20:
                    category = "DECODER_MISPICK (Good Peak Existed <=2.0px)"
                elif near_dist64 <= 3.0 and near_conf < 0.20:
                    category = "MODEL_FAINT_PEAK (Peak Existed <=3.0px but Conf < 0.20)"
                elif near_dist64 <= 3.0:
                    category = "DECODER_MISPICK (Peak Existed 2-3px)"
                elif near_dist64 > 3.0 and near_dist64 <= 6.0 and near_conf >= 0.20:
                    category = "MODEL_SPATIAL_DRIFT (Peak Shifted 3-6px from GT)"
                else:
                    category = "MODEL_MISSING_PEAK (No Peak within 3.0px)"

                all_kps_data.append({
                    'stem': lf.stem, 'kp_idx': k, 'kp_name': KP_NAMES[k],
                    'is_pck20': is_pck20, 'is_pck10': is_pck10, 'is_pck05': is_pck05,
                    'norm_dist': norm_dist, 'dec_dist_px': dec_dist_px,
                    'near_dist64': near_dist64, 'near_conf': near_conf,
                    'category': category,
                    'scale_cat': gt['scale_cat'],
                    'angle_cat': gt['angle_cat'],
                    'occl_cat': gt['occl_cat'],
                    'lum_cat': lum_cat,
                    'clutter_cat': clutter_cat
                })

    # Summary Statistics
    total_visible_kps = len(all_kps_data)
    total_feet = len(feet_data)
    print(f"\nEvaluated: {total_feet} feet, {total_visible_kps} visible keypoints across 253 validation images.")

    # 1. Overall Error Decomposition
    cat_counts = Counter(d['category'] for d in all_kps_data)
    print("\n" + "=" * 80)
    print("1. OVERALL VALIDATION KEYPOINT LOCALIZATION BREAKDOWN")
    print("=" * 80)
    print(f"{'Category':<55} | {'Count':<8} | {'Percent':<8}")
    print("-" * 80)
    for cat, count in cat_counts.most_common():
        print(f"{cat:<55} | {count:<8} | {count / total_visible_kps * 100:>6.2f}%")

    # Group into broad root causes
    decoder_recoverable = cat_counts["DECODER_MISPICK (Good Peak Existed <=2.0px)"] + cat_counts["DECODER_MISPICK (Peak Existed 2-3px)"]
    model_unrecoverable = (cat_counts["MODEL_MISSING_PEAK (No Peak within 3.0px)"] + 
                           cat_counts["MODEL_FAINT_PEAK (Peak Existed <=3.0px but Conf < 0.20)"] + 
                           cat_counts["MODEL_SPATIAL_DRIFT (Peak Shifted 3-6px from GT)"])
    grouping_miss = cat_counts["GROUPING_MISS (Foot Not Detected)"]
    accurate = cat_counts["ACCURATE (Passed PCK@0.20)"]

    print("\n" + "-" * 80)
    print(f"TOTAL ACCURATE (PCK@0.20)          : {accurate:5d} ({accurate / total_visible_kps * 100:.2f}%)")
    print(f"DECODER-RECOVERABLE FAILURES      : {decoder_recoverable:5d} ({decoder_recoverable / total_visible_kps * 100:.2f}%)")
    print(f"MODEL/HEATMAP-LEVEL FAILURES      : {model_unrecoverable:5d} ({model_unrecoverable / total_visible_kps * 100:.2f}%)")
    print(f"GROUPING MISSES (UNMATCHED FEET)  : {grouping_miss:5d} ({grouping_miss / total_visible_kps * 100:.2f}%)")
    print("-" * 80)

    # 2. Per-Keypoint Detailed Table
    print("\n" + "=" * 90)
    print("2. PER-KEYPOINT ACCURACY & FAILURE ATTRIBUTION TABLE")
    print("=" * 90)
    print(f"{'ID':<3} | {'Keypoint Name':<18} | {'N':<5} | {'PCK@.20':<8} | {'PCK@.10':<8} | {'PCK@.05':<8} | {'Decoder Err':<12} | {'Model Err':<12}")
    print("-" * 90)

    per_kp_res = {}
    for k in range(16):
        k_data = [d for d in all_kps_data if d['kp_idx'] == k]
        n_k = len(k_data)
        pck20 = sum(1 for d in k_data if d['is_pck20']) / max(1, n_k) * 100
        pck10 = sum(1 for d in k_data if d['is_pck10']) / max(1, n_k) * 100
        pck05 = sum(1 for d in k_data if d['is_pck05']) / max(1, n_k) * 100
        dec_err = sum(1 for d in k_data if "DECODER_MISPICK" in d['category'])
        mod_err = sum(1 for d in k_data if "MODEL_" in d['category'])
        
        per_kp_res[KP_NAMES[k]] = {
            'id': k, 'n': n_k, 'pck20': round(pck20, 2), 'pck10': round(pck10, 2), 'pck05': round(pck05, 2),
            'dec_err': dec_err, 'dec_err_pct': round(dec_err / max(1, n_k) * 100, 2),
            'mod_err': mod_err, 'mod_err_pct': round(mod_err / max(1, n_k) * 100, 2)
        }
        print(f"{k:<3} | {KP_NAMES[k]:<18} | {n_k:<5} | {pck20:>7.2f}% | {pck10:>7.2f}% | {pck05:>7.2f}% | {dec_err:>4} ({dec_err/n_k*100:>5.1f}%) | {mod_err:>4} ({mod_err/n_k*100:>5.1f}%)")

    # 3. Stratified Breakdowns
    dimensions = ["scale_cat", "angle_cat", "occl_cat", "lum_cat", "clutter_cat"]
    dim_titles = [
        "A. FOOT SCALE STRATIFICATION",
        "B. CAMERA ANGLE / ORIENTATION STRATIFICATION",
        "C. OCCLUSION SEVERITY STRATIFICATION",
        "D. ILLUMINATION / CONTRAST STRATIFICATION",
        "E. SCENE CLUTTER STRATIFICATION"
    ]

    stratified_res = {}
    for dim_key, dim_title in zip(dimensions, dim_titles):
        print("\n" + "=" * 90)
        print(dim_title)
        print("=" * 90)
        print(f"{'Subgroup':<35} | {'Visible KPs':<12} | {'PCK@.20':<10} | {'PCK@.10':<10} | {'Model Err %':<12} | {'Decoder Err %':<14}")
        print("-" * 90)

        groups = sorted(list(set(d[dim_key] for d in all_kps_data)))
        stratified_res[dim_key] = {}
        for g in groups:
            g_data = [d for d in all_kps_data if d[dim_key] == g]
            n_g = len(g_data)
            pck20 = sum(1 for d in g_data if d['is_pck20']) / max(1, n_g) * 100
            pck10 = sum(1 for d in g_data if d['is_pck10']) / max(1, n_g) * 100
            mod_err_pct = sum(1 for d in g_data if "MODEL_" in d['category']) / max(1, n_g) * 100
            dec_err_pct = sum(1 for d in g_data if "DECODER_MISPICK" in d['category']) / max(1, n_g) * 100

            stratified_res[dim_key][g] = {
                'n': n_g, 'pck20': round(pck20, 2), 'pck10': round(pck10, 2),
                'mod_err_pct': round(mod_err_pct, 2), 'dec_err_pct': round(dec_err_pct, 2)
            }
            print(f"{g:<35} | {n_g:<12} | {pck20:>9.2f}% | {pck10:>9.2f}% | {mod_err_pct:>11.2f}% | {dec_err_pct:>13.2f}%")

    # Foot-level Box IoU & Class Acc Stratification
    print("\n" + "=" * 90)
    print("FOOT-LEVEL METRICS (BOX IoU & CLASS ACCURACY) STRATIFIED")
    print("=" * 90)
    print(f"{'Dimension & Subgroup':<40} | {'Feet (N)':<10} | {'Mean Box IoU':<14} | {'Class Acc %':<14}")
    print("-" * 90)

    foot_stratified = {}
    for dim_key in dimensions:
        groups = sorted(list(set(f[dim_key] for f in feet_data)))
        foot_stratified[dim_key] = {}
        for g in groups:
            gf = [f for f in feet_data if f[dim_key] == g]
            n_f = len(gf)
            mean_iou = np.mean([f['box_iou'] for f in gf]) * 100 if gf else 0.0
            cls_acc = sum(1 for f in gf if f['class_correct']) / max(1, n_f) * 100
            foot_stratified[dim_key][g] = {'n_feet': n_f, 'mean_iou': round(mean_iou, 2), 'cls_acc': round(cls_acc, 2)}
            print(f"{g:<40} | {n_f:<10} | {mean_iou:>13.2f}% | {cls_acc:>13.2f}%")

    out_file = OUT_DIR / "val_failure_decomposition.json"
    with open(out_file, "w") as f:
        json.dump({
            'total_visible_kps': total_visible_kps,
            'total_feet': total_feet,
            'broad_categories': {
                'accurate': accurate, 'accurate_pct': round(accurate / total_visible_kps * 100, 2),
                'decoder_recoverable': decoder_recoverable, 'decoder_recoverable_pct': round(decoder_recoverable / total_visible_kps * 100, 2),
                'model_unrecoverable': model_unrecoverable, 'model_unrecoverable_pct': round(model_unrecoverable / total_visible_kps * 100, 2),
                'grouping_miss': grouping_miss, 'grouping_miss_pct': round(grouping_miss / total_visible_kps * 100, 2)
            },
            'detailed_categories': dict(cat_counts),
            'per_keypoint': per_kp_res,
            'stratified': stratified_res,
            'foot_stratified': foot_stratified
        }, f, indent=2)

    print(f"\nSaved validation failure decomposition to: {out_file}")

if __name__ == "__main__":
    run_val_failure_analysis()
