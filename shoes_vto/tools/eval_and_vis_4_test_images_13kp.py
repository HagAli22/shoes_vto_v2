"""
Evaluate and Visualize 4 Test Images with the Trained 13-Keypoint ARShoe M1v2 Model
Outputs saved to: shoes_vto/outputs/4images_13kp_eval

Generates:
1. Clean Original Image panel
2. Ground Truth Annotations panel (13 keypoints, 16 limbs)
3. ARShoe M1v2 13-KP Predictions panel (using production skeleton-consistent decoder)
4. 3-Panel side-by-side comparisons: [Clean | Ground Truth | Prediction]
5. All-in-one summary poster (all 4 samples stacked)
6. Exact numerical PCK evaluation metrics for these 4 samples.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Paths setup
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from utils.keypoint_grouping import group_keypoints, load_paf_config, compute_foot_box
from training.eval_keypoints import compute_multi_threshold_pck

# Colors (BGR)
COLOR_LEFT = (235, 140, 25)       # Sky Blue / Cyan for Left Foot prediction
COLOR_RIGHT = (35, 50, 240)       # Coral / Red for Right Foot prediction
COLOR_GT_LEFT = (220, 220, 0)     # Cyan/Green for GT Left
COLOR_GT_RIGHT = (0, 200, 20)     # Bright Green for GT Right
COLOR_KP_INNER = (255, 255, 255)
COLOR_HEADER_BG = (25, 25, 25)

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


def load_model(checkpoint_path, device='cpu'):
    model = ARShoeM1v2(pretrained=False, num_keypoints=13, num_limbs=16, num_classes=2)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(cleaned_sd, strict=True)
    if missing or unexpected:
        print(f"Warning: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def parse_ground_truth_label_13kp(label_path, img_w, img_h):
    instances = []
    if not label_path.exists():
        return instances

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            cls_id = int(tokens[0])
            cx, cy, w, h = [float(x) for x in tokens[1:5]]

            # Pixel bbox
            x1 = int(max(0, (cx - w / 2.0) * img_w))
            y1 = int(max(0, (cy - h / 2.0) * img_h))
            x2 = int(min(img_w - 1, (cx + w / 2.0) * img_w))
            y2 = int(min(img_h - 1, (cy + h / 2.0) * img_h))

            # 13 Keypoints
            kps = []
            for i in range(13):
                idx = 5 + i * 3
                if idx + 2 < len(tokens):
                    kx = float(tokens[idx]) * img_w
                    ky = float(tokens[idx + 1]) * img_h
                    kv = float(tokens[idx + 2])
                    kps.append([kx, ky, kv])
                else:
                    kps.append([0.0, 0.0, 0.0])

            instances.append({
                'class_id': cls_id,
                'bbox': (x1, y1, x2, y2),
                'norm_bbox': (cx, cy, w, h),
                'keypoints': kps
            })

    return instances


def draw_panel_header(img, title, subtitle=None, bg_color=COLOR_HEADER_BG):
    h, w = img.shape[:2]
    header_h = 44
    cv2.rectangle(img, (0, 0), (w, header_h), bg_color, -1)
    cv2.line(img, (0, header_h), (w, header_h), (80, 80, 80), 1)

    cv2.putText(img, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        (tw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(img, subtitle, (w - tw - 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1, cv2.LINE_AA)


def draw_ground_truth(base_img, gt_instances, limb_pairs):
    img = base_img.copy()
    img_h, img_w = img.shape[:2]

    for inst in gt_instances:
        cls_id = inst['class_id']
        side_name = "Left Foot" if cls_id == 0 else "Right Foot"
        color = COLOR_GT_LEFT if cls_id == 0 else COLOR_GT_RIGHT
        kps = inst['keypoints']

        # Bbox
        x1, y1, x2, y2 = inst['bbox']
        y1 = max(48, y1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        badge_text = f"GT: {side_name}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        by1 = max(48, y1 - th - 8)
        by2 = y1
        cv2.rectangle(img, (x1, by1), (x1 + tw + 10, by2), color, -1)
        cv2.putText(img, badge_text, (x1 + 5, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

        # Limbs
        for p1, p2, _ in limb_pairs:
            if p1 < len(kps) and p2 < len(kps):
                pt1 = kps[p1]
                pt2 = kps[p2]
                if pt1[2] > 0 and pt2[2] > 0:
                    c1 = (int(round(pt1[0])), int(round(pt1[1])))
                    c2 = (int(round(pt2[0])), int(round(pt2[1])))
                    cv2.line(img, c1, c2, color, 3, cv2.LINE_AA)

        # Keypoints
        for idx, (kx, ky, kv) in enumerate(kps):
            if kv > 0:
                cx_pt = int(round(kx))
                cy_pt = int(round(ky))
                cv2.circle(img, (cx_pt, cy_pt), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, (cx_pt, cy_pt), 4, color, -1, cv2.LINE_AA)
                cv2.putText(img, str(idx), (cx_pt + 5, cy_pt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_predictions(base_img, pred_instances, limb_pairs, scale_x, scale_y):
    img = base_img.copy()
    img_h, img_w = img.shape[:2]

    for inst in pred_instances:
        cls_id = inst.get('class_id', 0)
        conf = inst.get('class_confidence', 0.9)
        side_name = "Left Foot" if cls_id == 0 else "Right Foot"
        color = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        kps = inst['keypoints']

        scaled_kps = []
        for kp in kps:
            sx = kp[0] * scale_x
            sy = kp[1] * scale_y
            sc = kp[2]
            scaled_kps.append([sx, sy, sc])

        box = compute_foot_box(scaled_kps, img_w, img_h, scale_x=1.0, scale_y=1.0)
        if box is not None:
            x1, y1, x2, y2 = box
            y1 = max(48, y1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            badge_text = f"M1v2 (13KP): {side_name} ({conf * 100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            by1 = max(48, y1 - th - 8)
            by2 = y1
            cv2.rectangle(img, (x1, by1), (x1 + tw + 10, by2), color, -1)
            cv2.putText(img, badge_text, (x1 + 5, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Limbs
        for p1, p2, _ in limb_pairs:
            if p1 < len(scaled_kps) and p2 < len(scaled_kps):
                pt1 = scaled_kps[p1]
                pt2 = scaled_kps[p2]
                if pt1[2] > 0.15 and pt2[2] > 0.15:
                    c1 = (int(round(pt1[0])), int(round(pt1[1])))
                    c2 = (int(round(pt2[0])), int(round(pt2[1])))
                    cv2.line(img, c1, c2, color, 3, cv2.LINE_AA)

        # Keypoints
        for idx, (kx, ky, kv) in enumerate(scaled_kps):
            if kv > 0.15:
                cx_pt = int(round(kx))
                cy_pt = int(round(ky))
                cv2.circle(img, (cx_pt, cy_pt), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, (cx_pt, cy_pt), 4, color, -1, cv2.LINE_AA)
                cv2.putText(img, str(idx), (cx_pt + 5, cy_pt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate and visualize 4 test images")
    parser.add_argument('--checkpoint', type=str, default="shoes_vto/outputs/m1v2_13kp_training/checkpoints/best.pth")
    parser.add_argument('--img_dir', type=str, default="shoes_vto/outputs/4imagesfromtesttoseethem")
    parser.add_argument('--lbl_dir', type=str, default="dataset/merged_v3_v4_13kp/test/labels")
    parser.add_argument('--out_dir', type=str, default="shoes_vto/outputs/4images_13kp_eval")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    lbl_dir = Path(args.lbl_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("👟 Evaluating & Visualizing 4 Test Images (13-KP Model)")
    print("=" * 80)
    print(f"  Source Images      : {img_dir}")
    print(f"  Ground Truth Labels: {lbl_dir}")
    print(f"  Model Checkpoint   : {ckpt_path}")
    print(f"  Target Output Dir  : {out_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device             : {device}")

    # Load model
    model = load_model(ckpt_path, device=device)

    # Load 13-KP PAF config
    paf_cfg_path = Path("shoes_vto/configs/paf_connections_13kp.yaml")
    with open(paf_cfg_path, 'r', encoding='utf-8') as f:
        paf_cfg = yaml.safe_load(f)
    limb_pairs = paf_cfg['limb_pairs']

    image_files = sorted(list(img_dir.glob("*.jpg")))
    print(f"\nProcessing {len(image_files)} test images...\n")

    composite_rows = []
    all_pred_kps = []
    all_gt_kps = []
    all_gt_vis = []
    all_diags = []

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        orig_h, orig_w = img_bgr.shape[:2]

        # 1. Clean panel
        clean_panel = img_bgr.copy()
        draw_panel_header(clean_panel, "1. Clean Original Image", f"{orig_w}x{orig_h}")

        # 2. Ground Truth panel
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_instances = parse_ground_truth_label_13kp(lbl_path, orig_w, orig_h)
        gt_panel = draw_ground_truth(img_bgr, gt_instances, limb_pairs)
        draw_panel_header(gt_panel, "2. Ground Truth (13-KP)", f"{len(gt_instances)} feet labeled")

        # 3. Model Prediction panel
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            out = model(img_tensor)
            hm = out['heatmaps'].cpu()
            paf = out['pafs'].cpu()
            cls_probs = out.get('class_probs', out.get('class', None)).cpu()

        batch_instances = group_keypoints(
            heatmaps=hm,
            pafs=paf,
            class_probs=cls_probs,
            peak_threshold=0.10,
            paf_threshold=0.05,
            nms_window=3,
            min_keypoints=5,
            max_instances=2,
            subpixel=True,
            method='spatial',
            paf_config=paf_cfg
        )
        pred_instances = batch_instances[0]
        pred_instances = [inst for inst in pred_instances if inst.get('score', 0) >= 0.15]

        scale_x = orig_w / 64.0
        scale_y = orig_h / 64.0
        pred_panel = draw_predictions(img_bgr, pred_instances, limb_pairs, scale_x, scale_y)
        draw_panel_header(pred_panel, "3. ARShoe M1v2 (13-KP) Prediction", f"{len(pred_instances)} feet detected")

        # Match predictions to GT for PCK computation
        scale_x_norm = orig_w
        scale_y_norm = orig_h
        for gt_inst in gt_instances:
            gt_kp = np.array([kp[:2] for kp in gt_inst['keypoints']])
            gt_v = np.array([kp[2] for kp in gt_inst['keypoints']])
            norm_cx, norm_cy, norm_w, norm_h = gt_inst['norm_bbox']
            diag = float(np.hypot(norm_w * orig_w, norm_h * orig_h))

            # Find matching pred instance
            best_pred_kp = None
            best_match_score = -1e9
            for p_inst in pred_instances:
                if p_inst.get('class_id', 0) == gt_inst['class_id']:
                    p_kps = np.array([kp[:2] for kp in p_inst['keypoints']])
                    # Scale to original pixels
                    p_kps[:, 0] *= scale_x
                    p_kps[:, 1] *= scale_y
                    best_pred_kp = p_kps
                    break

            if best_pred_kp is None and len(pred_instances) > 0:
                p_kps = np.array([kp[:2] for kp in pred_instances[0]['keypoints']])
                p_kps[:, 0] *= scale_x
                p_kps[:, 1] *= scale_y
                best_pred_kp = p_kps

            if best_pred_kp is not None:
                all_pred_kps.append(best_pred_kp)
                all_gt_kps.append(gt_kp)
                all_gt_vis.append(gt_v)
                all_diags.append(diag)

        # 3-Panel composite: [Clean | GT | Prediction]
        side_by_side = np.hstack([clean_panel, gt_panel, pred_panel])

        # Save individual panels and comparison
        cv2.imwrite(str(out_dir / f"sample_{i}_clean.jpg"), clean_panel)
        cv2.imwrite(str(out_dir / f"sample_{i}_groundtruth.jpg"), gt_panel)
        cv2.imwrite(str(out_dir / f"sample_{i}_prediction.jpg"), pred_panel)

        out_comp = out_dir / f"comparison_sample_{i}.jpg"
        cv2.imwrite(str(out_comp), side_by_side)
        print(f"  ✅ Saved comparison: {out_comp.name}")

        scaled_row = cv2.resize(side_by_side, (1800, int(1800 * orig_h / (orig_w * 3))))
        composite_rows.append(scaled_row)

    # Save summary poster
    if composite_rows:
        all_samples_poster = np.vstack(composite_rows)
        poster_path = out_dir / "all_4_samples_comparison.jpg"
        cv2.imwrite(str(poster_path), all_samples_poster)
        print(f"\n🎉 Saved all-in-one poster: {poster_path.name}")

    # Compute numerical PCK for these 4 samples
    if len(all_pred_kps) > 0:
        pck_metrics = compute_multi_threshold_pck(
            pred_keypoints=all_pred_kps,
            gt_keypoints=all_gt_kps,
            gt_visibility=all_gt_vis,
            bbox_diags=all_diags,
            thresholds=(0.20, 0.10, 0.05)
        )
        summary_file = out_dir / "evaluation_summary_4images.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(pck_metrics, f, indent=2)

        print("\n" + "=" * 80)
        print("📊 NUMERICAL EVALUATION ON THE 4 TEST IMAGES:")
        print("=" * 80)
        ov = pck_metrics['overall']
        print(f"  • PCK@0.20 (Coarse) : {ov.get(0.2, ov.get('0.2', 0.0)):.2%}")
        print(f"  • PCK@0.10 (Medium) : {ov.get(0.1, ov.get('0.1', 0.0)):.2%}")
        print(f"  • PCK@0.05 (Precise): {ov.get(0.05, ov.get('0.05', 0.0)):.2%}")
        print(f"  • Total Feet Evaluated: {len(all_pred_kps)}")
        print(f"  • Total Visible Keypoints: {pck_metrics.get('total_keypoints_evaluated', 'N/A')}")
        print("=" * 80)

    print(f"\n📁 All deliverables saved in: {out_dir}")


if __name__ == '__main__':
    main()

