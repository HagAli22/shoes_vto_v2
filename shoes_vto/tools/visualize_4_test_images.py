"""
Visualize Clean Image, Ground Truth, and Model Predictions for 4 Test Images

Generates 3-panel comparisons:
[Panel 1: Clean Image] | [Panel 2: Ground Truth] | [Panel 3: Model Prediction]
for all 4 images in shoes_vto/outputs/4imagesfromtesttoseethem.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from utils.keypoint_grouping import group_keypoints, load_paf_config, compute_foot_box

# Colors (BGR)
COLOR_LEFT = (235, 140, 25)       # Cyan / Sky Blue for Left Foot
COLOR_RIGHT = (35, 50, 240)       # Red / Coral for Right Foot
COLOR_GT_LEFT = (220, 220, 0)     # Cyan/Green for GT Left
COLOR_GT_RIGHT = (0, 200, 20)     # Bright Green for GT Right
COLOR_KP_INNER = (255, 255, 255)
COLOR_HEADER_BG = (25, 25, 25)

KEYPOINT_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "ball_top", "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "ankle_center",
    "throat", "achilles", "shin_mid"
]


def load_model(checkpoint_path, device='cpu'):
    model = ARShoeM1v2()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    cleaned_sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(cleaned_sd)
    model.to(device)
    model.eval()
    return model


def parse_ground_truth_label(label_path, img_w, img_h):
    """
    Parse YOLO format keypoint annotations from text file.
    Returns list of instances:
    [{'class_id': int, 'bbox': (x1, y1, x2, y2), 'keypoints': [[x, y, v], ...]}, ...]
    """
    instances = []
    if not label_path.exists():
        return instances

    with open(label_path, 'r') as f:
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

            # Keypoints
            kps = []
            for i in range(16):
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
                'keypoints': kps
            })

    return instances


def draw_panel_header(img, title, subtitle=None, bg_color=COLOR_HEADER_BG):
    """Draw a professional title bar at the top of a panel."""
    h, w = img.shape[:2]
    header_h = 44
    cv2.rectangle(img, (0, 0), (w, header_h), bg_color, -1)
    cv2.line(img, (0, header_h), (w, header_h), (80, 80, 80), 1)

    cv2.putText(img, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        (tw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(img, subtitle, (w - tw - 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1, cv2.LINE_AA)


def draw_ground_truth(base_img, gt_instances, limb_pairs):
    """Draw Ground Truth annotations (limbs, keypoints, bboxes, labels)."""
    img = base_img.copy()
    img_h, img_w = img.shape[:2]

    for inst in gt_instances:
        cls_id = inst['class_id']
        side_name = "Left Foot" if cls_id == 0 else "Right Foot"
        color = COLOR_GT_LEFT if cls_id == 0 else COLOR_GT_RIGHT
        kps = inst['keypoints']
        x1, y1, x2, y2 = inst['bbox']

        # 1. Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        badge_text = f"GT: {side_name}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        by1 = max(48, y1 - th - 8)
        by2 = y1
        cv2.rectangle(img, (x1, by1), (x1 + tw + 10, by2), color, -1)
        cv2.putText(img, badge_text, (x1 + 5, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

        # 2. Limbs
        for p1, p2, _ in limb_pairs:
            if p1 < len(kps) and p2 < len(kps):
                pt1 = kps[p1]
                pt2 = kps[p2]
                if pt1[2] > 0 and pt2[2] > 0:
                    c1 = (int(round(pt1[0])), int(round(pt1[1])))
                    c2 = (int(round(pt2[0])), int(round(pt2[1])))
                    cv2.line(img, c1, c2, color, 3, cv2.LINE_AA)

        # 3. Keypoints
        for idx, (kx, ky, kv) in enumerate(kps):
            if kv > 0:
                cx_pt = int(round(kx))
                cy_pt = int(round(ky))
                cv2.circle(img, (cx_pt, cy_pt), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, (cx_pt, cy_pt), 4, color, -1, cv2.LINE_AA)
                cv2.putText(img, str(idx), (cx_pt + 5, cy_pt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_predictions(base_img, pred_instances, limb_pairs, scale_x, scale_y):
    """Draw Model Predicted annotations (limbs, keypoints, bboxes, side labels)."""
    img = base_img.copy()
    img_h, img_w = img.shape[:2]

    for inst in pred_instances:
        cls_id = inst.get('class_id', 0)
        conf = inst.get('class_confidence', 0.9)
        side_name = "Left Foot" if cls_id == 0 else "Right Foot"
        color = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        kps = inst['keypoints']

        # Scale keypoints to image coordinates
        scaled_kps = []
        visible_pts = []
        for kp in kps:
            sx = kp[0] * scale_x
            sy = kp[1] * scale_y
            sc = kp[2]
            scaled_kps.append([sx, sy, sc])
            if sc > 0.15:
                visible_pts.append((sx, sy))

        # Calibrated bounding box computed from 14 foot keypoints (excluding ankle_center & shin_mid)
        box = compute_foot_box(scaled_kps, img_w, img_h, scale_x=1.0, scale_y=1.0)
        if box is not None:
            x1, y1, x2, y2 = box
            y1 = max(48, y1)

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            badge_text = f"M1v2: {side_name} ({conf * 100:.0f}%)"
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


def process_4_images():
    img_dir = Path("shoes_vto/outputs/4imagesfromtesttoseethem")
    lbl_dir = Path("dataset/merged_v3_v4/test/labels")
    ckpt_path = Path("shoes_vto/outputs/m1v2_training/checkpoints/best.pth")
    out_dir = img_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎨 Generating Clean vs GT vs Model Prediction Visualizations")
    print("=" * 80)
    print(f"  Images Directory : {img_dir}")
    print(f"  Labels Directory : {lbl_dir}")
    print(f"  Output Directory : {out_dir}")
    print(f"  Model Checkpoint : {ckpt_path}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model on {device}...")
    model = load_model(ckpt_path, device=device)

    # PAF config
    paf_config = load_paf_config()
    limb_pairs = paf_config['limb_pairs']

    image_files = sorted(list(img_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} test images.\n")

    composite_rows = []

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        orig_h, orig_w = img_bgr.shape[:2]

        # 1. Clean panel
        clean_panel = img_bgr.copy()
        draw_panel_header(clean_panel, "1. Clean Original Image", f"{orig_w}x{orig_h}")

        # 2. Ground Truth panel
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_instances = parse_ground_truth_label(lbl_path, orig_w, orig_h)
        gt_panel = draw_ground_truth(img_bgr, gt_instances, limb_pairs)
        draw_panel_header(gt_panel, "2. Ground Truth Annotations", f"{len(gt_instances)} feet labeled")

        # 3. Model Prediction panel
        # Resize to 256x256 for model
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            out = model(img_tensor)
            hm = out['heatmaps'].cpu()
            paf = out['pafs'].cpu()
            cls_probs = out['class'].cpu()

        batch_instances = group_keypoints(
            heatmaps=hm,
            pafs=paf,
            class_probs=cls_probs,
            peak_threshold=0.10,
            paf_threshold=0.05,
            nms_window=3,
            min_keypoints=5,
            max_instances=2,
            subpixel=True
        )
        pred_instances = batch_instances[0]
        # Filter weak predictions
        pred_instances = [inst for inst in pred_instances if inst.get('score', 0) >= 0.15]

        scale_x = orig_w / 64.0
        scale_y = orig_h / 64.0
        pred_panel = draw_predictions(img_bgr, pred_instances, limb_pairs, scale_x, scale_y)
        draw_panel_header(pred_panel, "3. ARShoe M1v2 Prediction", f"{len(pred_instances)} feet detected")

        # Side-by-side composite: [Clean | GT | Prediction]
        side_by_side = np.hstack([clean_panel, gt_panel, pred_panel])
        
        # Save individual panels
        cv2.imwrite(str(out_dir / f"sample_{i}_clean.jpg"), clean_panel)
        cv2.imwrite(str(out_dir / f"sample_{i}_groundtruth.jpg"), gt_panel)
        cv2.imwrite(str(out_dir / f"sample_{i}_prediction.jpg"), pred_panel)
        
        # Save side-by-side comparison
        out_comp = out_dir / f"comparison_sample_{i}.jpg"
        cv2.imwrite(str(out_comp), side_by_side)
        print(f"  ✅ Saved 3-panel comparison -> {out_comp.name}")

        # Store a scaled copy for all-in-one summary
        scaled_row = cv2.resize(side_by_side, (1800, int(1800 * orig_h / (orig_w * 3))))
        composite_rows.append(scaled_row)

    # Save all-in-one summary poster
    if composite_rows:
        all_samples_poster = np.vstack(composite_rows)
        poster_path = out_dir / "all_4_samples_comparison.jpg"
        cv2.imwrite(str(poster_path), all_samples_poster)
        print(f"\n🎉 Saved all 4 samples comparison poster -> {poster_path.name}")
        print(f"📁 All visualizations ready in: {out_dir}")


if __name__ == '__main__':
    process_4_images()

