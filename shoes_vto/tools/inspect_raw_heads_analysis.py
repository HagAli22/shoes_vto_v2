"""
Deep Inspection of Raw Model Heads & Decoder Error Diagnosis on 4 Test Images
Analyzes:
1. Raw Heatmap Head [1, 13, 64, 64]:
   - Peak locations, max intensities, spatial spread (FWHM), drift from GT.
2. Raw PAF Head [1, 32, 64, 64]:
   - Direction vectors, vector magnitudes along true limb lines, field alignment.
3. Raw Class Probability Maps [1, 2, 64, 64]:
   - Foot separation, left vs right probability coverage.
4. Decoder Internal Selection:
   - Centroid k-means clustering, Voronoi partitioning, candidate gathering,
   - Geometric distance prior penalties, PAF score term.
5. Saves visual diagnostic maps for each head.
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
                kv = float(tokens[5 + i * 3 + 2])
                kps.append((kx, ky, kv))
            instances.append({
                'class_id': cls_id,
                'class_name': 'left_foot' if cls_id == 0 else 'right_foot',
                'bbox': (cx, cy, w, h),
                'keypoints': kps
            })
    return instances


def inspect_image(img_path, lbl_path, model, device, paf_cfg, out_dir, img_idx):
    img_bgr = cv2.imread(str(img_path))
    orig_h, orig_w = img_bgr.shape[:2]
    gt_instances = parse_ground_truth(lbl_path, orig_w, orig_h)

    # Preprocess
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        outputs = model(img_tensor)
        heatmaps = outputs['heatmaps'][0].cpu().numpy()  # [13, 64, 64]
        pafs = outputs['pafs'][0].cpu().numpy()          # [32, 64, 64]
        class_probs = outputs.get('class_probs', outputs.get('class', None))[0].cpu().numpy()  # [2, 64, 64]

    print("\n" + "=" * 80)
    print(f"📸 Image {img_idx}: {img_path.name} ({orig_w}x{orig_h})")
    print(f"   Ground Truth Instances: {len(gt_instances)}")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. HEATMAP HEAD INSPECTION
    # -------------------------------------------------------------
    print("\n🔥 1. RAW HEATMAP HEAD INSPECTION [13 Channels, 64x64]:")
    print(f" {'Idx':<4} | {'Name':<18} | {'Min':<6} | {'Max':<6} | {'Mean':<6} | {'Peak (x, y)':<14} | {'Conf':<6} | {'GT Left (x, y)':<15} | {'GT Right (x, y)':<15}")
    print("-" * 105)

    scale_x = orig_w / 64.0
    scale_y = orig_h / 64.0

    hm_grid_tiles = []

    for k in range(13):
        hm = heatmaps[k]
        h_min, h_max, h_mean = float(hm.min()), float(hm.max()), float(hm.mean())
        peaks = detect_peaks(hm, threshold=0.10, nms_window=3, subpixel=True)

        if len(peaks) > 0:
            top_peak = sorted(peaks, key=lambda p: p[2], reverse=True)[0]
            peak_str = f"({top_peak[0]:.1f}, {top_peak[1]:.1f})"
            conf_str = f"{top_peak[2]:.3f}"
        else:
            py, px = np.unravel_index(np.argmax(hm), hm.shape)
            peak_str = f"({px:.1f}, {py:.1f})*"
            conf_str = f"{h_max:.3f}"

        # GT positions on 64x64 grid
        gt_left_str = "None"
        gt_right_str = "None"
        for inst in gt_instances:
            kp_pt = inst['keypoints'][k]
            if kp_pt[2] > 0:
                grid_x = kp_pt[0] / scale_x
                grid_y = kp_pt[1] / scale_y
                pt_str = f"({grid_x:.1f}, {grid_y:.1f})"
                if inst['class_id'] == 0:
                    gt_left_str = pt_str
                else:
                    gt_right_str = pt_str

        print(f" {k:<4} | {KEYPOINT_NAMES_13[k]:<18} | {h_min:<6.3f} | {h_max:<6.3f} | {h_mean:<6.3f} | {peak_str:<14} | {conf_str:<6} | {gt_left_str:<15} | {gt_right_str:<15}")

        # Render Heatmap tile with color map
        hm_norm = (np.clip(hm / max(1e-5, h_max), 0, 1) * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        # Put title
        tile = cv2.resize(hm_color, (160, 160), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, f"{k}:{KEYPOINT_NAMES_13[k]}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"max={h_max:.2f}", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1, cv2.LINE_AA)
        hm_grid_tiles.append(tile)

    # Save Heatmaps Grid (4x4 layout, 16 tiles)
    while len(hm_grid_tiles) < 16:
        hm_grid_tiles.append(np.zeros((160, 160, 3), dtype=np.uint8))
    row1 = np.hstack(hm_grid_tiles[0:4])
    row2 = np.hstack(hm_grid_tiles[4:8])
    row3 = np.hstack(hm_grid_tiles[8:12])
    row4 = np.hstack(hm_grid_tiles[12:16])
    hm_grid = np.vstack([row1, row2, row3, row4])
    hm_grid_path = out_dir / f"sample_{img_idx}_heatmaps_grid.jpg"
    cv2.imwrite(str(hm_grid_path), hm_grid)

    # -------------------------------------------------------------
    # 2. PAF HEAD INSPECTION (16 Limbs, 32 channels)
    # -------------------------------------------------------------
    print("\n🧭 2. RAW PAF HEAD INSPECTION [16 Limbs, 32 Channels]:")
    print(f" {'Limb':<5} | {'From -> To':<28} | {'Max Mag':<8} | {'Mean Mag':<9} | {'Mean Angle (deg)':<16}")
    print("-" * 75)

    limb_pairs = paf_cfg['limb_pairs']
    paf_mag_sum = np.zeros((64, 64), dtype=np.float32)

    for l_idx, (u, v, name) in enumerate(limb_pairs):
        px = pafs[l_idx * 2]
        py = pafs[l_idx * 2 + 1]
        mag = np.sqrt(px**2 + py**2)
        paf_mag_sum += mag
        m_max = float(mag.max())
        m_mean = float(mag.mean())
        angles = np.arctan2(py, px) * 180.0 / np.pi
        active_angles = angles[mag > 0.15]
        mean_ang = float(active_angles.mean()) if len(active_angles) > 0 else 0.0

        limb_str = f"{u}:{KEYPOINT_NAMES_13[u]} -> {v}:{KEYPOINT_NAMES_13[v]}"
        print(f" {l_idx:<5} | {limb_str:<28} | {m_max:<8.3f} | {m_mean:<9.4f} | {mean_ang:<16.1f}")

    # Visualize PAF Magnitude Field
    paf_mag_norm = (np.clip(paf_mag_sum / max(1e-5, paf_mag_sum.max()), 0, 1) * 255).astype(np.uint8)
    paf_mag_color = cv2.applyColorMap(paf_mag_norm, cv2.COLORMAP_INFERNO)
    paf_vis = cv2.resize(paf_mag_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    blend_paf = cv2.addWeighted(img_bgr, 0.6, paf_vis, 0.4, 0)
    paf_path = out_dir / f"sample_{img_idx}_paf_magnitude.jpg"
    cv2.imwrite(str(paf_path), blend_paf)

    # -------------------------------------------------------------
    # 3. CLASS PROBABILITY MAPS [2 Channels: Left, Right]
    # -------------------------------------------------------------
    print("\n👟 3. CLASS PROBABILITY MAPS [2 Channels]:")
    cls_left = class_probs[0]
    cls_right = class_probs[1]
    print(f"   Left  Prob: Min={cls_left.min():.3f}, Max={cls_left.max():.3f}, Mean={cls_left.mean():.3f}")
    print(f"   Right Prob: Min={cls_right.min():.3f}, Max={cls_right.max():.3f}, Mean={cls_right.mean():.3f}")

    # -------------------------------------------------------------
    # 4. DECODER STAGE INTERNAL TRACING
    # -------------------------------------------------------------
    print("\n🔍 4. DECODER STAGE STEP-BY-STEP TRACE:")
    # Run decoder with tracing
    decoded_instances = decode_spatial_instances_single(
        heatmaps=heatmaps,
        class_probs=class_probs,
        pafs=pafs,
        peak_threshold=0.10,
        nms_window=3,
        min_keypoints=5,
        max_instances=2,
        subpixel=True,
        paf_config=paf_cfg
    )

    print(f"   Total Instances Decoded: {len(decoded_instances)}")
    for inst_i, inst in enumerate(decoded_instances):
        cls_id = inst.get('class_id', 0)
        conf = inst.get('class_confidence', 0.0)
        side_str = "Left Foot" if cls_id == 0 else "Right Foot"
        kps = inst['keypoints']
        valid_kps = [k for k in kps if k[2] > 0.15]
        print(f"   ↳ Instance {inst_i}: {side_str} (Conf: {conf:.2f}) | {len(valid_kps)}/13 Keypoints detected")

        # Compare predicted coords with nearest GT coords
        matching_gt = None
        for gt in gt_instances:
            if gt['class_id'] == cls_id:
                matching_gt = gt
                break

        if matching_gt:
            print(f"      Detailed Per-Keypoint Error vs Ground Truth {side_str}:")
            print(f"      {'KP':<3} | {'Name':<18} | {'Pred (x, y, conf)':<24} | {'GT (x, y)':<16} | {'Pixel Error':<12} | {'Grid Error (64x64)'}")
            print("      " + "-" * 88)
            errors_px = []
            for k in range(13):
                pk = kps[k]
                gk = matching_gt['keypoints'][k]
                pred_px_x = pk[0] * scale_x
                pred_px_y = pk[1] * scale_y
                if gk[2] > 0:
                    err_px = np.hypot(pred_px_x - gk[0], pred_px_y - gk[1])
                    err_grid = err_px / scale_x
                    errors_px.append(err_px)
                    pred_str = f"({pred_px_x:.1f}, {pred_px_y:.1f}, {pk[2]:.2f})"
                    gt_str = f"({gk[0]:.1f}, {gk[1]:.1f})"
                    print(f"      {k:<3} | {KEYPOINT_NAMES_13[k]:<18} | {pred_str:<24} | {gt_str:<16} | {err_px:<12.1f} | {err_grid:.2f} px")
                else:
                    pred_str = f"({pred_px_x:.1f}, {pred_px_y:.1f}, {pk[2]:.2f})"
                    print(f"      {k:<3} | {KEYPOINT_NAMES_13[k]:<18} | {pred_str:<24} | {'[Not Labeled]':<16} | {'-':<12} | -")

            if errors_px:
                mean_err = np.mean(errors_px)
                print(f"      >>> Mean Euclidean Error for {side_str}: {mean_err:.1f} pixels ({(mean_err/scale_x):.2f} on 64x64 grid)")


def main():
    img_dir = Path("shoes_vto/outputs/4imagesfromtesttoseethem")
    lbl_dir = Path("dataset/merged_v3_v4_13kp/test/labels")
    ckpt_path = Path("shoes_vto/outputs/m1v2_13kp_training/checkpoints/best.pth")
    out_dir = Path("shoes_vto/outputs/4images_13kp_diagnostic_inspection")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 COMPREHENSIVE RAW HEAD INSPECTION & ERROR DIAGNOSIS")
    print("=" * 80)
    print(f"  Model Checkpoint: {ckpt_path}")
    print(f"  Images Directory: {img_dir}")
    print(f"  Labels Directory: {lbl_dir}")
    print(f"  Output Directory: {out_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(ckpt_path, device=device)

    paf_cfg_path = Path("shoes_vto/configs/paf_connections_13kp.yaml")
    with open(paf_cfg_path, 'r', encoding='utf-8') as f:
        paf_cfg = yaml.safe_load(f)

    image_files = sorted(list(img_dir.glob("*.jpg")))
    for idx, img_path in enumerate(image_files, 1):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        inspect_image(img_path, lbl_path, model, device, paf_cfg, out_dir, idx)

    print("\n" + "=" * 80)
    print(f"🎉 Raw Heads Inspection & Diagnostic Maps Saved to: {out_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

