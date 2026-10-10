"""
Calibrate scale-aware bounding box margins from foot keypoints using the TRAINING split only.
"""
from pathlib import Path
import os
import numpy as np
import json

FOOT_KP_INDICES = [i for i in range(16) if i not in (12, 15)]

def safe_open(p, mode="r", encoding="utf-8"):
    p_str = os.path.abspath(str(p))
    if os.name == "nt" and not p_str.startswith("\\\\?\\"):
        p_str = "\\\\?\\" + p_str
    return open(p_str, mode, encoding=encoding)

def calibrate_on_train():
    train_label_dir = Path("dataset/merged_v3_v4/train/labels")
    label_files = list(train_label_dir.glob("*.txt"))
    print(f"Found {len(label_files)} train label files.")

    pad_left_ratios = []
    pad_right_ratios = []
    pad_top_ratios = []
    pad_bottom_ratios = []

    valid_instances = 0
    skipped_files = 0

    for lf in label_files:
        try:
            with safe_open(lf, "r") as f:
                lines = f.readlines()
        except Exception as e:
            skipped_files += 1
            continue

        for line in lines:
            t = line.strip().split()
            if len(t) < 53:
                continue
            cx, cy, bw, bh = map(float, t[1:5])
            # Keypoints in normalized coords [0, 1]
            kps = np.array([[float(t[5+i*3]), float(t[6+i*3]), float(t[7+i*3])] for i in range(16)])
            
            # Select foot-only keypoints
            foot_kps = kps[FOOT_KP_INDICES]
            vis = foot_kps[foot_kps[:, 2] > 0]
            if len(vis) < 5:
                continue

            k_xmin, k_ymin = vis[:, 0].min(), vis[:, 1].min()
            k_xmax, k_ymax = vis[:, 0].max(), vis[:, 1].max()
            kw = k_xmax - k_xmin
            kh = k_ymax - k_ymin

            if kw < 1e-4 or kh < 1e-4:
                continue

            # GT box
            gx1 = cx - bw / 2.0
            gy1 = cy - bh / 2.0
            gx2 = cx + bw / 2.0
            gy2 = cy + bh / 2.0

            # Compute expansions relative to keypoint width and height
            dl = (k_xmin - gx1) / kw
            dr = (gx2 - k_xmax) / kw
            dt = (k_ymin - gy1) / kh
            db = (gy2 - k_ymax) / kh

            pad_left_ratios.append(dl)
            pad_right_ratios.append(dr)
            pad_top_ratios.append(dt)
            pad_bottom_ratios.append(db)
            valid_instances += 1

    print(f"Total valid training foot instances calibrated: {valid_instances} (skipped files: {skipped_files})")
    
    # Robust median margins
    calib = {
        "pad_left": float(np.median(pad_left_ratios)),
        "pad_right": float(np.median(pad_right_ratios)),
        "pad_top": float(np.median(pad_top_ratios)),
        "pad_bottom": float(np.median(pad_bottom_ratios)),
        "mean_pad_left": float(np.mean(pad_left_ratios)),
        "mean_pad_right": float(np.mean(pad_right_ratios)),
        "mean_pad_top": float(np.mean(pad_top_ratios)),
        "mean_pad_bottom": float(np.mean(pad_bottom_ratios)),
    }
    
    out_path = Path("shoes_vto/configs/box_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(out_path, "w") as f:
        json.dump(calib, f, indent=2)

    print("Calibration Results (Medians):")
    print(f"  pad_left   (relative to kw): {calib['pad_left']:.4f}")
    print(f"  pad_right  (relative to kw): {calib['pad_right']:.4f}")
    print(f"  pad_top    (relative to kh): {calib['pad_top']:.4f}")
    print(f"  pad_bottom (relative to kh): {calib['pad_bottom']:.4f}")
    print(f"Saved calibration config to: {out_path}")

if __name__ == "__main__":
    calibrate_on_train()

