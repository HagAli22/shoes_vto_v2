import os
import glob
import math
import json
import numpy as np
from PIL import Image

TRAIN_IMG_DIR = "dataset/merged_v3_v4/train/images"
TRAIN_LBL_DIR = "dataset/merged_v3_v4/train/labels"
VAL_IMG_DIR = "dataset/merged_v3_v4/valid/images"
VAL_LBL_DIR = "dataset/merged_v3_v4/valid/labels"
OUTPUT_JSON = "shoes_vto/outputs/error_analysis/train_vs_val_distribution.json"

KEYPOINT_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "malleolus_medial",
    "malleolus_lateral", "first_metatarsal", "fifth_metatarsal",
    "navicular", "peroneus_brevis", "distal_phalanx", "calcaneus_post",
    "toe_tip", "ball_medial", "ball_lateral", "ankle_center", "shin_mid"
]

def safe_open(path, mode='r', **kwargs):
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        abs_path = '\\\\?\\' + abs_path
    return open(abs_path, mode, **kwargs)

def parse_yolo_labels(lbl_dir):
    records = []
    lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    for f in lbl_files:
        base_name = os.path.splitext(os.path.basename(f))[0]
        try:
            with safe_open(f, 'r') as fp:
                lines = [l.strip() for l in fp if l.strip()]
        except Exception as e:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            kp_data = []
            raw_kp = [float(x) for x in parts[5:]]
            # Keypoints are either (x, y, v) or (x, y)
            step = 3 if len(raw_kp) >= 48 else 2
            for i in range(16):
                idx = i * step
                if idx + 1 < len(raw_kp):
                    kx = raw_kp[idx]
                    ky = raw_kp[idx+1]
                    v = int(raw_kp[idx+2]) if step == 3 and idx+2 < len(raw_kp) else (2 if (kx > 0 or ky > 0) else 0)
                    kp_data.append((kx, ky, v))
                else:
                    kp_data.append((0.0, 0.0, 0))
            records.append({
                "image": base_name,
                "class_id": cls_id,
                "bbox": (cx, cy, w, h),
                "keypoints": kp_data
            })
    return records

def compute_distribution(records):
    total_feet = len(records)
    images = set(r["image"] for r in records)
    total_images = len(images)
    
    # Scale: sqrt(w * h)
    scales = []
    angles = []
    kp_vis = {name: {"v0": 0, "v1": 0, "v2": 0} for name in KEYPOINT_NAMES}
    kp_collisions = {name: 0 for name in KEYPOINT_NAMES} # instances where KP is identical to another visible KP
    
    for r in records:
        w, h = r["bbox"][2], r["bbox"][3]
        scale_pct = math.sqrt(w * h) * 100.0
        scales.append(scale_pct)
        
        # Camera angle / foot inclination angle
        # Estimated by angle of line between ankle_center (14) and heel_ground (2) or shin_mid (15) relative to vertical
        kps = r["keypoints"]
        # If heel_ground (2) and toe_ground (0) are visible, elevation angle
        hg = kps[2]
        tg = kps[0]
        if hg[2] > 0 and tg[2] > 0:
            dx = abs(tg[0] - hg[0])
            dy = abs(tg[1] - hg[1])
            # angle with horizontal ground plane
            angle_deg = math.degrees(math.atan2(dy, max(dx, 1e-4)))
            angles.append(angle_deg)
            
        for i, (kx, ky, v) in enumerate(kps):
            name = KEYPOINT_NAMES[i]
            if v == 0 or (kx == 0 and ky == 0):
                kp_vis[name]["v0"] += 1
            elif v == 1:
                kp_vis[name]["v1"] += 1
            else:
                kp_vis[name]["v2"] += 1
                
        # Check coordinate collisions (identical coordinates within 1e-4)
        for i in range(16):
            if kps[i][2] > 0 and (kps[i][0] > 0 or kps[i][1] > 0):
                for j in range(i + 1, 16):
                    if kps[j][2] > 0 and (kps[j][0] > 0 or kps[j][1] > 0):
                        dist = math.hypot(kps[i][0] - kps[j][0], kps[i][1] - kps[j][1])
                        if dist < 0.005: # less than 0.5% of image width (~1.3 px on 256)
                            kp_collisions[KEYPOINT_NAMES[i]] += 1
                            kp_collisions[KEYPOINT_NAMES[j]] += 1

    scales = np.array(scales) if scales else np.array([0])
    angles = np.array(angles) if angles else np.array([0])
    
    # Scale brackets
    small_scale = float(np.mean(scales < 22.0) * 100.0)
    med_scale = float(np.mean((scales >= 22.0) & (scales <= 32.0)) * 100.0)
    large_scale = float(np.mean(scales > 32.0) * 100.0)
    
    # Angle brackets
    flat_angle = float(np.mean(angles < 25.0) * 100.0)
    oblique_angle = float(np.mean((angles >= 25.0) & (angles <= 55.0)) * 100.0)
    steep_angle = float(np.mean(angles > 55.0) * 100.0)
    
    return {
        "total_images": total_images,
        "total_feet": total_feet,
        "scale_distribution": {
            "mean": float(np.mean(scales)),
            "std": float(np.std(scales)),
            "small_pct (<22%)": small_scale,
            "medium_pct (22-32%)": med_scale,
            "large_pct (>32%)": large_scale
        },
        "angle_distribution": {
            "mean_deg": float(np.mean(angles)),
            "std_deg": float(np.std(angles)),
            "flat_pct (<25 deg)": flat_angle,
            "oblique_pct (25-55 deg)": oblique_angle,
            "steep_pct (>55 deg)": steep_angle
        },
        "keypoint_visibility": {
            k: {
                "visible_pct": round((v["v1"] + v["v2"]) / max(1, total_feet) * 100.0, 2),
                "occluded_v0_pct": round(v["v0"] / max(1, total_feet) * 100.0, 2),
                "counts": v
            }
            for k, v in kp_vis.items()
        },
        "keypoint_collisions_count": kp_collisions
    }

def main():
    print(f"Parsing Train labels from {TRAIN_LBL_DIR}...")
    train_records = parse_yolo_labels(TRAIN_LBL_DIR)
    print(f"Loaded {len(train_records)} train feet.")
    
    print(f"Parsing Val labels from {VAL_LBL_DIR}...")
    val_records = parse_yolo_labels(VAL_LBL_DIR)
    print(f"Loaded {len(val_records)} val feet.")
    
    print("Computing distributions...")
    train_dist = compute_distribution(train_records)
    val_dist = compute_distribution(val_records)
    
    result = {
        "train": train_dist,
        "val": val_dist
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"Saved distribution comparison to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
