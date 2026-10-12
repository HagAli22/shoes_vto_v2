import json
from pathlib import Path
import numpy as np
import yaml

def win_path(p):
    s = str(Path(p).resolve())
    if len(s) > 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s

def compute_priors():
    labels_dir = Path("dataset/merged_v3_v4_13kp/train/labels")
    paf_cfg_path = Path("shoes_vto/configs/paf_connections_13kp.yaml")
    out_path = Path("shoes_vto/configs/skeleton_priors_train_13kp.json")

    with open(paf_cfg_path, "r") as f:
        paf_cfg = yaml.safe_load(f)

    limb_pairs = paf_cfg["limb_pairs"]
    edges_to_measure = set()
    for u, v, _ in limb_pairs:
        edges_to_measure.add((min(u, v), max(u, v)))
    
    # Extra edges for 13-KP: (2, 5), (0, 5), (10, 5)
    for u, v in [(2, 5), (0, 5), (10, 5)]:
        edges_to_measure.add((min(u, v), max(u, v)))

    distances = {f"{u}_{v}": [] for u, v in edges_to_measure}

    for label_file in labels_dir.glob("*.txt"):
        with open(win_path(label_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 44:
                continue
            # parts: class_id, cx, cy, w, h, then 13 * 3 keypoints
            # in pixel coords on 64x64 grid (or normalized)
            # YOLO coords are normalized [0, 1]. In keypoint grouping, coords are on 64x64 feature map:
            # Let's check how coordinates are represented:
            # If normalized [0, 1], multiplying by 64 gives 64x64 feature map coords.
            kp_raw = [float(x) for x in parts[5:]]
            kps = []
            for i in range(13):
                x = kp_raw[i * 3] * 64.0
                y = kp_raw[i * 3 + 1] * 64.0
                vis = kp_raw[i * 3 + 2]
                kps.append((x, y, vis))

            # foot scale from anchors [5, 0, 10, 2]
            anchors = [kps[idx][:2] for idx in [5, 0, 10, 2] if kps[idx][2] > 0]
            if len(anchors) >= 2:
                arr = np.array(anchors)
                foot_scale = float(max(10.0, np.hypot(arr[:, 0].max() - arr[:, 0].min(),
                                                     arr[:, 1].max() - arr[:, 1].min()) * 1.3))
            else:
                foot_scale = 22.0

            for u, v in edges_to_measure:
                if kps[u][2] > 0 and kps[v][2] > 0:
                    d = np.hypot(kps[u][0] - kps[v][0], kps[u][1] - kps[v][1])
                    norm_d = d / foot_scale
                    distances[f"{u}_{v}"].append(norm_d)

    priors = {}
    for key, dist_list in distances.items():
        if len(dist_list) >= 10:
            arr = np.array(dist_list)
            u_str, v_str = key.split("_")
            priors[key] = {
                "u": int(u_str),
                "v": int(v_str),
                "mean_dist": round(float(np.mean(arr)), 4),
                "std_dist": round(float(np.std(arr)), 4),
                "min_dist": round(float(np.min(arr)), 4),
                "max_dist": round(float(np.max(arr)), 4),
                "count": len(dist_list)
            }
            # Also store symmetric key so lookup by either u_v or v_u succeeds
            rev_key = f"{v_str}_{u_str}"
            priors[rev_key] = {
                "u": int(v_str),
                "v": int(u_str),
                "mean_dist": priors[key]["mean_dist"],
                "std_dist": priors[key]["std_dist"],
                "min_dist": priors[key]["min_dist"],
                "max_dist": priors[key]["max_dist"],
                "count": len(dist_list)
            }

    with open(out_path, "w") as f:
        json.dump(priors, f, indent=2)

    print(f"Generated 13-KP skeleton priors for {len(distances)} limb edges to {out_path}")

if __name__ == "__main__":
    compute_priors()
