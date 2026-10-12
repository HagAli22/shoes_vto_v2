"""
Unified Pipeline for Google Colab and Local Environments:
Convert raw shuffled_v3 and shuffled_v4 directly into the verified 13-Keypoint dataset.

Features:
1. Automatic path discovery for Google Colab (/content/shoes_vto_v2/shoes_vto/dataset/...)
2. Cross-platform safe path handling (Linux POSIX + Windows long paths)
3. SHA-256 deduplication and strict split-isolation check (0 data leakage)
4. Dynamic 16 -> 13 keypoint filtering:
   - Removes: ball_top (KP 5), ankle_center (KP 12), shin_mid (KP 15)
   - Remaps remaining 13 keypoints to indices 0..12
5. Computes skeleton distance priors to configs/skeleton_priors_train_13kp.json
6. Performs full integrity validation with detailed statistics
"""

import os
import sys
import shutil
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def safe_path(p):
    p_str = str(Path(p).resolve())
    if sys.platform == 'win32' and len(p_str) > 240 and not p_str.startswith('\\\\?\\'):
        return '\\\\?\\' + p_str
    return p_str


def compute_hash(filepath):
    h = hashlib.sha256()
    with open(safe_path(filepath), 'rb') as f:
        while chunk := f.read(32768):
            h.update(chunk)
    return h.hexdigest()


# 13-KP selection mapping from 16-KP original format:
# Kept indices: 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14
KEPT_16_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
REMOVED_16_INDICES = [5, 12, 15]  # ball_top, ankle_center, shin_mid


def find_dataset_dir(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists() and (p / "train" / "images").exists():
            return p.resolve()
    return None


def convert_yolo_line_16_to_13(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    class_id = parts[0]
    bbox = parts[1:5]
    kp_data = parts[5:]

    num_kps_in_file = len(kp_data) // 3
    if num_kps_in_file < 16:
        return None

    new_kps = []
    for old_idx in KEPT_16_INDICES:
        start = old_idx * 3
        if start + 2 < len(kp_data):
            x = kp_data[start]
            y = kp_data[start + 1]
            v = kp_data[start + 2]
            new_kps.extend([x, y, v])
        else:
            new_kps.extend(["0.0", "0.0", "0"])

    tokens = [class_id] + bbox + new_kps
    return " ".join(tokens) + "\n"


def compute_priors_13kp(train_labels_dir, paf_cfg_path, out_priors_path):
    if not paf_cfg_path.exists():
        print(f"Warning: PAF config {paf_cfg_path} not found. Skipping priors computation.")
        return

    with open(safe_path(paf_cfg_path), "r", encoding="utf-8") as f:
        paf_cfg = yaml.safe_load(f)

    limb_pairs = paf_cfg.get("limb_pairs", [])
    edges_to_measure = set()
    for u, v, _ in limb_pairs:
        edges_to_measure.add((min(u, v), max(u, v)))
    for u, v in [(2, 5), (0, 5), (10, 5)]:
        edges_to_measure.add((min(u, v), max(u, v)))

    distances = {f"{u}_{v}": [] for u, v in edges_to_measure}

    for label_file in Path(train_labels_dir).glob("*.txt"):
        with open(safe_path(label_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 44:
                continue
            kp_raw = [float(x) for x in parts[5:]]
            kps = []
            for i in range(13):
                x = kp_raw[i * 3] * 64.0
                y = kp_raw[i * 3 + 1] * 64.0
                vis = kp_raw[i * 3 + 2]
                kps.append((x, y, vis))

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

    out_priors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(safe_path(out_priors_path), "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2)
    print(f"✅ Generated 13-KP skeleton distance priors at: {out_priors_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare 13-KP Dataset from shuffled_v3 and shuffled_v4 in Colab or Local")
    parser.add_argument('--v3_dir', type=str, default=None, help="Path to shuffled_v3 directory")
    parser.add_argument('--v4_dir', type=str, default=None, help="Path to shuffled_v4 directory")
    parser.add_argument('--output_dir', type=str, default=None, help="Path to output 13-KP dataset")
    args = parser.parse_args()

    # Smart discovery for Colab and Local
    cwd = Path.cwd()
    v3_candidates = [
        args.v3_dir,
        cwd / "dataset" / "shuffled_v3",
        cwd / "shoes_vto" / "dataset" / "shuffled_v3",
        Path("/content/shoes_vto_v2/shoes_vto/dataset/shuffled_v3"),
        Path("/content/dataset/shuffled_v3"),
    ]
    v4_candidates = [
        args.v4_dir,
        cwd / "dataset" / "shuffled_v4",
        cwd / "shoes_vto" / "dataset" / "shuffled_v4",
        Path("/content/shoes_vto_v2/shoes_vto/dataset/shuffled_v4"),
        Path("/content/dataset/shuffled_v4"),
    ]

    v3_path = find_dataset_dir([c for c in v3_candidates if c is not None])
    v4_path = find_dataset_dir([c for c in v4_candidates if c is not None])

    if v3_path is None or v4_path is None:
        print("❌ Error: Could not locate shuffled_v3 or shuffled_v4 datasets.")
        print(f"Checked v3 candidates: {[str(c) for c in v3_candidates if c is not None]}")
        print(f"Checked v4 candidates: {[str(c) for c in v4_candidates if c is not None]}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        # Save alongside existing dataset directory
        output_dir = v3_path.parent / "merged_v3_v4_13kp"

    print("=" * 80)
    print("👟 13-KEYPOINT DATASET PREPARATION PIPELINE (COLAB & LOCAL)")
    print("=" * 80)
    print(f"  Source v3 : {v3_path}")
    print(f"  Source v4 : {v4_path}")
    print(f"  Target 13-KP Output : {output_dir}")
    print(f"  Removed Keypoints   : ball_top (5), ankle_center (12), shin_mid (15)")
    print(f"  Target Representation: 13 Keypoints (44 YOLO tokens per line)")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'valid', 'test']
    seen_hashes = {}  # sha256 -> split
    stats = {s: {'images': 0, 'instances': 0, 'duplicates': 0} for s in splits}

    for split in splits:
        dst_img_dir = output_dir / split / "images"
        dst_lbl_dir = output_dir / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for ds_name, src_root in [('v3', v3_path), ('v4', v4_path)]:
            src_img_dir = src_root / split / "images"
            src_lbl_dir = src_root / split / "labels"

            if not src_img_dir.exists():
                continue

            for img_path in src_img_dir.glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                stem = img_path.stem
                lbl_path = src_lbl_dir / f"{stem}.txt"
                if not lbl_path.exists():
                    continue

                # Hash check for deduplication
                h = compute_hash(img_path)
                if h in seen_hashes:
                    prev_split = seen_hashes[h]
                    stats[split]['duplicates'] += 1
                    if prev_split != split:
                        raise ValueError(f"CRITICAL: Data leakage detected! Image {img_path.name} in both {prev_split} and {split}!")
                    continue

                seen_hashes[h] = split

                # Convert label lines from 16 to 13 keypoints
                with open(safe_path(lbl_path), 'r', encoding='utf-8') as f:
                    raw_lines = f.readlines()

                converted_lines = []
                for line in raw_lines:
                    converted = convert_yolo_line_16_to_13(line)
                    if converted:
                        converted_lines.append(converted)

                if len(converted_lines) == 0:
                    continue

                # Copy image
                target_img_name = img_path.name
                target_lbl_name = f"{stem}.txt"
                target_img_path = dst_img_dir / target_img_name
                target_lbl_path = dst_lbl_dir / target_lbl_name

                if target_img_path.exists():
                    target_img_name = f"{ds_name}_{img_path.name}"
                    target_lbl_name = f"{ds_name}_{stem}.txt"
                    target_img_path = dst_img_dir / target_img_name
                    target_lbl_path = dst_lbl_dir / target_lbl_name

                shutil.copy2(safe_path(img_path), safe_path(target_img_path))
                with open(safe_path(target_lbl_path), 'w', encoding='utf-8') as f:
                    f.writelines(converted_lines)

                stats[split]['images'] += 1
                stats[split]['instances'] += len(converted_lines)

    print("\n📊 Preparation Summary:")
    print("-" * 60)
    for s in splits:
        st = stats[s]
        print(f"  • {s:<6} : {st['images']:>5} images | {st['instances']:>5} instances | {st['duplicates']:>4} duplicates skipped")
    print("-" * 60)

    # Compute 13-KP skeleton priors if configs dir is found
    configs_candidates = [
        cwd / "shoes_vto" / "configs",
        cwd / "configs",
        output_dir.parent / "configs",
        Path("/content/shoes_vto_v2/shoes_vto/configs")
    ]
    cfg_dir = None
    for cand in configs_candidates:
        if cand.exists():
            cfg_dir = cand
            break

    if cfg_dir:
        paf_cfg = cfg_dir / "paf_connections_13kp.yaml"
        priors_out = cfg_dir / "skeleton_priors_train_13kp.json"
        train_lbls = output_dir / "train" / "labels"
        print("\n⚙️ Computing data-driven 13-KP skeleton distance priors...")
        compute_priors_13kp(train_lbls, paf_cfg, priors_out)

    print(f"\n🎉 13-Keypoint Dataset is 100% prepared at: {output_dir}")
    print("👉 Next step in Colab: Launch training with:")
    print(f"   !python shoes_vto/train_m1v2_13kp.py --dataset_root {output_dir} --output_dir shoes_vto/outputs/m1v2_13kp_training --epochs 100 --batch_size 16 --lr 0.0005\n")


if __name__ == "__main__":
    main()
