"""
Dataset Conversion Tool: Convert YOLO 16-Keypoint Dataset to 13 Keypoints

Removes 3 ambiguous/redundant keypoints:
- ball_top (old index 5)
- ankle_center (old index 12)
- shin_mid (old index 15)

Keeps 13 keypoints: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
Output format: YOLO pose format with 44 tokens per instance (class cx cy w h + 13 * (x y v))

Usage:
    python convert_dataset_to_13kp.py --src /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1 \
                                      --dst /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1_13kp \
                                      --symlink
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

KEPT_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
assert len(KEPT_INDICES) == 13

KEYPOINT_NAMES_13 = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "instep_top", "arch_medial", "midfoot_lateral", "malleolus_medial",
    "malleolus_lateral", "toe_tip", "throat", "achilles"
]


def convert_dataset(src_dir, dst_dir, use_symlink=True):
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    print("=" * 80)
    print("👟 CONVERTING YOLO KEYPOINT DATASET: 16-KP -> 13-KP")
    print("=" * 80)
    print(f"  Source Directory : {src_path}")
    print(f"  Target Directory : {dst_path}")
    print(f"  Image Link Mode  : {'Symlink (Fast)' if use_symlink and os.name != 'nt' else 'Copy'}")
    print(f"  Kept KP Indices  : {KEPT_INDICES}")
    print("=" * 80)

    # Detect splits: check train, valid, test
    splits = []
    for split_candidate in ['train', 'valid', 'val', 'test']:
        if (src_path / split_candidate / 'images').exists() or (src_path / split_candidate / 'labels').exists():
            splits.append(split_candidate)

    if not splits:
        # Check if src_path itself contains images/ and labels/
        if (src_path / 'images').exists() and (src_path / 'labels').exists():
            splits = ['.']
        else:
            raise FileNotFoundError(f"No train/valid/test splits or images/labels found in {src_path}")

    total_images = 0
    total_labels = 0
    total_instances = 0

    for split in splits:
        src_img_dir = (src_path / split / "images") if split != '.' else (src_path / "images")
        src_lbl_dir = (src_path / split / "labels") if split != '.' else (src_path / "labels")
        
        dst_img_dir = (dst_path / split / "images") if split != '.' else (dst_path / "images")
        dst_lbl_dir = (dst_path / split / "labels") if split != '.' else (dst_path / "labels")

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"Split {split}: No images directory, skipping.")
            continue

        img_files = sorted([f for f in src_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        print(f"\nProcessing [{split}]: {len(img_files)} images...")

        split_instances = 0
        for img_p in img_files:
            stem = img_p.stem
            dst_img_p = dst_img_dir / img_p.name

            # 1. Link or copy image
            if not dst_img_p.exists():
                if use_symlink and os.name != 'nt':
                    try:
                        os.symlink(img_p, dst_img_p)
                    except Exception:
                        shutil.copy2(img_p, dst_img_p)
                else:
                    shutil.copy2(img_p, dst_img_p)
            total_images += 1

            # 2. Process corresponding label
            lbl_p = src_lbl_dir / f"{stem}.txt"
            dst_lbl_p = dst_lbl_dir / f"{stem}.txt"

            new_lines = []
            if lbl_p.exists():
                with open(lbl_p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        tokens = line.split()
                        if len(tokens) < 5:
                            continue
                        cls_id = tokens[0]
                        bbox = tokens[1:5]
                        raw_kps = tokens[5:]

                        if len(raw_kps) == 48:
                            # 16 keypoints -> extract 13 keypoints
                            kept_tokens = []
                            for idx in KEPT_INDICES:
                                kept_tokens.extend(raw_kps[idx * 3 : (idx + 1) * 3])
                            new_line = f"{cls_id} " + " ".join(bbox) + " " + " ".join(kept_tokens)
                            new_lines.append(new_line)
                            split_instances += 1
                        elif len(raw_kps) == 39:
                            # Already 13 keypoints
                            new_lines.append(line)
                            split_instances += 1
                        else:
                            print(f"Warning: Unexpected token count in {lbl_p.name}: {len(tokens)} tokens ({len(raw_kps)} KP floats)")

            with open(dst_lbl_p, 'w', encoding='utf-8') as f:
                for nl in new_lines:
                    f.write(nl + "\n")
            total_labels += 1

        total_instances += split_instances
        print(f"  [{split}] Done: {len(img_files)} images, {split_instances} instances converted to 13 keypoints.")

    print("\n" + "=" * 80)
    print("✅ DATASET CONVERSION COMPLETE!")
    print(f"  Total Images    : {total_images}")
    print(f"  Total Labels    : {total_labels}")
    print(f"  Total Instances : {total_instances}")
    print(f"  Output Path     : {dst_path}")
    print("=" * 80)
    return str(dst_path)


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO 16-KP dataset to 13-KP dataset")
    parser.add_argument('--src', type=str, required=True, help="Source dataset root (e.g. /content/shoes_vto_v2/shoes_vto/data/shuffled_v4_1)")
    parser.add_argument('--dst', type=str, default=None, help="Target 13-KP dataset root")
    parser.add_argument('--symlink', action='store_true', default=True, help="Symlink images instead of copying (Linux only, fast)")
    parser.add_argument('--no_symlink', dest='symlink', action='store_false', help="Copy images instead of symlinking")
    args = parser.parse_args()

    dst = args.dst
    if dst is None:
        src_path = Path(args.src)
        dst = str(src_path.parent / (src_path.name + "_13kp"))

    convert_dataset(args.src, dst, use_symlink=args.symlink)


if __name__ == '__main__':
    main()

