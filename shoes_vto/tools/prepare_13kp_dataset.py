"""
Generate 13-Keypoint Dataset from dataset/merged_v3_v4
Removes 3 keypoints:
- ball_top (old index 5)
- ankle_center (old index 12)
- shin_mid (old index 15)

Keeps 13 keypoints: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SRC_ROOT = Path("dataset/merged_v3_v4").resolve()
DST_ROOT = Path("dataset/merged_v3_v4_13kp").resolve()

KEPT_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
assert len(KEPT_INDICES) == 13

def safe_path(p):
    p_str = str(p)
    if os.name == 'nt' and not p_str.startswith('\\\\?\\'):
        abs_p = os.path.abspath(p_str)
        return '\\\\?\\' + abs_p
    return p_str

def main():
    print("=" * 80)
    print("👟 Preparing 13-Keypoint Dataset: merged_v3_v4 -> merged_v3_v4_13kp")
    print("=" * 80)
    print(f"Source: {SRC_ROOT}")
    print(f"Target: {DST_ROOT}")
    print(f"Kept 16-KP indices: {KEPT_INDICES}")
    
    splits = ['train', 'valid', 'test']
    
    total_imgs_processed = 0
    total_lbls_processed = 0
    total_instances_processed = 0

    for split in splits:
        src_img_dir = SRC_ROOT / split / "images"
        src_lbl_dir = SRC_ROOT / split / "labels"
        
        dst_img_dir = DST_ROOT / split / "images"
        dst_lbl_dir = DST_ROOT / split / "labels"
        
        os.makedirs(safe_path(dst_img_dir), exist_ok=True)
        os.makedirs(safe_path(dst_lbl_dir), exist_ok=True)
        
        safe_src_img_str = safe_path(src_img_dir)
        safe_src_lbl_str = safe_path(src_lbl_dir)
        safe_dst_img_str = safe_path(dst_img_dir)
        safe_dst_lbl_str = safe_path(dst_lbl_dir)
        
        img_files = [f for f in os.listdir(safe_src_img_str) if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
        print(f"\nProcessing [{split}]: {len(img_files)} images...")
        
        split_instances = 0
        
        for fname in img_files:
            stem = os.path.splitext(fname)[0]
            src_img_file = os.path.join(safe_src_img_str, fname)
            dst_img_file = os.path.join(safe_dst_img_str, fname)
            
            # Copy image if not exists or size mismatch
            if not os.path.exists(dst_img_file) or os.path.getsize(dst_img_file) != os.path.getsize(src_img_file):
                shutil.copy2(src_img_file, dst_img_file)
            total_imgs_processed += 1
            
            # Process label file
            src_lbl_file = os.path.join(safe_src_lbl_str, stem + ".txt")
            dst_lbl_file = os.path.join(safe_dst_lbl_str, stem + ".txt")
            
            new_lines = []
            if os.path.exists(src_lbl_file):
                with open(src_lbl_file, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    
                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = parts[0]
                    bbox = parts[1:5]
                    raw_kps = parts[5:]
                    
                    # Ensure 16 keypoints (48 tokens)
                    if len(raw_kps) != 48:
                        raise ValueError(f"Expected 48 KP tokens in {src_lbl_file}, got {len(raw_kps)}")
                        
                    # Slice kept keypoints
                    new_kp_tokens = []
                    for idx in KEPT_INDICES:
                        start_pos = idx * 3
                        x = raw_kps[start_pos]
                        y = raw_kps[start_pos + 1]
                        v = raw_kps[start_pos + 2]
                        new_kp_tokens.extend([x, y, v])
                        
                    assert len(new_kp_tokens) == 39  # 13 keypoints * 3
                    new_line = f"{cls_id} {' '.join(bbox)} {' '.join(new_kp_tokens)}"
                    new_lines.append(new_line)
                    split_instances += 1
                    total_instances_processed += 1
                    
            with open(dst_lbl_file, 'w', encoding='utf-8') as f:
                for nl in new_lines:
                    f.write(nl + "\n")
            total_lbls_processed += 1

        print(f"  [{split}] Done: {len(img_files)} images, {split_instances} instances transformed.")

    # Write target data.yaml
    yaml_content = """path: dataset/merged_v3_v4_13kp
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: left_foot
  1: right_foot

kpt_shape: [13, 3]
flip_idx: [0, 1, 2, 4, 3, 5, 7, 6, 9, 8, 10, 11, 12]
"""
    yaml_path = os.path.join(safe_path(DST_ROOT), "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print("\n" + "=" * 80)
    print("✅ 13-Keypoint Dataset Generation Completed Successfully!")
    print(f"Total Images: {total_imgs_processed}")
    print(f"Total Labels: {total_lbls_processed}")
    print(f"Total Instances: {total_instances_processed}")
    print(f"Configuration written to: {yaml_path}")
    print("=" * 80)

if __name__ == '__main__':
    main()

