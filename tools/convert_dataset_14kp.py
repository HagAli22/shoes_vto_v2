"""
Dataset converter: shuffled_v3 (16 KPs) -> shuffled_v3_14kp (14 KPs)

Removes:
  - ankle_center (old index 12)
  - shin_mid (old index 15)

Retains indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
New indices: 0..13
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

KEEP_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
assert len(KEEP_INDICES) == 14

SRC_DIR = Path("dataset/shuffled_v3")
DST_DIR = Path("dataset/shuffled_v3_14kp")

def convert_label_line(line: str) -> str:
    tokens = line.strip().split()
    if len(tokens) == 0:
        return ""
    # Format: cls cx cy w h (x0 y0 v0) ... (x15 y15 v15)
    # Total tokens: 1 + 4 + 16*3 = 53
    cls_id = tokens[0]
    bbox = tokens[1:5]
    kp_tokens = tokens[5:]
    
    assert len(kp_tokens) == 16 * 3, f"Expected 48 keypoint values, got {len(kp_tokens)}"
    
    # Parse keypoints
    kps = []
    for i in range(16):
        x = kp_tokens[i * 3]
        y = kp_tokens[i * 3 + 1]
        v = kp_tokens[i * 3 + 2]
        kps.append((x, y, v))
        
    # Filter keypoints
    filtered_kps = [kps[idx] for idx in KEEP_INDICES]
    
    # Reassemble line
    out_tokens = [cls_id] + bbox
    for x, y, v in filtered_kps:
        out_tokens.extend([x, y, v])
        
    return " ".join(out_tokens)

def main():
    print(f"Converting dataset from {SRC_DIR} to {DST_DIR}...")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process splits: train, valid, test
    splits = ["train", "valid", "test"]
    for split in splits:
        src_split_img = SRC_DIR / split / "images"
        src_split_lbl = SRC_DIR / split / "labels"
        
        dst_split_img = DST_DIR / split / "images"
        dst_split_lbl = DST_DIR / split / "labels"
        
        dst_split_img.mkdir(parents=True, exist_ok=True)
        dst_split_lbl.mkdir(parents=True, exist_ok=True)
        
        img_files = sorted(list(src_split_img.glob("*.*")))
        print(f"\nProcessing {split}: {len(img_files)} images...")
        
        for img_path in tqdm(img_files, desc=f"{split} images"):
            dst_img = dst_split_img / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            
            # Process corresponding label
            lbl_name = img_path.stem + ".txt"
            src_lbl = src_split_lbl / lbl_name
            dst_lbl = dst_split_lbl / lbl_name
            
            if src_lbl.exists():
                with open(src_lbl, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                converted_lines = []
                for line in lines:
                    line_str = line.strip()
                    if line_str:
                        converted = convert_label_line(line_str)
                        if converted:
                            converted_lines.append(converted + "\n")
                            
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    f.writelines(converted_lines)
            else:
                # Empty file if no annotations
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    pass

    # Create data.yaml
    data_yaml = {
        'path': str(DST_DIR.resolve()).replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['left_foot', 'right_foot'],
        'kpt_shape': [14, 3],
        # Bilateral flip map for 14 KPs:
        # 3 (ball_medial) <-> 4 (ball_lateral)
        # 7 (arch_medial) <-> 8 (midfoot_lateral)
        # 9 (malleolus_medial) <-> 10 (malleolus_lateral)
        'flip_idx': [0, 1, 2, 4, 3, 5, 6, 8, 7, 10, 9, 11, 12, 13]
    }
    
    with open(DST_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"\n[OK] Successfully created {DST_DIR} with data.yaml!")

if __name__ == "__main__":
    main()

