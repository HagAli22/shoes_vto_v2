"""
Merge shuffled_v3 and shuffled_v4 datasets with deduplication, split isolation, and safe path handling.
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


def compute_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(16384):
            h.update(chunk)
    return h.hexdigest()


def merge_datasets():
    base_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
    out_dir = base_dir / "merged_v3_v4"
    
    print("=" * 80)
    print("👟 Merging shuffled_v3 and shuffled_v4 Datasets")
    print("=" * 80)
    print(f"Target Output: {out_dir}")
    
    # Clean/create target directory
    safe_out = "\\\\?\\" + str(out_dir.resolve())
    if os.path.exists(safe_out):
        print(f"Directory already exists: {out_dir}")
    else:
        os.makedirs(safe_out, exist_ok=True)
        
    stats = {}
    seen_hashes = {}  # hash -> split
    
    for split in ['train', 'valid', 'test']:
        img_out = os.path.join(safe_out, split, "images")
        lbl_out = os.path.join(safe_out, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        
        split_seen_hashes = set()
        count = 0
        duplicates = 0
        
        for ds_name in ['shuffled_v3', 'shuffled_v4']:
            ds_dir = "\\\\?\\" + str((base_dir / ds_name / split).resolve())
            img_dir = os.path.join(ds_dir, "images")
            lbl_dir = os.path.join(ds_dir, "labels")
            
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} does not exist!")
                continue
                
            for fname in os.listdir(img_dir):
                if not any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    continue
                    
                src_img = os.path.join(img_dir, fname)
                stem = os.path.splitext(fname)[0]
                src_lbl = os.path.join(lbl_dir, stem + ".txt")
                
                if not os.path.exists(src_lbl):
                    print(f"Warning: missing label for {src_img}")
                    continue
                    
                img_hash = compute_hash(src_img)
                
                # Check for duplicate within split or across splits
                if img_hash in seen_hashes:
                    prev_split = seen_hashes[img_hash]
                    duplicates += 1
                    if prev_split != split:
                        raise ValueError(f"CRITICAL: Data leakage! Image {fname} is in both {prev_split} and {split}!")
                    continue
                    
                seen_hashes[img_hash] = split
                split_seen_hashes.add(img_hash)
                
                # Destination filename
                dst_img = os.path.join(img_out, fname)
                dst_lbl = os.path.join(lbl_out, stem + ".txt")
                
                # Handle filename collision with different content
                if os.path.exists(dst_img):
                    unique_name = f"{ds_name}_{fname}"
                    unique_stem = f"{ds_name}_{stem}"
                    dst_img = os.path.join(img_out, unique_name)
                    dst_lbl = os.path.join(lbl_out, unique_stem + ".txt")
                    
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                count += 1
                
        stats[split] = {'count': count, 'duplicates_skipped': duplicates}
        print(f"  Split [{split.upper()}]: {count} images copied, {duplicates} duplicates deduplicated")
        
    # Write data.yaml
    data_yaml_path = os.path.join(safe_out, "data.yaml")
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        f.write("""path: dataset/merged_v3_v4
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: left_foot
  1: right_foot

kpt_shape: [16, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
""")
    print("\n✅ Dataset merge complete!")
    print(f"📁 Merged dataset saved at: {out_dir}")
    print(f"📊 Summary: Train={stats['train']['count']}, Valid={stats['valid']['count']}, Test={stats['test']['count']}")
    return stats


if __name__ == '__main__':
    merge_datasets()
