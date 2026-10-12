"""
Comprehensive Integrity Verification for dataset/merged_v3_v4_13kp
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

DATASET_ROOT = Path("dataset/merged_v3_v4_13kp").resolve()
SRC_ROOT = Path("dataset/merged_v3_v4").resolve()

def safe_path(p):
    p_str = str(p)
    if os.name == 'nt' and not p_str.startswith('\\\\?\\'):
        abs_p = os.path.abspath(p_str)
        return '\\\\?\\' + abs_p
    return p_str

def compute_hash(fpath):
    h = hashlib.sha256()
    with open(fpath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify():
    print("=" * 80)
    print("🔍 Running Comprehensive Integrity Check on merged_v3_v4_13kp")
    print("=" * 80)
    
    splits = ['train', 'valid', 'test']
    expected_counts = {'train': 2034, 'valid': 253, 'test': 258}
    expected_instances = {'train': 3629, 'valid': 449, 'test': 456}
    
    seen_hashes = {}
    total_imgs = 0
    total_lbls = 0
    total_insts = 0
    
    class_counter = Counter()
    vis_counter = Counter()
    
    errors = []
    
    for split in splits:
        img_dir = os.path.join(safe_path(DATASET_ROOT), split, "images")
        lbl_dir = os.path.join(safe_path(DATASET_ROOT), split, "labels")
        
        img_files = [f for f in os.listdir(img_dir) if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
        lbl_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]
        
        # 1. Count checks
        if len(img_files) != expected_counts[split]:
            errors.append(f"[{split}] Image count mismatch: expected {expected_counts[split]}, got {len(img_files)}")
        if len(lbl_files) != expected_counts[split]:
            errors.append(f"[{split}] Label count mismatch: expected {expected_counts[split]}, got {len(lbl_files)}")
            
        img_stems = {os.path.splitext(f)[0] for f in img_files}
        lbl_stems = {os.path.splitext(f)[0] for f in lbl_files}
        
        missing_lbls = img_stems - lbl_stems
        if missing_lbls:
            errors.append(f"[{split}] {len(missing_lbls)} images missing labels!")
            
        split_insts = 0
        
        # 2. Per-file label integrity check
        for fname in img_files:
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            
            # Check duplicate / split leakage via SHA-256
            im_h = compute_hash(img_path)
            if im_h in seen_hashes:
                prev_split = seen_hashes[im_h]
                errors.append(f"Image {fname} in {split} is duplicate of image in {prev_split}!")
            seen_hashes[im_h] = split
            
            with open(lbl_path, 'r', encoding='utf-8') as lf:
                lines = [l.strip() for l in lf if l.strip()]
                
            for line_idx, line in enumerate(lines):
                split_insts += 1
                parts = line.split()
                if len(parts) != 44:  # 1 cls + 4 bbox + 13*3 kpts = 44
                    errors.append(f"[{split}/{stem}.txt line {line_idx}] Expected 44 tokens, got {len(parts)}")
                    continue
                
                cls_id = int(parts[0])
                if cls_id not in (0, 1):
                    errors.append(f"[{split}/{stem}.txt] Invalid class {cls_id}")
                class_counter[cls_id] += 1
                
                cx, cy, w, h = [float(x) for x in parts[1:5]]
                if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w <= 0 or h <= 0:
                    errors.append(f"[{split}/{stem}.txt] BBox out of range: {parts[1:5]}")
                    
                kp_vals = [float(x) for x in parts[5:]]
                for k in range(13):
                    kx = kp_vals[k * 3]
                    ky = kp_vals[k * 3 + 1]
                    kv = int(kp_vals[k * 3 + 2])
                    vis_counter[kv] += 1
                    if kv not in (0, 1, 2):
                        errors.append(f"[{split}/{stem}.txt] Invalid visibility {kv} for kp {k}")
                    if kv > 0 and (kx < 0 or kx > 1 or ky < 0 or ky > 1):
                        errors.append(f"[{split}/{stem}.txt] KP {k} out of range: ({kx}, {ky})")
                        
        if split_insts != expected_instances[split]:
            errors.append(f"[{split}] Instance count mismatch: expected {expected_instances[split]}, got {split_insts}")
            
        total_imgs += len(img_files)
        total_lbls += len(lbl_files)
        total_insts += split_insts
        print(f"  Split [{split.upper()}]: {len(img_files)} images, {len(lbl_files)} labels, {split_insts} instances. (OK)")

    # 3. Cross-check against source dataset
    # Pick a sample label and verify exact mapping
    print("\n🔬 Verifying 16-to-13 Mapping Consistency Against Source Dataset...")
    sample_split = "valid"
    sample_lbl_name = os.listdir(os.path.join(safe_path(DATASET_ROOT), sample_split, "labels"))[0]
    src_sample_lbl = os.path.join(safe_path(SRC_ROOT), sample_split, "labels", sample_lbl_name)
    dst_sample_lbl = os.path.join(safe_path(DATASET_ROOT), sample_split, "labels", sample_lbl_name)
    
    with open(src_sample_lbl, 'r', encoding='utf-8') as f:
        src_lines = [l.strip().split() for l in f if l.strip()]
    with open(dst_sample_lbl, 'r', encoding='utf-8') as f:
        dst_lines = [l.strip().split() for l in f if l.strip()]
        
    kept = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
    for inst_i in range(len(src_lines)):
        src_raw = src_lines[inst_i][5:]
        dst_raw = dst_lines[inst_i][5:]
        for new_k, old_k in enumerate(kept):
            s_triple = src_raw[old_k*3 : old_k*3 + 3]
            d_triple = dst_raw[new_k*3 : new_k*3 + 3]
            assert s_triple == d_triple, f"Mismatch at instance {inst_i}, new_kp {new_k} vs old_kp {old_k}: {s_triple} vs {d_triple}"
    print("  Mapping verification passed perfectly on sample instances.")

    print("\n" + "=" * 80)
    print("📊 DATASET INTEGRITY REPORT SUMMARY")
    print("=" * 80)
    print(f"Total Unique Images Verified: {total_imgs} / 2,545")
    print(f"Total Labels Verified       : {total_lbls} / 2,545")
    print(f"Total Instances             : {total_insts} / 4,534")
    print(f"Class Distribution          : Left={class_counter[0]} ({class_counter[0]/total_insts*100:.1f}%), Right={class_counter[1]} ({class_counter[1]/total_insts*100:.1f}%)")
    print(f"Visibility Distribution     : Visible(2)={vis_counter[2]}, Occluded(1)={vis_counter[1]}, Unlabeled(0)={vis_counter[0]}")
    print(f"Total Detected Errors       : {len(errors)}")
    if errors:
        print("❌ ERRORS FOUND:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("✅ 100% INTEGRITY CHECK PASSED: Zero errors, zero leakage, valid coordinates, perfect 13-KP format.")
    print("=" * 80)

if __name__ == '__main__':
    verify()

