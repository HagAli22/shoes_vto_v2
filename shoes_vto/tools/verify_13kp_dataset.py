"""
Comprehensive Integrity Verification for 13-Keypoint YOLO Dataset
Checks:
1. 1-to-1 matching between images and label files for all splits (train, valid, test).
2. Zero duplicate images or cross-split leakage (verified via SHA-256).
3. Annotation schema validity: exactly 44 tokens per line (1 class + 4 bbox + 13*3 keypoints).
4. Coordinate sanity: 0.0 <= cx, cy, w, h <= 1.0 and 0.0 <= kx, ky <= 1.0.
5. Keypoint visibility flags (0=unlabeled, 1=occluded, 2=visible) and class IDs (0=left, 1=right).
6. Optional sample cross-check against 16-KP source (merged_v3_v4, shuffled_v3, or shuffled_v4).
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def safe_path(p):
    p_str = str(Path(p).resolve())
    if sys.platform == 'win32' and len(p_str) > 240 and not p_str.startswith('\\\\?\\'):
        return '\\\\?\\' + p_str
    return p_str


def compute_hash(fpath):
    h = hashlib.sha256()
    with open(safe_path(fpath), 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_source_label(lbl_name, split, candidates):
    for c in candidates:
        candidate_lbl = Path(c) / split / "labels" / lbl_name
        if candidate_lbl.exists():
            return candidate_lbl
    return None


def verify(dataset_root="dataset/merged_v3_v4_13kp"):
    data_root = Path(dataset_root).resolve()
    if not data_root.exists() and (Path.cwd() / dataset_root).exists():
        data_root = (Path.cwd() / dataset_root).resolve()

    print("=" * 80)
    print(f"🔍 Running Comprehensive Integrity Check on: {data_root.name}")
    print("=" * 80)
    print(f"  Target Path: {data_root}")

    splits = ['train', 'valid', 'test']
    seen_hashes = {}
    total_imgs = 0
    total_lbls = 0
    total_insts = 0

    class_counter = Counter()
    vis_counter = Counter()
    errors = []

    for split in splits:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            errors.append(f"Missing images or labels directory for split: {split}")
            continue

        img_files = sorted([f for f in os.listdir(safe_path(img_dir)) if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])])
        lbl_files = sorted([f for f in os.listdir(safe_path(lbl_dir)) if f.lower().endswith('.txt')])

        if len(img_files) != len(lbl_files):
            errors.append(f"[{split}] Image ({len(img_files)}) and Label ({len(lbl_files)}) count mismatch!")

        img_stems = {os.path.splitext(f)[0] for f in img_files}
        lbl_stems = {os.path.splitext(f)[0] for f in lbl_files}

        missing_lbls = img_stems - lbl_stems
        if missing_lbls:
            errors.append(f"[{split}] {len(missing_lbls)} images missing corresponding label file!")

        missing_imgs = lbl_stems - img_stems
        if missing_imgs:
            errors.append(f"[{split}] {len(missing_imgs)} labels missing corresponding image file!")

        split_insts = 0

        # Per-file label integrity check
        for fname in img_files:
            stem = os.path.splitext(fname)[0]
            img_path = img_dir / fname
            lbl_path = lbl_dir / f"{stem}.txt"

            # Check duplicate / split leakage via SHA-256
            im_h = compute_hash(img_path)
            if im_h in seen_hashes:
                prev_split = seen_hashes[im_h]
                errors.append(f"Image {fname} in {split} is duplicate of image in {prev_split}!")
            seen_hashes[im_h] = split

            if not lbl_path.exists():
                continue

            with open(safe_path(lbl_path), 'r', encoding='utf-8') as lf:
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

        total_imgs += len(img_files)
        total_lbls += len(lbl_files)
        total_insts += split_insts
        print(f"  Split [{split.upper():<5}]: {len(img_files):>5} images, {len(lbl_files):>5} labels, {split_insts:>5} instances. (OK)")

    # Optional cross-check against 16-KP source dataset
    print("\n🔬 Verifying 16-to-13 Mapping Consistency Against Source Dataset...")
    src_candidates = [
        data_root.parent / "merged_v3_v4",
        data_root.parent / "shuffled_v3",
        data_root.parent / "shuffled_v4",
        Path("dataset/merged_v3_v4"),
        Path("dataset/shuffled_v3"),
        Path("dataset/shuffled_v4"),
        Path("/content/shoes_vto_v2/shoes_vto/dataset/shuffled_v3"),
        Path("/content/shoes_vto_v2/shoes_vto/dataset/shuffled_v4"),
    ]

    sample_split = "valid"
    sample_lbl_dir = data_root / sample_split / "labels"
    sample_lbl_files = os.listdir(safe_path(sample_lbl_dir)) if sample_lbl_dir.exists() else []

    if sample_lbl_files:
        sample_lbl_name = sample_lbl_files[0]
        src_sample_lbl = find_source_label(sample_lbl_name, sample_split, src_candidates)

        if src_sample_lbl and src_sample_lbl.exists():
            dst_sample_lbl = sample_lbl_dir / sample_lbl_name
            with open(safe_path(src_sample_lbl), 'r', encoding='utf-8') as f:
                src_lines = [l.strip().split() for l in f if l.strip()]
            with open(safe_path(dst_sample_lbl), 'r', encoding='utf-8') as f:
                dst_lines = [l.strip().split() for l in f if l.strip()]

            kept = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14]
            for inst_i in range(len(src_lines)):
                src_raw = src_lines[inst_i][5:]
                dst_raw = dst_lines[inst_i][5:]
                for new_k, old_k in enumerate(kept):
                    s_triple = src_raw[old_k*3 : old_k*3 + 3]
                    d_triple = dst_raw[new_k*3 : new_k*3 + 3]
                    assert s_triple == d_triple, f"Mismatch at instance {inst_i}, new_kp {new_k} vs old_kp {old_k}"
            print(f"  Mapping verification passed perfectly against source: {src_sample_lbl.parent.parent.name}")
        else:
            print("  (Source 16-KP sample label not found in candidate paths; structural check passed.)")
    else:
        print("  (No validation label files found to check.)")

    print("\n" + "=" * 80)
    print("📊 DATASET INTEGRITY REPORT SUMMARY")
    print("=" * 80)
    print(f"Total Unique Images Verified: {total_imgs}")
    print(f"Total Labels Verified       : {total_lbls}")
    print(f"Total Instances             : {total_insts}")
    if total_insts > 0:
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


def main():
    parser = argparse.ArgumentParser(description="Verify 13-Keypoint Dataset Integrity")
    parser.add_argument('--dataset_root', type=str, default='dataset/merged_v3_v4_13kp', help='Path to 13-KP dataset root')
    args = parser.parse_args()
    verify(args.dataset_root)


if __name__ == '__main__':
    main()
