"""
Package 13-KP Dataset and shoes_vto Codebase for Google Colab
Creates a clean, portable zip archive containing:
1. dataset/merged_v3_v4_13kp (train + valid images & labels)
2. shoes_vto (codebase, configs, pretrained weights, baseline evaluation summaries)
"""

import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def win_path(p):
    s = str(Path(p).resolve())
    if sys.platform == 'win32' and len(s) > 240 and not s.startswith('\\\\?\\'):
        return '\\\\?\\' + s
    return s

def make_archive(output_zip="colab_13kp_training_pack.zip"):
    workspace_root = Path(__file__).resolve().parent.parent.parent
    output_path = workspace_root / output_zip

    print(f"[Packaging] Packaging codebase and 13-KP dataset into: {output_path}")

    items_to_pack = [
        ("shoes_vto", workspace_root / "shoes_vto"),
        ("dataset/merged_v3_v4_13kp", workspace_root / "dataset" / "merged_v3_v4_13kp"),
    ]

    files_to_add = []
    for arc_prefix, folder in items_to_pack:
        if not folder.exists():
            print(f"⚠️ Warning: Folder {folder} not found, skipping.")
            continue
        for file in folder.rglob("*"):
            if file.is_file():
                # Skip caches, checkpoints, and temporary dirs
                rel = file.relative_to(folder)
                rel_str = str(rel).replace("\\", "/")
                if any(x in rel_str for x in ["__pycache__", ".git", ".pytest_cache", "smoke_test", "smoke_eval"]):
                    continue
                arcname = f"{arc_prefix}/{rel_str}"
                files_to_add.append((file, arcname))

    print(f"Total files to archive: {len(files_to_add)}")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path, arcname in tqdm(files_to_add, desc="Compressing"):
            zipf.write(win_path(file_path), arcname)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Done] Archive created successfully: {output_path} ({size_mb:.2f} MB)")
    print(f"[Info] Upload {output_zip} to Google Drive or directly to your Colab runtime session.")

if __name__ == "__main__":
    make_archive()
