"""
Visualize Ground Truth Supervision Targets (Heatmaps, PAFs, Masks)
for Sample 0 (Single foot) and Sample 2 (Two feet)
Saves visual artifacts to the artifact directory for verification.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.heads.paf_head import generate_pafs_batch, load_paf_config
from models.heads.heatmap_head import generate_heatmaps_batch
from tools.evaluate_decoders_on_validation import parse_label_file

ARTIFACT_DIR = Path(r"C:\Users\Matrix Store\.gemini\antigravity\brain\729e2265-efb4-4e0a-8890-a75ed53005c3")
OUT_DIR = BASE_DIR / "outputs" / "gt_targets_inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    PROJECT_ROOT / "dataset" / "shuffled_v4_1_13kp" / "valid",
    PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp" / "valid",
    PROJECT_ROOT / "dataset" / "merged_v3_v4_13kp" / "train"
]

TEST_SAMPLES = [
    {
        'name': 'Sample 0 (Single Foot)',
        'prefix': 'sample0',
        'pattern': '*-3_jpg*'
    },
    {
        'name': 'Sample 2 (Two Feet)',
        'prefix': 'sample2',
        'pattern': '*-tattoo-feettattoo_jpg*'
    }
]

paf_config = load_paf_config("paf_connections_13kp.yaml")
limb_pairs = paf_config['limb_pairs']

for info in TEST_SAMPLES:
    img_matches = []
    lbl_path = None
    for s_dir in SEARCH_DIRS:
        matches = list((s_dir / "images").glob(info['pattern']))
        if matches:
            img_matches = matches
            lbl_path = s_dir / "labels" / (matches[0].stem + ".txt")
            break
            
    if not img_matches or not lbl_path.exists():
        print(f"Warning: could not find {info['pattern']} in any search directory")
        continue
        
    img_path = img_matches[0]
    instances = parse_label_file(lbl_path, img_size=256)
    print(f"\nProcessing {info['name']}: {img_path.name}")
    print(f"  Source dir: {img_path.parent.parent}")
    print(f"  Instances found: {len(instances)}")
    for i, inst in enumerate(instances):
        kps = inst['keypoints']
        vis_count = sum(1 for kp in kps if kp[2] > 0)
        print(f"  Instance {i} ({inst['class_name']}): {vis_count}/13 keypoints visible")
        
    batch = [instances]
    
    # Generate Heatmaps & Masks
    gt_hm, hm_masks = generate_heatmaps_batch(batch, heatmap_size=64, sigma=3.5, device='cpu', num_keypoints=13)
    # Generate PAFs & Masks
    gt_paf, paf_masks = generate_pafs_batch(batch, heatmap_size=64, paf_width=6, device='cpu', paf_config=paf_config, num_limbs=16)
    
    hm_comp = gt_hm[0].max(dim=0)[0].numpy()
    hm_mask_mean = hm_masks[0].mean(dim=0).numpy()
    
    # PAF magnitude composite
    paf_mag_sum = np.zeros((64, 64), dtype=np.float32)
    for l_idx in range(16):
        px = gt_paf[0, l_idx * 2].numpy()
        py = gt_paf[0, l_idx * 2 + 1].numpy()
        paf_mag_sum += np.sqrt(px**2 + py**2)
        
    paf_mask_mean = paf_masks[0].mean(dim=0).numpy()
    
    # Load input image resized to 256
    orig_img = Image.open(img_path).convert("RGB").resize((256, 256))
    img_np = np.array(orig_img)
    
    # Render 2x3 panel:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{info['name']} - GT Supervision Targets & Masks", fontsize=16, fontweight='bold')
    
    # 1. Input Image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"Input Image (256x256)\n{len(instances)} Instance(s)")
    axes[0, 0].axis('off')
    
    # 2. Composite GT Heatmaps
    im_hm = axes[0, 1].imshow(hm_comp, cmap='hot', vmin=0, vmax=1.0)
    axes[0, 1].set_title("Composite GT Heatmaps (64x64)\n(Max over 13 Channels)")
    plt.colorbar(im_hm, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].axis('off')
    
    # 3. Heatmap Mask
    im_hm_m = axes[0, 2].imshow(hm_mask_mean, cmap='viridis', vmin=0, vmax=1.0)
    axes[0, 2].set_title("Heatmap Mask (Mean across Channels)\n1=Supervised (incl. BG), 0=Ignored")
    plt.colorbar(im_hm_m, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].axis('off')
    
    # 4. Input Image with Skeleton Overlay
    axes[1, 0].imshow(img_np)
    for inst in instances:
        kps = inst['keypoints']
        for u, v, _ in limb_pairs:
            if u < len(kps) and v < len(kps) and kps[u][2] > 0 and kps[v][2] > 0:
                axes[1, 0].plot([kps[u][0], kps[v][0]], [kps[u][1], kps[v][1]], 'c-', linewidth=2)
        for k_idx, (kx, ky, kv) in enumerate(kps):
            if kv > 0:
                axes[1, 0].plot(kx, ky, 'ro', markersize=4)
    axes[1, 0].set_title("GT Skeleton Overlay (256x256)\nVisible Limbs")
    axes[1, 0].axis('off')
    
    # 5. Composite PAF Magnitude
    im_paf = axes[1, 1].imshow(paf_mag_sum, cmap='inferno')
    axes[1, 1].set_title("Composite PAF Magnitude (64x64)\n(Sum over 16 Limbs)")
    plt.colorbar(im_paf, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].axis('off')
    
    # 6. PAF Mask
    im_paf_m = axes[1, 2].imshow(paf_mask_mean, cmap='viridis', vmin=0, vmax=1.0)
    axes[1, 2].set_title("PAF Mask (Mean across Limbs)\n1=Supervised (incl. BG), 0=Ignored")
    plt.colorbar(im_paf_m, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    out_file = OUT_DIR / f"{info['prefix']}_gt_supervision_targets.png"
    artifact_file = ARTIFACT_DIR / f"{info['prefix']}_gt_supervision_targets.png"
    plt.savefig(out_file, dpi=150)
    plt.savefig(artifact_file, dpi=150)
    plt.close()
    print(f"  Saved visualization: {artifact_file}")

print("\nGT target visualization complete!")

