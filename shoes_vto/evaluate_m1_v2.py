"""
Evaluation Script for ARShoe M1_V2 (14 Keypoints)

Evaluates:
1. Overall PCK@0.20, PCK@0.10, and PCK@0.05
2. Per-keypoint PCK for all 14 keypoints
3. Ground Truth vs Prediction Visualizations (Overlays & 3-Panel Comparisons)

Usage:
    python evaluate_m1_v2.py
    python evaluate_m1_v2.py --checkpoint outputs/m1_v2_training/checkpoints/best.pth --num_vis 10
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add src and current directory to path
_this_dir = Path(__file__).resolve().parent
_src_dir = _this_dir / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

# Evict third-party 'datasets' package from sys.modules if it shadowed local datasets
if 'datasets' in sys.modules and 'site-packages' in getattr(sys.modules['datasets'], '__file__', ''):
    del sys.modules['datasets']

from models.arshoe_m1_v2 import ARShoeM1V2

try:
    from datasets.yolo_dataset import YOLOFootDataset
except (ImportError, ModuleNotFoundError):
    from src.datasets.yolo_dataset import YOLOFootDataset
from training.eval_keypoints import (
    decode_heatmaps_to_keypoints,
    compute_multi_threshold_pck,
    create_side_by_side_visualization,
    KEYPOINT_NAMES_14
)


def load_model(checkpoint_path, device='cpu'):
    """Load ARShoeM1V2 model from checkpoint."""
    model = ARShoeM1V2(
        encoder_channels=128,
        num_keypoints=14,
        num_limbs=14,
        num_classes=2
    )
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'N/A')
        best_val_loss = checkpoint.get('best_val_loss', 'N/A')
        print(f"  Checkpoint Epoch: {epoch}, Best Val Loss: {best_val_loss}")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
        
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_key = k.replace('_orig_mod.', '')
        cleaned_state_dict[cleaned_key] = v
        
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"  Warning - Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        print("  Model weights loaded successfully.")
        
    model.to(device)
    model.eval()
    return model


def run_evaluation(model, val_dataset, device, output_dir, num_vis=8):
    output_dir = Path(output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    all_pred_kps = []
    all_gt_kps = []
    all_gt_vis = []
    all_diags = []
    vis_samples = []
    
    print(f"\nRunning inference on {len(val_dataset)} validation samples...")
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Evaluating"):
            sample = val_dataset[i]
            img_tensor = sample['image'].unsqueeze(0).to(device)
            instances = sample['instances']
            img_path = Path(sample['image_path'])
            
            outputs = model(img_tensor)
            hm_img = outputs['heatmaps'][0]
            
            img_disp = (sample['image'].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            for inst_idx, inst in enumerate(instances):
                if 'keypoints' in inst and len(inst['keypoints']) == 14:
                    gt_kp = [kp[:2] for kp in inst['keypoints']]
                    gt_v = [kp[2] for kp in inst['keypoints']]
                    cx, cy, w, h = inst['bbox']
                    diag = float(((w * 256.0)**2 + (h * 256.0)**2)**0.5)
                    cname = inst.get('class_name', f"foot_{inst.get('class_id', 0)}")
                    
                    # Instance-scoped decoding within bounding box
                    x1 = int(max(0, (cx - w * 0.575) * 64))
                    y1 = int(max(0, (cy - h * 0.575) * 64))
                    x2 = int(min(64, (cx + w * 0.575) * 64))
                    y2 = int(min(64, (cy + h * 0.575) * 64))
                    
                    hm_inst = torch.zeros_like(hm_img)
                    hm_inst[:, y1:y2, x1:x2] = hm_img[:, y1:y2, x1:x2]
                    
                    pred_inst_tensor, pred_conf_tensor = decode_heatmaps_to_keypoints(hm_inst.unsqueeze(0), image_size=256)
                    pred_kps_np = pred_inst_tensor[0].cpu().numpy()
                    pred_conf_np = pred_conf_tensor[0].cpu().numpy()
                    
                    all_pred_kps.append(pred_kps_np)
                    all_gt_kps.append(gt_kp)
                    all_gt_vis.append(gt_v)
                    all_diags.append(diag)
                    
                    if len(vis_samples) < num_vis:
                        vis_samples.append({
                            'img_disp': img_disp,
                            'gt_kps': gt_kp,
                            'gt_vis': gt_v,
                            'pred_kps': pred_kps_np,
                            'pred_conf': pred_conf_np,
                            'bbox': inst['bbox'],
                            'name': f"{img_path.stem}_{cname}",
                            'class_name': cname
                        })
                        
    metrics = compute_multi_threshold_pck(
        pred_keypoints=all_pred_kps,
        gt_keypoints=all_gt_kps,
        gt_visibility=all_gt_vis,
        bbox_diags=all_diags,
        thresholds=(0.20, 0.10, 0.05)
    )
    
    print(f"\nGenerating and saving {len(vis_samples)} visualizations to {vis_dir}...")
    for idx, s in enumerate(vis_samples):
        side_by_side = create_side_by_side_visualization(
            image_rgb=s['img_disp'],
            gt_kps=s['gt_kps'],
            gt_vis=s['gt_vis'],
            pred_kps=s['pred_kps'],
            pred_confs=s['pred_conf'],
            bbox=s['bbox'],
            img_name=s['name']
        )
        bgr_side = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        out_file = vis_dir / f"eval_vis_{idx+1:02d}_{s['name']}.jpg"
        cv2.imwrite(str(out_file), bgr_side)
        
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed JSON evaluation saved to: {summary_path}")
    
    return metrics, vis_dir


def print_evaluation_summary(metrics, num_images, vis_dir):
    overall = metrics['overall']
    per_kp = metrics['per_keypoint']
    total_kp = metrics['total_keypoints_evaluated']
    total_inst = metrics['total_instances']
    
    print("\n" + "=" * 80)
    print("             SHOE VTO M1_V2 (14-KP) EVALUATION REPORT")
    print("=" * 80)
    
    print("\n📊 OVERALL PCK AT MULTIPLE THRESHOLDS:")
    print(f"  • Overall PCK@0.20 (Coarse) : {overall[0.2]:.2%} ({overall[0.2]*100:.1f}%)")
    print(f"  • Overall PCK@0.10 (Medium) : {overall[0.1]:.2%} ({overall[0.1]*100:.1f}%)")
    print(f"  • Overall PCK@0.05 (Precise): {overall[0.05]:.2%} ({overall[0.05]*100:.1f}%)")
    
    print("\n🎯 PER-KEYPOINT PCK BREAKDOWN (All 14 Keypoints):")
    print("-" * 80)
    print(f" {'Idx':<4} | {'Anatomical Name':<18} | {'Valid #':<7} | {'PCK@0.20':<10} | {'PCK@0.10':<10} | {'PCK@0.05':<10} | {'Status'}")
    print("-" * 80)
    
    for idx in range(14):
        k = per_kp[idx]
        name = k['name']
        cnt = k['valid_count']
        p20 = k['pck@0.2']
        p10 = k['pck@0.1']
        p05 = k['pck@0.05']
        
        if p20 >= 0.60:
            status = "🟢 Strong"
        elif p20 >= 0.40:
            status = "🟡 Moderate"
        else:
            status = "🔴 Poor"
            
        print(f" {idx:<4} | {name:<18} | {cnt:<7} | {p20:<10.1%} | {p10:<10.1%} | {p05:<10.1%} | {status}")
    print("-" * 80)
    
    print(f"\n📦 DATASET STATISTICS:")
    print(f"  • Total Validation Images:     {num_images}")
    print(f"  • Total Foot Instances:        {total_inst}")
    print(f"  • Total Keypoints Evaluated:   {total_kp:,}")
    print(f"  • Visualizations Saved to:     {vis_dir}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARShoe M1_V2 (14 Keypoints)")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="outputs/m1_v2_training/checkpoints/best.pth",
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/m1_v2_training",
        help="Directory to save evaluation results and visualizations"
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=8,
        help="Number of validation images to render and save"
    )
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    project_root = Path(__file__).parent.parent
    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root)
    else:
        v3_14kp = project_root / "dataset" / "shuffled_v3_14kp"
        if v3_14kp.exists():
            dataset_root = v3_14kp
        else:
            dataset_root = project_root / "dataset" / "shuffled_v3"
            
    val_images_dir = dataset_root / "valid" / "images"
    val_labels_dir = dataset_root / "valid" / "labels"
    
    print(f"Loading validation dataset from: {dataset_root}")
    val_dataset = YOLOFootDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        img_size=256,
        augment=False,
        target_keypoints=14
    )
    print(f"Loaded {len(val_dataset)} validation samples.")
    
    model = load_model(args.checkpoint, device=device)
    metrics, vis_dir = run_evaluation(
        model=model,
        val_dataset=val_dataset,
        device=device,
        output_dir=args.output_dir,
        num_vis=args.num_vis
    )
    print_evaluation_summary(metrics, len(val_dataset), vis_dir)


if __name__ == "__main__":
    main()

