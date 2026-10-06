"""
Extended Keypoint Evaluation & Visualization for Shoe VTO M1 Model

Evaluates:
1. Per-keypoint PCK (0 to 15) with anatomical names
2. Multi-threshold PCK: PCK@0.20, PCK@0.10, and PCK@0.05
3. Ground Truth vs Prediction Visualizations (Overlays & 3-Panel Comparisons)

Usage:
    python evaluate_m1.py
    python evaluate_m1.py --checkpoint outputs/m1_training/checkpoints/best.pth --num_vis 10
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

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1 import ARShoeM1
from datasets.yolo_dataset import YOLOFootDataset
from training.eval_keypoints import (
    decode_heatmaps_to_keypoints,
    compute_multi_threshold_pck,
    create_side_by_side_visualization,
    draw_keypoints_on_image,
    KEYPOINT_NAMES,
    SKELETON_LIMBS
)


def load_model(checkpoint_path, device='cpu'):
    """Load ARShoeM1 model from checkpoint, handling torch.compile prefixes."""
    model = ARShoeM1(
        encoder_channels=128,
        num_keypoints=16,
        num_limbs=15,
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
        
    # Strip torch.compile '_orig_mod.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_key = k.replace('_orig_mod.', '')
        cleaned_state_dict[cleaned_key] = v
        
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"  Warning - Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        print("  Model weights loaded strictly and successfully.")
        
    model.to(device)
    model.eval()
    return model


def run_evaluation(model, val_dataset, device, output_dir, num_vis=8):
    """
    Run evaluation across validation dataset:
    - Computes multi-threshold PCK (0.2, 0.1, 0.05)
    - Computes per-keypoint PCK
    - Generates and saves visual comparisons
    """
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
            
            # Forward pass
            outputs = model(img_tensor)
            
            # Heatmaps for this image [16, 64, 64]
            hm_img = outputs['heatmaps'][0]
            
            # Reconstruct original image for visualization (RGB, 0-255 uint8)
            img_disp = (sample['image'].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # Process instances
            for inst_idx, inst in enumerate(instances):
                if 'keypoints' in inst and len(inst['keypoints']) == 16:
                    gt_kp = [kp[:2] for kp in inst['keypoints']]
                    gt_v = [kp[2] for kp in inst['keypoints']]
                    cx, cy, w, h = inst['bbox']
                    diag = float(((w * 256.0)**2 + (h * 256.0)**2)**0.5)
                    cname = inst.get('class_name', f"foot_{inst.get('class_id', 0)}")
                    
                    # Instance-scoped heatmap decoding:
                    # When an image has multiple feet (left and right), we search for peaks
                    # within the foot instance region (expanded slightly by 15% margin)
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
                        
    # Compute multi-threshold PCK
    metrics = compute_multi_threshold_pck(
        pred_keypoints=all_pred_kps,
        gt_keypoints=all_gt_kps,
        gt_visibility=all_gt_vis,
        bbox_diags=all_diags,
        thresholds=(0.20, 0.10, 0.05)
    )
    
    # Save visualizations
    print(f"\nGenerating and saving {len(vis_samples)} visualizations to {vis_dir}...")
    for idx, s in enumerate(vis_samples):
        # 1. 3-panel visualization: [GT | Prediction | Overlay Error]
        side_by_side = create_side_by_side_visualization(
            image_rgb=s['img_disp'],
            gt_kps=s['gt_kps'],
            gt_vis=s['gt_vis'],
            pred_kps=s['pred_kps'],
            pred_confs=s['pred_conf'],
            bbox=s['bbox'],
            img_name=s['name']
        )
        # Convert RGB to BGR for cv2.imwrite
        bgr_side = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        out_file = vis_dir / f"eval_vis_{idx+1:02d}_{s['name']}.jpg"
        cv2.imwrite(str(out_file), bgr_side)
        
    # Save JSON summary
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed JSON evaluation saved to: {summary_path}")
    
    return metrics, vis_dir


def print_evaluation_summary(metrics, num_images, vis_dir):
    """Print formatted concise summary matching user requirements."""
    overall = metrics['overall']
    per_kp = metrics['per_keypoint']
    total_kp = metrics['total_keypoints_evaluated']
    total_inst = metrics['total_instances']
    
    print("\n" + "=" * 80)
    print("                SHOE VTO M1 KEYPOINT EVALUATION REPORT")
    print("=" * 80)
    
    print("\n📊 OVERALL PCK AT MULTIPLE THRESHOLDS:")
    print(f"  • Overall PCK@0.20 (Coarse) : {overall[0.2]:.2%} ({overall[0.2]*100:.1f}%)")
    print(f"  • Overall PCK@0.10 (Medium) : {overall[0.1]:.2%} ({overall[0.1]*100:.1f}%)")
    print(f"  • Overall PCK@0.05 (Precise): {overall[0.05]:.2%} ({overall[0.05]*100:.1f}%)")
    
    print("\n🎯 PER-KEYPOINT PCK BREAKDOWN (All 16 Keypoints):")
    print("-" * 80)
    print(f" {'Idx':<4} | {'Anatomical Name':<18} | {'Valid #':<7} | {'PCK@0.20':<10} | {'PCK@0.10':<10} | {'PCK@0.05':<10} | {'Status'}")
    print("-" * 80)
    
    for idx in range(16):
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
    
    # Highlight poorest keypoints
    poor = [item for item in metrics['ranked_by_pck@0.2'] if item[1] < 0.40]
    strong = [item for item in metrics['ranked_by_pck@0.2'] if item[1] >= 0.60]
    
    print("\n⚠️  KEYPOINT PERFORMANCE HIGHLIGHTS:")
    if poor:
        print(f"  • Poorest Keypoints (PCK@0.2 < 40%):")
        for name, score, cnt in poor:
            print(f"      - {name:<18}: {score:.1%} ({cnt} evaluated)")
    else:
        print("  • All keypoints achieved >= 40% PCK@0.2")
        
    if strong:
        print(f"  • Best Keypoints (PCK@0.2 >= 60%):")
        for name, score, cnt in strong:
            print(f"      + {name:<18}: {score:.1%} ({cnt} evaluated)")
            
    print("\n📋 EVALUATION METADATA:")
    print(f"  • Number of Validation Images Evaluated   : {num_images}")
    print(f"  • Number of Foot Instances Evaluated     : {total_inst}")
    print(f"  • Number of Labeled Keypoints Evaluated  : {total_kp}")
    print(f"  • Path to Saved Visualization Results    : {vis_dir}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shoe VTO M1 Model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/m1_training/checkpoints/best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='../dataset/shuffled_v3',
        help='Path to dataset root'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/m1_eval_results',
        help='Directory to save evaluation visualizations and summary'
    )
    parser.add_argument(
        '--num_vis',
        type=int,
        default=8,
        help='Number of validation images to visualize'
    )
    args = parser.parse_args()
    
    # Resolve paths relative to shoes_vto directory
    base_dir = Path(__file__).parent
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = base_dir / ckpt_path
        
    data_root = Path(args.dataset_root)
    if not data_root.is_absolute():
        data_root = base_dir / data_root
        
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # Load dataset
    val_images = data_root / "valid" / "images"
    val_labels = data_root / "valid" / "labels"
    if not val_images.exists():
        # Fallback to local inspect folder path
        val_images = Path("dataset/shuffled_v3/valid/images").resolve()
        val_labels = Path("dataset/shuffled_v3/valid/labels").resolve()
        
    val_dataset = YOLOFootDataset(
        images_dir=val_images,
        labels_dir=val_labels,
        img_size=256,
        augment=False,
        cache_in_memory=True
    )
    
    # Load model
    model = load_model(ckpt_path, device=device)
    
    # Run evaluation
    metrics, vis_dir = run_evaluation(
        model=model,
        val_dataset=val_dataset,
        device=device,
        output_dir=out_dir,
        num_vis=args.num_vis
    )
    
    # Print concise summary
    print_evaluation_summary(metrics, num_images=len(val_dataset), vis_dir=vis_dir)


if __name__ == '__main__':
    main()
