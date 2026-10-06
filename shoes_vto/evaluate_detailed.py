"""
Detailed evaluation of ARShoe M1 predictions vs ground truth
Analyzes raw outputs to identify training issues
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.arshoe_m1 import ARShoeM1
from datasets.yolo_dataset import YOLOFootDataset
from models.heads.heatmap_head import generate_heatmaps_batch
from models.heads.paf_head import generate_pafs_batch
from models.heads.class_head import generate_class_maps_batch


def load_model(checkpoint_path, device='cuda'):
    """Load trained model"""
    model = ARShoeM1()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def analyze_single_image(model, sample, device='cuda'):
    """Detailed analysis of single image prediction"""
    image = sample['image']
    gt_instances = sample['instances']
    
    # Forward pass
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        outputs = model(x)
    
    # Generate GT
    gt_heatmaps, hm_masks = generate_heatmaps_batch([gt_instances], heatmap_size=64, sigma=2.0)
    gt_pafs, paf_masks = generate_pafs_batch([gt_instances], heatmap_size=64, paf_width=8)
    gt_class_maps, class_masks = generate_class_maps_batch([gt_instances], heatmap_size=64, image_size=256)
    
    # Move to device
    gt_heatmaps = gt_heatmaps.to(device)
    gt_pafs = gt_pafs.to(device)
    gt_class_maps = gt_class_maps.to(device)
    hm_masks = hm_masks.to(device)
    
    # Extract predictions
    pred_heatmaps = outputs['heatmaps'][0]  # [16, 64, 64]
    pred_pafs = outputs['pafs'][0]  # [30, 64, 64]
    pred_class_probs = outputs['class_probs'][0]  # [2, 64, 64]
    
    gt_heatmaps = gt_heatmaps[0]  # [16, 64, 64]
    gt_pafs = gt_pafs[0]  # [30, 64, 64]
    gt_class_maps = gt_class_maps[0]  # [64, 64]
    hm_masks = hm_masks[0]  # [16, 64, 64]
    
    analysis = {
        'num_gt_instances': len(gt_instances),
        'heatmap_analysis': {},
        'class_analysis': {},
        'instances': []
    }
    
    # Analyze each ground truth instance
    for inst_idx, inst in enumerate(gt_instances):
        inst_analysis = {
            'gt_class': inst['class_id'],
            'gt_class_name': 'Left' if inst['class_id'] == 0 else 'Right',
            'keypoints': []
        }
        
        # Analyze each keypoint
        for kp_idx, (x, y, v) in enumerate(inst['keypoints']):
            if v > 0:  # Visible keypoint
                # Convert to heatmap coordinates
                hm_x = int(x * (64 / 256))
                hm_y = int(y * (64 / 256))
                hm_x = max(0, min(63, hm_x))
                hm_y = max(0, min(63, hm_y))
                
                # Get GT and predicted values at this location
                gt_val = gt_heatmaps[kp_idx, hm_y, hm_x].item()
                pred_val = pred_heatmaps[kp_idx, hm_y, hm_x].item()
                
                # Get max predicted value for this keypoint channel
                pred_max = pred_heatmaps[kp_idx].max().item()
                pred_max_loc = pred_heatmaps[kp_idx].argmax().item()
                pred_max_y = pred_max_loc // 64
                pred_max_x = pred_max_loc % 64
                
                # Distance between GT and predicted peak
                dist = np.sqrt((pred_max_x - hm_x)**2 + (pred_max_y - hm_y)**2)
                
                kp_analysis = {
                    'kp_idx': kp_idx,
                    'gt_location': (hm_x, hm_y),
                    'gt_value': gt_val,
                    'pred_at_gt': pred_val,
                    'pred_max': pred_max,
                    'pred_max_loc': (pred_max_x, pred_max_y),
                    'distance': dist,
                    'error': abs(gt_val - pred_val)
                }
                inst_analysis['keypoints'].append(kp_analysis)
        
        # Analyze class prediction at instance location
        if len(inst['keypoints']) > 0:
            # Get instance center
            visible_kps = [(x, y) for x, y, v in inst['keypoints'] if v > 0]
            if visible_kps:
                center_x = int(np.mean([x for x, y in visible_kps]) * (64 / 256))
                center_y = int(np.mean([y for x, y in visible_kps]) * (64 / 256))
                center_x = max(0, min(63, center_x))
                center_y = max(0, min(63, center_y))
                
                # Get class predictions at center
                pred_left_prob = pred_class_probs[0, center_y, center_x].item()
                pred_right_prob = pred_class_probs[1, center_y, center_x].item()
                pred_class = 0 if pred_left_prob > pred_right_prob else 1
                
                inst_analysis['center_location'] = (center_x, center_y)
                inst_analysis['pred_class'] = pred_class
                inst_analysis['pred_class_name'] = 'Left' if pred_class == 0 else 'Right'
                inst_analysis['pred_left_prob'] = pred_left_prob
                inst_analysis['pred_right_prob'] = pred_right_prob
                inst_analysis['class_correct'] = (pred_class == inst['class_id'])
        
        analysis['instances'].append(inst_analysis)
    
    # Overall heatmap statistics
    for kp_idx in range(16):
        mask = hm_masks[kp_idx]
        if mask.sum() > 0:  # Keypoint is labeled
            gt_vals = gt_heatmaps[kp_idx][mask > 0]
            pred_vals = pred_heatmaps[kp_idx][mask > 0]
            
            analysis['heatmap_analysis'][kp_idx] = {
                'gt_max': gt_vals.max().item(),
                'gt_mean': gt_vals.mean().item(),
                'pred_max': pred_heatmaps[kp_idx].max().item(),
                'pred_mean': pred_vals.mean().item(),
                'mse': ((gt_vals - pred_vals)**2).mean().item()
            }
    
    # Overall class statistics
    # Note: class_masks is a single channel mask [64, 64]
    if gt_class_maps[gt_class_maps >= 0].numel() > 0:  # Has valid class labels
        # Get valid regions (not ignore_index=-1)
        valid_mask = gt_class_maps >= 0
        
        # gt_class_maps is [64, 64], pred_class_probs is [2, 64, 64]
        gt_class_valid = gt_class_maps[valid_mask]
        pred_class_argmax = pred_class_probs.argmax(dim=0)  # [64, 64]
        pred_class_valid = pred_class_argmax[valid_mask]
        
        accuracy = (pred_class_valid == gt_class_valid).float().mean().item()
        
        # Per-class accuracy
        left_mask = gt_class_valid == 0
        right_mask = gt_class_valid == 1
        
        left_acc = (pred_class_valid[left_mask] == 0).float().mean().item() if left_mask.sum() > 0 else 0.0
        right_acc = (pred_class_valid[right_mask] == 1).float().mean().item() if right_mask.sum() > 0 else 0.0
        
        analysis['class_analysis'] = {
            'overall_accuracy': accuracy,
            'left_accuracy': left_acc,
            'right_accuracy': right_acc,
            'num_left_pixels': left_mask.sum().item(),
            'num_right_pixels': right_mask.sum().item()
        }
    
    return analysis


def print_analysis(image_idx, analysis):
    """Print detailed analysis"""
    print(f"\n{'='*80}")
    print(f"IMAGE {image_idx} - Detailed Analysis")
    print(f"{'='*80}")
    
    print(f"\nGround Truth: {analysis['num_gt_instances']} instances")
    
    # Analyze each instance
    for i, inst in enumerate(analysis['instances']):
        print(f"\nInstance {i+1}: GT={inst['gt_class_name']}")
        
        if 'pred_class_name' in inst:
            correct = "[OK]" if inst['class_correct'] else "[WRONG]"
            print(f"   Class Prediction: {inst['pred_class_name']} {correct}")
            print(f"   Probabilities: Left={inst['pred_left_prob']:.3f}, Right={inst['pred_right_prob']:.3f}")
        
        print(f"\n   Keypoint Analysis ({len(inst['keypoints'])} visible):")
        print(f"   {'KP':<4} {'GT Val':<8} {'Pred@GT':<9} {'PredMax':<9} {'Distance':<10} {'Status':<10}")
        print(f"   {'-'*60}")
        
        for kp in inst['keypoints']:
            status = "[OK]" if kp['distance'] < 3 else "[WARN]" if kp['distance'] < 5 else "[BAD]"
            print(f"   {kp['kp_idx']:<4} {kp['gt_value']:.3f}    "
                  f"{kp['pred_at_gt']:.3f}     {kp['pred_max']:.3f}     "
                  f"{kp['distance']:.1f} px      {status}")
    
    # Overall heatmap stats
    if analysis['heatmap_analysis']:
        print(f"\nHeatmap Statistics (per keypoint):")
        print(f"   {'KP':<4} {'GT Max':<8} {'Pred Max':<10} {'Pred Mean':<11} {'MSE':<8}")
        print(f"   {'-'*50}")
        
        for kp_idx, stats in sorted(analysis['heatmap_analysis'].items())[:5]:  # Show first 5
            print(f"   {kp_idx:<4} {stats['gt_max']:.3f}    "
                  f"{stats['pred_max']:.3f}      {stats['pred_mean']:.4f}      "
                  f"{stats['mse']:.4f}")
        if len(analysis['heatmap_analysis']) > 5:
            print(f"   ... ({len(analysis['heatmap_analysis']) - 5} more)")
    
    # Class statistics
    if analysis['class_analysis']:
        print(f"\nClass Prediction Accuracy:")
        ca = analysis['class_analysis']
        print(f"   Overall: {ca['overall_accuracy']*100:.1f}%")
        print(f"   Left:    {ca['left_accuracy']*100:.1f}% ({ca['num_left_pixels']} pixels)")
        print(f"   Right:   {ca['right_accuracy']*100:.1f}% ({ca['num_right_pixels']} pixels)")


def main():
    print("="*80)
    print("ARShoe M1 - Detailed Evaluation")
    print("="*80)
    
    # Paths
    checkpoint_path = Path("outputs/m1_training/checkpoints/best.pth")
    dataset_root = Path("..") / "dataset" / "shuffled_v3"
    
    # Load model
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device=device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = YOLOFootDataset(
        images_dir=dataset_root / "test" / "images",
        labels_dir=dataset_root / "test" / "labels",
        img_size=256
    )
    
    # Analyze specific images
    print(f"\nAnalyzing 5 test images (total: {len(test_dataset)})")
    
    # Use same indices as before for comparison
    test_indices = [64, 99, 40, 38, 45]
    
    for idx in test_indices:
        sample = test_dataset[idx]
        analysis = analyze_single_image(model, sample, device=device)
        print_analysis(idx, analysis)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
