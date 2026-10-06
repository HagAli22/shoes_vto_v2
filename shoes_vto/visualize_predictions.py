"""
Visualize ARShoe M1 Model Predictions

Loads trained model and visualizes predictions on test images
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.arshoe_m1 import ARShoeM1
from datasets.yolo_dataset import YOLOFootDataset
from utils.keypoint_grouping import group_keypoints


# Keypoint names for visualization
KEYPOINT_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "ball_top", "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "ankle_center",
    "throat", "achilles", "shin_mid"
]

# Color map: left foot = blue, right foot = red
CLASS_COLORS = {
    0: (255, 0, 0),    # Left = Blue (BGR)
    1: (0, 0, 255)     # Right = Red (BGR)
}

CLASS_NAMES = {0: "Left", 1: "Right"}


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    model = ARShoeM1()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def visualize_prediction(image, instances, gt_instances=None, save_path=None):
    """
    Visualize predictions on image
    
    Args:
        image: [3, 256, 256] tensor or [256, 256, 3] numpy array
        instances: List of predicted instances
        gt_instances: List of ground truth instances (optional)
        save_path: Path to save visualization
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy()
        if img.shape[0] == 3:  # [3, H, W]
            img = img.transpose(1, 2, 0)  # [H, W, 3]
        img = (img * 255).astype(np.uint8)
    else:
        img = image.copy()
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create figure with subplots
    if gt_instances is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Draw GT
        img_gt = img_bgr.copy()
        for inst in gt_instances:
            class_id = inst['class_id']
            color = CLASS_COLORS[class_id]
            keypoints = inst['keypoints']
            
            # Draw keypoints
            for kp_idx, (x, y, v) in enumerate(keypoints):
                if v > 0:  # Visible
                    x_px = int(x)
                    y_px = int(y)
                    cv2.circle(img_gt, (x_px, y_px), 3, color, -1)
                    cv2.putText(img_gt, str(kp_idx), (x_px+5, y_px-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        ax1.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
        ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Draw predictions
        img_pred = img_bgr.copy()
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        img_pred = img_bgr.copy()
    
    # Draw predicted instances
    for inst_idx, inst in enumerate(instances):
        class_id = inst['class_id']
        color = CLASS_COLORS[class_id]
        class_conf = inst['class_confidence']
        keypoints = inst['keypoints']
        
        # Draw keypoints
        visible_kps = []
        for kp_idx, (x, y, conf) in enumerate(keypoints):
            if conf > 0:  # Valid keypoint
                x_px = int(x * 4)  # Scale from 64x64 to 256x256
                y_px = int(y * 4)
                visible_kps.append((kp_idx, x_px, y_px))
                
                # Draw circle
                cv2.circle(img_pred, (x_px, y_px), 4, color, -1)
                # Draw keypoint index
                cv2.putText(img_pred, str(kp_idx), (x_px+6, y_px-6),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw instance info
        if len(visible_kps) > 0:
            # Find bounding box
            xs = [kp[1] for kp in visible_kps]
            ys = [kp[2] for kp in visible_kps]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Draw label
            label = f"{CLASS_NAMES[class_id]} ({class_conf:.2f})"
            cv2.putText(img_pred, label, (x_min, y_min-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw bbox
            cv2.rectangle(img_pred, (x_min-5, y_min-5), (x_max+5, y_max+5), color, 2)
    
    ax2.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    title = f'Predictions ({len(instances)} instances)'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved to {save_path}")
    
    plt.show()


def test_on_images(model, dataset, num_images=5, output_dir='outputs/visualizations'):
    """Test model on multiple images"""
    device = next(model.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Testing on {num_images} images from {len(dataset)} total")
    print("=" * 80)
    
    # Select random images
    indices = np.random.choice(len(dataset), size=min(num_images, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        image = sample['image']
        gt_instances = sample['instances']
        
        print(f"\n[Image {idx}]")
        print(f"  GT instances: {len(gt_instances)}")
        
        # Forward pass
        with torch.no_grad():
            x = image.unsqueeze(0).to(device)
            outputs = model(x)
        
        # Check heatmap peaks for debugging
        hm_max = outputs['heatmaps'].max().item()
        hm_mean = outputs['heatmaps'].mean().item()
        print(f"  Heatmap stats: max={hm_max:.3f}, mean={hm_mean:.3f}")
        
        # Group keypoints with lower thresholds
        pred_instances = group_keypoints(
            outputs['heatmaps'],
            outputs['pafs'],
            outputs['class_probs'],
            peak_threshold=0.05,  # Lowered from 0.1
            paf_threshold=0.01,   # Lowered from 0.05
            min_keypoints=3       # Lowered from 5
        )[0]  # Get first (only) batch item
        
        print(f"  Predicted instances: {len(pred_instances)}")
        
        # Show predictions
        for i, inst in enumerate(pred_instances):
            visible = sum(1 for kp in inst['keypoints'] if kp[2] > 0)
            print(f"    Instance {i}: {CLASS_NAMES[inst['class_id']]} "
                  f"(conf: {inst['class_confidence']:.2f}, "
                  f"keypoints: {visible}/{len(inst['keypoints'])})")
        
        # Visualize
        save_path = output_dir / f"prediction_{idx}.png"
        visualize_prediction(image, pred_instances, gt_instances, save_path=save_path)


def main():
    print("=" * 80)
    print("ARShoe M1 - Prediction Visualization")
    print("=" * 80)
    
    # Paths
    checkpoint_path = Path("outputs/m1_training/checkpoints/best.pth")
    dataset_root = Path("..") / "dataset" / "shuffled_v3"
    
    # Load model
    print("\n📦 Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device=device)
    
    # Load test dataset
    print("\n📁 Loading test dataset...")
    test_dataset = YOLOFootDataset(
        images_dir=dataset_root / "test" / "images",
        labels_dir=dataset_root / "test" / "labels",
        img_size=256
    )
    print(f"  Test images: {len(test_dataset)}")
    
    # Test on images
    test_on_images(model, test_dataset, num_images=5)
    
    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
