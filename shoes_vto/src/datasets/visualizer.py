"""
Visualization utilities for foot keypoints, skeletons, and heatmaps
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


# Load keypoint schema and PAF connections
def load_configs():
    schema_path = Path("shoes_vto/configs/keypoint_schema.yaml")
    paf_path = Path("shoes_vto/configs/paf_connections.yaml")
    
    with open(schema_path, 'r') as f:
        schema = yaml.safe_load(f)
    
    with open(paf_path, 'r') as f:
        paf_config = yaml.safe_load(f)
    
    return schema, paf_config


# Colors for visualization
KEYPOINT_COLORS = [
    (255, 0, 0),    # 0: toe_ground - Red
    (255, 127, 0),  # 1: heel_back - Orange
    (255, 255, 0),  # 2: heel_ground - Yellow
    (0, 255, 0),    # 3: ball_medial - Green
    (0, 255, 127),  # 4: ball_lateral - Cyan
    (0, 255, 255),  # 5: ball_top - Aqua
    (0, 127, 255),  # 6: instep_top - Light Blue
    (0, 0, 255),    # 7: arch_medial - Blue
    (127, 0, 255),  # 8: midfoot_lateral - Purple
    (255, 0, 255),  # 9: malleolus_medial - Magenta
    (255, 0, 127),  # 10: malleolus_lateral - Pink
    (255, 255, 255),# 11: toe_tip - WHITE (MOST IMPORTANT)
    (127, 127, 127),# 12: ankle_center - Gray
    (191, 191, 191),# 13: throat - Light Gray
    (63, 63, 63),   # 14: achilles - Dark Gray
    (191, 127, 63), # 15: shin_mid - Brown
]

CLASS_COLORS = {
    'left_foot': (0, 255, 0),   # Green
    'right_foot': (255, 0, 0),  # Red (BGR format)
}


def draw_keypoints(image, keypoints, color=None, radius=5, thickness=-1):
    """
    Draw keypoints on image
    
    Args:
        image: numpy array (H, W, 3)
        keypoints: List of [x, y, visibility], 16 keypoints
        color: Single color tuple or None (will use per-keypoint colors)
        radius: Circle radius
        thickness: -1 for filled circle
    
    Returns:
        image with keypoints drawn
    """
    img = image.copy()
    
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # Only draw if visible or covered (v > 0)
            x, y = int(x), int(y)
            kp_color = color if color else KEYPOINT_COLORS[i]
            
            # Draw circle
            cv2.circle(img, (x, y), radius, kp_color, thickness)
            
            # Draw keypoint index
            cv2.putText(img, str(i), (x + 7, y - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, kp_color, 1)
    
    return img


def draw_skeleton(image, keypoints, limb_pairs, color=(0, 255, 255), thickness=2):
    """
    Draw skeleton connections between keypoints
    
    Args:
        image: numpy array (H, W, 3)
        keypoints: List of [x, y, visibility], 16 keypoints
        limb_pairs: List of [from_idx, to_idx, name] from PAF config
        color: Line color
        thickness: Line thickness
    
    Returns:
        image with skeleton drawn
    """
    img = image.copy()
    
    for from_idx, to_idx, name in limb_pairs:
        kp_from = keypoints[from_idx]
        kp_to = keypoints[to_idx]
        
        # Only draw if both keypoints are visible
        if kp_from[2] > 0 and kp_to[2] > 0:
            pt1 = (int(kp_from[0]), int(kp_from[1]))
            pt2 = (int(kp_to[0]), int(kp_to[1]))
            cv2.line(img, pt1, pt2, color, thickness)
    
    return img


def draw_bbox(image, bbox, class_name, color=None, thickness=2):
    """
    Draw bounding box
    
    Args:
        bbox: [cx, cy, w, h] in pixel coordinates
        class_name: 'left_foot' or 'right_foot'
    """
    img = image.copy()
    cx, cy, w, h = bbox
    
    # Convert to corner coordinates
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    
    # Get color
    if color is None:
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
    
    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    label = class_name
    cv2.putText(img, label, (x1, y1 - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img


def visualize_sample(sample, show_bbox=True, show_skeleton=True, show_keypoints=True):
    """
    Visualize a sample from the dataset
    
    Args:
        sample: Dict with 'image', 'instances' from YOLOFootDataset
        show_bbox: Whether to draw bounding boxes
        show_skeleton: Whether to draw skeleton connections
        show_keypoints: Whether to draw keypoint circles
    
    Returns:
        Visualization image (numpy array)
    """
    # Get image (convert from tensor if needed)
    if isinstance(sample['image'], np.ndarray):
        image = sample['image']
    else:
        # Tensor: CHW -> HWC
        image = sample['image'].permute(1, 2, 0).numpy()
    
    # Convert to uint8 if float
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Load PAF config for skeleton
    _, paf_config = load_configs()
    limb_pairs = paf_config['limb_pairs']
    
    # Draw each instance
    for inst in sample['instances']:
        class_name = inst['class_name']
        bbox = inst['bbox']
        keypoints = inst['keypoints']
        
        # Convert bbox from normalized to pixel coordinates
        img_h, img_w = image.shape[:2]
        bbox_pixel = [
            bbox[0] * img_w,  # cx
            bbox[1] * img_h,  # cy
            bbox[2] * img_w,  # w
            bbox[3] * img_h   # h
        ]
        
        # Draw bbox
        if show_bbox:
            image = draw_bbox(image, bbox_pixel, class_name)
        
        # Draw skeleton
        if show_skeleton:
            image = draw_skeleton(image, keypoints, limb_pairs)
        
        # Draw keypoints
        if show_keypoints:
            image = draw_keypoints(image, keypoints)
    
    return image


def visualize_multiple_samples(dataset, num_samples=5, save_path=None):
    """
    Visualize multiple samples in a grid
    
    Args:
        dataset: YOLOFootDataset instance
        num_samples: Number of samples to visualize
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        sample = dataset[i]
        vis_img = visualize_sample(sample)
        
        # Convert BGR to RGB for matplotlib
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(vis_img)
        axes[i].set_title(f"Sample {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def print_dataset_statistics(dataset):
    """
    Print dataset statistics
    """
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    
    # Count instances per class
    left_count = 0
    right_count = 0
    total_keypoints = 0
    visible_keypoints = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        for inst in sample['instances']:
            if inst['class_id'] == 0:
                left_count += 1
            else:
                right_count += 1
            
            for kp in inst['keypoints']:
                total_keypoints += 1
                if kp[2] > 0:
                    visible_keypoints += 1
    
    print(f"\nInstances:")
    print(f"  Left foot: {left_count}")
    print(f"  Right foot: {right_count}")
    print(f"  Total: {left_count + right_count}")
    
    print(f"\nKeypoints:")
    print(f"  Total keypoints: {total_keypoints}")
    print(f"  Visible keypoints: {visible_keypoints}")
    print(f"  Visibility rate: {visible_keypoints / total_keypoints * 100:.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from shoes_vto.src.dataset.yolo_dataset import YOLOFootDataset
    
    # Load dataset
    dataset = YOLOFootDataset(
        images_dir="dataset/shuffled_v3/train/images",
        labels_dir="dataset/shuffled_v3/train/labels",
        img_size=256,
        return_original_image=False
    )
    
    # Print statistics
    print_dataset_statistics(dataset)
    
    # Visualize samples
    print("\nVisualizing 5 samples...")
    visualize_multiple_samples(dataset, num_samples=5, 
                              save_path="shoes_vto/outputs/dataset_samples.png")
