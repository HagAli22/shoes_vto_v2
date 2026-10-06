"""
YOLO Format Dataset Loader for 16-Keypoint Foot Detection
Loads images and labels in YOLO pose format from Roboflow
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import random


# Keypoint mirroring map for horizontal flip
# When flipping horizontally, left foot becomes right foot and bilateral keypoints swap.
# Sagittal midline keypoints remain at the same index.
# Schema reference: dataset/keypoint-order-audit.md & configs/keypoint_schema.yaml:
#   0: toe_ground      (midline)           -> stays 0
#   1: heel_back       (midline)           -> stays 1
#   2: heel_ground     (midline)           -> stays 2
#   3: ball_medial     (bilateral pair)    <-> 4 (ball_lateral)
#   4: ball_lateral    (bilateral pair)    <-> 3 (ball_medial)
#   5: ball_top        (midline)           -> stays 5
#   6: instep_top      (midline)           -> stays 6
#   7: arch_medial     (bilateral pair)    <-> 8 (midfoot_lateral)
#   8: midfoot_lateral (bilateral pair)    <-> 7 (arch_medial)
#   9: malleolus_med   (bilateral pair)    <-> 10 (malleolus_lat)
#  10: malleolus_lat   (bilateral pair)    <-> 9 (malleolus_med)
#  11: toe_tip         (midline)           -> stays 11
#  12: ankle_center    (midline)           -> stays 12
#  13: throat          (midline)           -> stays 13
#  14: achilles        (midline)           -> stays 14
#  15: shin_mid        (midline)           -> stays 15
KEYPOINT_FLIP_MAP = {
    0: 0,    # toe_ground
    1: 1,    # heel_back
    2: 2,    # heel_ground
    3: 4,    # ball_medial <-> ball_lateral
    4: 3,    # ball_lateral <-> ball_medial
    5: 5,    # ball_top
    6: 6,    # instep_top
    7: 8,    # arch_medial <-> midfoot_lateral
    8: 7,    # midfoot_lateral <-> arch_medial
    9: 10,   # malleolus_medial <-> malleolus_lateral
    10: 9,   # malleolus_lateral <-> malleolus_medial
    11: 11,  # toe_tip
    12: 12,  # ankle_center
    13: 13,  # throat
    14: 14,  # achilles
    15: 15,  # shin_mid
}


class YOLOFootDataset(Dataset):
    """
    YOLO format foot dataset with 16 keypoints
    
    Label format (per line in .txt file):
    class_id cx cy w h kp0_x kp0_y kp0_v kp1_x kp1_y kp1_v ... kp15_x kp15_y kp15_v
    
    Where:
    - class_id: 0=left_foot, 1=right_foot
    - cx, cy, w, h: Normalized bbox (0-1)
    - kpX_x, kpX_y: Normalized keypoint coordinates (0-1)
    - kpX_v: Visibility (0=not labeled, 1=covered, 2=visible)
    """
    
    def __init__(self, 
                 images_dir,
                 labels_dir,
                 img_size=256,
                 transform=None,
                 return_original_image=False,
                 augment=False,
                 flip_prob=0.5,
                 cache_in_memory=True):
        """
        Args:
            images_dir: Path to images directory
            labels_dir: Path to labels directory
            img_size: Target image size (will be resized to img_size × img_size)
            transform: Optional transform to apply
            return_original_image: If True, also return original image (for visualization)
            augment: If True, apply data augmentation (horizontal flip)
            flip_prob: Probability of horizontal flip when augment=True
            cache_in_memory: If True, caches images and labels in RAM after first read
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.transform = transform
        self.return_original_image = return_original_image
        self.augment = augment
        self.flip_prob = flip_prob
        self.cache_in_memory = cache_in_memory
        self.cache = {}
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        if len(self.image_files) == 0:
            self.image_files = sorted(list(self.images_dir.glob("*.png")))
        
        print(f"Loaded {len(self.image_files)} images from {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # In-memory RAM caching for ultra-fast training
        if self.cache_in_memory and idx in self.cache:
            cached_img, instances_raw, (orig_width, orig_height) = self.cache[idx]
            image = cached_img.copy()
            instances = [
                {
                    'class_id': inst['class_id'],
                    'class_name': inst['class_name'],
                    'bbox': list(inst['bbox']),
                    'keypoints': [list(kp) for kp in inst['keypoints']]
                }
                for inst in instances_raw
            ]
        else:
            image_raw = Image.open(img_path).convert("RGB")
            orig_width, orig_height = image_raw.size
            label_path = self.labels_dir / (img_path.stem + ".txt")
            instances_raw = self._parse_label_file(label_path)
            cached_img = image_raw.resize((self.img_size, self.img_size), Image.BILINEAR)
            if self.cache_in_memory:
                self.cache[idx] = (cached_img, instances_raw, (orig_width, orig_height))
            image = cached_img.copy()
            instances = [
                {
                    'class_id': inst['class_id'],
                    'class_name': inst['class_name'],
                    'bbox': list(inst['bbox']),
                    'keypoints': [list(kp) for kp in inst['keypoints']]
                }
                for inst in instances_raw
            ]
        
        # Apply horizontal flip augmentation
        flip = self.augment and random.random() < self.flip_prob
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            instances = self._flip_instances(instances)
        
        # Apply random scale and translation jitter
        if self.augment and random.random() < 0.5:
            image, instances = self._affine_augment(image, instances)
        
        # Apply color jitter (brightness, contrast, saturation)
        if self.augment and random.random() < 0.5:
            image = self._color_jitter(image)
        
        # Resize image
        image_resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image_resized)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Scale keypoints to resized image coordinates (0 to img_size)
        for inst in instances:
            # Bbox (keep normalized for now)
            pass
            
            # Keypoints (convert from normalized to pixel coordinates on resized image)
            for kp in inst['keypoints']:
                kp[0] = kp[0] * self.img_size  # x
                kp[1] = kp[1] * self.img_size  # y
                # kp[2] is visibility, keep as is
        
        sample = {
            'image': image_tensor,
            'instances': instances,
            'image_path': str(img_path),
            'orig_size': (orig_width, orig_height)
        }
        
        if self.return_original_image:
            sample['original_image'] = np.array(image)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _flip_instances(self, instances):
        """
        Apply horizontal flip to instances
        - Flip x-coordinates: x_new = 1.0 - x_old (normalized coords)
        - Swap left/right class: 0 <-> 1
        - Rearrange keypoints according to KEYPOINT_FLIP_MAP
        
        Args:
            instances: List of instance dicts
        
        Returns:
            Flipped instances
        """
        flipped = []
        
        for inst in instances:
            # Flip class: left <-> right
            new_class_id = 1 - inst['class_id']  # 0->1, 1->0
            new_class_name = 'left_foot' if new_class_id == 0 else 'right_foot'
            
            # Flip bbox x-coordinate
            cx, cy, w, h = inst['bbox']
            new_bbox = [1.0 - cx, cy, w, h]
            
            # Flip and rearrange keypoints
            old_kps = inst['keypoints']
            new_kps = [[0.0, 0.0, 0] for _ in range(16)]  # Initialize independent sublists
            
            for old_idx, kp in enumerate(old_kps):
                x, y, v = kp
                new_idx = KEYPOINT_FLIP_MAP[old_idx]
                new_x = 1.0 - x if v > 0 else 0.0  # Only flip if visible
                new_kps[new_idx] = [new_x, y, v]
            
            flipped.append({
                'class_id': new_class_id,
                'class_name': new_class_name,
                'bbox': new_bbox,
                'keypoints': new_kps
            })
        
        return flipped
    
    def _affine_augment(self, image, instances):
        """
        Apply random scaling (0.85x to 1.15x) and translation (+/-6%)
        """
        scale = random.uniform(0.85, 1.15)
        dx = random.uniform(-0.06, 0.06)
        dy = random.uniform(-0.06, 0.06)
        W, H = image.size
        
        # PIL affine matrix mapping output coords to input coords
        a = 1.0 / scale
        b = 0.0
        c = -(0.5 * W + dx * W) / scale + 0.5 * W
        d = 0.0
        e = 1.0 / scale
        f = -(0.5 * H + dy * H) / scale + 0.5 * H
        
        image = image.transform((W, H), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
        
        aug_instances = []
        for inst in instances:
            cx, cy, w, h = inst['bbox']
            new_cx = (cx - 0.5) * scale + 0.5 + dx
            new_cy = (cy - 0.5) * scale + 0.5 + dy
            new_w = w * scale
            new_h = h * scale
            
            new_kps = []
            for kp in inst['keypoints']:
                kx, ky, kv = kp
                if kv > 0:
                    nkx = (kx - 0.5) * scale + 0.5 + dx
                    nky = (ky - 0.5) * scale + 0.5 + dy
                    if 0.0 <= nkx <= 1.0 and 0.0 <= nky <= 1.0:
                        new_kps.append([nkx, nky, kv])
                    else:
                        new_kps.append([0.0, 0.0, 0])
                else:
                    new_kps.append([0.0, 0.0, 0])
            
            aug_instances.append({
                'class_id': inst['class_id'],
                'class_name': inst['class_name'],
                'bbox': [new_cx, new_cy, new_w, new_h],
                'keypoints': new_kps
            })
        return image, aug_instances
    
    def _color_jitter(self, image):
        """
        Apply random brightness, contrast and saturation jitter
        """
        from PIL import ImageEnhance
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
        return image
    
    def _parse_label_file(self, label_path):
        """
        Parse YOLO format label file
        
        Returns:
            List of instances, each is a dict with:
            - 'class_id': int (0=left_foot, 1=right_foot)
            - 'bbox': [cx, cy, w, h] (normalized 0-1)
            - 'keypoints': [[x, y, v], ...] 16 keypoints (normalized 0-1)
        """
        instances = []
        
        if not label_path.exists():
            return instances
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            # Parse class and bbox
            class_id = int(parts[0])
            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            
            # Parse keypoints (16 keypoints × 3 = 48 values)
            keypoints = []
            kp_start_idx = 5
            for i in range(16):
                idx = kp_start_idx + i * 3
                if idx + 2 < len(parts):
                    x = float(parts[idx])
                    y = float(parts[idx + 1])
                    v = int(float(parts[idx + 2]))  # visibility
                    keypoints.append([x, y, v])
                else:
                    # Missing keypoint data, mark as not labeled
                    keypoints.append([0.0, 0.0, 0])
            
            instances.append({
                'class_id': class_id,
                'class_name': 'left_foot' if class_id == 0 else 'right_foot',
                'bbox': bbox,
                'keypoints': keypoints
            })
        
        return instances


def collate_fn(batch):
    """
    Custom collate function for batching
    Since different images may have different numbers of instances,
    we keep instances as a list rather than stacking
    """
    images = torch.stack([item['image'] for item in batch])
    instances_list = [item['instances'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    orig_sizes = [item['orig_size'] for item in batch]
    
    return {
        'image': images,
        'instances': instances_list,
        'image_path': image_paths,
        'orig_size': orig_sizes
    }


if __name__ == "__main__":
    # Test dataset loader
    print("Testing YOLO Foot Dataset Loader...")
    
    dataset = YOLOFootDataset(
        images_dir="dataset/shuffled_v3/train/images",
        labels_dir="dataset/shuffled_v3/train/labels",
        img_size=256,
        return_original_image=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Number of instances: {len(sample['instances'])}")
    
    for i, inst in enumerate(sample['instances']):
        print(f"\n  Instance {i}:")
        print(f"    Class: {inst['class_name']} ({inst['class_id']})")
        print(f"    Bbox: {inst['bbox']}")
        print(f"    Keypoints: {len(inst['keypoints'])}")
        
        # Count visible keypoints
        visible = sum(1 for kp in inst['keypoints'] if kp[2] > 0)
        print(f"    Visible keypoints: {visible}/16")
        
        # Check toe_tip (Index 11)
        toe_tip = inst['keypoints'][11]
        print(f"    toe_tip (Index 11): x={toe_tip[0]:.1f}, y={toe_tip[1]:.1f}, v={toe_tip[2]}")
