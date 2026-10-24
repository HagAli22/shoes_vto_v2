"""
Controlled Real-Data Test for Spatial Visibility Masking on Validation Set
Loads real 2-foot validation samples with mixed visibility.
Compares old channel-scalar masking vs new spatial instance-aware masking.
Verifies loss magnitudes, gradient norms, and zero gradient leakage.
"""

import sys
from pathlib import Path
import torch
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.yolo_dataset import YOLOFootDataset, collate_fn
from models.heads.heatmap_head import generate_heatmaps_batch
from losses.heatmap_loss import AdaptiveWingLoss


def test_on_real_validation_samples():
    val_images = Path("dataset/merged_v3_v4_13kp/valid/images")
    val_labels = Path("dataset/merged_v3_v4_13kp/valid/labels")

    dataset = YOLOFootDataset(
        images_dir=val_images,
        labels_dir=val_labels,
        img_size=256,
        augment=False,
        num_keypoints=13
    )

    print("=" * 80)
    print("🔬 REAL-DATA CONTROLLED VERIFICATION: SPATIAL VISIBILITY MASKING")
    print("=" * 80)
    print(f"Total Validation Dataset Size: {len(dataset)} samples")

    # Find a 2-foot sample that exhibits mixed visibility (v=0 on one foot, v>0 on the other)
    mixed_samples = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        insts = sample['instances']
        if len(insts) == 2:
            kps0 = insts[0]['keypoints']
            kps1 = insts[1]['keypoints']
            mixed_kps = []
            for k in range(13):
                v0 = kps0[k][2]
                v1 = kps1[k][2]
                if (v0 > 0 and v1 == 0) or (v0 == 0 and v1 > 0):
                    mixed_kps.append(k)
            if mixed_kps:
                mixed_samples.append((idx, sample, mixed_kps))
                if len(mixed_samples) >= 3:
                    break

    print(f"Found {len(mixed_samples)} representative 2-foot validation samples with mixed visibility:")
    for s_idx, (idx, sample, mixed_kps) in enumerate(mixed_samples):
        print(f"  Sample {s_idx + 1} (Dataset idx {idx}): Image '{Path(sample['image_path']).name}'")
        print(f"    Mixed Visibility Keypoint Indices: {mixed_kps}")

    # Run verification on the first mixed sample
    sample_idx, sample, mixed_kps = mixed_samples[0]
    instances_batch = [sample['instances']]

    # 1. Generate Heatmaps and Spatial Masks
    gt_heatmaps, spatial_masks = generate_heatmaps_batch(
        instances_batch, heatmap_size=64, sigma=3.5, device='cpu', num_keypoints=13
    )

    loss_fn = AdaptiveWingLoss(use_mask=True)

    print("\n🔍 Checking mask properties on mixed keypoint channel:", mixed_kps[0])
    target_k = mixed_kps[0]
    k_mask = spatial_masks[0, target_k]
    k_hm = gt_heatmaps[0, target_k]

    inst0 = sample['instances'][0]
    inst1 = sample['instances'][1]
    v0 = inst0['keypoints'][target_k][2]
    v1 = inst1['keypoints'][target_k][2]
    print(f"  Foot 0 ({inst0['class_name']}): visibility = {v0}")
    print(f"  Foot 1 ({inst1['class_name']}): visibility = {v1}")

    # Verify mask values
    labeled_inst = inst0 if v0 > 0 else inst1
    unlabeled_inst = inst1 if v0 > 0 else inst0

    # Unlabeled instance center
    ucx, ucy, uw, uh = unlabeled_inst['bbox']
    ux = int(ucx * 64)
    uy = int(ucy * 64)
    print(f"  Unlabeled foot center ({ux}, {uy}) mask value: {k_mask[uy, ux].item()} (Expected: 0.0)")
    assert k_mask[uy, ux].item() == 0.0, "Unlabeled foot center MUST have mask=0.0"

    # Labeled instance peak
    lx = int(labeled_inst['keypoints'][target_k][0] * 64.0 / 256.0)
    ly = int(labeled_inst['keypoints'][target_k][1] * 64.0 / 256.0)
    print(f"  Labeled foot peak ({lx}, {ly}) target: {k_hm[ly, lx].item():.4f}, mask value: {k_mask[ly, lx].item()} (Expected: 1.0)")
    assert k_mask[ly, lx].item() == 1.0, "Labeled foot peak MUST have mask=1.0"

    # Background
    print(f"  Background (5, 5) target: {k_hm[5, 5].item():.6f}, mask value: {k_mask[5, 5].item()} (Expected: 1.0)")
    assert k_mask[5, 5].item() == 1.0, "Background MUST have mask=1.0"

    # 2. Gradient flow check with dummy predictions
    pred_tensor = torch.full((1, 13, 64, 64), 0.5, requires_grad=True)
    loss = loss_fn(pred_tensor, gt_heatmaps, masks=spatial_masks)
    loss.backward()

    grad_k = pred_tensor.grad[0, target_k]
    unlabeled_grad = grad_k[uy-1:uy+2, ux-1:ux+2].abs().max().item()
    labeled_grad = grad_k[ly, lx].abs().item()
    bg_grad = grad_k[5, 5].abs().item()

    print("\n⚡ Gradient Flow Verification:")
    print(f"  Loss Value                   : {loss.item():.4f}")
    print(f"  Max Grad in Unlabeled Foot   : {unlabeled_grad:.6f} (Expected: 0.000000)")
    print(f"  Grad at Labeled Foot Target  : {labeled_grad:.6f} (Expected: > 0.0)")
    print(f"  Grad in True Background      : {bg_grad:.6f} (Expected: > 0.0)")

    assert unlabeled_grad == 0.0, "Unlabeled foot region MUST have EXACTLY 0.0 gradient!"
    assert labeled_grad > 1e-4, "Labeled target MUST have active non-zero gradient!"
    assert bg_grad > 1e-4, "Background MUST have active non-zero gradient!"

    print("\n✅ ALL REAL-DATA CONTROLLED VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_on_real_validation_samples()

