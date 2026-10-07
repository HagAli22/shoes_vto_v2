"""
Train ARShoe M1v2 Architecture from Scratch on Merged Dataset (v3 + v4)

Usage:
    python train_m1v2.py --dataset_root ../dataset/merged_v3_v4 \
                         --output_dir outputs/m1v2_training \
                         --epochs 100 \
                         --batch_size 16 \
                         --lr 0.0005
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.optim as optim

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add src and base directory to path with robust absolute resolution
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from models.arshoe_m1v2 import ARShoeM1v2
from training.trainer_m1 import ARShoeM1Trainer
from datasets.yolo_dataset import YOLOFootDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARShoe M1v2 on merged dataset")
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='../dataset/merged_v3_v4',
        help='Path to merged dataset root containing train/ and valid/'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/m1v2_training',
        help='Output directory to save checkpoints and training summaries'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0005,
        help='Peak learning rate for heads (backbone uses 0.5x lr)'
    )
    parser.add_argument(
        '--eta_min',
        type=float,
        default=1e-6,
        help='Minimum learning rate for Cosine Annealing'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay regularization'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Input image resolution'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4 if sys.platform != 'win32' else 0,
        help='DataLoader worker count'
    )
    parser.add_argument(
        '--no_amp',
        action='store_true',
        help='Disable mixed precision training'
    )
    parser.add_argument(
        '--no_pretrained_backbone',
        action='store_true',
        help='Train MobileNetV3 backbone purely from random initialization instead of ImageNet weights'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        if (dataset_root / "train" / "images").exists():
            dataset_root = dataset_root.resolve()
        elif (base_dir / dataset_root / "train" / "images").exists():
            dataset_root = (base_dir / dataset_root).resolve()
        elif Path("dataset/merged_v3_v4/train/images").exists():
            dataset_root = Path("dataset/merged_v3_v4").resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    print("=" * 80)
    print("👟 ARShoe M1v2 Fresh Training Pipeline")
    print("=" * 80)
    print(f"  Target Dataset Root   : {dataset_root}")
    print(f"  Output Directory      : {output_dir}")
    print(f"  Total Epochs          : {args.epochs}")
    print(f"  Learning Rate         : {args.lr} (Cosine decay -> {args.eta_min})")
    print(f"  Batch Size            : {args.batch_size}")
    print(f"  Workers               : {args.num_workers}")
    print(f"  Mixed Precision (AMP) : {not args.no_amp}")
    print(f"  Pretrained Backbone   : {not args.no_pretrained_backbone}")
    print("=" * 80)

    # 1. Dataset loading
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_images = dataset_root / "valid" / "images"
    val_labels = dataset_root / "valid" / "labels"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(f"Train or valid directories not found in {dataset_root}")

    print("\n📁 Loading datasets...")
    train_dataset = YOLOFootDataset(
        images_dir=train_images,
        labels_dir=train_labels,
        img_size=args.image_size,
        augment=True,
        flip_prob=0.5
    )

    val_dataset = YOLOFootDataset(
        images_dir=val_images,
        labels_dir=val_labels,
        img_size=args.image_size,
        augment=False
    )

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val  : {len(val_dataset)} images")

    # 2. Model initialization
    print("\n🏗️  Initializing ARShoe M1v2 Architecture...")
    model = ARShoeM1v2(
        pretrained=not args.no_pretrained_backbone,
        num_keypoints=16,
        num_limbs=15,
        num_classes=2
    )

    total_params, breakdown = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Backbone (MobileNetV3): {breakdown['backbone']:,}")
    print(f"  FPN-lite Fusion       : {breakdown['fpn']:,}")
    print(f"  Heatmap Head (DSConv) : {breakdown['heatmap_head']:,}")
    print(f"  PAF Head (DSConv)     : {breakdown['paf_head']:,}")
    print(f"  Class Head (Global+DS): {breakdown['class_head']:,}")
    print(f"  Budget Utilization    : {total_params / 1_300_000 * 100:.1f}% of 1.3M budget")

    # 3. Configure Trainer
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_epochs': args.epochs,
        'eta_min': args.eta_min,
        'image_size': args.image_size,
        'num_workers': args.num_workers,
        'output_dir': str(output_dir),
        'use_amp': not args.no_amp,
        'prefetch_factor': 2 if args.num_workers > 0 else None,
        'loss_weights': {
            'heatmap': 4.0,
            'paf': 2.0,
            'class': 1.5
        }
    }

    print("\n🎯 Initializing Trainer with Differential Learning Rates...")
    # Differential parameter groups: 0.5x LR for backbone, 1.0x LR for FPN and heads
    backbone_params = (
        list(model.s4.parameters()) + 
        list(model.s8.parameters()) + 
        list(model.s16.parameters()) + 
        list(model.s32.parameters())
    )
    head_params = (
        list(model.lat32.parameters()) + 
        list(model.lat16.parameters()) + 
        list(model.lat8.parameters()) + 
        list(model.lat4.parameters()) + 
        list(model.fpn_conv.parameters()) + 
        list(model.hm_head.parameters()) + 
        list(model.paf_head.parameters()) + 
        list(model.cls_global.parameters()) + 
        list(model.cls_spatial.parameters())
    )

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.5},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min
    )

    trainer = ARShoeM1Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    # Assign our differential optimizer and scheduler
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    # 4. Start training
    print(f"\n🚀 Launching Fresh M1v2 Training for {args.epochs} epochs...\n")
    trainer.train(num_epochs=config['num_epochs'])

    print(f"\n🎉 M1v2 Training finished successfully!")
    print(f"💾 Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()

