"""
Train ARShoe M1 Model

Usage:
    python train_m1.py
"""

import torch
import yaml
import sys
from pathlib import Path

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

from models.arshoe_m1 import ARShoeM1
from training.trainer_m1 import ARShoeM1Trainer
from datasets.yolo_dataset import YOLOFootDataset


def main():
    # Training configuration — v3: CosineAnnealingLR + wider sigma + more augmentation
    config = {
        'batch_size': 16,        # ~55 steps/epoch for 880 images
        'learning_rate': 0.001,  # Cosine peak LR; decays to 1e-5 by epoch 200
        'weight_decay': 1e-4,
        'num_epochs': 200,       # Full cosine cycle (was 100 — LR never decayed!)
        'image_size': 256,
        'num_workers': 4,
        'output_dir': 'outputs/m1_training',
        'use_amp': True,
        'prefetch_factor': 2,
        'loss_weights': {
            'heatmap': 4.0,      # Primary task
            'paf': 2.0,          # Limb association
            'class': 1.5         # Raised from 0.5: class head needs more gradient to converge
        }
    }
    
    # Dataset root
    dataset_root = Path(__file__).parent.parent / "dataset" / "shuffled_v3"
    
    print("=" * 80)
    print("ARShoe M1 Training")
    print("=" * 80)
    
    # Create datasets
    print("\n📁 Loading datasets...")
    train_dataset = YOLOFootDataset(
        images_dir=dataset_root / "train" / "images",
        labels_dir=dataset_root / "train" / "labels",
        img_size=config['image_size'],
        augment=True,  # Enable horizontal flip augmentation
        flip_prob=0.5  # 50% chance of flipping
    )
    
    val_dataset = YOLOFootDataset(
        images_dir=dataset_root / "valid" / "images",
        labels_dir=dataset_root / "valid" / "labels",
        img_size=config['image_size'],
        augment=False  # No augmentation for validation
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = ARShoeM1(
        encoder_channels=128,
        num_keypoints=16,
        num_limbs=15,
        num_classes=2
    )
    
    total_params, breakdown = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder: {breakdown['encoder']:,}")
    print(f"  Heatmap Head: {breakdown['heatmap_head']:,}")
    print(f"  PAF Head: {breakdown['paf_head']:,}")
    print(f"  Class Head: {breakdown['class_head']:,}")
    print(f"  Budget: {total_params / 1_300_000 * 100:.1f}% of 1.3M")
    
    # Create trainer
    print("\n🎯 Initializing trainer...")
    trainer = ARShoeM1Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Start training
    print("\n🚀 Starting training...\n")
    trainer.train(num_epochs=config['num_epochs'])
    
    print(f"\n✅ Training complete!")
    print(f"📁 Checkpoints saved to: {trainer.checkpoint_dir}")
    

if __name__ == "__main__":
    main()
