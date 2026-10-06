"""
Train ARShoe M1_V2 Model (14 Keypoints)

Usage:
    python train_m1_v2.py
    python train_m1_v2.py --dry_run
    python train_m1_v2.py --num_epochs 200 --batch_size 16
"""

import argparse
import sys
from pathlib import Path
import torch

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
from training.trainer_m1_v2 import ARShoeM1V2Trainer

try:
    from datasets.yolo_dataset import YOLOFootDataset
except (ImportError, ModuleNotFoundError):
    from src.datasets.yolo_dataset import YOLOFootDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARShoe M1_V2 (14 Keypoints)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Peak learning rate")
    parser.add_argument("--num_epochs", type=int, default=200, help="Total epochs (default: 200)")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--output_dir", type=str, default="outputs/m1_v2_training", help="Output directory")
    parser.add_argument("--dataset_root", type=str, default=None, help="Root path to 14-KP dataset")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch test without full training")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'num_epochs': 1 if args.dry_run else args.num_epochs,
        'image_size': args.image_size,
        'num_workers': 0 if sys.platform == 'win32' else args.num_workers,
        'output_dir': args.output_dir,
        'paf_config': 'paf_connections_14kp.yaml',
        'use_amp': True,
        'max_batches': 2 if args.dry_run else None,
        'prefetch_factor': 2 if (sys.platform != 'win32' and args.num_workers > 0) else None,
        'loss_weights': {
            'heatmap': 4.0,
            'paf': 2.0,
            'class': 1.5
        }
    }
    
    # Locate dataset: prefer shuffled_v3_14kp, fallback to shuffled_v3 with target_keypoints=14
    project_root = Path(__file__).parent.parent
    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root)
    else:
        v3_14kp = project_root / "dataset" / "shuffled_v3_14kp"
        if v3_14kp.exists():
            dataset_root = v3_14kp
        else:
            dataset_root = project_root / "dataset" / "shuffled_v3"
            
    print("=" * 80)
    print("ARShoe M1_V2 Training Pipeline (14 Keypoints)")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Target keypoints: 14 (ankle_center & shin_mid removed)")
    
    # Create datasets
    print("\n📁 Loading datasets...")
    train_dataset = YOLOFootDataset(
        images_dir=dataset_root / "train" / "images",
        labels_dir=dataset_root / "train" / "labels",
        img_size=config['image_size'],
        augment=True,
        flip_prob=0.5,
        target_keypoints=14
    )
    
    val_dataset = YOLOFootDataset(
        images_dir=dataset_root / "valid" / "images",
        labels_dir=dataset_root / "valid" / "labels",
        img_size=config['image_size'],
        augment=False,
        target_keypoints=14
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    
    # Create M1_V2 model
    print("\n🏗️  Creating ARShoe M1_V2 Model...")
    model = ARShoeM1V2(
        encoder_channels=128,
        num_keypoints=14,
        num_limbs=14,
        num_classes=2
    )
    
    total_params, breakdown = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder:      {breakdown['encoder']:,}")
    print(f"  Heatmap Head: {breakdown['heatmap_head']:,} (14 channels)")
    print(f"  PAF Head:     {breakdown['paf_head']:,} (28 channels)")
    print(f"  Class Head:   {breakdown['class_head']:,}")
    print(f"  Budget:       {total_params / 1_300_000 * 100:.1f}% of 1.3M")
    
    # Initialize trainer
    print("\n🎯 Initializing trainer...")
    trainer = ARShoeM1V2Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    if args.dry_run:
        print("\n🧪 Running 1-epoch dry run verification...")
        trainer.train(num_epochs=1)
        print("\n✅ Dry run PASSED successfully!")
    else:
        print("\n🚀 Starting training...\n")
        trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()
