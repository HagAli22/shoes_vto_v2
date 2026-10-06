"""
Fine-tune ARShoe M1 Model on New/Expanded Dataset

Loads pretrained weights from existing checkpoint (e.g. best.pth from v3),
resets optimizer and scheduler for fine-tuning, and trains on target dataset (e.g. shuffled_v4).

Usage:
    python fine_tune_m1.py --pretrained outputs/m1_training/checkpoints/best.pth \
                           --dataset_root ../dataset/shuffled_v4 \
                           --output_dir outputs/m1_finetune_v4 \
                           --epochs 75 \
                           --lr 0.0002
"""

import os
import sys
import argparse
from pathlib import Path
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models.arshoe_m1 import ARShoeM1
from training.trainer_m1 import ARShoeM1Trainer
from datasets.yolo_dataset import YOLOFootDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ARShoe M1 on target dataset")
    parser.add_argument(
        '--pretrained',
        type=str,
        default='outputs/m1_training/checkpoints/best.pth',
        help='Path to pretrained checkpoint to load weights from'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='../dataset/shuffled_v4',
        help='Path to dataset root directory (containing train/ and valid/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/m1_finetune_v4',
        help='Output directory to save fine-tuned checkpoints and logs'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=75,
        help='Number of fine-tuning epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0002,
        help='Initial learning rate for fine-tuning'
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
        '--freeze_encoder_epochs',
        type=int,
        default=0,
        help='Optional: Number of initial epochs to freeze encoder backbone'
    )
    return parser.parse_args()


def load_pretrained_weights(model, checkpoint_path, device='cpu'):
    """Load pretrained model weights, stripping torch.compile prefixes."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"📦 Loading pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        orig_epoch = checkpoint.get('epoch', 'N/A')
        orig_loss = checkpoint.get('best_val_loss', 'N/A')
        print(f"   Source checkpoint info: Epoch={orig_epoch}, Best Val Loss={orig_loss}")
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
        print(f"⚠️ Warning: Missing keys={len(missing)}, Unexpected keys={len(unexpected)}")
    else:
        print("✅ Pretrained weights loaded strictly and verified successfully!")


def main():
    args = parse_args()
    base_dir = Path(__file__).parent.resolve()

    ckpt_path = Path(args.pretrained)
    if not ckpt_path.is_absolute():
        ckpt_path = (base_dir / ckpt_path).resolve()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = (base_dir / dataset_root).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    print("=" * 80)
    print("👟 ARShoe M1 Fine-Tuning Pipeline")
    print("=" * 80)
    print(f"  Pretrained Checkpoint : {ckpt_path}")
    print(f"  Target Dataset Root   : {dataset_root}")
    print(f"  Output Directory      : {output_dir}")
    print(f"  Epochs                : {args.epochs}")
    print(f"  Learning Rate         : {args.lr} (Cosine decay -> {args.eta_min})")
    print(f"  Batch Size            : {args.batch_size}")
    print(f"  Workers               : {args.num_workers}")
    print(f"  Mixed Precision (AMP) : {not args.no_amp}")
    print("=" * 80)

    # 1. Dataset loading
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_images = dataset_root / "valid" / "images"
    val_labels = dataset_root / "valid" / "labels"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(f"Train or validation directories not found in {dataset_root}")

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
    print("\n🏗️ Initializing ARShoe M1 architecture...")
    model = ARShoeM1(
        encoder_channels=128,
        num_keypoints=16,
        num_limbs=15,
        num_classes=2
    )

    total_params, breakdown = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder: {breakdown['encoder']:,} | Heatmap: {breakdown['heatmap_head']:,} | PAF: {breakdown['paf_head']:,} | Class: {breakdown['class_head']:,}")

    # 3. Load pretrained weights
    load_pretrained_weights(model, ckpt_path)

    # 4. Configure Trainer
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

    print("\n🎯 Initializing Trainer with fresh fine-tuning optimizer & scheduler...")
    trainer = ARShoeM1Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )

    # 5. Start fine-tuning
    print(f"\n🚀 Launching Fine-Tuning for {args.epochs} epochs...\n")
    trainer.train(num_epochs=config['num_epochs'])

    print(f"\n🎉 Fine-tuning finished successfully!")
    print(f"💾 Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()

