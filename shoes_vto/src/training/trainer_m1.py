"""
ARShoe M1 Trainer

Training pipeline for basic 16-keypoint detection model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import time
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.arshoe_m1 import ARShoeM1
from models.heads.heatmap_head import generate_heatmaps_batch
from models.heads.paf_head import generate_pafs_batch
from models.heads.class_head import generate_class_maps_batch
from losses.heatmap_loss import HeatmapLoss, FocalHeatmapLoss, AdaptiveWingLoss
from losses.paf_loss import PAFLoss
from losses.class_loss import ClassLoss, compute_class_accuracy
from datasets.yolo_dataset import YOLOFootDataset, collate_fn
from training.eval_keypoints import decode_heatmaps_to_keypoints, compute_pck


class ARShoeM1Trainer:
    """
    Trainer for ARShoe M1 model
    
    Handles:
    - Training loop with multi-task loss
    - Validation with metrics
    - Checkpoint saving
    - Tensorboard logging (optional)
    """
    
    def __init__(self, 
                 model,
                 train_dataset,
                 val_dataset,
                 config):
        """
        Initialize trainer
        
        Args:
            model: ARShoeM1 instance
            train_dataset: YOLOFootDataset for training
            val_dataset: YOLOFootDataset for validation
            config: dict with training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # A100 & Modern NVIDIA GPU Optimizations
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("  [GPU] Enabled TF32 & cuDNN benchmark for A100 maximum throughput!")
        
        # PyTorch 2.x Compile on Linux (Colab A100)
        if sys.platform != 'win32' and config.get('use_compile', True):
            try:
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model)
                    print("  [GPU] Model compiled with torch.compile() for A100 maximum speed!")
            except Exception as e:
                print(f"  [GPU] torch.compile() skipped: {e}")
        else:
            print("  [INFO] Running model without torch.compile() (standard PyTorch mode)")
        
        # Data loaders - OPTIMIZED FOR SPEED
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=collate_fn,
            persistent_workers=True if config.get('num_workers', 0) > 0 else False,
            prefetch_factor=config.get('prefetch_factor', 2) if config.get('num_workers', 0) > 0 else None
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=collate_fn,
            persistent_workers=True if config.get('num_workers', 0) > 0 else False,
            prefetch_factor=config.get('prefetch_factor', 2) if config.get('num_workers', 0) > 0 else None
        )
        
        # Loss functions
        # Use Adaptive Wing Loss for robust heatmap regression with masking
        self.heatmap_loss_fn = AdaptiveWingLoss(use_mask=True)
        self.paf_loss_fn = PAFLoss(use_mask=True)
        self.class_loss_fn = ClassLoss()
        
        # Loss weights
        self.loss_weights = config.get('loss_weights', {
            'heatmap': 1.0,
            'paf': 0.5,
            'class': 0.3
        })
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler — cosine annealing decays LR unconditionally every epoch
        # This avoids the ReduceLROnPlateau issue where tiny loss improvements
        # constantly reset patience and the LR never decreases.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 200),
            eta_min=config.get('eta_min', 1e-5)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'outputs/m1_training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch - OPTIMIZED FOR SPEED"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'heatmap': 0.0,
            'paf': 0.0,
            'class': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to GPU with non_blocking for speed
            images = batch['image'].to(self.device, non_blocking=True)
            instances = batch['instances']  # List of instances per image
            
            # Pre-generate all GT on GPU in parallel
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Generate all ground truth tensors DIRECTLY ON GPU
                gt_heatmaps, hm_masks = generate_heatmaps_batch(
                    instances,
                    heatmap_size=64,
                    sigma=6.0,       # Wider Gaussian (was 3.5): 5-6px peaks → clearer gradient signal
                    device=self.device  # Generate directly on GPU
                )
                gt_pafs, paf_masks = generate_pafs_batch(
                    instances,
                    heatmap_size=64,
                    paf_width=8,
                    device=self.device  # 🚀 Generate directly on GPU
                )
                gt_class_maps, _ = generate_class_maps_batch(
                    instances,
                    heatmap_size=64,
                    image_size=256,
                    device=self.device  # 🚀 Generate directly on GPU
                )
                
                # No need for .to(device) - already on GPU!
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute all losses in parallel on GPU with validity masking
                heatmap_loss = self.heatmap_loss_fn(outputs['heatmaps'], gt_heatmaps, masks=hm_masks)
                paf_loss = self.paf_loss_fn(outputs['pafs'], gt_pafs, masks=paf_masks)
                class_loss = self.class_loss_fn(outputs['class_logits'], gt_class_maps)
                
                # Weighted total loss
                total_loss = (
                    self.loss_weights['heatmap'] * heatmap_loss +
                    self.loss_weights['paf'] * paf_loss +
                    self.loss_weights['class'] * class_loss
                )
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['heatmap'] += heatmap_loss.item()
            epoch_losses['paf'] += paf_loss.item()
            epoch_losses['class'] += class_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'hm': f"{heatmap_loss.item():.4f}",
                'paf': f"{paf_loss.item():.4f}",
                'cls': f"{class_loss.item():.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self):
        """Validate on validation set - OPTIMIZED FOR SPEED"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'heatmap': 0.0,
            'paf': 0.0,
            'class': 0.0
        }
        
        val_metrics = {
            'class_accuracy': 0.0,
            'pck_0.2': 0.0
        }
        
        num_batches = len(self.val_loader)
        all_pred_kps = []
        all_gt_kps = []
        all_gt_vis = []
        all_diags = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', leave=False):
                # Non-blocking GPU transfer
                images = batch['image'].to(self.device, non_blocking=True)
                instances = batch['instances']
                
                # Generate all GT DIRECTLY ON GPU with validity masks
                gt_heatmaps, hm_masks = generate_heatmaps_batch(
                    instances,
                    heatmap_size=64,
                    sigma=6.0,       # Match train sigma for consistent evaluation
                    device=self.device  # Generate directly on GPU
                )
                gt_pafs, paf_masks = generate_pafs_batch(
                    instances,
                    heatmap_size=64,
                    paf_width=8,
                    device=self.device  # 🚀 Generate directly on GPU
                )
                gt_class_maps, _ = generate_class_maps_batch(
                    instances,
                    heatmap_size=64,
                    image_size=256,
                    device=self.device  # 🚀 Generate directly on GPU
                )
                
                # No need for .to(device) - already on GPU!
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses on GPU with masking
                heatmap_loss = self.heatmap_loss_fn(outputs['heatmaps'], gt_heatmaps, masks=hm_masks)
                paf_loss = self.paf_loss_fn(outputs['pafs'], gt_pafs, masks=paf_masks)
                class_loss = self.class_loss_fn(outputs['class_logits'], gt_class_maps)
                
                total_loss = (
                    self.loss_weights['heatmap'] * heatmap_loss +
                    self.loss_weights['paf'] * paf_loss +
                    self.loss_weights['class'] * class_loss
                )
                
                # Compute classification metric
                class_acc, _ = compute_class_accuracy(outputs['class_probs'], gt_class_maps)
                
                # Decode keypoints per foot instance within its bounding box region
                hm_batch = outputs['heatmaps']
                img_sz = self.config.get('image_size', 256)
                for b, insts in enumerate(instances):
                    hm_img = hm_batch[b]
                    for inst in insts:
                        if 'keypoints' in inst and len(inst['keypoints']) == 16:
                            gt_kp = [kp[:2] for kp in inst['keypoints']]
                            gt_v = [kp[2] for kp in inst['keypoints']]
                            cx, cy, w, h = inst['bbox']
                            w_px = w * img_sz
                            h_px = h * img_sz
                            diag = (w_px**2 + h_px**2)**0.5
                            
                            # Instance-scoped search within bounding box (with 15% margin)
                            x1 = int(max(0, (cx - w * 0.575) * 64))
                            y1 = int(max(0, (cy - h * 0.575) * 64))
                            x2 = int(min(64, (cx + w * 0.575) * 64))
                            y2 = int(min(64, (cy + h * 0.575) * 64))
                            
                            # Ankle/Leg window (for ankle_center:12 and shin_mid:15)
                            leg_x1 = int(max(0, (cx - w * 0.75) * 64))
                            leg_x2 = int(min(64, (cx + w * 0.75) * 64))
                            leg_y1 = int(max(0, (cy - h * 1.5) * 64))
                            leg_y2 = int(min(64, (cy + h * 0.575) * 64))
                            
                            hm_inst = torch.zeros_like(hm_img)
                            foot_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
                            leg_indices = [12, 15]
                            hm_inst[foot_indices, y1:y2, x1:x2] = hm_img[foot_indices, y1:y2, x1:x2]
                            hm_inst[leg_indices, leg_y1:leg_y2, leg_x1:leg_x2] = hm_img[leg_indices, leg_y1:leg_y2, leg_x1:leg_x2]
                            
                            pred_inst, _ = decode_heatmaps_to_keypoints(hm_inst.unsqueeze(0), image_size=img_sz)
                            all_pred_kps.append(pred_inst[0].cpu().numpy())
                            all_gt_kps.append(gt_kp)
                            all_gt_vis.append(gt_v)
                            all_diags.append(diag)
                
                # Accumulate
                val_losses['total'] += total_loss.item()
                val_losses['heatmap'] += heatmap_loss.item()
                val_losses['paf'] += paf_loss.item()
                val_losses['class'] += class_loss.item()
                val_metrics['class_accuracy'] += class_acc
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
        val_metrics['class_accuracy'] /= num_batches
        
        if len(all_pred_kps) > 0:
            val_pck, _ = compute_pck(all_pred_kps, all_gt_kps, all_gt_vis, all_diags, threshold=0.2)
        else:
            val_pck = 0.0
        val_metrics['pck_0.2'] = val_pck
        
        return val_losses, val_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, num_epochs):
        """Main training loop"""
        print("=" * 80)
        print(f"Training ARShoe M1 for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Loss weights: {self.loss_weights}")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_losses = self.train_epoch()
            
            # Step cosine LR scheduler every epoch (not tied to validation)
            self.scheduler.step()
            
            # Validate every 2 epochs (skip validation to save time)
            if self.current_epoch % 2 == 0 or self.current_epoch == num_epochs:
                val_losses, val_metrics = self.validate()
                
                # Print epoch summary
                print(f"\nEpoch {self.current_epoch} Summary:")
                print(f"  Train - Total: {train_losses['total']:.4f}, "
                      f"HM: {train_losses['heatmap']:.4f}, "
                      f"PAF: {train_losses['paf']:.4f}, "
                      f"Cls: {train_losses['class']:.4f}")
                print(f"  Val   - Total: {val_losses['total']:.4f}, "
                      f"HM: {val_losses['heatmap']:.4f}, "
                      f"PAF: {val_losses['paf']:.4f}, "
                      f"Cls: {val_losses['class']:.4f}")
                print(f"  Metrics - Class Acc: {val_metrics['class_accuracy']:.2%}, PCK@0.2: {val_metrics['pck_0.2']:.2%}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Save checkpoint
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                
                self.save_checkpoint(is_best=is_best)
            else:
                # Skip validation, just print train losses
                print(f"\nEpoch {self.current_epoch} - Train: {train_losses['total']:.4f} "
                      f"(HM: {train_losses['heatmap']:.4f}, PAF: {train_losses['paf']:.4f}, "
                      f"Cls: {train_losses['class']:.4f})"
                      f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        print("\n" + "=" * 80)
        print(f"✅ Training complete! Best val loss: {self.best_val_loss:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    print("Trainer module - use train_m1.py to start training")
