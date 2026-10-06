"""
ARShoe M1_V2 Trainer
Training pipeline for 14-keypoint detection model (without ankle_center and shin_mid)
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
import json
import csv
from tqdm import tqdm

# Add src and shoes_vto to path
_src_dir = Path(__file__).resolve().parent.parent
_project_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

# Evict third-party 'datasets' package from sys.modules if it shadowed local datasets
if 'datasets' in sys.modules and 'site-packages' in getattr(sys.modules['datasets'], '__file__', ''):
    del sys.modules['datasets']

from models.arshoe_m1_v2 import ARShoeM1V2
from models.heads.heatmap_head import generate_heatmaps_batch
from models.heads.paf_head import generate_pafs_batch
from models.heads.class_head import generate_class_maps_batch
from losses.heatmap_loss import HeatmapLoss, AdaptiveWingLoss
from losses.paf_loss import PAFLoss
from losses.class_loss import ClassLoss, compute_class_accuracy

try:
    from datasets.yolo_dataset import YOLOFootDataset, collate_fn
except (ImportError, ModuleNotFoundError):
    from src.datasets.yolo_dataset import YOLOFootDataset, collate_fn

from training.eval_keypoints import decode_heatmaps_to_keypoints, compute_pck, compute_multi_threshold_pck


class ARShoeM1V2Trainer:
    """
    Trainer for ARShoe M1_V2 model (14 keypoints, 14 limbs / 28 channels)
    """
    
    def __init__(self, 
                 model,
                 train_dataset,
                 val_dataset,
                 config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.num_keypoints = getattr(model, 'num_keypoints', 14)
        self.paf_config_name = config.get('paf_config', 'paf_connections_14kp.yaml')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # A100 & Modern NVIDIA GPU Optimizations
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("  [GPU] Enabled TF32 & cuDNN benchmark for maximum throughput!")
        
        # Data loaders
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
        self.heatmap_loss_fn = AdaptiveWingLoss(use_mask=True)
        self.paf_loss_fn = PAFLoss(use_mask=True)
        self.class_loss_fn = ClassLoss()
        
        # Loss weights
        self.loss_weights = config.get('loss_weights', {
            'heatmap': 4.0,
            'paf': 2.0,
            'class': 1.5
        })
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Cosine Annealing LR Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 200),
            eta_min=1e-5
        )
        
        # Mixed precision scaler
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print("  [AMP] Mixed precision training enabled (FP16/BF16)!")
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'outputs/m1_v2_training'))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_pck = 0.0
        self.current_epoch = 0
        self.train_history = []
        
        # CSV log
        self.csv_log_path = self.output_dir / "training_log.csv"
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'train_hm', 'train_paf', 'train_cls',
                             'val_loss', 'val_hm', 'val_paf', 'val_cls', 'val_pck02', 'val_cls_acc'])

    def train_epoch(self):
        self.model.train()
        epoch_losses = {'total': 0.0, 'heatmap': 0.0, 'paf': 0.0, 'class': 0.0}
        num_batches = len(self.train_loader)
        
        max_batches = self.config.get('max_batches', None)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch['image'].to(self.device, non_blocking=True)
            instances = batch['instances']
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                gt_heatmaps, hm_masks = generate_heatmaps_batch(
                    instances,
                    heatmap_size=64,
                    sigma=6.0,
                    device=self.device,
                    num_keypoints=self.num_keypoints
                )
                gt_pafs, paf_masks = generate_pafs_batch(
                    instances,
                    heatmap_size=64,
                    paf_width=8,
                    device=self.device,
                    config_name=self.paf_config_name
                )
                gt_class_maps, _ = generate_class_maps_batch(
                    instances,
                    heatmap_size=64,
                    image_size=256,
                    device=self.device
                )
                
                outputs = self.model(images)
                
                heatmap_loss = self.heatmap_loss_fn(outputs['heatmaps'], gt_heatmaps, masks=hm_masks)
                paf_loss = self.paf_loss_fn(outputs['pafs'], gt_pafs, masks=paf_masks)
                class_loss = self.class_loss_fn(outputs['class_logits'], gt_class_maps)
                
                total_loss = (
                    self.loss_weights['heatmap'] * heatmap_loss +
                    self.loss_weights['paf'] * paf_loss +
                    self.loss_weights['class'] * class_loss
                )
            
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
                
            epoch_losses['total'] += total_loss.item()
            epoch_losses['heatmap'] += heatmap_loss.item()
            epoch_losses['paf'] += paf_loss.item()
            epoch_losses['class'] += class_loss.item()
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'hm': f"{heatmap_loss.item():.4f}",
                'paf': f"{paf_loss.item():.4f}",
                'cls': f"{class_loss.item():.4f}"
            })
            
        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)
        return epoch_losses

    def validate(self):
        self.model.eval()
        val_losses = {'total': 0.0, 'heatmap': 0.0, 'paf': 0.0, 'class': 0.0}
        val_metrics = {'class_accuracy': 0.0, 'pck_0.2': 0.0}
        num_batches = len(self.val_loader)
        
        all_pred_kps = []
        all_gt_kps = []
        all_gt_vis = []
        all_diags = []
        
        max_batches = self.config.get('max_batches', None)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]", leave=False)):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                images = batch['image'].to(self.device, non_blocking=True)
                instances = batch['instances']
                
                gt_heatmaps, hm_masks = generate_heatmaps_batch(
                    instances,
                    heatmap_size=64,
                    sigma=6.0,
                    device=self.device,
                    num_keypoints=self.num_keypoints
                )
                gt_pafs, paf_masks = generate_pafs_batch(
                    instances,
                    heatmap_size=64,
                    paf_width=8,
                    device=self.device,
                    config_name=self.paf_config_name
                )
                gt_class_maps, _ = generate_class_maps_batch(
                    instances,
                    heatmap_size=64,
                    image_size=256,
                    device=self.device
                )
                
                outputs = self.model(images)
                
                heatmap_loss = self.heatmap_loss_fn(outputs['heatmaps'], gt_heatmaps, masks=hm_masks)
                paf_loss = self.paf_loss_fn(outputs['pafs'], gt_pafs, masks=paf_masks)
                class_loss = self.class_loss_fn(outputs['class_logits'], gt_class_maps)
                
                total_loss = (
                    self.loss_weights['heatmap'] * heatmap_loss +
                    self.loss_weights['paf'] * paf_loss +
                    self.loss_weights['class'] * class_loss
                )
                
                class_acc, _ = compute_class_accuracy(outputs['class_probs'], gt_class_maps)
                
                # Instance-scoped keypoint evaluation within bounding box
                hm_batch = outputs['heatmaps']
                img_sz = self.config.get('image_size', 256)
                for b, insts in enumerate(instances):
                    hm_img = hm_batch[b]
                    for inst in insts:
                        if 'keypoints' in inst and len(inst['keypoints']) == self.num_keypoints:
                            gt_kp = [kp[:2] for kp in inst['keypoints']]
                            gt_v = [kp[2] for kp in inst['keypoints']]
                            cx, cy, w, h = inst['bbox']
                            w_px = w * img_sz
                            h_px = h * img_sz
                            diag = (w_px**2 + h_px**2)**0.5
                            
                            x1 = int(max(0, (cx - w * 0.575) * 64))
                            y1 = int(max(0, (cy - h * 0.575) * 64))
                            x2 = int(min(64, (cx + w * 0.575) * 64))
                            y2 = int(min(64, (cy + h * 0.575) * 64))
                            
                            hm_inst = torch.zeros_like(hm_img)
                            hm_inst[:, y1:y2, x1:x2] = hm_img[:, y1:y2, x1:x2]
                            
                            pred_inst, _ = decode_heatmaps_to_keypoints(hm_inst.unsqueeze(0), image_size=img_sz)
                            all_pred_kps.append(pred_inst[0].cpu().numpy())
                            all_gt_kps.append(gt_kp)
                            all_gt_vis.append(gt_v)
                            all_diags.append(diag)
                            
                val_losses['total'] += total_loss.item()
                val_losses['heatmap'] += heatmap_loss.item()
                val_losses['paf'] += paf_loss.item()
                val_losses['class'] += class_loss.item()
                val_metrics['class_accuracy'] += class_acc
                
        for k in val_losses:
            val_losses[k] /= max(1, num_batches)
        val_metrics['class_accuracy'] /= max(1, num_batches)
        
        if len(all_pred_kps) > 0:
            val_pck, _ = compute_pck(all_pred_kps, all_gt_kps, all_gt_vis, all_diags, threshold=0.2)
        else:
            val_pck = 0.0
        val_metrics['pck_0.2'] = val_pck
        return val_losses, val_metrics

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_pck': self.best_pck,
            'config': self.config,
            'num_keypoints': self.num_keypoints,
        }
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")

    def train(self, num_epochs=200):
        print(f"Starting M1_V2 Training ({self.num_keypoints} Keypoints) for {num_epochs} epochs...")
        print(f"Output directory: {self.output_dir}\n")
        
        start_time = time.time()
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            train_losses = self.train_epoch()
            val_losses, val_metrics = self.validate()
            self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            if val_metrics['pck_0.2'] > self.best_pck:
                self.best_pck = val_metrics['pck_0.2']
                
            self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | LR: {lr:.6f} | "
                  f"Train Loss: {train_losses['total']:.4f} (hm: {train_losses['heatmap']:.4f}, paf: {train_losses['paf']:.4f}, cls: {train_losses['class']:.4f}) | "
                  f"Val Loss: {val_losses['total']:.4f} | PCK@0.2: {val_metrics['pck_0.2']*100:.2f}% | Cls Acc: {val_metrics['class_accuracy']*100:.2f}%"
                  f"{' [BEST]' if is_best else ''}")
                  
            # Append CSV
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, f"{lr:.6f}",
                    f"{train_losses['total']:.4f}", f"{train_losses['heatmap']:.4f}", f"{train_losses['paf']:.4f}", f"{train_losses['class']:.4f}",
                    f"{val_losses['total']:.4f}", f"{val_losses['heatmap']:.4f}", f"{val_losses['paf']:.4f}", f"{val_losses['class']:.4f}",
                    f"{val_metrics['pck_0.2']:.4f}", f"{val_metrics['class_accuracy']:.4f}"
                ])
                
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/3600:.2f} hours.")
        print(f"Best Val Loss: {self.best_val_loss:.4f} | Best Val PCK@0.2: {self.best_pck*100:.2f}%")
