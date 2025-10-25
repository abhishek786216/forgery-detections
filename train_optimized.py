"""
Optimized Training Script for Maximum Performance
Uses pre-trained backbone and advanced training techniques for best results
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.nn.functional as F

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dataset import ImageForgeryDataset
from models.enhanced_model import get_enhanced_model
from utils.logging_utils import setup_enhanced_logging
from utils.metrics import iou_score, dice_score, pixel_accuracy


class OptimizedTrainer:
    """
    Optimized trainer with best practices for maximum performance
    """

    def __init__(self, args):
        self.args = args

        # Enhanced device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 Training Device: {self.device} ({gpu_name}, {gpu_memory:.1f}GB)")

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        else:
            self.device = torch.device('cpu')
            print(f"🖥️  Training Device: {self.device} (CPU)")

        # Mixed precision setup
        self.use_mixed_precision = getattr(args, 'use_mixed_precision', self.device.type == 'cuda')
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None

        if self.use_mixed_precision:
            print("⚡ Mixed Precision Training: ENABLED")
        else:
            print("📝 Mixed Precision Training: DISABLED")

        # Setup logging
        self.logger, self.log_dir = setup_enhanced_logging(args.experiment_name, "optimized_training")

        # Setup model, data, optimizer, loss
        self.model = self._setup_model()
        self.train_loader, self.val_loader = self._setup_data_loaders()
        self.optimizer, self.scheduler = self._setup_optimizer()
        self.criterion = self._setup_loss_functions()

        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.patience_counter = 0
        self.train_metrics = []
        self.val_metrics = []

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    # ===============================================================
    # MODEL SETUP
    # ===============================================================
    def _setup_model(self):
        print("🤖 Setting up Enhanced Model...")
        model = get_enhanced_model(pretrained=True, freeze_backbone_epochs=self.args.freeze_epochs)
        model = model.to(self.device)
        if self.args.resume_checkpoint:
            self._load_checkpoint()
        return model

    # ===============================================================
    # DATA LOADING
    # ===============================================================
    def _setup_data_loaders(self):
        print("📊 Setting up Data Loaders...")

        train_dataset = ImageForgeryDataset(
            images_dir=self.args.train_images_dir,
            masks_dir=self.args.train_masks_dir,
            transform_type='enhanced_train',
            image_size=(self.args.img_size, self.args.img_size)
        )

        val_dataset = ImageForgeryDataset(
            images_dir=self.args.val_images_dir,
            masks_dir=self.args.val_masks_dir,
            transform_type='val',
            image_size=(self.args.img_size, self.args.img_size)
        )

        pin_memory = getattr(self.args, 'pin_memory', True) and self.device.type == 'cuda'
        num_workers = 0 if os.name == 'nt' else self.args.num_workers

        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=False, prefetch_factor=2 if num_workers > 0 else None
        )

        print(f"  ✅ Training samples: {len(train_dataset):,}")
        print(f"  ✅ Validation samples: {len(val_dataset):,}")
        print(f"  ✅ Batch size: {self.args.batch_size}")
        print(f"  ✅ Training batches per epoch: {len(train_loader):,}")

        return train_loader, val_loader

    # ===============================================================
    # OPTIMIZER
    # ===============================================================
    def _setup_optimizer(self):
        print("⚡ Setting up Optimizer...")
        backbone_params, decoder_params = [], []

        for name, param in self.model.named_parameters():
            (backbone_params if 'backbone_features' in name else decoder_params).append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.args.learning_rate * 0.1, 'weight_decay': 0.01},
            {'params': decoder_params, 'lr': self.args.learning_rate, 'weight_decay': 0.01}
        ], betas=(0.9, 0.999), eps=1e-8)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.args.epochs // 4, T_mult=2,
            eta_min=self.args.learning_rate * 0.001
        )

        print(f"  ✅ Optimizer: AdamW with differential learning rates")
        print(f"  ✅ Backbone LR: {self.args.learning_rate * 0.1:.6f}")
        print(f"  ✅ Decoder LR: {self.args.learning_rate:.6f}")
        print(f"  ✅ Scheduler: CosineAnnealingWarmRestarts")

        return optimizer, scheduler

    # ===============================================================
    # LOSS FUNCTIONS
    # ===============================================================
    def _setup_loss_functions(self):
        print("🎯 Setting up Loss Functions...")

        class IoULoss(nn.Module):
            def __init__(self, smooth=1e-8):
                super().__init__()
                self.smooth = smooth

            def forward(self, pred, target):
                pred = torch.sigmoid(pred)  # ensure in [0,1]
                pred_flat = pred.view(pred.size(0), -1)
                target_flat = target.view(target.size(0), -1)
                intersection = (pred_flat * target_flat).sum(dim=1)
                union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
                iou = (intersection + self.smooth) / (union + self.smooth)
                return 1 - iou.mean()

        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, pred, target):
                # Compute BCE safely on logits
                bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

# then compute focal weight using sigmoid probabilities
                prob = torch.sigmoid(pred)
                p_t = prob * target + (1 - prob) * (1 - target)
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                focal_weight = alpha_t * (1 - p_t) ** self.gamma
                return (focal_weight * bce).mean()


        class CombinedLoss(nn.Module):
            """Simpler loss: BCEWithLogits + IoU (autocast-safe)"""
            def __init__(self):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss()
                self.iou = IoULoss()

            def forward(self, pred, target):
                bce_loss = self.bce(pred, target)
                iou_loss = self.iou(pred, target)
                total_loss = 0.5 * bce_loss + 0.5 * iou_loss
                return {
                    'total': total_loss,
                    'bce': bce_loss,
                    'iou': iou_loss,
                    'focal': torch.tensor(0.0, device=pred.device)
                }

        criterion = CombinedLoss()
        print("  ✅ Combined Loss: BCEWithLogits (50%) + IoU (50%)")
        return criterion

    # ===============================================================
    # TRAINING + VALIDATION LOOP (unchanged)
    # ===============================================================
    def train_epoch(self, epoch):
        self.model.train()
        self.model.set_epoch(epoch)
        total_loss, loss_components = 0.0, {'bce': 0.0, 'iou': 0.0, 'focal': 0.0}
        progress_interval = max(1, len(self.train_loader) // 20)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, masks)

                self.scaler.scale(loss_dict['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, masks)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            total_loss += loss_dict['total'].item()
            for k in loss_components:
                loss_components[k] += loss_dict[k].item()

            if batch_idx % progress_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                prog = 100.0 * batch_idx / len(self.train_loader)
                self.logger.info(f"Epoch {epoch} [{prog:.1f}%] - Loss: {loss_dict['total'].item():.6f}, LR: {lr:.8f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}
        return avg_loss, avg_components

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss, loss_components = 0.0, {'bce': 0.0, 'iou': 0.0, 'focal': 0.0}
        all_ious, all_dices = [], []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, masks)
                total_loss += loss_dict['total'].item()
                for k in loss_components:
                    loss_components[k] += loss_dict[k].item()
                all_ious.append(iou_score(outputs, masks))
                all_dices.append(dice_score(outputs, masks))

        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in loss_components.items()}
        avg_iou = sum(all_ious) / len(all_ious)
        avg_dice = sum(all_dices) / len(all_dices)
        return avg_loss, avg_components, avg_iou, avg_dice

    # Rest of code (save_checkpoint, train, main) remains unchanged
    # ===============================================================

# MAIN FUNCTION (unchanged)
def main():
    parser = argparse.ArgumentParser(description='Optimized Image Forgery Detection Training')
    # [arguments same as before...]
    args = parser.parse_args()
    print("🎯 OPTIMIZED TRAINING CONFIGURATION\n" + "="*50)
    trainer = OptimizedTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
