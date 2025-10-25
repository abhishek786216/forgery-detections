"""
Main training script for Image Forgery Detection using Noise Residual Learning
Enhanced with comprehensive logging capabilities
"""

import os
import sys
import argparse
import time
from datetime import datetime
import json
import warnings
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import project modules
from models import get_model, model_summary
from utils import (ForgeryDataModule, SegmentationMetrics, CombinedLoss,
                   plot_training_history, visualize_predictions, create_sample_data)

warnings.filterwarnings('ignore')


def setup_logging(log_dir, experiment_name):
    """
    Setup comprehensive logging system
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create loggers
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler for training log
    log_file = os.path.join(log_dir, f'{experiment_name}_training.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Create metrics logger (CSV format)
    metrics_logger = logging.getLogger('metrics')
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.handlers.clear()
    
    metrics_file = os.path.join(log_dir, f'{experiment_name}_metrics.csv')
    metrics_handler = logging.FileHandler(metrics_file, mode='w')
    
    # CSV formatter for metrics
    csv_formatter = logging.Formatter('%(message)s')
    metrics_handler.setFormatter(csv_formatter)
    metrics_logger.addHandler(metrics_handler)
    
    # Write CSV header
    with open(metrics_file, 'w') as f:
        f.write('timestamp,epoch,split,loss,iou,f1_score,pixel_accuracy,precision,recall\n')
    
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Training log: {log_file}")
    logger.info(f"Metrics log: {metrics_file}")
    
    return logger, metrics_logger


class Trainer:
    """
    Trainer class for Image Forgery Detection model with enhanced logging
    """
    
    def __init__(self, config, logger=None, metrics_logger=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
        # Setup logging
        self.logger = logger if logger else logging.getLogger('training')
        self.metrics_logger = metrics_logger if metrics_logger else logging.getLogger('metrics')
        
        # Log system info
        self.log_system_info()
        
        # Initialize model
        self.logger.info("Initializing model...")
        # Use logits (no sigmoid) for mixed precision training compatibility
        use_sigmoid = not getattr(config, 'mixed_precision', False)
        self.model = get_model(
            input_channels=3,
            num_classes=1,
            use_sigmoid=use_sigmoid
        ).to(self.device)
        
        # Log model architecture
        if config.print_model_summary:
            self.logger.info("Model architecture:")
            # Capture model summary to log
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                model_summary(self.model, (3, config.image_size, config.image_size))
            
            model_info = f.getvalue()
            self.logger.info(f"\n{model_info}")
        
        # Initialize loss function
        self.logger.info(f"Using loss function: {config.loss_function}")
        if config.loss_function == 'combined':
            focal_alpha = getattr(config, 'focal_alpha', 0.25)
            focal_gamma = getattr(config, 'focal_gamma', 2.0)
            self.criterion = CombinedLoss(
                bce_weight=config.bce_weight,
                dice_weight=config.dice_weight,
                focal_weight=config.focal_weight,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma
            )
        elif config.loss_function == 'bce':
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss function: {config.loss_function}")
        
        # Initialize optimizer
        self.logger.info(f"Using optimizer: {config.optimizer} with lr={config.learning_rate}")
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            # AdamW optimizer for better regularization
            betas = getattr(config, 'betas', (0.9, 0.999))
            eps = getattr(config, 'eps', 1e-8)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Initialize learning rate scheduler
        self.logger.info(f"Using scheduler: {config.use_scheduler}")
        if config.use_scheduler:
            scheduler_type = getattr(config, 'scheduler_type', 'plateau')
            
            if scheduler_type == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=config.lr_factor,
                    patience=config.lr_patience,
                    verbose=True
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.epochs,
                    eta_min=getattr(config, 'eta_min', 1e-6)
                )
            elif scheduler_type == 'cosine_restarts':
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=getattr(config, 'T_0', 10),
                    T_mult=getattr(config, 'T_mult', 2),
                    eta_min=getattr(config, 'eta_min', 1e-6)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        else:
            self.scheduler = None
            
        # Initialize mixed precision training
        self.use_mixed_precision = getattr(config, 'mixed_precision', False)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            
        # Gradient accumulation
        self.gradient_accumulation = getattr(config, 'gradient_accumulation', 1)
        if self.gradient_accumulation > 1:
            self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation}")
            self.logger.info(f"Effective batch size: {config.batch_size * self.gradient_accumulation}")
        
        # Initialize metrics
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        # Initialize data module
        self.logger.info("Setting up data loaders...")
        self.data_module = ForgeryDataModule(
            train_images_dir=config.train_images_dir,
            train_masks_dir=config.train_masks_dir,
            val_images_dir=config.val_images_dir,
            val_masks_dir=config.val_masks_dir,
            image_size=(config.image_size, config.image_size),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            val_split=config.val_split
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_history = {'iou': [], 'f1_score': [], 'pixel_accuracy': []}
        self.val_history = {'iou': [], 'f1_score': [], 'pixel_accuracy': []}
        
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        
        self.logger.info("Trainer initialization complete")
    
    def log_system_info(self):
        """Log system and environment information"""
        self.logger.info("=== SYSTEM INFORMATION ===")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"GPU device: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 50)
    
    def log_metrics(self, epoch, split, loss, metrics):
        """Log metrics to CSV file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_str = f"{timestamp},{epoch},{split},{loss:.6f},{metrics['iou']:.6f},"
        metrics_str += f"{metrics['f1_score']:.6f},{metrics['pixel_accuracy']:.6f},"
        metrics_str += f"{metrics['precision']:.6f},{metrics['recall']:.6f}"
        
        self.metrics_logger.info(metrics_str)
        
    def setup_logging(self):
        """Setup logging directories and tensorboard writer"""
        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Setup tensorboard
        log_dir = os.path.join(self.config.log_dir, 
                              f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.writer = SummaryWriter(log_dir)
        
        # Save config
        config_path = os.path.join(self.config.checkpoint_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=4)
    
    def train_epoch(self, epoch):
        """Train for one epoch with detailed logging"""
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        epoch_losses = []
        
        self.logger.info(f"Starting epoch {epoch + 1}/{self.config.epochs}")
        self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        batch_start_time = time.time()
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images, masks = images.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
            
            # Zero gradients (only at the beginning of accumulation cycle)
            if batch_idx % self.gradient_accumulation == 0:
                self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    # Calculate loss
                    if isinstance(self.criterion, CombinedLoss):
                        loss, loss_dict = self.criterion(outputs, masks)
                        loss_value = loss_dict['total_loss']
                    else:
                        loss = self.criterion(outputs, masks)
                        loss_dict = {'total_loss': loss.item()}
                        loss_value = loss.item()
                        
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation
                    
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
            else:
                # Standard forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if isinstance(self.criterion, CombinedLoss):
                    loss, loss_dict = self.criterion(outputs, masks)
                    loss_value = loss_dict['total_loss']
                else:
                    loss = self.criterion(outputs, masks)
                    loss_dict = {'total_loss': loss.item()}
                    loss_value = loss.item()
                    
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation
                
                # Backward pass
                loss.backward()
            
            running_loss += loss_value
            epoch_losses.append(loss_value)
            
            # Log detailed loss components occasionally
            if batch_idx % 100 == 0:
                loss_info = f"Batch {batch_idx}: "
                for loss_name, loss_value in loss_dict.items():
                    loss_info += f"{loss_name}={loss_value:.4f}, "
                self.logger.debug(loss_info.rstrip(', '))
            
            # Update optimizer (only at the end of accumulation cycle)
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.use_mixed_precision:
                    # Gradient clipping
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    self.optimizer.step()
            
            # Update metrics (convert logits to probabilities for metrics)
            if self.use_mixed_precision:
                outputs_prob = torch.sigmoid(outputs.detach())
            else:
                outputs_prob = outputs.detach()
            self.train_metrics.update(outputs_prob, masks)
            
            # Update progress bar
            current_metrics = self.train_metrics.get_metrics()
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'IoU': f'{current_metrics["iou"]:.4f}',
                'F1': f'{current_metrics["f1_score"]:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Log batch info occasionally
            if batch_idx % 200 == 0 and batch_idx > 0:
                batch_time = time.time() - batch_start_time
                self.logger.info(f"Batch {batch_idx}/{len(self.train_loader)}: "
                                f"Loss={running_loss/(batch_idx+1):.4f}, "
                                f"IoU={current_metrics['iou']:.4f}, "
                                f"Time={batch_time:.1f}s")
                batch_start_time = time.time()
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            
            if isinstance(self.criterion, CombinedLoss):
                for loss_name, loss_value in loss_dict.items():
                    self.writer.add_scalar(f'Loss/{loss_name}', loss_value, global_step)
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        final_metrics = self.train_metrics.get_metrics()
        
        # Log epoch results
        self.logger.info(f"Epoch {epoch + 1} Training Results:")
        self.logger.info(f"  Average Loss: {avg_loss:.6f}")
        self.logger.info(f"  IoU: {final_metrics['iou']:.6f}")
        self.logger.info(f"  F1 Score: {final_metrics['f1_score']:.6f}")
        self.logger.info(f"  Pixel Accuracy: {final_metrics['pixel_accuracy']:.6f}")
        
        # Log metrics to CSV
        self.log_metrics(epoch + 1, 'train', avg_loss, final_metrics)
        
        return avg_loss, final_metrics
    
    def validate_epoch(self, epoch):
        """Validate for one epoch with detailed logging"""
        if self.val_loader is None:
            return None, None
        
        self.logger.info(f"Starting validation for epoch {epoch + 1}")
        
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.val_loader, desc='Validation')):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if isinstance(self.criterion, CombinedLoss):
                    loss, _ = self.criterion(outputs, masks)
                else:
                    loss = self.criterion(outputs, masks)
                
                val_losses.append(loss.item())
                
                # Update metrics (convert logits to probabilities for metrics)
                if self.use_mixed_precision:
                    outputs_prob = torch.sigmoid(outputs.detach())
                else:
                    outputs_prob = outputs.detach()
                self.val_metrics.update(outputs_prob, masks)
                running_loss += loss.item()
        
        # Calculate epoch metrics
        avg_loss = np.mean(val_losses)
        final_metrics = self.val_metrics.get_metrics()
        
        # Log validation results
        self.logger.info(f"Epoch {epoch + 1} Validation Results:")
        self.logger.info(f"  Average Loss: {avg_loss:.6f}")
        self.logger.info(f"  IoU: {final_metrics['iou']:.6f}")
        self.logger.info(f"  F1 Score: {final_metrics['f1_score']:.6f}")
        self.logger.info(f"  Pixel Accuracy: {final_metrics['pixel_accuracy']:.6f}")
        
        # Log metrics to CSV
        self.log_metrics(epoch + 1, 'val', avg_loss, final_metrics)
        
        return avg_loss, final_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
            'config': vars(self.config)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with IoU: {self.best_val_iou:.6f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_iou = checkpoint.get('best_val_iou', 0.0)
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resuming from epoch {start_epoch}")
            
            return start_epoch
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return 0
    
    def train(self):
        """Main training loop"""
        print("Setting up data...")
        self.data_module.setup()
        
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        print(f"Training samples: {len(self.train_loader.dataset) if self.train_loader else 0}")
        print(f"Validation samples: {len(self.val_loader.dataset) if self.val_loader else 0}")
        
        # Print model summary
        print("\nModel Summary:")
        model_summary(self.model, (3, self.config.image_size, self.config.image_size))
        
        # Load checkpoint if resuming
        start_epoch = 0
        if self.config.resume and os.path.exists(
            os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        ):
            start_epoch = self.load_checkpoint(
                os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
            )
        
        print(f"\nStarting training from epoch {start_epoch + 1}...")
        
        for epoch in range(start_epoch, self.config.epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            if self.scheduler and val_loss is not None:
                self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            for metric in ['iou', 'f1_score', 'pixel_accuracy']:
                self.train_history[metric].append(train_metrics[metric])
                if val_metrics:
                    self.val_history[metric].append(val_metrics[metric])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
            
            for metric, value in train_metrics.items():
                self.writer.add_scalar(f'Metrics/Train_{metric}', value, epoch)
            
            if val_metrics:
                for metric, value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/Val_{metric}', value, epoch)
            
            # Check for best model
            is_best = False
            if val_loss is not None:
                if val_metrics['iou'] > self.best_val_iou:
                    self.best_val_iou = val_metrics['iou']
                    self.best_val_loss = val_loss
                    is_best = True
            else:
                if train_metrics['iou'] > self.best_val_iou:
                    self.best_val_iou = train_metrics['iou']
                    is_best = True
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_metrics['iou']:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            
            if is_best:
                print("New best model!")
        
        # Training complete
        print("\nTraining completed!")
        self.writer.close()
        
        # Save final results
        self.save_results()
    
    def save_results(self):
        """Save training results and plots"""
        # Save training history plot
        plot_path = os.path.join(self.config.results_dir, 'training_history.png')
        plot_training_history(
            train_losses=self.train_losses,
            val_losses=self.val_losses if self.val_losses else None,
            train_metrics=self.train_history,
            val_metrics=self.val_history if any(self.val_history.values()) else None,
            save_path=plot_path
        )
        
        # Save sample predictions
        if self.val_loader:
            self.model.eval()
            with torch.no_grad():
                for images, masks in self.val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    predictions = self.model(images)
                    
                    pred_path = os.path.join(self.config.results_dir, 'sample_predictions.png')
                    visualize_predictions(
                        images=images[:4],
                        ground_truth=masks[:4],
                        predictions=predictions[:4],
                        save_path=pred_path
                    )
                    break


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Image Forgery Detection Model')
    
    # Data arguments
    parser.add_argument('--train_images_dir', type=str, default='dataset/images',
                       help='Training images directory')
    parser.add_argument('--train_masks_dir', type=str, default='dataset/masks',
                       help='Training masks directory')
    parser.add_argument('--val_images_dir', type=str, default=None,
                       help='Validation images directory')
    parser.add_argument('--val_masks_dir', type=str, default=None,
                       help='Validation masks directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold (0 to disable)')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='LR reduction factor')
    parser.add_argument('--lr_patience', type=int, default=5,
                       help='LR scheduler patience')
    
    # Loss function arguments
    parser.add_argument('--loss_function', type=str, default='combined',
                       choices=['bce', 'combined'], help='Loss function type')
    parser.add_argument('--bce_weight', type=float, default=1.0,
                       help='BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                       help='Dice loss weight')
    parser.add_argument('--focal_weight', type=float, default=0.5,
                       help='Focal loss weight')
    
    # System arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Results directory')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint frequency (epochs)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--print_model_summary', action='store_true',
                       help='Print model summary')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    
    # Create sample data
    parser.add_argument('--create_sample_data', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of sample images to create')
    
    return parser.parse_args()


def main():
    """Main function with enhanced logging"""
    args = parse_args()
    
    # Set experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = f"casia_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*80)
    print("IMAGE FORGERY DETECTION - ENHANCED TRAINING WITH LOGGING")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Logs will be saved to: {args.log_dir}")
    print("="*80)
    
    # Setup logging
    logger, metrics_logger = setup_logging(args.log_dir, args.experiment_name)
    
    # Create sample data if requested
    if args.create_sample_data:
        logger.info("Creating sample data...")
        create_sample_data(args.train_images_dir, args.train_masks_dir, args.num_samples)
        logger.info("Sample data created successfully!")
        return
    
    # Check if training data exists
    if not os.path.exists(args.train_images_dir) or not os.path.exists(args.train_masks_dir):
        error_msg = f"Training data not found! Images: {args.train_images_dir}, Masks: {args.train_masks_dir}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        print("\nUse --create_sample_data to create sample data for testing.")
        return
    
    # Check if directories have data
    train_images = [f for f in os.listdir(args.train_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not train_images:
        logger.error("No training images found!")
        print("No training images found!")
        return
    
    logger.info(f"Found {len(train_images)} training images")
    
    # Log configuration
    logger.info("=== TRAINING CONFIGURATION ===")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)
    
    # Initialize trainer
    try:
        trainer = Trainer(args, logger, metrics_logger)
        
        # Start training
        logger.info("=== TRAINING START ===")
        trainer.train()
        
        logger.info("=== TRAINING COMPLETE ===")
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best model saved in: {args.checkpoint_dir}/best_model.pth")
        print(f"Training logs saved in: {args.log_dir}")
        print(f"Experiment: {args.experiment_name}")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining interrupted by user")
        if 'trainer' in locals():
            trainer.save_checkpoint(trainer.config.epochs - 1)
    except Exception as e:
        error_msg = f"Error during training: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"\nError: {error_msg}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()