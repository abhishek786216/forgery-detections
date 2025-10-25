"""
GPU-Optimized Training Script for Image Forgery Detection
Optimized for RTX 3050 6GB with best hyperparameters
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from datetime import datetime
import argparse

# Import project modules
from models import get_model, model_summary
from utils import (ForgeryDataModule, SegmentationMetrics, CombinedLoss,
                   plot_training_history, visualize_predictions)


def setup_gpu_optimization():
    """Setup GPU optimizations for RTX 3050 6GB"""
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"GPU Optimization Enabled")
        print(f"   Device: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        return True
    else:
        print("CUDA not available, falling back to CPU")
        return False


def get_optimal_hyperparameters(gpu_memory_gb=6):
    """
    Get optimal hyperparameters based on GPU memory and dataset size
    Optimized for RTX 3050 6GB and ~8K training images
    """
    
    # Base configuration for RTX 3050 6GB
    config = {
        # Model & Data
        'image_size': 256,  # Good balance of quality vs memory
        'batch_size': 8,    # Optimal for 6GB GPU
        'num_workers': 0,   # Windows compatibility (use 0 to avoid multiprocessing issues)
        
        # Training Schedule - More epochs for better convergence
        'epochs': 75,       # Increased for better results
        'warmup_epochs': 5, # Warm up learning rate
        
        # Learning Rate & Optimization
        'learning_rate': 2e-4,     # Lower initial LR for stability
        'weight_decay': 1e-4,      # Regularization
        'optimizer': 'adamw',      # AdamW is better than Adam
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        
        # Scheduler
        'use_scheduler': True,
        'scheduler_type': 'cosine_restarts',  # CosineAnnealingWarmRestarts
        'T_0': 10,          # Restart every 10 epochs
        'T_mult': 2,        # Double the period after each restart
        'eta_min': 1e-6,    # Minimum learning rate
        
        # Loss Function Weights - Optimized for forgery detection
        'loss_function': 'combined',
        'bce_weight': 1.0,
        'dice_weight': 2.0,     # Increased dice weight for segmentation
        'focal_weight': 1.5,    # Increased focal weight for hard examples
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        
        # Gradient & Memory Management
        'grad_clip': 0.5,       # Lower gradient clipping
        'mixed_precision': True, # Use AMP for memory efficiency
        'gradient_accumulation': 2, # Effective batch size = 16
        
        # Data Augmentation - Enhanced for robustness
        'augmentation': 'enhanced',
        'dropout_rate': 0.3,
        
        # Checkpointing
        'save_freq': 10,
        'early_stopping_patience': 15,
        'monitor_metric': 'val_f1',
    }
    
    return config


def create_optimized_trainer(config):
    """Create trainer with optimized configuration"""
    
    print("OPTIMIZED GPU TRAINING FOR IMAGE FORGERY DETECTION")
    print("=" * 80)
    print(f"Configuration Summary:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation']})")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Optimizer: {config['optimizer']}")
    print(f"   Scheduler: {config['scheduler_type']}")
    print(f"   Mixed Precision: {config['mixed_precision']}")
    print(f"   Image Size: {config['image_size']}x{config['image_size']}")
    print("=" * 80)
    
    # Enhanced trainer configuration with all required fields
    trainer_config = {
        'train_images_dir': 'dataset/train/images',
        'train_masks_dir': 'dataset/train/masks', 
        'val_images_dir': 'dataset/val/images',
        'val_masks_dir': 'dataset/val/masks',
        'experiment_name': f"gpu_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        
        # Add missing required fields
        'use_gpu': True,
        'val_split': 0.2,
        'momentum': 0.9,
        'lr_factor': 0.5,
        'lr_patience': 5,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'results_dir': 'results',
        'print_model_summary': True,
        'resume': False,
        'create_sample_data': False,
        'num_samples': 20,
        
        **config
    }
    
    return trainer_config


def train_with_optimal_config(args):
    """Main training function with optimal configuration"""
    
    # Setup GPU optimization
    gpu_available = setup_gpu_optimization()
    
    if not gpu_available:
        print("GPU not available, this script is optimized for GPU training")
        return False
    
    # Get optimal hyperparameters
    config = get_optimal_hyperparameters()
    
    # Override config with command line arguments
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size  
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Create trainer configuration
    trainer_config = create_optimized_trainer(config)
    
    # Import and setup trainer with enhanced logging
    from train_model import setup_logging, Trainer
    import argparse
    
    # Convert dict to namespace for compatibility
    config_namespace = argparse.Namespace(**trainer_config)
    
    # Setup logging
    log_dir = 'logs'
    experiment_name = trainer_config['experiment_name']
    logger, metrics_logger = setup_logging(log_dir, experiment_name)
    
    # Create trainer
    trainer = Trainer(config_namespace, logger, metrics_logger)
    
    # Log optimization info
    logger.info("GPU-OPTIMIZED TRAINING STARTED")
    logger.info(f"Target GPU: RTX 3050 6GB")
    logger.info(f"Expected training time: ~2-3 hours")
    logger.info(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation']}")
    
    # Start training
    try:
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        training_time = (end_time - start_time) / 3600  # Convert to hours
        logger.info(f"Training completed in {training_time:.2f} hours")
        
        # Generate final results
        trainer.save_results()
        
        print("\nTRAINING COMPLETED SUCCESSFULLY!")
        print(f"Total time: {training_time:.2f} hours")
        print(f"Results saved to: logs/{experiment_name}")
        print(f"Best model: checkpoints/best_model.pth")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining stopped by user")
        return False
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\nTraining failed: {str(e)}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='GPU-Optimized Training')
    parser.add_argument('--epochs', type=int, default=75, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    print(f"Starting GPU-Optimized Training")
    print(f"   Estimated time: 2-3 hours for {args.epochs} epochs")
    print(f"   Press Ctrl+C to stop anytime")
    print()
    
    success = train_with_optimal_config(args)
    
    if success:
        print("\nNext Steps:")
        print("1. View training results in logs/")
        print("2. Test the model with: python test_model.py")
        print("3. Try the web interface: streamlit run streamlit_app.py")
    
    return success


if __name__ == "__main__":
    main()