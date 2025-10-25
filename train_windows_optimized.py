"""
Windows-Optimized Training Resume Script
Fixes multiprocessing issues and resumes training from checkpoint
"""

import os
import torch
import argparse
from datetime import datetime

def resume_training():
    """Resume training with Windows optimizations"""
    
    print("🔄 RESUMING GPU-OPTIMIZED TRAINING (Windows Compatible)")
    print("=" * 70)
    
    # Check for existing checkpoints
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoints directory found")
        return False
        
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoint_files:
        print("❌ No checkpoint files found")
        print("🔄 Starting fresh training with Windows optimizations...")
        return start_fresh_windows_training()
    
    # Find the latest checkpoint
    latest_checkpoint = None
    latest_epoch = -1
    
    for ckpt in checkpoint_files:
        if 'epoch_' in ckpt:
            try:
                epoch_num = int(ckpt.split('epoch_')[1].split('.')[0])
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint = ckpt
            except:
                continue
    
    if latest_checkpoint:
        print(f"📁 Found checkpoint: {latest_checkpoint} (Epoch {latest_epoch})")
        print("🔄 Resuming training...")
        return resume_from_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint))
    else:
        print("🔄 Starting fresh training with Windows optimizations...")
        return start_fresh_windows_training()


def start_fresh_windows_training():
    """Start fresh training optimized for Windows"""
    
    # Import after setting multiprocessing
    import multiprocessing
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)
    
    from train_model import setup_logging, Trainer
    
    # Windows-optimized configuration
    config = {
        # Data paths
        'train_images_dir': 'dataset/train/images',
        'train_masks_dir': 'dataset/train/masks', 
        'val_images_dir': 'dataset/val/images',
        'val_masks_dir': 'dataset/val/masks',
        
        # Training parameters
        'epochs': 75,
        'batch_size': 8,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'grad_clip': 0.5,
        
        # Optimizer
        'optimizer': 'adamw',
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        
        # Scheduler
        'use_scheduler': True,
        'scheduler_type': 'cosine_restarts',
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-6,
        
        # Loss function
        'loss_function': 'combined',
        'bce_weight': 1.0,
        'dice_weight': 2.0,
        'focal_weight': 1.5,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        
        # Windows optimizations
        'num_workers': 0,  # Critical for Windows!
        'mixed_precision': True,
        'gradient_accumulation': 2,
        
        # Other settings
        'image_size': 256,
        'use_gpu': True,
        'val_split': 0.2,
        'momentum': 0.9,
        'lr_factor': 0.5,
        'lr_patience': 5,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'results_dir': 'results',
        'save_freq': 10,
        'print_model_summary': True,
        'resume': False,
        'create_sample_data': False,
        'num_samples': 20,
        'experiment_name': f"windows_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # Convert to namespace
    config_namespace = argparse.Namespace(**config)
    
    # Setup logging
    logger, metrics_logger = setup_logging(config['log_dir'], config['experiment_name'])
    
    # Create trainer
    trainer = Trainer(config_namespace, logger, metrics_logger)
    
    # Log optimization info
    logger.info("🖥️ WINDOWS-OPTIMIZED TRAINING STARTED")
    logger.info("⚡ GPU optimizations enabled")
    logger.info("🔧 Multiprocessing disabled (Windows compatibility)")
    logger.info(f"🎯 Target: {config['epochs']} epochs")
    
    try:
        # Start training
        trainer.train()
        print("\n🎉 Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        return False


def resume_from_checkpoint(checkpoint_path):
    """Resume training from a specific checkpoint"""
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        resume_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"📈 Resuming from epoch {resume_epoch}")
        
        # Update configuration for resuming
        config = checkpoint.get('config', {})
        config.update({
            'resume': True,
            'resume_path': checkpoint_path,
            'start_epoch': resume_epoch,
            'num_workers': 0,  # Windows compatibility
            'experiment_name': f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })
        
        # Convert to namespace
        config_namespace = argparse.Namespace(**config)
        
        # Setup logging
        from train_model import setup_logging, Trainer
        logger, metrics_logger = setup_logging(config['log_dir'], config['experiment_name'])
        
        # Create trainer
        trainer = Trainer(config_namespace, logger, metrics_logger)
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        # Resume training
        logger.info(f"🔄 RESUMING TRAINING FROM EPOCH {resume_epoch}")
        trainer.train()
        
        print(f"\n🎉 Training resumed and completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to resume training: {e}")
        return False


if __name__ == '__main__':
    # Set multiprocessing start method for Windows
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    success = resume_training()
    
    if success:
        print("\n🎯 Training completed! Next steps:")
        print("1. Check logs/ directory for training metrics")
        print("2. Run: python test_model.py to evaluate the model")
        print("3. Run: streamlit run streamlit_app.py for web interface")
    else:
        print("\n❌ Training failed. Check logs for details.")