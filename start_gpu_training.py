"""
GPU-Optimized Training Launcher
Automatically detects GPU/CPU and launches optimized training
"""

import os
import sys
import subprocess
from pathlib import Path


def check_gpu_status():
    """Check GPU availability"""
    try:
        import torch
        
        print(f"🔧 PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 GPU DETECTED: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB") 
            print(f"⚡ CUDA Version: {torch.version.cuda}")
            
            return {
                'available': True,
                'name': gpu_name,
                'memory_gb': gpu_memory,
                'count': gpu_count
            }
        else:
            print("🖥️  GPU: Not available (using CPU)")
            return {'available': False}
            
    except ImportError:
        print("❌ PyTorch not installed")
        return {'available': False, 'error': 'pytorch_missing'}


def get_training_parameters(gpu_info):
    """Get optimal training parameters"""
    
    if gpu_info['available']:
        memory_gb = gpu_info['memory_gb']
        
        if memory_gb >= 12:
            # High-end GPU (RTX 4070 Ti+, RTX 3080+)
            return {
                'epochs': 100,
                'batch_size': 12,
                'img_size': 384,
                'learning_rate': 1e-4,
                'freeze_epochs': 15,
                'patience': 25,
                'num_workers': 0,  # Windows compatibility
                'profile': 'High-end GPU'
            }
        elif memory_gb >= 8:
            # Mid-range GPU (RTX 4060 Ti, RTX 3070)
            return {
                'epochs': 80,
                'batch_size': 8,
                'img_size': 320,
                'learning_rate': 1e-4,
                'freeze_epochs': 12,
                'patience': 20,
                'num_workers': 0,  # Windows compatibility
                'profile': 'Mid-range GPU'
            }
        elif memory_gb >= 6:
            # Entry-level GPU (RTX 4060, RTX 3060)
            return {
                'epochs': 60,
                'batch_size': 6,
                'img_size': 288,
                'learning_rate': 8e-5,
                'freeze_epochs': 10,
                'patience': 15,
                'num_workers': 0,  # Windows compatibility
                'profile': 'Entry-level GPU'
            }
        else:
            # Low-end GPU
            return {
                'epochs': 50,
                'batch_size': 4,
                'img_size': 256,
                'learning_rate': 8e-5,
                'freeze_epochs': 8,
                'patience': 12,
                'num_workers': 0,  # Windows compatibility
                'profile': 'Low-end GPU'
            }
    else:
        # CPU fallback
        cpu_count = os.cpu_count() or 4
        return {
            'epochs': 40,
            'batch_size': 2,
            'img_size': 256,
            'learning_rate': 5e-5,
            'freeze_epochs': 5,
            'patience': 10,
            'num_workers': 0,  # Windows compatibility
            'profile': f'CPU ({cpu_count} cores)'
        }


def launch_training():
    """Launch optimized training"""
    print("🎯 GPU-OPTIMIZED FORGERY DETECTION TRAINING")
    print("=" * 60)
    
    # Check GPU status
    gpu_info = check_gpu_status()
    
    # Get parameters
    params = get_training_parameters(gpu_info)
    
    print(f"\n📋 Training Configuration ({params['profile']}):")
    print(f"   🔢 Epochs: {params['epochs']}")
    print(f"   📦 Batch Size: {params['batch_size']}")
    print(f"   🖼️  Image Size: {params['img_size']}x{params['img_size']}")
    print(f"   ⚡ Learning Rate: {params['learning_rate']}")
    print(f"   🧊 Freeze Epochs: {params['freeze_epochs']}")
    print(f"   ⏳ Patience: {params['patience']}")
    print(f"   👥 Workers: {params['num_workers']}")
    
    # Estimate training time
    if gpu_info['available']:
        if gpu_info['memory_gb'] >= 12:
            time_estimate = "30-60 minutes"
        elif gpu_info['memory_gb'] >= 8:
            time_estimate = "45-90 minutes"
        else:
            time_estimate = "60-120 minutes"
    else:
        time_estimate = "2-4 hours"
    
    print(f"\n⏰ Estimated Training Time: {time_estimate}")
    
    # Build command
    cmd = [
        sys.executable, "train_optimized.py",
        "--train_images_dir", "dataset/train/images",
        "--train_masks_dir", "dataset/train/masks",
        "--val_images_dir", "dataset/val/images", 
        "--val_masks_dir", "dataset/val/masks",
        "--epochs", str(params['epochs']),
        "--batch_size", str(params['batch_size']),
        "--img_size", str(params['img_size']),
        "--learning_rate", str(params['learning_rate']),
        "--freeze_epochs", str(params['freeze_epochs']),
        "--patience", str(params['patience']),
        "--num_workers", str(params['num_workers']),
        "--experiment_name", f"gpu_optimized_{params['profile'].lower().replace(' ', '_').replace('-', '_')}"
    ]
    
    if gpu_info['available']:
        cmd.extend(["--use_mixed_precision", "--pin_memory"])
    
    print(f"\n🚀 Starting Training...")
    print(f"📊 Monitor progress in terminal")
    print(f"🛑 Press Ctrl+C to stop gracefully")
    print("=" * 60)
    
    try:
        # Start training
        subprocess.run(cmd, check=True)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"📦 Model saved in: checkpoints/")
        print(f"📊 Logs saved in: logs/")
        
        return True
        
    except subprocess.CalledProcessError:
        print(f"\n❌ Training failed - check logs for details")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
        return False


def main():
    """Main function"""
    try:
        success = launch_training()
        
        if success:
            print(f"\n🌐 Next: Launch Streamlit interface")
            print(f"   Command: streamlit run streamlit_app.py")
            
            choice = input(f"\nLaunch Streamlit now? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
                print(f"✅ Streamlit launched at: http://localhost:8501")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    main()