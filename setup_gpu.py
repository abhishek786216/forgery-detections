"""
GPU Setup Helper for Optimized Training
Helps install and configure GPU support for maximum training performance
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


def print_banner():
    """Print setup banner"""
    print(">> GPU SETUP HELPER FOR OPTIMIZED TRAINING")
    print("=" * 60)
    print(">> Enables CUDA acceleration for maximum performance")
    print(">> Pre-trained ResNet50 + GPU = 10x faster training")
    print("=" * 60)


def check_system_info():
    """Check system information"""
    print("\n[INFO] System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Architecture: {platform.architecture()[0]}")
    
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] NVIDIA GPU detected!")
            
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'Driver Version' in line:
                    print(f"   📊 {line.strip()}")
                elif 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        gpu_name = ' '.join([p for p in parts if any(c.isalpha() for c in p)][:3])
                        print(f"   🎮 GPU: {gpu_name}")
            
            return True
        else:
            print("[ERROR] No NVIDIA GPU detected")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[ERROR] NVIDIA drivers not installed or GPU not detected")
        return False


def check_current_pytorch():
    """Check current PyTorch installation"""
    print("\n[INFO] Current PyTorch Status:")
    
    try:
        import torch
        print(f"   [OK] PyTorch Version: {torch.__version__}")
        print(f"   🔧 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   [FAST] CUDA Version: {torch.version.cuda}")
            print(f"   🎮 GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   📊 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            return True
        else:
            print("   [WARNING]  CUDA not available - CPU-only PyTorch installed")
            return False
            
    except ImportError:
        print("   [ERROR] PyTorch not installed")
        return False


def get_cuda_installation_command():
    """Get the appropriate CUDA PyTorch installation command"""
    print("\n[PACKAGE] Recommended PyTorch Installation:")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Recommend CUDA 11.8 for best compatibility
    cuda_version = "cu118"
    
    commands = {
        'pip': f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version}",
        'conda': f"conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    }
    
    print(f"   🐍 Python {python_version} detected")
    print(f"   [TARGET] CUDA 11.8 recommended for best compatibility")
    print(f"\n   📋 Installation Options:")
    print(f"   1. Using pip (recommended):")
    print(f"      {commands['pip']}")
    print(f"\n   2. Using conda:")
    print(f"      {commands['conda']}")
    
    return commands


def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch"""
    print("\n[SETUP] Installing CUDA PyTorch...")
    
    commands = get_cuda_installation_command()
    
    choice = input("\n   Use pip (1) or conda (2)? [1]: ").strip()
    
    if choice == '2':
        cmd = commands['conda'].split()
    else:
        cmd = commands['pip'].split()
    
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("[OK] CUDA PyTorch installation completed!")
        
        # Verify installation
        print("\n[INFO] Verifying installation...")
        verify_result = subprocess.run([
            sys.executable, "-c", 
            "import torch; print(f'[OK] PyTorch {torch.__version__}'); print(f'[OK] CUDA Available: {torch.cuda.is_available()}')"
        ], capture_output=True, text=True)
        
        print(verify_result.stdout)
        
        if "CUDA Available: True" in verify_result.stdout:
            print("🎉 GPU acceleration is now ENABLED!")
            return True
        else:
            print("[WARNING]  GPU acceleration not detected - may need system restart")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Installation failed: {e}")
        return False


def check_additional_dependencies():
    """Check and install additional GPU-optimized dependencies"""
    print("\n[INFO] Checking Additional Dependencies...")
    
    gpu_packages = [
        ('cupy', 'GPU-accelerated NumPy'),
        ('tensorboard', 'Training visualization'),
        ('torchinfo', 'Model summary and analysis')
    ]
    
    missing_packages = []
    
    for package, description in gpu_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   [OK] {package}: {description}")
        except ImportError:
            print(f"   [WARNING]  {package}: {description} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[PACKAGE] Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        
        install_choice = input("\n   Install missing packages now? (y/n): ").strip().lower()
        if install_choice in ['y', 'yes']:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
                print("[OK] Additional packages installed!")
            except subprocess.CalledProcessError:
                print("[WARNING]  Some packages failed to install")


def create_gpu_optimized_config():
    """Create GPU-optimized training configuration"""
    config = {
        "device": "cuda",
        "mixed_precision": True,
        "cudnn_benchmark": True,
        "pin_memory": True,
        "non_blocking": True,
        "compile_model": False  # PyTorch 2.0+ feature
    }
    
    config_file = Path("config_gpu.json")
    
    import json
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📝 GPU config saved: {config_file}")
    return config_file


def main():
    """Main setup function"""
    print_banner()
    
    # Check system
    has_gpu = check_system_info()
    has_cuda_pytorch = check_current_pytorch()
    
    if not has_gpu:
        print("\n[ERROR] No NVIDIA GPU detected")
        print("💡 For GPU training, you need:")
        print("   1. NVIDIA GPU (GTX 1060+ or RTX series)")
        print("   2. NVIDIA drivers installed")
        print("   3. CUDA-compatible PyTorch")
        
        choice = input("\n🤔 Continue with CPU setup optimization? (y/n): ").strip().lower()
        if choice not in ['y', 'yes']:
            return False
    
    if has_gpu and not has_cuda_pytorch:
        print("\n[TARGET] GPU detected but CUDA PyTorch not installed")
        choice = input("   Install CUDA PyTorch now? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            if not install_cuda_pytorch():
                print("[ERROR] Failed to install CUDA PyTorch")
                return False
        else:
            print("[WARNING]  Continuing with CPU PyTorch")
    
    # Check additional dependencies
    check_additional_dependencies()
    
    # Create optimized config
    config_file = create_gpu_optimized_config()
    
    # Final summary
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 40)
    
    if has_gpu and (has_cuda_pytorch or torch.cuda.is_available()):
        print("[OK] GPU Training: ENABLED")
        print("[SETUP] Expected speedup: 5-10x faster")
        print("[FAST] Mixed precision: ENABLED")
        print("[TARGET] Ready for optimized training!")
    else:
        print("[GPU]  CPU Training: OPTIMIZED")
        print("📊 Performance: Good for smaller models")
        
    print(f"\n[SETUP] Next Steps:")
    print(f"   1. Run: python launch_optimized_training.py")
    print(f"   2. Enjoy faster training with pre-trained ResNet50!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Setup error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)