"""
Image Forgery Detection - Complete Pipeline Status
Shows comprehensive status of the entire end-to-end system
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch


def print_header():
    """Print system header"""
    print("🎯 IMAGE FORGERY DETECTION SYSTEM")
    print("=" * 60)
    print("🔬 Complete End-to-End Pipeline Status")
    print("=" * 60)


def show_dataset_status():
    """Show dataset processing status"""
    print("\n📊 DATASET STATUS")
    print("-" * 30)
    
    # Dataset directories and counts
    dataset_info = {
        "Training Images": ("dataset/train/images", "*.jpg"),
        "Training Masks": ("dataset/train/masks", "*.png"), 
        "Validation Images": ("dataset/val/images", "*.jpg"),
        "Validation Masks": ("dataset/val/masks", "*.png"),
        "Test Images": ("dataset/test/images", "*.jpg"),
        "Test Masks": ("dataset/test/masks", "*.png")
    }
    
    total_images = 0
    total_masks = 0
    
    for name, (path_str, pattern) in dataset_info.items():
        path = Path(path_str)
        if path.exists():
            count = len(list(path.glob(pattern)))
            print(f"  ✅ {name:<20}: {count:>6,} files")
            
            if "Images" in name:
                total_images += count
            else:
                total_masks += count
        else:
            print(f"  ❌ {name:<20}: Not found")
    
    print(f"\n  📈 Total Images: {total_images:,}")
    print(f"  🎨 Total Masks:  {total_masks:,}")
    
    # Show preprocessing logs
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.rglob("*preprocessing*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
            print(f"  📝 Latest Log: {latest_log.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")


def show_model_status():
    """Show model training status"""
    print("\n🤖 MODEL STATUS")
    print("-" * 30)
    
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("  ❌ No checkpoints directory found")
        return
    
    # Find model files
    model_files = list(checkpoint_dir.glob("*.pth"))
    if not model_files:
        print("  ❌ No trained models found")
        return
    
    print(f"  ✅ Found {len(model_files)} model file(s):")
    
    for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime):
        size_mb = model_file.stat().st_size / 1024 / 1024
        mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
        print(f"    📦 {model_file.name}")
        print(f"       Size: {size_mb:.1f} MB | Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Test model loading
    try:
        from models import get_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  🖥️  Device: {device}")
        
        model = get_model().to(device)
        
        # Load latest model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        checkpoint = torch.load(latest_model, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"  🎯 Loaded: Epoch {checkpoint['epoch']}")
            if 'train_loss' in checkpoint:
                print(f"  📊 Train Loss: {checkpoint['train_loss']:.6f}")
            if 'val_loss' in checkpoint:
                print(f"  📊 Val Loss: {checkpoint['val_loss']:.6f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"  ✅ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  🔢 Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        model.eval()
        
        # Test inference
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            output = model(dummy_input)
            print(f"  ✅ Inference Test: Input {dummy_input.shape} → Output {output.shape}")
        
    except Exception as e:
        print(f"  ❌ Model Error: {e}")


def show_training_logs():
    """Show training log summary"""
    print("\n📈 TRAINING LOGS")
    print("-" * 30)
    
    log_dir = Path("logs")
    if not log_dir.exists():
        print("  ❌ No logs directory found")
        return
    
    # Find training logs
    training_logs = list(log_dir.rglob("*training*.log"))
    csv_files = list(log_dir.rglob("*metrics*.csv"))
    
    if training_logs:
        print(f"  ✅ Training Logs: {len(training_logs)} files")
        latest_log = max(training_logs, key=lambda x: x.stat().st_mtime)
        mod_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
        print(f"    📝 Latest: {latest_log.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    if csv_files:
        print(f"  ✅ Metrics Files: {len(csv_files)} files")
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"    📊 Latest: {latest_csv.name}")


def show_streamlit_status():
    """Show Streamlit app status"""
    print("\n🌐 STREAMLIT STATUS")
    print("-" * 30)
    
    streamlit_file = Path("streamlit_app.py")
    if streamlit_file.exists():
        size_kb = streamlit_file.stat().st_size / 1024
        print(f"  ✅ App File: {streamlit_file.name} ({size_kb:.1f} KB)")
    else:
        print("  ❌ streamlit_app.py not found")
        return
    
    # Check required packages
    required_packages = ['streamlit', 'torch', 'torchvision', 'PIL', 'cv2', 'numpy']
    available_packages = []
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                from PIL import Image
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            available_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    print(f"  ✅ Available Packages: {', '.join(available_packages)}")
    if missing_packages:
        print(f"  ❌ Missing Packages: {', '.join(missing_packages)}")
    
    print(f"\n  🚀 Launch Command: streamlit run streamlit_app.py")
    print(f"  🌐 URL: http://localhost:8501")


def show_quick_commands():
    """Show quick command reference"""
    print("\n🚀 QUICK COMMANDS")
    print("-" * 30)
    
    commands = [
        ("Start Preprocessing", "python process_casia.py"),
        ("Train Model", "python train_model.py --epochs 25 --batch_size 8 --train_images_dir dataset/train/images --train_masks_dir dataset/train/masks --val_images_dir dataset/val/images --val_masks_dir dataset/val/masks"),
        ("Monitor & Launch", "python monitor_and_launch.py"),
        ("Run Streamlit", "streamlit run streamlit_app.py"),
        ("Verify Pipeline", "python verify_pipeline.py"),
        ("Analyze Logs", "python log_analyzer.py")
    ]
    
    for i, (description, command) in enumerate(commands, 1):
        print(f"  {i}. {description}")
        print(f"     {command}")
        print()


def main():
    """Main status function"""
    print_header()
    
    show_dataset_status()
    show_model_status()
    show_training_logs()
    show_streamlit_status()
    show_quick_commands()
    
    # Final summary
    print("🎉 SYSTEM STATUS SUMMARY")
    print("=" * 60)
    
    # Check all components
    dataset_ok = Path("dataset/train/images").exists() and len(list(Path("dataset/train/images").glob("*.jpg"))) > 0
    model_ok = Path("checkpoints").exists() and len(list(Path("checkpoints").glob("*.pth"))) > 0
    streamlit_ok = Path("streamlit_app.py").exists()
    
    status_items = [
        ("📊 Dataset Processed", dataset_ok),
        ("🤖 Model Trained", model_ok),
        ("🌐 Streamlit Ready", streamlit_ok)
    ]
    
    all_ok = True
    for item, status in status_items:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {item}")
        if not status:
            all_ok = False
    
    if all_ok:
        print(f"\n🎉 ALL SYSTEMS OPERATIONAL!")
        print(f"🚀 Ready for image forgery detection!")
    else:
        print(f"\n⚠️  Some components need attention")
    
    print("=" * 60)


if __name__ == "__main__":
    main()