"""
End-to-End Pipeline Verification
Tests the complete pipeline from data processing to Streamlit interface
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
from datetime import datetime


def verify_preprocessing():
    """Verify preprocessing results"""
    print("🔍 Verifying Preprocessing...")
    
    # Check dataset structure
    dataset_dirs = {
        "train_images": Path("dataset/train/images"),
        "train_masks": Path("dataset/train/masks"),
        "val_images": Path("dataset/val/images"),
        "val_masks": Path("dataset/val/masks"),
        "test_images": Path("dataset/test/images"),
        "test_masks": Path("dataset/test/masks")
    }
    
    for name, path in dataset_dirs.items():
        if path.exists():
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            print(f"  ✅ {name}: {count} files")
        else:
            print(f"  ❌ {name}: Directory not found")
    
    # Check preprocessing logs
    log_files = list(Path("logs").rglob("*preprocessing*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"  ✅ Latest preprocessing log: {latest_log}")
    else:
        print("  ⚠️  No preprocessing logs found")


def verify_training():
    """Verify training results"""
    print("\n🤖 Verifying Training...")
    
    # Check checkpoints
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("  ❌ No checkpoints directory")
        return False
    
    # Find model files
    model_files = list(checkpoint_dir.glob("*.pth"))
    if not model_files:
        print("  ❌ No model files found")
        return False
    
    print(f"  ✅ Found {len(model_files)} model files:")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"    - {model_file.name}: {size_mb:.1f} MB")
    
    # Test model loading
    try:
        from models import get_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model().to(device)
        
        # Load latest checkpoint
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        checkpoint = torch.load(latest_model, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"  ✅ Model loaded from epoch {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)
            print(f"  ✅ Model loaded from {latest_model.name}")
        
        model.eval()
        
        # Test inference
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            output = model(dummy_input)
            print(f"  ✅ Model inference test passed: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False


def verify_streamlit():
    """Verify Streamlit app configuration"""
    print("\n🌐 Verifying Streamlit App...")
    
    # Check streamlit app exists
    streamlit_file = Path("streamlit_app.py")
    if not streamlit_file.exists():
        print("  ❌ streamlit_app.py not found")
        return False
    
    print(f"  ✅ Streamlit app found: {streamlit_file}")
    
    # Check if required packages are available
    required_packages = ['streamlit', 'torch', 'torchvision', 'PIL', 'cv2']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                from PIL import Image
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"  ✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} missing")
    
    if missing_packages:
        print(f"  ⚠️  Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n📊 Generating Test Report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "preprocessing": {
            "dataset_dirs": {},
            "log_files": []
        },
        "training": {
            "model_files": [],
            "model_loaded": False
        },
        "streamlit": {
            "app_exists": Path("streamlit_app.py").exists(),
            "packages_available": True
        }
    }
    
    # Dataset info
    dataset_dirs = {
        "train_images": Path("dataset/train/images"),
        "train_masks": Path("dataset/train/masks"),
        "val_images": Path("dataset/val/images"),
        "val_masks": Path("dataset/val/masks"),
        "test_images": Path("dataset/test/images"),
        "test_masks": Path("dataset/test/masks")
    }
    
    for name, path in dataset_dirs.items():
        if path.exists():
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            report["preprocessing"]["dataset_dirs"][name] = count
    
    # Model files info
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        model_files = list(checkpoint_dir.glob("*.pth"))
        for model_file in model_files:
            size_mb = model_file.stat().st_size / 1024 / 1024
            report["training"]["model_files"].append({
                "name": model_file.name,
                "size_mb": round(size_mb, 1)
            })
    
    # Save report
    report_file = Path("logs/end_to_end_verification.json")
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✅ Report saved: {report_file}")
    return report


def main():
    """Main verification function"""
    print("🎯 End-to-End Pipeline Verification")
    print("=" * 60)
    
    # Run verification steps
    verify_preprocessing()
    model_ok = verify_training()
    streamlit_ok = verify_streamlit()
    
    # Generate report
    report = generate_test_report()
    
    # Summary
    print("\n🎉 Verification Summary")
    print("=" * 30)
    
    preprocessing_ok = len(report["preprocessing"]["dataset_dirs"]) >= 4
    print(f"📊 Preprocessing: {'✅ PASS' if preprocessing_ok else '❌ FAIL'}")
    print(f"🤖 Training: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"🌐 Streamlit: {'✅ PASS' if streamlit_ok else '❌ FAIL'}")
    
    if preprocessing_ok and model_ok and streamlit_ok:
        print("\n🎉 ALL SYSTEMS GO!")
        print("🚀 Your end-to-end pipeline is ready!")
        print("\nNext steps:")
        print("1. Run: python monitor_and_launch.py")
        print("2. Open: http://localhost:8501")
        print("3. Upload images to test forgery detection!")
    else:
        print("\n⚠️  Some issues found. Check the details above.")
    
    return preprocessing_ok and model_ok and streamlit_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)