"""
Project Cleanup Script
Removes unnecessary files and folders while keeping essential components
"""

import os
import shutil
from pathlib import Path


def print_banner():
    """Print cleanup banner"""
    print("🧹 PROJECT CLEANUP - REMOVING UNNECESSARY FILES")
    print("=" * 60)


def get_files_to_remove():
    """Get list of unnecessary files and folders to remove"""
    
    # Files that are duplicates or outdated
    unnecessary_files = [
        # Duplicate training scripts
        "train_enhanced.py",  # Superseded by train_optimized.py
        "run_training_with_logs.py",  # Superseded by start_gpu_training.py
        "run_pipeline.py",  # Superseded by run_complete_pipeline.py
        
        # Old configuration files
        "config.py",  # Basic config, superseded by enhanced versions
        "app.py",  # Old app version, superseded by streamlit_app.py
        
        # Shell scripts (keeping only Python versions)
        "run_streamlit.bat",
        "run_streamlit.sh",
        
        # Test/demo files that are no longer needed
        "create_demo.py",  # Demo creation, no longer needed
        "test_model.py",   # Basic test, superseded by verify_pipeline.py
        "predict.py",      # Basic prediction, integrated into streamlit_app.py
        "setup_dataset.py", # Superseded by process_casia.py
        
        # Personal files
        "personl.txt",
        
        # Cache directories
        "__pycache__",
    ]
    
    # Empty directories to remove
    empty_dirs = [
        "results",  # Empty results folder
        "checkpoints" if not list(Path("checkpoints").glob("*")) else None,
    ]
    
    # Remove None values
    empty_dirs = [d for d in empty_dirs if d is not None]
    
    return unnecessary_files, empty_dirs


def get_logs_to_clean():
    """Get old log directories to clean up"""
    log_dirs_to_remove = []
    
    logs_path = Path("logs")
    if logs_path.exists():
        for item in logs_path.iterdir():
            if item.is_dir():
                # Keep only the most recent comprehensive logs
                if item.name in ["casia_with_logging"]:
                    continue  # Keep this one
                elif item.name.startswith("run_") or item.name in ["casia_end_to_end_training", "casia_full_training"]:
                    log_dirs_to_remove.append(f"logs/{item.name}")
    
    return log_dirs_to_remove


def confirm_deletion(items, item_type="files"):
    """Confirm deletion with user"""
    if not items:
        return False
    
    print(f"\n📋 {item_type.title()} to remove:")
    for item in items:
        if Path(item).exists():
            if Path(item).is_file():
                size = Path(item).stat().st_size / 1024
                print(f"   📄 {item} ({size:.1f} KB)")
            else:
                print(f"   📁 {item}/")
        else:
            print(f"   ⚠️  {item} (not found)")
    
    choice = input(f"\n🗑️  Remove these {item_type}? (y/n): ").strip().lower()
    return choice in ['y', 'yes']


def remove_files(files):
    """Remove specified files"""
    removed_count = 0
    total_size = 0
    
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            try:
                if path.is_file():
                    size = path.stat().st_size
                    path.unlink()
                    total_size += size
                    print(f"   ✅ Removed: {file_path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    print(f"   ✅ Removed: {file_path}/")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"   ⚠️  Not found: {file_path}")
    
    if total_size > 0:
        print(f"\n💾 Freed up: {total_size / 1024 / 1024:.2f} MB")
    
    return removed_count


def organize_remaining_files():
    """Organize remaining files into logical groups"""
    print("\n📁 ORGANIZING REMAINING FILES:")
    print("-" * 30)
    
    # Core training files
    training_files = [
        "train_optimized.py",
        "start_gpu_training.py", 
        "launch_optimized_training.py",
        "run_complete_pipeline.py"
    ]
    
    # Data processing files  
    data_files = [
        "process_casia.py",
        "run_preprocessing.py",
        "dataset.py"
    ]
    
    # Interface files
    interface_files = [
        "streamlit_app.py",
        "monitor_and_launch.py"
    ]
    
    # Utility files
    utility_files = [
        "system_status.py",
        "verify_pipeline.py",
        "log_analyzer.py",
        "setup_gpu.py"
    ]
    
    # Documentation
    doc_files = [
        "README.md",
        "RUN_PIPELINE.md",
        "STREAMLIT_SUCCESS.md",
        "requirements.txt"
    ]
    
    categories = [
        ("🚀 Training & Optimization", training_files),
        ("📊 Data Processing", data_files),
        ("🌐 User Interface", interface_files),
        ("🔧 Utilities & Analysis", utility_files),
        ("📚 Documentation", doc_files)
    ]
    
    for category, files in categories:
        print(f"\n{category}:")
        for file in files:
            if Path(file).exists():
                size = Path(file).stat().st_size / 1024
                print(f"   ✅ {file} ({size:.1f} KB)")
        
    # Show directories
    print(f"\n📁 Directories:")
    dirs = ["models/", "utils/", "dataset/", "logs/", "demo_images/"]
    for dir_name in dirs:
        if Path(dir_name).exists():
            file_count = len(list(Path(dir_name).rglob("*")))
            print(f"   ✅ {dir_name} ({file_count} items)")


def create_project_structure_summary():
    """Create a summary of the cleaned project structure"""
    summary = """
# 🎯 IMAGE FORGERY DETECTION - CLEAN PROJECT STRUCTURE

## 🚀 Core Training Files
- `train_optimized.py` - GPU-optimized training with pre-trained ResNet50
- `start_gpu_training.py` - Simple GPU training launcher
- `launch_optimized_training.py` - Advanced training configuration
- `run_complete_pipeline.py` - Full end-to-end automation

## 📊 Data Processing
- `process_casia.py` - CASIA dataset processing with logging
- `run_preprocessing.py` - Preprocessing with terminal logging
- `dataset.py` - Dataset class imports

## 🌐 User Interface
- `streamlit_app.py` - Main web interface
- `monitor_and_launch.py` - Training monitor and auto-launcher

## 🔧 Utilities
- `system_status.py` - Complete system overview
- `verify_pipeline.py` - End-to-end verification
- `log_analyzer.py` - Training log analysis
- `setup_gpu.py` - GPU setup helper

## 📁 Directories
- `models/` - Model architectures (enhanced_model.py, noise_residual_cnn.py)
- `utils/` - Utilities (logging, metrics, preprocessing, dataset)
- `dataset/` - Processed training data
- `logs/` - Training logs and metrics
- `demo_images/` - Sample images for testing

## 🚀 Quick Commands
```bash
# Start GPU training
python start_gpu_training.py

# Complete pipeline 
python run_complete_pipeline.py

# Check status
python system_status.py

# Launch interface
streamlit run streamlit_app.py
```

## 📈 Performance
- Pre-trained ResNet50 backbone
- GPU acceleration (5-10x faster)
- Mixed precision training
- Advanced loss functions
- Real-time web interface
"""
    
    with open("PROJECT_STRUCTURE.md", "w") as f:
        f.write(summary.strip())
    
    print(f"\n📝 Project structure saved: PROJECT_STRUCTURE.md")


def main():
    """Main cleanup function"""
    print_banner()
    
    # Get files to remove
    unnecessary_files, empty_dirs = get_files_to_remove()
    log_dirs = get_logs_to_clean()
    
    all_items_to_remove = []
    
    # Process unnecessary files
    if unnecessary_files:
        existing_files = [f for f in unnecessary_files if Path(f).exists()]
        if existing_files and confirm_deletion(existing_files, "unnecessary files"):
            all_items_to_remove.extend(existing_files)
    
    # Process empty directories
    if empty_dirs:
        existing_dirs = [d for d in empty_dirs if Path(d).exists()]
        if existing_dirs and confirm_deletion(existing_dirs, "empty directories"):
            all_items_to_remove.extend(existing_dirs)
    
    # Process old logs
    if log_dirs:
        existing_log_dirs = [d for d in log_dirs if Path(d).exists()]
        if existing_log_dirs and confirm_deletion(existing_log_dirs, "old log directories"):
            all_items_to_remove.extend(existing_log_dirs)
    
    # Remove confirmed items
    if all_items_to_remove:
        print(f"\n🗑️  REMOVING FILES...")
        print("-" * 30)
        removed_count = remove_files(all_items_to_remove)
        print(f"\n✅ Successfully removed {removed_count} items")
    else:
        print(f"\n✅ No files to remove - project is already clean!")
    
    # Organize remaining files
    organize_remaining_files()
    
    # Create project structure summary
    create_project_structure_summary()
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"=" * 60)
    print(f"✅ Project is now clean and organized")
    print(f"🚀 Ready for optimized GPU training")
    print(f"📚 See PROJECT_STRUCTURE.md for overview")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n🛑 Cleanup cancelled by user")
    except Exception as e:
        print(f"\n❌ Cleanup error: {e}")
        import traceback
        traceback.print_exc()