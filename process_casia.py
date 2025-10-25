"""
CASIA Dataset Processor for Image Forgery Detection with Enhanced Logging
Processes CASIA v2.0 dataset and organizes it for training with comprehensive logging
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import random
import logging
import sys
import time
from datetime import datetime
import json

def setup_preprocessing_logging(log_dir="logs", experiment_name=None):
    """
    Setup comprehensive logging for preprocessing
    """
    if not experiment_name:
        experiment_name = f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create log directory
    log_path = Path(log_dir) / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Setup main logger
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = log_path / f"{experiment_name}_preprocessing.log"
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
    
    # Stats logger for CSV output
    stats_logger = logging.getLogger('preprocessing_stats')
    stats_logger.setLevel(logging.INFO)
    stats_logger.handlers.clear()
    
    stats_file = log_path / f"{experiment_name}_preprocessing_stats.csv"
    stats_handler = logging.FileHandler(stats_file, mode='w')
    csv_formatter = logging.Formatter('%(message)s')
    stats_handler.setFormatter(csv_formatter)
    stats_logger.addHandler(stats_handler)
    
    # Write CSV header
    stats_logger.info('timestamp,operation,file_name,file_size_kb,processing_time_ms,success,error_message')
    
    logger.info(f"Preprocessing logging initialized: {experiment_name}")
    logger.info(f"Main log: {log_file}")
    logger.info(f"Stats log: {stats_file}")
    
    return logger, stats_logger, log_path


def log_file_operation(stats_logger, operation, file_name, file_size_kb, processing_time_ms, success, error_msg=""):
    """Log individual file operation statistics"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stats_logger.info(f"{timestamp},{operation},{file_name},{file_size_kb:.2f},{processing_time_ms:.2f},{success},{error_msg}")


def process_casia_dataset(log_dir="logs", experiment_name=None):
    """Process CASIA v2.0 dataset for training with comprehensive logging"""
    
    # Setup logging
    logger, stats_logger, log_path = setup_preprocessing_logging(log_dir, experiment_name)
    
    logger.info("🔄 Processing CASIA v2.0 Dataset with Enhanced Logging")
    logger.info("=" * 80)
    
    # Log system information
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"PIL version: {Image.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info("=" * 50)
    
    # Dataset paths
    casia_root = Path("dataset/CASIA2")
    au_path = casia_root / "Au"
    tp_path = casia_root / "Tp"
    gt_path = casia_root / "CASIA 2 Groundtruth"
    
    logger.info("=== DATASET PATHS ===")
    logger.info(f"CASIA root: {casia_root}")
    logger.info(f"Authentic images: {au_path}")
    logger.info(f"Tampered images: {tp_path}")
    logger.info(f"Ground truth: {gt_path}")
    logger.info("=" * 50)
    
    # Output paths
    train_img_path = Path("dataset/train/images")
    train_mask_path = Path("dataset/train/masks")
    val_img_path = Path("dataset/val/images")
    val_mask_path = Path("dataset/val/masks")
    test_img_path = Path("dataset/test/images")
    test_mask_path = Path("dataset/test/masks")
    
    logger.info("=== OUTPUT PATHS ===")
    for path in [train_img_path, train_mask_path, val_img_path, val_mask_path, test_img_path, test_mask_path]:
        logger.info(f"Creating: {path}")
    logger.info("=" * 50)
    
    # Create output directories
    for path in [train_img_path, train_mask_path, val_img_path, val_mask_path, test_img_path, test_mask_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Check if paths exist
    if not au_path.exists():
        error_msg = f"❌ Authentic images path not found: {au_path}"
        logger.error(error_msg)
        return False, log_path
    
    if not tp_path.exists():
        error_msg = f"❌ Tampered images path not found: {tp_path}"
        logger.error(error_msg)
        return False, log_path
    
    # Get authentic images with logging
    logger.info("🔍 Scanning for images...")
    start_time = time.time()
    
    au_images = list(au_path.glob("*.jpg")) + list(au_path.glob("*.tif"))
    logger.info(f"📁 Found {len(au_images)} authentic images")
    
    # Get tampered images
    tp_images = list(tp_path.glob("*.jpg")) + list(tp_path.glob("*.tif"))
    logger.info(f"📁 Found {len(tp_images)} tampered images")
    
    scan_time = time.time() - start_time
    logger.info(f"⏱️ Image scanning completed in {scan_time:.2f} seconds")
    
    if len(au_images) == 0:
        error_msg = "❌ No authentic images found!"
        logger.error(error_msg)
        return False, log_path
    
    if len(tp_images) == 0:
        error_msg = "❌ No tampered images found!"
        logger.error(error_msg)
        return False, log_path
    
    # Split ratios (70% train, 15% validation, 15% test)
    def split_dataset(image_list, train_ratio=0.7, val_ratio=0.15):
        """Split dataset into train/val/test"""
        random.shuffle(image_list)
        n = len(image_list)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': image_list[:train_end],
            'val': image_list[train_end:val_end], 
            'test': image_list[val_end:]
        }
    
    # Split authentic and tampered images
    logger.info("📊 Splitting dataset...")
    split_start = time.time()
    
    au_splits = split_dataset(au_images)
    tp_splits = split_dataset(tp_images)
    
    split_time = time.time() - split_start
    logger.info(f"⏱️ Dataset splitting completed in {split_time:.2f} seconds")
    
    # Log split information
    logger.info("=== DATASET SPLITS ===")
    total_images = 0
    for split in ['train', 'val', 'test']:
        au_count = len(au_splits[split])
        tp_count = len(tp_splits[split])
        split_total = au_count + tp_count
        total_images += split_total
        logger.info(f"{split:>5}: {au_count:>4} authentic + {tp_count:>4} tampered = {split_total:>4} total")
    
    logger.info(f"Total images to process: {total_images}")
    logger.info("=" * 50)
    
    # Start overall processing timer
    overall_start = time.time()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        logger.info(f"🔄 Processing {split} set...")
        split_start_time = time.time()
        
        # Get output paths for this split
        if split == 'train':
            img_out = train_img_path
            mask_out = train_mask_path
        elif split == 'val':
            img_out = val_img_path
            mask_out = val_mask_path
        else:
            img_out = test_img_path
            mask_out = test_mask_path
        
        # Process authentic images
        logger.info(f"   Processing {len(au_splits[split])} authentic images...")
        au_success_count = 0
        au_error_count = 0
        
        for img_path in tqdm(au_splits[split], desc=f"Authentic ({split})"):
            start_time = time.time()
            file_size_kb = img_path.stat().st_size / 1024
            
            try:
                # Copy image
                dest_img = img_out / f"Au_{img_path.name}"
                shutil.copy2(img_path, dest_img)
                
                # Create black mask (no forgery)
                img = Image.open(img_path)
                mask = Image.new('L', img.size, 0)  # Black mask
                mask_dest = mask_out / f"Au_{img_path.stem}_mask.png"
                mask.save(mask_dest)
                
                processing_time = (time.time() - start_time) * 1000
                log_file_operation(stats_logger, f"authentic_{split}", img_path.name, 
                                 file_size_kb, processing_time, True)
                au_success_count += 1
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                error_msg = str(e).replace(',', ';')  # Remove commas for CSV
                log_file_operation(stats_logger, f"authentic_{split}", img_path.name, 
                                 file_size_kb, processing_time, False, error_msg)
                logger.error(f"Error processing {img_path.name}: {e}")
                au_error_count += 1
        
        logger.info(f"   Authentic images: {au_success_count} success, {au_error_count} errors")
        
        # Process tampered images
        logger.info(f"   Processing {len(tp_splits[split])} tampered images...")
        tp_success_count = 0
        tp_error_count = 0
        masks_found = 0
        masks_missing = 0
        
        for img_path in tqdm(tp_splits[split], desc=f"Tampered ({split})"):
            start_time = time.time()
            file_size_kb = img_path.stat().st_size / 1024
            
            try:
                # Copy image
                dest_img = img_out / f"Tp_{img_path.name}"
                shutil.copy2(img_path, dest_img)
                
                # Look for corresponding ground truth mask
                mask_found = False
                
                # Try different mask naming patterns
                possible_mask_names = [
                    f"{img_path.stem}_gt.png",
                    f"{img_path.stem}.png", 
                    f"{img_path.name}_gt.png",
                    f"{img_path.name}.png"
                ]
                
                for mask_name in possible_mask_names:
                    mask_path = gt_path / mask_name
                    if mask_path.exists():
                        mask_dest = mask_out / f"Tp_{img_path.stem}_mask.png"
                        
                        try:
                            # Load and process mask
                            mask = Image.open(mask_path).convert('L')
                            
                            # Ensure mask is binary (0 or 255)
                            mask_array = np.array(mask)
                            mask_array = (mask_array > 128).astype(np.uint8) * 255
                            mask_processed = Image.fromarray(mask_array)
                            
                            # Resize mask to match image size if needed
                            img = Image.open(img_path)
                            if mask_processed.size != img.size:
                                mask_processed = mask_processed.resize(img.size, Image.NEAREST)
                            
                            mask_processed.save(mask_dest)
                            mask_found = True
                            masks_found += 1
                            break
                            
                        except Exception as e:
                            logger.warning(f"Error processing mask {mask_name} for {img_path.name}: {e}")
                            continue
                
                # If no mask found, create white mask (assume entire image is forged)
                if not mask_found:
                    img = Image.open(img_path)
                    mask = Image.new('L', img.size, 255)  # White mask
                    mask_dest = mask_out / f"Tp_{img_path.stem}_mask.png"
                    mask.save(mask_dest)
                    masks_missing += 1
                    logger.warning(f"No ground truth mask found for {img_path.name}, using default white mask")
                
                processing_time = (time.time() - start_time) * 1000
                log_file_operation(stats_logger, f"tampered_{split}", img_path.name, 
                                 file_size_kb, processing_time, True)
                tp_success_count += 1
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                error_msg = str(e).replace(',', ';')  # Remove commas for CSV
                log_file_operation(stats_logger, f"tampered_{split}", img_path.name, 
                                 file_size_kb, processing_time, False, error_msg)
                logger.error(f"Error processing {img_path.name}: {e}")
                tp_error_count += 1
        
        logger.info(f"   Tampered images: {tp_success_count} success, {tp_error_count} errors")
        logger.info(f"   Ground truth masks: {masks_found} found, {masks_missing} missing")
        
        split_time = time.time() - split_start_time
        logger.info(f"   Split processing time: {split_time:.2f} seconds")
    
    # Calculate total processing time
    total_time = time.time() - overall_start
    
    logger.info("✅ CASIA dataset processing complete!")
    logger.info(f"⏱️ Total processing time: {total_time:.2f} seconds")
    
    # Calculate and log final statistics
    logger.info("=== FINAL DATASET STATISTICS ===")
    total_images = 0
    total_masks = 0
    
    for split in ['train', 'val', 'test']:
        if split == 'train':
            img_path = train_img_path
            mask_path = train_mask_path
        elif split == 'val':
            img_path = val_img_path
            mask_path = val_mask_path
        else:
            img_path = test_img_path
            mask_path = test_mask_path
        
        img_count = len(list(img_path.glob("*")))
        mask_count = len(list(mask_path.glob("*")))
        total_images += img_count
        total_masks += mask_count
        
        logger.info(f"   {split:>5}: {img_count:>4} images, {mask_count:>4} masks")
    
    logger.info(f"Total: {total_images} images, {total_masks} masks")
    logger.info("=" * 50)
    
    # Create processing summary
    summary = {
        'total_processing_time': total_time,
        'total_images': total_images,
        'total_masks': total_masks,
        'splits': {
            'train': len(au_splits['train']) + len(tp_splits['train']),
            'val': len(au_splits['val']) + len(tp_splits['val']),
            'test': len(au_splits['test']) + len(tp_splits['test'])
        },
        'completion_time': datetime.now().isoformat()
    }
    
    # Save summary to JSON
    summary_file = log_path / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processing summary saved to: {summary_file}")
    
    return True, log_path

def verify_dataset():
    """Verify the processed dataset"""
    
    print("\n🔍 Verifying processed dataset...")
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        print(f"\n   Checking {split} set:")
        
        if split == 'train':
            img_path = Path("dataset/train/images")
            mask_path = Path("dataset/train/masks")
        elif split == 'val':
            img_path = Path("dataset/val/images")
            mask_path = Path("dataset/val/masks")
        else:
            img_path = Path("dataset/test/images")
            mask_path = Path("dataset/test/masks")
        
        # Get all images and masks
        images = list(img_path.glob("*"))
        masks = list(mask_path.glob("*"))
        
        print(f"      Images: {len(images)}")
        print(f"      Masks: {len(masks)}")
        
        # Check if each image has corresponding mask
        missing_masks = 0
        size_mismatches = 0
        
        for img_file in images:
            # Find corresponding mask
            mask_file = mask_path / f"{img_file.stem}_mask.png"
            
            if not mask_file.exists():
                missing_masks += 1
                continue
            
            # Check if sizes match
            try:
                img = Image.open(img_file)
                mask = Image.open(mask_file)
                
                if img.size != mask.size:
                    size_mismatches += 1
                    
            except Exception as e:
                issues.append(f"Error checking {img_file.name}: {e}")
        
        if missing_masks > 0:
            issues.append(f"{split}: {missing_masks} images missing masks")
        
        if size_mismatches > 0:
            issues.append(f"{split}: {size_mismatches} size mismatches")
        
        if missing_masks == 0 and size_mismatches == 0:
            print(f"      ✅ All good!")
    
    if issues:
        print(f"\n⚠️ Found {len(issues)} issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Dataset verification complete - no issues found!")
    
    return len(issues) == 0

def main():
    """Main function with enhanced logging"""
    
    print("="*80)
    print("CASIA DATASET PREPROCESSING WITH ENHANCED LOGGING")
    print("="*80)
    
    # Check if CASIA dataset exists
    casia_path = Path("dataset/CASIA2")
    
    if not casia_path.exists():
        print("❌ CASIA dataset not found!")
        print(f"Expected path: {casia_path.absolute()}")
        print("Please ensure CASIA v2.0 is extracted to dataset/CASIA2/")
        return False
    
    # Process the dataset with logging
    experiment_name = f"casia_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    success, log_path = process_casia_dataset(experiment_name=experiment_name)
    
    if success:
        # Verify the processed dataset
        verify_success = verify_dataset()
        
        print("\n🎉 CASIA dataset ready for training!")
        print(f"📊 Processing logs saved to: {log_path}")
        print("\n🚀 Next steps:")
        print("   1. Start training: python train_model.py --epochs 25")
        print("   2. Monitor progress: tensorboard --logdir logs")
        print("   3. Test model: python test_model.py --model_path checkpoints/best_model.pth")
        print("   4. Analyze preprocessing logs: python log_analyzer.py --action analyze --experiment preprocessing_logs")
        
        return True and verify_success
    
    return False
    
    return False

if __name__ == "__main__":
    main()