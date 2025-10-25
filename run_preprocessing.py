"""
Enhanced CASIA Dataset Preprocessing with Full Terminal Logging
Captures all preprocessing output and provides comprehensive analysis
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import logging
import time


class PreprocessingLogger:
    """
    Captures and logs all preprocessing terminal output
    """
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        # Create log file
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        
        # Redirect stdout and stderr
        sys.stdout = TeeOutput(self.original_stdout, self.log_handle)
        sys.stderr = TeeOutput(self.original_stderr, self.log_handle)
        
        print(f"=== PREPROCESSING LOG STARTED: {datetime.now()} ===")
        print(f"Log file: {self.log_file}")
        print("=" * 80)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("=" * 80)
        print(f"=== PREPROCESSING LOG ENDED: {datetime.now()} ===")
        
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Close log file
        self.log_handle.close()


class TeeOutput:
    """
    Writes output to both original stream and log file
    """
    
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file
    
    def write(self, message):
        # Write to original stream (console)
        self.original.write(message)
        self.original.flush()
        
        # Write to log file
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.original.flush()
        self.log_file.flush()


def check_prerequisites():
    """Check if CASIA dataset exists"""
    casia_path = Path("dataset/CASIA2")
    
    if not casia_path.exists():
        print("❌ CASIA dataset not found!")
        print(f"Expected path: {casia_path.absolute()}")
        print("\nPlease ensure CASIA v2.0 is extracted to dataset/CASIA2/")
        print("Expected structure:")
        print("dataset/CASIA2/")
        print("├── Au/                    # Authentic images")
        print("├── Tp/                    # Tampered images")  
        print("└── CASIA 2 Groundtruth/  # Ground truth masks")
        return False
    
    # Check subdirectories
    au_path = casia_path / "Au"
    tp_path = casia_path / "Tp"
    gt_path = casia_path / "CASIA 2 Groundtruth"
    
    missing_dirs = []
    if not au_path.exists():
        missing_dirs.append("Au")
    if not tp_path.exists():
        missing_dirs.append("Tp")
    if not gt_path.exists():
        missing_dirs.append("CASIA 2 Groundtruth")
    
    if missing_dirs:
        print(f"❌ Missing subdirectories: {', '.join(missing_dirs)}")
        return False
    
    # Count images
    au_images = list(au_path.glob("*.jpg")) + list(au_path.glob("*.tif"))
    tp_images = list(tp_path.glob("*.jpg")) + list(tp_path.glob("*.tif"))
    
    print(f"✅ CASIA dataset found:")
    print(f"   Authentic images: {len(au_images)}")
    print(f"   Tampered images: {len(tp_images)}")
    
    return True


def run_preprocessing_with_logging(args):
    """
    Run preprocessing with comprehensive logging
    """
    # Create experiment directory
    experiment_name = args.experiment_name or f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(args.log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create terminal log file
    terminal_log = log_dir / f"{experiment_name}_terminal.log"
    
    print("="*80)
    print("CASIA DATASET PREPROCESSING - FULL TERMINAL LOGGING")
    print("="*80)
    print(f"Experiment: {experiment_name}")
    print(f"Log directory: {log_dir}")
    print(f"Terminal output will be saved to: {terminal_log}")
    print("="*80)
    
    # Check prerequisites first
    if not check_prerequisites():
        return False
    
    # Capture all terminal output
    with PreprocessingLogger(terminal_log):
        print(f"Starting CASIA dataset preprocessing...")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            # Import and run preprocessing directly
            from process_casia import process_casia_dataset, verify_dataset
            
            # Run preprocessing with logging
            print("🔄 Starting dataset processing...")
            success, processing_log_path = process_casia_dataset(
                log_dir=str(log_dir.parent), 
                experiment_name=experiment_name
            )
            
            if success:
                print("\n🔍 Verifying processed dataset...")
                verify_success = verify_dataset()
                
                processing_time = time.time() - start_time
                
                print("\n" + "="*80)
                print("PREPROCESSING COMPLETED SUCCESSFULLY!")
                print("="*80)
                print(f"⏱️ Total processing time: {processing_time:.2f} seconds")
                print(f"📊 Processing logs: {processing_log_path}")
                print(f"📊 Terminal log: {terminal_log}")
                print("="*80)
                
                return True and verify_success
            else:
                print("❌ Preprocessing failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False


def generate_preprocessing_report(log_dir, experiment_name):
    """Generate comprehensive preprocessing report"""
    
    experiment_path = Path(log_dir) / experiment_name
    
    if not experiment_path.exists():
        print(f"❌ Experiment directory not found: {experiment_path}")
        return
    
    # Find log files
    stats_csv = experiment_path / f"{experiment_name}_preprocessing_stats.csv"
    summary_json = experiment_path / "processing_summary.json"
    terminal_log = experiment_path / f"{experiment_name}_terminal.log"
    
    print("="*80)
    print(f"PREPROCESSING REPORT: {experiment_name}")
    print("="*80)
    
    # Basic info
    print("📁 Log Files:")
    if terminal_log.exists():
        print(f"   Terminal log: {terminal_log}")
    if stats_csv.exists():
        print(f"   Statistics CSV: {stats_csv}")
    if summary_json.exists():
        print(f"   Summary JSON: {summary_json}")
    
    # Parse summary if available
    if summary_json.exists():
        import json
        with open(summary_json, 'r') as f:
            summary = json.load(f)
        
        print("\n📊 Processing Summary:")
        print(f"   Total processing time: {summary['total_processing_time']:.2f} seconds")
        print(f"   Total images: {summary['total_images']}")
        print(f"   Total masks: {summary['total_masks']}")
        print(f"   Train split: {summary['splits']['train']} images")
        print(f"   Val split: {summary['splits']['val']} images")
        print(f"   Test split: {summary['splits']['test']} images")
    
    # Parse statistics if available
    if stats_csv.exists():
        try:
            import pandas as pd
            
            # Check if header exists, if not add it
            with open(stats_csv, 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('timestamp,'):
                df = pd.read_csv(stats_csv)
            else:
                # Add header manually
                columns = ['timestamp', 'operation', 'file_name', 'file_size_kb', 'processing_time_ms', 'success', 'error_message']
                df = pd.read_csv(stats_csv, names=columns)
            
            print("\n📈 Processing Statistics:")
            print(f"   Total file operations: {len(df)}")
            
            # Success/failure rates
            success_rate = (df['success'] == True).sum() / len(df) * 100
            print(f"   Success rate: {success_rate:.1f}%")
            
            # Average processing times by operation
            print("\n   Average processing times (ms):")
            for operation in df['operation'].unique():
                op_df = df[df['operation'] == operation]
                avg_time = op_df['processing_time_ms'].mean()
                print(f"     {operation}: {avg_time:.2f} ms")
            
            # File size analysis
            print(f"\n   File sizes:")
            print(f"     Average: {df['file_size_kb'].mean():.1f} KB")
            print(f"     Min: {df['file_size_kb'].min():.1f} KB")
            print(f"     Max: {df['file_size_kb'].max():.1f} KB")
            
        except ImportError:
            print("\n📈 Install pandas to view detailed statistics")
        except Exception as e:
            print(f"\n❌ Error parsing statistics: {e}")
    
    print("="*80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced CASIA Preprocessing with Logging')
    
    parser.add_argument('--action', type=str, default='preprocess',
                       choices=['preprocess', 'report'],
                       help='Action to perform')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    if args.action == 'preprocess':
        success = run_preprocessing_with_logging(args)
        
        if success:
            print("\n✅ Preprocessing completed successfully!")
            print("\n🚀 Next steps:")
            print("   1. Start training: python run_training_with_logs.py --epochs 25")
            print("   2. View preprocessing report: python run_preprocessing.py --action report")
        else:
            print("\n❌ Preprocessing failed!")
        
        return success
    
    elif args.action == 'report':
        if not args.experiment_name:
            # Find latest experiment
            log_path = Path(args.log_dir)
            if log_path.exists():
                experiments = [d for d in log_path.iterdir() 
                             if d.is_dir() and 'preprocessing' in d.name]
                if experiments:
                    # Sort by modification time, get latest
                    latest = max(experiments, key=lambda x: x.stat().st_mtime)
                    args.experiment_name = latest.name
                    print(f"Using latest experiment: {args.experiment_name}")
                else:
                    print("❌ No preprocessing experiments found!")
                    return False
            else:
                print("❌ Log directory not found!")
                return False
        
        generate_preprocessing_report(args.log_dir, args.experiment_name)
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)