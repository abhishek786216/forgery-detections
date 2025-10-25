"""
Training Progress Monitor
Real-time monitoring of GPU-optimized training
"""

import time
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def get_latest_experiment():
    """Get the most recent experiment"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        return None
        
    experiments = [f for f in os.listdir(log_dir) if f.startswith('gpu_optimized_')]
    if not experiments:
        return None
        
    # Get the most recent experiment
    experiments.sort()
    return experiments[-1]


def monitor_training_progress():
    """Monitor training progress in real-time"""
    print("🔍 TRAINING PROGRESS MONITOR")
    print("=" * 60)
    
    experiment = get_latest_experiment()
    if not experiment:
        print("❌ No GPU-optimized experiments found")
        return
        
    print(f"📊 Monitoring experiment: {experiment}")
    
    metrics_file = f"logs/{experiment}_metrics.csv"
    log_file = f"logs/{experiment}_training.log"
    
    last_size = 0
    last_epoch = 0
    
    while True:
        try:
            # Check if metrics file exists and has new data
            if os.path.exists(metrics_file):
                current_size = os.path.getsize(metrics_file)
                
                if current_size > last_size:
                    # Read metrics
                    try:
                        df = pd.read_csv(metrics_file)
                        if len(df) > 0:
                            latest = df.iloc[-1]
                            
                            if latest['epoch'] > last_epoch:
                                print(f"\n📈 Epoch {int(latest['epoch'])}/75:")
                                print(f"   Train Loss: {latest['train_loss']:.4f}")
                                print(f"   Val Loss: {latest['val_loss']:.4f}" if 'val_loss' in df.columns else "")
                                print(f"   Train IoU: {latest['train_iou']:.4f}")
                                print(f"   Val IoU: {latest['val_iou']:.4f}" if 'val_iou' in df.columns else "")
                                print(f"   Train F1: {latest['train_f1_score']:.4f}")
                                print(f"   Val F1: {latest['val_f1_score']:.4f}" if 'val_f1_score' in df.columns else "")
                                
                                last_epoch = latest['epoch']
                                
                                # Show progress
                                progress = (latest['epoch'] / 75) * 100
                                print(f"   Progress: {progress:.1f}%")
                                
                    except Exception as e:
                        pass
                    
                    last_size = current_size
            
            # Check training log for errors
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:  # Check last 10 lines
                        if 'ERROR' in line or 'Exception' in line:
                            print(f"⚠️  {line.strip()}")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)


def show_training_summary():
    """Show training summary and plots"""
    experiment = get_latest_experiment()
    if not experiment:
        print("❌ No experiments found")
        return
        
    metrics_file = f"logs/{experiment}_metrics.csv"
    
    if not os.path.exists(metrics_file):
        print("❌ No metrics file found")
        return
        
    # Read and display metrics
    df = pd.read_csv(metrics_file)
    
    print(f"\n📊 TRAINING SUMMARY: {experiment}")
    print("=" * 60)
    
    if len(df) > 0:
        latest = df.iloc[-1]
        print(f"Current Epoch: {int(latest['epoch'])}/75")
        print(f"Best Train IoU: {df['train_iou'].max():.4f}")
        print(f"Best Val IoU: {df['val_iou'].max():.4f}" if 'val_iou' in df.columns else "")
        print(f"Current Train Loss: {latest['train_loss']:.4f}")
        print(f"Current Val Loss: {latest['val_loss']:.4f}" if 'val_loss' in df.columns else "")
        
        # Create training plots
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 3, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        if 'val_loss' in df.columns:
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # IoU plot
        plt.subplot(2, 3, 2)
        plt.plot(df['epoch'], df['train_iou'], label='Train IoU')
        if 'val_iou' in df.columns:
            plt.plot(df['epoch'], df['val_iou'], label='Val IoU')
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        # F1 Score plot
        plt.subplot(2, 3, 3)
        plt.plot(df['epoch'], df['train_f1_score'], label='Train F1')
        if 'val_f1_score' in df.columns:
            plt.plot(df['epoch'], df['val_f1_score'], label='Val F1')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'logs/{experiment}_training_plots.png', dpi=300, bbox_inches='tight')
        print(f"\n📈 Training plots saved to: logs/{experiment}_training_plots.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        show_training_summary()
    else:
        monitor_training_progress()