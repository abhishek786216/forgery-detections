"""
Training Monitor and Streamlit Launcher
Monitors training progress and automatically launches Streamlit when model is ready
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import torch
import webbrowser
from datetime import datetime


class TrainingMonitor:
    """
    Monitor training progress and launch Streamlit when ready
    """
    
    def __init__(self):
        self.model_path = Path("checkpoints/best_model.pth")
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.streamlit_launched = False
    
    def check_model_exists(self):
        """Check if trained model exists"""
        # Check for best model first
        if self.model_path.exists():
            return True
        
        # Check for any checkpoint files
        if self.checkpoint_dir.exists():
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
            if checkpoint_files:
                # Use the latest checkpoint
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                self.model_path = latest_checkpoint
                return True
        
        return False
    
    def check_training_in_progress(self):
        """Check if training is currently running"""
        # Look for recent log files
        if not self.log_dir.exists():
            return False
        
        recent_logs = []
        for log_file in self.log_dir.rglob("*_training.log"):
            if log_file.exists():
                # Check if file was modified in last 5 minutes
                mod_time = log_file.stat().st_mtime
                if time.time() - mod_time < 300:  # 5 minutes
                    recent_logs.append(log_file)
        
        return len(recent_logs) > 0
    
    def get_training_status(self):
        """Get current training status"""
        if self.check_model_exists():
            return "model_ready"
        elif self.check_training_in_progress():
            return "training"
        else:
            return "no_training"
    
    def launch_streamlit(self):
        """Launch Streamlit app"""
        if self.streamlit_launched:
            return
        
        print("🚀 Launching Streamlit app...")
        
        try:
            # Launch Streamlit in background
            process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for Streamlit to start
            time.sleep(3)
            
            # Open browser
            webbrowser.open("http://localhost:8501")
            
            self.streamlit_launched = True
            print("✅ Streamlit app launched successfully!")
            print("🌐 Open your browser to: http://localhost:8501")
            
            return process
            
        except Exception as e:
            print(f"❌ Error launching Streamlit: {e}")
            return None
    
    def monitor_and_launch(self, check_interval=30, max_wait_time=1800):
        """
        Monitor training and launch Streamlit when ready
        
        Args:
            check_interval: How often to check (seconds)
            max_wait_time: Maximum time to wait for training (seconds)
        """
        print("🔍 Training Monitor Started")
        print("=" * 50)
        
        start_time = time.time()
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            status = self.get_training_status()
            
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Status: {status} | Elapsed: {elapsed_time:.0f}s", end="")
            
            if status == "model_ready":
                print(f"\n✅ Model is ready! Found at: {self.model_path}")
                
                # Test model loading
                try:
                    from models import get_model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = get_model().to(device)
                    checkpoint = torch.load(self.model_path, map_location=device)
                    
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    print("✅ Model loaded successfully!")
                    
                    # Launch Streamlit
                    streamlit_process = self.launch_streamlit()
                    
                    if streamlit_process:
                        print("\n🎉 End-to-end pipeline complete!")
                        print("=" * 50)
                        print("📊 Training: ✅ Complete")
                        print("🤖 Model: ✅ Loaded") 
                        print("🌐 Streamlit: ✅ Running")
                        print("=" * 50)
                        
                        # Keep monitoring Streamlit
                        try:
                            while True:
                                if streamlit_process.poll() is not None:
                                    print("\n🛑 Streamlit process ended")
                                    break
                                time.sleep(5)
                        except KeyboardInterrupt:
                            print("\n🛑 Stopping monitor...")
                            streamlit_process.terminate()
                    
                    break
                    
                except Exception as e:
                    print(f"\n❌ Error loading model: {e}")
                    print("Model file exists but cannot be loaded. Waiting for training to complete...")
            
            elif status == "training":
                print(" | Training in progress...")
            
            elif status == "no_training":
                print(" | No active training detected")
                print(f"\n⚠️  No training detected. Please start training with:")
                print("   python train_model.py --epochs 25 --batch_size 8 \\")
                print("     --train_images_dir dataset/train/images \\")
                print("     --train_masks_dir dataset/train/masks \\") 
                print("     --val_images_dir dataset/val/images \\")
                print("     --val_masks_dir dataset/val/masks")
                break
            
            # Check timeout
            if elapsed_time > max_wait_time:
                print(f"\n⏰ Timeout reached ({max_wait_time}s). Launching Streamlit with demo model...")
                streamlit_process = self.launch_streamlit()
                
                if streamlit_process:
                    print("\n📝 Note: Using demo model since training is not complete")
                    print("   Complete training to use the real model!")
                
                break
            
            time.sleep(check_interval)


def main():
    """Main function"""
    print("🎯 Image Forgery Detection - End-to-End Pipeline")
    print("=" * 60)
    
    monitor = TrainingMonitor()
    
    # Check current status
    status = monitor.get_training_status()
    
    if status == "model_ready":
        print("🎉 Trained model already exists!")
        monitor.launch_streamlit()
    elif status == "training":
        print("🔄 Training detected! Monitoring progress...")
        monitor.monitor_and_launch()
    else:
        print("⚠️  No training or model detected.")
        print("\nOptions:")
        print("1. Start training first:")
        print("   python train_model.py --epochs 25 --batch_size 8 \\")
        print("     --train_images_dir dataset/train/images \\")
        print("     --train_masks_dir dataset/train/masks \\")
        print("     --val_images_dir dataset/val/images \\")
        print("     --val_masks_dir dataset/val/masks")
        print("\n2. Launch Streamlit with demo model:")
        print("   python -m streamlit run streamlit_app.py")
        
        # Ask user what to do
        choice = input("\nLaunch Streamlit with demo model? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            monitor.launch_streamlit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()