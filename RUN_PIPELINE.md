# 🎯 Complete Image Forgery Detection Pipeline Guide

## 🚀 Quick Start (Everything is Ready!)

Since your pipeline is already set up, you can jump straight to using it:

### Option 1: Auto-Launch (Recommended)
```bash
python monitor_and_launch.py
```
This will automatically detect your trained model and launch the Streamlit interface.

### Option 2: Direct Streamlit Launch
```bash
streamlit run streamlit_app.py
```
Then open: http://localhost:8501

---

## 📋 Full Pipeline from Scratch

If you want to run the complete pipeline from the beginning:

### Step 1: Data Preprocessing
```bash
# Process the CASIA v2.0 dataset with enhanced logging
python process_casia.py

# Alternative: Run with full terminal logging
python run_preprocessing.py
```

**Expected Output:**
- ✅ 12,555+ images processed
- ✅ Train/Val/Test splits created
- ✅ Comprehensive logs generated

### Step 2: Model Training
```bash
# Train the model (full training - 25 epochs)
python train_model.py --epochs 25 --batch_size 8 --train_images_dir dataset/train/images --train_masks_dir dataset/train/masks --val_images_dir dataset/val/images --val_masks_dir dataset/val/masks --experiment_name "full_training"

# Or quick test (5 epochs)
python train_model.py --epochs 5 --batch_size 8 --train_images_dir dataset/train/images --train_masks_dir dataset/train/masks --val_images_dir dataset/val/images --val_masks_dir dataset/val/masks --experiment_name "quick_test"
```

**Expected Output:**
- ✅ Model trained with 31.4M parameters
- ✅ Checkpoints saved in `checkpoints/`
- ✅ Training logs and metrics saved

### Step 3: Launch Web Interface
```bash
# Monitor training and auto-launch when ready
python monitor_and_launch.py

# Or launch directly
streamlit run streamlit_app.py
```

**Expected Output:**
- ✅ Web interface at http://localhost:8501
- ✅ Real-time image forgery detection
- ✅ Visual mask overlays

---

## 🔧 Utility Commands

### Check System Status
```bash
python system_status.py
```

### Verify Everything Works
```bash
python verify_pipeline.py
```

### Analyze Logs
```bash
python log_analyzer.py
```

---

## 📁 Current System State

Your system already has:

```
📊 DATASET: ✅ Ready
├── Training Images: 8,655 files
├── Training Masks: 11,438 files
├── Validation Images: 2,649 files
├── Validation Masks: 3,464 files
├── Test Images: 2,649 files
└── Test Masks: 3,513 files

🤖 MODEL: ✅ Trained
├── checkpoint_epoch_5.pth (360.3 MB)
├── 31.4M parameters
└── Inference tested ✅

🌐 STREAMLIT: ✅ Ready
├── streamlit_app.py (10.5 KB)
├── All packages available
└── Auto model loading
```

---

## 🎪 How to Use the Web Interface

1. **Launch the app:**
   ```bash
   python monitor_and_launch.py
   ```

2. **Open your browser to:** http://localhost:8501

3. **Upload an image** using the file uploader

4. **View results:**
   - Original image display
   - Forgery prediction mask
   - Confidence score
   - Authenticity assessment

---

## 🐛 Troubleshooting

### If Training Fails:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reduce batch size if out of memory
python train_model.py --epochs 5 --batch_size 4 --train_images_dir dataset/train/images --train_masks_dir dataset/train/masks --val_images_dir dataset/val/images --val_masks_dir dataset/val/masks
```

### If Streamlit Won't Start:
```bash
# Install missing packages
pip install streamlit torch torchvision opencv-python pillow

# Check port availability
netstat -an | findstr :8501
```

### If Model Loading Fails:
```bash
# Verify model file
python verify_pipeline.py

# Check model architecture
python -c "from models import get_model; print(get_model())"
```

---

## 🎯 Expected Performance

- **Dataset:** 12,555+ images processed
- **Training Time:** ~30-60 minutes (5 epochs)
- **Model Size:** 360+ MB
- **Inference Speed:** ~1-2 seconds per image
- **Web Interface:** Real-time response

---

## 📈 Next Steps

Once the pipeline is running:

1. **Test with your own images**
2. **Experiment with different architectures**
3. **Fine-tune hyperparameters**
4. **Add more datasets**
5. **Deploy to cloud platforms**

---

## 🎉 Success Indicators

You'll know everything is working when you see:

```
🎯 IMAGE FORGERY DETECTION SYSTEM
============================================================
  ✅ 📊 Dataset Processed
  ✅ 🤖 Model Trained  
  ✅ 🌐 Streamlit Ready

🎉 ALL SYSTEMS OPERATIONAL!
🚀 Ready for image forgery detection!
```

**Your pipeline is ready to detect image forgeries! 🚀**