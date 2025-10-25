# Image Forgery Detection - AI Coding Instructions

## Project Overview
This is a PyTorch-based image forgery detection system using **Noise Residual Learning** and CNNs. The core innovation is extracting noise patterns with high-pass filtering to identify forged regions in images.

## Architecture & Key Components

### Model Architecture (`models/noise_residual_cnn.py`)
- **NoiseResidualBlock**: Applies high-pass filtering with frozen weights for noise extraction
- **ForgeryLocalizationCNN**: U-Net style encoder-decoder with skip connections
- Predefined high-pass filter kernel `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]` is frozen during training

### Data Pipeline (`utils/dataset.py`)
- **ForgeryDetectionDataset**: Custom dataset class with mask-image pairing
- **ForgeryDataModule**: Wrapper for train/val/test dataloaders with preprocessing
- Expects dataset structure: `dataset/{train,val,test}/{images,masks}/`
- Masks are binary (0=authentic, 255=forged regions)

### Training System (`train_model.py`)
- **Trainer class**: Comprehensive training with logging, checkpointing, metrics
- Uses `CombinedLoss` (BCE + Dice + Focal) for robust training
- Extensive logging system with experiment tracking in `logs/` directory

## Critical Workflows

### Training Pipeline
```bash
# Complete pipeline (preferred)
python run_complete_pipeline.py

# Manual training with logging
python train_model.py --experiment_name "exp_name" --epochs 25 --batch_size 8

# Enhanced training with TensorBoard
python train_enhanced.py --use_tensorboard --experiment_name "detailed_exp"
```

### Dataset Processing
```bash
# CASIA dataset preprocessing
python process_casia.py

# Preprocessing with logging
python run_preprocessing.py --action preprocess --experiment_name "prep_v1"
```

### Analysis & Debugging
```bash
# View experiment logs
python log_analyzer.py --action view_terminal --experiment "exp_name"

# Generate training plots
python log_analyzer.py --action analyze --experiment "exp_name"

# Pipeline verification
python verify_pipeline.py
```

## Project-Specific Patterns

### Import Convention
```python
# Models
from models import get_model, model_summary

# Utils (comprehensive imports)
from utils import (ForgeryDataModule, SegmentationMetrics, CombinedLoss,
                   plot_training_history, visualize_predictions)
```

### Experiment Management
- All experiments create timestamped directories in `logs/`
- Log types: `*_terminal.log`, `*_training.log`, `*_metrics.csv`
- Use `--experiment_name` parameter for consistent naming

### GPU/CPU Handling
- `setup_gpu.py` handles GPU detection and optimization
- Falls back to CPU with optimized settings if GPU unavailable
- Device detection pattern: Check CUDA availability early in scripts

### File Naming Conventions
- Masks: `{image_name}_mask.{ext}` or `{image_name}.{ext}` (flexible matching)
- Checkpoints: `best_model.pth`, `checkpoint_epoch_{n}.pth`
- Logs: `{experiment_name}_{type}.{ext}`

## Integration Points

### Streamlit Interface (`streamlit_app.py`)
- Demo model creates fake predictions when actual model unavailable
- Real-time visualization with threshold adjustment
- Upload → Process → Visualize → Download workflow

### Logging Infrastructure
- **LogAnalyzer**: Central analysis tool for all experiment logs
- CSV metrics tracking with pandas integration
- Terminal output capture for debugging

### Model Factory Pattern (`models/__init__.py`)
- `get_model()` function returns configured model instances
- `model_summary()` provides architecture details
- Supports multiple architectures (noise_residual_cnn, enhanced_model)

## Key Dependencies & Environment

### Core Stack
- PyTorch (CUDA support detection)
- OpenCV, PIL for image processing
- Streamlit for web interface
- Matplotlib/Seaborn for visualization

### Windows-Specific Considerations
- Uses PowerShell commands in pipeline scripts
- Path handling with `pathlib.Path` for cross-platform compatibility
- Batch files for streamlit launching (`run_streamlit.bat`)

## Common Debugging Patterns

### Dataset Issues
1. Check `dataset/{split}/{type}/` structure exists
2. Verify mask-image filename matching in logs
3. Use `create_sample_data()` for testing

### Training Issues
1. Check GPU availability with `setup_gpu.py`
2. Monitor logs with `log_analyzer.py --action view_terminal`
3. Validate metrics in CSV files

### Model Loading
1. Checkpoints saved with full state dict including optimizer
2. Use `load_checkpoint()` method from Trainer class
3. Model architecture must match exactly for loading