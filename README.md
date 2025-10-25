# 🧠 Image Forgery Localization via Noise Residual Learning

This project detects and localizes forged regions in digital images using **Noise Residual Learning** and **Convolutional Neural Networks (CNNs)**.  
The implementation is done in **Python (PyTorch)** and focuses on improving the accuracy of forgery localization by analyzing noise patterns.

## 🎯 Features

- **Noise Residual Learning**: Extracts noise patterns to identify forged regions
- **CNN-based Architecture**: Deep learning model for precise forgery localization
- **Real-time Detection**: Fast inference for practical applications
- **Visualization Tools**: Built-in functions to visualize detection results
- **Evaluation Metrics**: Comprehensive metrics for model performance assessment

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/abhishek786216/Image-Forgery-Localization.git
cd Image-Forgery-Localization
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On Linux/Mac
```

### 3️⃣ Install Required Dependencies

Install all necessary libraries using pip:
```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python tqdm matplotlib scikit-learn pillow
```

Or install all at once from requirements.txt:
```bash
pip install -r requirements.txt
```

## 📦 Project Structure

```
imageforgerydetections/
│
├── README.md
├── models/
│   ├── __init__.py
│   └── noise_residual_cnn.py     # Main CNN model architecture
├── utils/
│   ├── __init__.py
│   ├── dataset.py               # Dataset handling
│   ├── preprocessing.py         # Image preprocessing utilities
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Visualization tools
├── dataset/                     # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
├── logs/                        # Training logs and metrics
├── checkpoints/                 # Saved model checkpoints
├── results/                     # Output results and visualizations
├── train_model.py              # Standard training script
├── train_enhanced.py           # Enhanced training with detailed logging
├── test_model.py               # Model testing script
├── predict.py                  # Prediction script
├── streamlit_app.py            # Web interface
├── process_casia.py            # CASIA dataset processing
├── setup_dataset.py            # Dataset management utilities
├── run_training_with_logs.py   # Training with full terminal logging
├── log_analyzer.py             # Log analysis and visualization
└── requirements.txt            # Python dependencies
├── requirements.txt
├── train_model.py          # Main training script
├── test_model.py           # Testing and evaluation script
├── predict.py              # Single image prediction
│
├── models/
│   ├── __init__.py
│   └── noise_residual_cnn.py  # CNN model architecture
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py          # Custom dataset class
│   ├── preprocessing.py    # Image preprocessing functions
│   ├── visualization.py    # Plotting and visualization utilities
│   └── metrics.py          # Evaluation metrics
│
├── dataset/
│   ├── images/             # Original images
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── masks/              # Ground truth masks
│       ├── img_001_mask.png
│       ├── img_002_mask.png
│       └── ...
│
├── checkpoints/            # Model checkpoints
├── results/                # Output results and visualizations
└── logs/                   # Training logs
```

## 🗂️ Dataset Structure

Make sure your dataset is organized as:

```
dataset/
│
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
│
└── masks/
    ├── img_001_mask.png
    ├── img_002_mask.png
    └── ...
```

**Note**: Mask images should be binary (0 for authentic, 255 for forged regions) and have the same dimensions as corresponding input images.

## 🚀 Usage

### Web Interface (Streamlit App)
Launch the interactive web application:

**Windows:**
```bash
run_streamlit.bat
```

**Linux/Mac:**
```bash
chmod +x run_streamlit.sh
./run_streamlit.sh
```

**Or manually:**
```bash
streamlit run streamlit_app.py
```

The web interface provides:
- 📤 **Easy file upload** with drag-and-drop support
- 🎯 **Real-time detection** with adjustable threshold
- 📊 **Interactive visualizations** including heatmaps and overlays
- 💾 **Download results** as PNG files
- ⚙️ **Configurable parameters** in the sidebar

### Command Line Usage

#### Training with Enhanced Logging
```bash
# Standard training with logging
python train_model.py --epochs 25 --batch_size 8 --learning_rate 0.001 \
    --train_images_dir dataset/train/images \
    --train_masks_dir dataset/train/masks \
    --val_images_dir dataset/val/images \
    --val_masks_dir dataset/val/masks \
    --experiment_name "casia_experiment_1" \
    --print_model_summary

# Enhanced training with full terminal logging
python run_training_with_logs.py --epochs 25 --batch_size 8 \
    --experiment_name "casia_full_logging" \
    --print_model_summary

# Ultra-detailed training (recommended for debugging)
python train_enhanced.py --epochs 25 --batch_size 8 \
    --experiment_name "casia_detailed" \
    --use_tensorboard --print_model_summary
```

#### Dataset Preprocessing with Logging
```bash
# Run CASIA preprocessing with comprehensive logging
python run_preprocessing.py --action preprocess --experiment_name "casia_preprocessing_v1"

# Generate preprocessing report
python run_preprocessing.py --action report --experiment_name "casia_preprocessing_v1"

# Alternative: Basic preprocessing (existing script)
python process_casia.py
```

#### Log Analysis and Visualization
```bash
# List all experiments
python log_analyzer.py --action list

# View terminal output
python log_analyzer.py --action view_terminal --experiment "casia_experiment_1"

# Analyze training metrics with plots
python log_analyzer.py --action analyze --experiment "casia_experiment_1"

# Analyze preprocessing statistics
python log_analyzer.py --action analyze_preprocessing --experiment "casia_preprocessing_v1"

# Generate comprehensive report
python log_analyzer.py --action report --experiment "casia_experiment_1"
```

#### Testing the Model
```bash
python test_model.py --model_path checkpoints/best_model.pth --test_dir dataset/test
```

#### Single Image Prediction
```bash
python predict.py --image_path path/to/image.jpg --model_path checkpoints/best_model.pth
```

## � Comprehensive Logging System

The project includes a robust logging system that captures all training details:

### 🗂️ Log Types Generated

**Training Logs:**
1. **Terminal Logs** (`*_terminal.log`)
   - Complete terminal output capture
   - Real-time training progress
   - All print statements and errors

2. **Training Logs** (`*_training.log`)
   - Structured training information
   - System configuration details
   - Epoch-by-epoch progress
   - Error tracking with stack traces

3. **Metrics CSV** (`*_metrics.csv`)
   - Epoch-wise performance metrics
   - IoU, F1-score, pixel accuracy, precision, recall
   - Separate train/validation tracking
   - Timestamped entries

4. **TensorBoard Logs** (when enabled)
   - Real-time loss visualization
   - Learning rate tracking
   - Model architecture graphs

**Preprocessing Logs:**
1. **Preprocessing Logs** (`*_preprocessing.log`)
   - System information (Python, OpenCV, PIL versions)
   - Dataset path validation
   - Image scanning and splitting details
   - Processing progress with timing

2. **Processing Stats CSV** (`*_preprocessing_stats.csv`)
   - File-by-file processing statistics
   - Processing times per operation
   - File sizes and success rates
   - Error tracking for failed operations

3. **Processing Summary JSON** (`processing_summary.json`)
   - Overall processing statistics
   - Dataset split information
   - Total processing time
   - Completion timestamp

### 📈 Log Analysis Features

- **Interactive visualizations** of training metrics
- **Automated report generation** with performance summaries
- **Experiment comparison** across different runs
- **Error analysis** and debugging assistance
- **Performance trend analysis** with statistical insights

### 🔍 Usage Examples

```bash
# Train with full logging
python run_training_with_logs.py --epochs 25 --experiment_name "my_experiment"

# Analyze results
python log_analyzer.py --action analyze --experiment "my_experiment"

# Generate PDF report (if you have additional dependencies)
python log_analyzer.py --action report --experiment "my_experiment"
```

## 🔧 Model Architecture

The model uses a CNN with noise residual learning that:
1. Extracts noise patterns from input images using high-pass filtering
2. Applies encoder-decoder architecture with skip connections
3. Generates pixel-level predictions for forgery localization
4. Uses multiple loss functions (BCE, Dice, Focal) for robust training

## 📊 Performance Metrics

The model is evaluated using:
- **Pixel Accuracy**: Overall pixel-level accuracy
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth masks
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

## 🧑‍💻 Authors

- **Abhishek Kumar** (231212001)
- **Banothu Ramu** (231212009)  
- **Raghava Priya** (231212015)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- [Noise Residual Learning for Image Forgery Detection](https://example.com)
- [CNN-based Approaches for Image Manipulation Detection](https://example.com)