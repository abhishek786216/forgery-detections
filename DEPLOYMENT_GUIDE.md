# Streamlit Cloud Deployment Guide

## Files Required for Deployment

### 1. requirements.txt
- Updated to use `opencv-python-headless` for cloud compatibility
- All necessary Python packages listed

### 2. packages.txt
- System-level dependencies for OpenCV
- Required for Streamlit Cloud environment

### 3. streamlit_app.py
- Main application file
- Contains fallback mechanisms for OpenCV issues
- Handles import errors gracefully

## Deployment Steps

1. **Push to GitHub** (✅ Already done)
   ```bash
   git add .
   git commit -m "Update for Streamlit Cloud compatibility"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Connect your GitHub account
   - Select repository: `abhishek786216/forgery-detections`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Troubleshooting**
   - If OpenCV issues persist, the app will use fallback mode
   - Check the logs in Streamlit Cloud dashboard
   - Ensure all files are in the repository root

## Features Available in Cloud Mode

### ✅ Working Features:
- Model loading and prediction
- Image upload and display
- Basic image processing
- Results visualization
- Threshold adjustment

### ⚠️ Limited Features (if OpenCV fails):
- Advanced image analysis will use simplified algorithms
- Some edge detection features may be reduced
- Basic functionality remains intact

## Repository Structure
```
forgery-detections/
├── streamlit_app.py          # Main app (with fallbacks)
├── requirements.txt          # Python dependencies  
├── packages.txt             # System dependencies
├── models/                  # Model architecture
├── utils/                   # Utilities
└── README.md               # Documentation
```

## Environment Variables (if needed)
- None required for basic deployment
- Model will use CPU by default in cloud

## Performance Notes
- Cloud deployment uses CPU (no GPU)
- Demo model will be used if trained weights not available
- Response time: 2-5 seconds per image analysis