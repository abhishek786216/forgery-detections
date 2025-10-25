
"""
Simple Streamlit App for Image Forgery Detection
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

# Set page config
st.set_page_config(
    page_title="Image Forgery Detection",
    page_icon="🔍",
    layout="wide"
)

def create_demo_model():
    """Create a demo model for testing when actual model is not available"""
    class DemoModel:
        def __init__(self):
            self.device = 'cpu'
            # Add some randomness to make predictions more realistic
            np.random.seed(42)  # For consistent demo results
        
        def eval(self):
            pass
        
        def __call__(self, x):
            # Create a more realistic demo prediction
            batch_size, channels, height, width = x.shape
            
            # Start with low background noise
            fake_prediction = torch.zeros(batch_size, 1, height, width)
            
            # Add subtle background noise (authentic regions have very low values)
            background_noise = torch.rand_like(fake_prediction) * 0.1
            fake_prediction += background_noise
            
            for i in range(batch_size):
                # 60% chance of authentic image (no significant forgery)
                if np.random.random() < 0.6:
                    # Authentic image - keep mostly low values
                    fake_prediction[i] = fake_prediction[i] * 0.3
                else:
                    # Forged image - add some suspicious regions
                    num_regions = np.random.randint(1, 3)
                    for _ in range(num_regions):
                        x1 = np.random.randint(0, width // 2)
                        y1 = np.random.randint(0, height // 2)
                        x2 = x1 + np.random.randint(30, width // 3)
                        y2 = y1 + np.random.randint(30, height // 3)
                        
                        x2 = min(x2, width)
                        y2 = min(y2, height)
                        
                        # More realistic forgery intensity
                        intensity = np.random.uniform(0.4, 0.8)
                        fake_prediction[i, 0, y1:y2, x1:x2] = intensity
            
            return fake_prediction
    
    return DemoModel()
    
    return DemoModel()

@st.cache_resource
def load_model():
    """Load the model or create demo model"""
    try:
        # Try to import the actual model
        from models import get_model
        
        model_path = "checkpoints/best_model.pth"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if os.path.exists(model_path):
            model = get_model().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model, device, "real"
        else:
            # Use demo model
            model = create_demo_model()
            return model, 'cpu', "demo"
    
    except ImportError:
        # Use demo model if imports fail
        model = create_demo_model()
        return model, 'cpu', "demo"

def preprocess_image(image, target_size=(256, 256)):
    """Simple preprocessing function"""
    # Resize image
    image_resized = image.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def predict_forgery(model, image, threshold=0.5, model_type="real"):
    """Predict forgery in image with improved logic"""
    original_size = image.size
    
    # Get model device (handle both real and demo models)
    try:
        device = next(model.parameters()).device
    except:
        device = torch.device('cpu')
    
    # Preprocess
    if model_type == "real":
        # Use proper preprocessing for real model
        try:
            from utils.preprocessing import ImagePreprocessor
            preprocessor = ImagePreprocessor(target_size=(256, 256), normalize=True)
            image_tensor = preprocessor.preprocess_image(image).unsqueeze(0)
        except ImportError:
            # Fallback to simple preprocessing
            image_tensor = preprocess_image(image)
    else:
        # Simple preprocessing for demo
        image_tensor = preprocess_image(image)
    
    # Move tensor to model device
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        raw_prediction = model(image_tensor)
        
        # Debug: Print raw prediction statistics
        print(f"Raw prediction - Min: {raw_prediction.min().item():.4f}, Max: {raw_prediction.max().item():.4f}, Mean: {raw_prediction.mean().item():.4f}")
        
        # Apply different post-processing based on model type
        if model_type == "real":
            # For real model, check if sigmoid is needed
            # If raw values are already in [0,1] range, don't apply sigmoid
            if raw_prediction.min() >= 0 and raw_prediction.max() <= 1:
                prediction = raw_prediction
                print("Raw prediction already in [0,1] range, skipping sigmoid")
            else:
                prediction = torch.sigmoid(raw_prediction)
                print("Applied sigmoid activation")
        else:
            # For demo model, apply some normalization to make it more realistic
            prediction = torch.clamp(raw_prediction, 0, 1)
        
        print(f"Final prediction - Min: {prediction.min().item():.4f}, Max: {prediction.max().item():.4f}, Mean: {prediction.mean().item():.4f}")
        
        # Move prediction back to CPU for numpy conversion
        prediction = prediction.cpu()
    
    # Process output
    prediction_np = prediction.squeeze().numpy()
    
    # Resize back to original
    prediction_resized = cv2.resize(prediction_np, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Apply adaptive thresholding for better results
    if model_type == "demo":
        # For demo, make threshold more sensitive to show variety
        adaptive_threshold = threshold * 0.7  # Lower threshold for demo
    else:
        # For real model, use adaptive thresholding based on prediction statistics
        mean_prediction = np.mean(prediction_resized)
        std_prediction = np.std(prediction_resized)
        min_prediction = np.min(prediction_resized)
        max_prediction = np.max(prediction_resized)
        
        print(f"Prediction stats - Min: {min_prediction:.4f}, Max: {max_prediction:.4f}, Mean: {mean_prediction:.4f}, Std: {std_prediction:.4f}")
        
        # If all values are very high (like your case), use a higher threshold
        if mean_prediction > 0.8:
            adaptive_threshold = 0.9  # Very high threshold for oversaturated predictions
            print(f"High mean prediction detected, using high threshold: {adaptive_threshold}")
        elif mean_prediction < 0.1:  # Very low overall prediction
            adaptive_threshold = threshold * 0.5  # More sensitive
            print(f"Low mean prediction detected, using sensitive threshold: {adaptive_threshold}")
        elif std_prediction < 0.05:  # Very uniform predictions
            # Use percentile-based threshold
            adaptive_threshold = np.percentile(prediction_resized, 75)  # Top 25% as forgery
            print(f"Uniform predictions detected, using percentile threshold: {adaptive_threshold}")
        else:
            adaptive_threshold = threshold
            print(f"Using standard threshold: {adaptive_threshold}")
    
    # Create binary mask
    binary_mask = (prediction_resized > adaptive_threshold).astype(np.uint8) * 255
    
    # Add forgery percentage calculation
    forgery_percentage = (np.sum(binary_mask > 0) / binary_mask.size) * 100
    
    return np.array(image), prediction_resized, binary_mask, forgery_percentage

def create_visualization(image, prediction, binary_mask, threshold):
    """Create result visualization"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction heatmap
    im = axes[1].imshow(prediction, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Forgery Probability')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title(f'Binary Mask (t={threshold})')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    red_overlay = np.zeros_like(overlay)
    red_overlay[:, :, 0] = binary_mask
    
    # Blend
    mask_3d = np.stack([binary_mask] * 3, axis=-1) > 0
    overlay[mask_3d] = (0.6 * overlay[mask_3d] + 0.4 * red_overlay[mask_3d]).astype(np.uint8)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    st.title("🔍 Image Forgery Detection System")
    
    st.markdown("""
    Upload an image to detect potential forgery regions using deep learning.
    This system uses **Noise Residual Learning** to identify manipulated areas.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, device, model_type = load_model()
    
    if model_type == "demo":
        st.warning("⚠️ Using demo mode. For real detection, train a model first using `train_model.py`")
    else:
        st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload an image to analyze for potential forgery"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, width="stretch")
        
        with col2:
            st.subheader("Analysis")
            
            with st.spinner("Analyzing image..."):
                # Predict
                image_np, prediction, binary_mask, forgery_percentage = predict_forgery(
                    model, image, threshold, model_type
                )
                
                # Calculate stats
                forgery_ratio = forgery_percentage / 100.0  # Convert percentage to ratio
                max_prob = np.max(prediction)
                mean_prob = np.mean(prediction)
            
            # Show results
            if forgery_ratio > 0.01:
                st.error(f"⚠️ **POTENTIAL FORGERY DETECTED**")
            else:
                st.success("✅ **IMAGE APPEARS AUTHENTIC**")
            
            # Statistics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Forgery Ratio", f"{forgery_ratio:.1%}")
                st.metric("Max Probability", f"{max_prob:.3f}")
            with col_b:
                st.metric("Mean Probability", f"{mean_prob:.3f}")
                st.metric("Image Size", f"{image.size[0]}×{image.size[1]}")
        
        # Visualization
        st.subheader("Detection Results")
        
        fig = create_visualization(image_np, prediction, binary_mask, threshold)
        st.pyplot(fig)
        
        # Download buttons
        st.subheader("Download Results")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            mask_pil = Image.fromarray(binary_mask)
            buf = io.BytesIO()
            mask_pil.save(buf, format='PNG')
            st.download_button(
                "Download Mask",
                data=buf.getvalue(),
                file_name="forgery_mask.png",
                mime="image/png"
            )
        
        with col_dl2:
            heatmap = (prediction * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            heatmap_pil = Image.fromarray(heatmap_rgb)
            
            buf2 = io.BytesIO()
            heatmap_pil.save(buf2, format='PNG')
            st.download_button(
                "Download Heatmap",
                data=buf2.getvalue(),
                file_name="forgery_heatmap.png",
                mime="image/png"
            )
        
        with col_dl3:
            # Create proper overlay
            overlay = image_np.copy()
            red_overlay = np.zeros_like(overlay)
            red_overlay[:, :, 0] = binary_mask
            
            mask_3d = np.stack([binary_mask] * 3, axis=-1) > 0
            overlay[mask_3d] = (0.6 * overlay[mask_3d] + 0.4 * red_overlay[mask_3d]).astype(np.uint8)
            
            overlay_pil = Image.fromarray(overlay)
            buf3 = io.BytesIO()
            overlay_pil.save(buf3, format='PNG')
            st.download_button(
                "Download Overlay",
                data=buf3.getvalue(),
                file_name="forgery_overlay.png",
                mime="image/png"
            )
    
    else:
        st.info("👆 Please upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This System
    
    This Image Forgery Detection System uses:
    - **Noise Residual Learning** to extract manipulation artifacts
    - **Convolutional Neural Networks** for deep pattern analysis  
    - **Pixel-level localization** to pinpoint forged regions
    
    **Interpretation:**
    - 🔴 Red areas: Potential forgery detected
    - 🌡️ Heatmap: Warmer colors = higher forgery probability
    - ⚙️ Threshold: Adjustable sensitivity
    
    *Developed by: Abhishek Kumar, Banothu Ramu, Raghava Priya*
    """)

if __name__ == "__main__":
    main()