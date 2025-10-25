"""
Visualization utilities for image forgery detection
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
from PIL import Image
import seaborn as sns
from typing import List, Tuple, Optional


def plot_training_history(train_losses: List[float], 
                         val_losses: List[float] = None,
                         train_metrics: dict = None,
                         val_metrics: dict = None,
                         save_path: str = None):
    """
    Plot training history including losses and metrics
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot IoU scores
    if train_metrics and 'iou' in train_metrics:
        axes[0, 1].plot(epochs, train_metrics['iou'], 'b-', label='Training IoU', linewidth=2)
        if val_metrics and 'iou' in val_metrics:
            axes[0, 1].plot(epochs, val_metrics['iou'], 'r-', label='Validation IoU', linewidth=2)
        axes[0, 1].set_title('IoU Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 scores
    if train_metrics and 'f1_score' in train_metrics:
        axes[1, 0].plot(epochs, train_metrics['f1_score'], 'b-', label='Training F1', linewidth=2)
        if val_metrics and 'f1_score' in val_metrics:
            axes[1, 0].plot(epochs, val_metrics['f1_score'], 'r-', label='Validation F1', linewidth=2)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if train_metrics and 'pixel_accuracy' in train_metrics:
        axes[1, 1].plot(epochs, train_metrics['pixel_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if val_metrics and 'pixel_accuracy' in val_metrics:
            axes[1, 1].plot(epochs, val_metrics['pixel_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1, 1].set_title('Pixel Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def visualize_predictions(images: torch.Tensor,
                         ground_truth: torch.Tensor,
                         predictions: torch.Tensor,
                         num_samples: int = 4,
                         threshold: float = 0.5,
                         save_path: str = None):
    """
    Visualize model predictions alongside ground truth
    
    Args:
        images: Input images tensor
        ground_truth: Ground truth masks tensor
        predictions: Predicted masks tensor
        num_samples: Number of samples to visualize
        threshold: Threshold for binary prediction
        save_path: Path to save the plot
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Convert tensors to numpy
        image = tensor_to_numpy(images[i])
        gt_mask = ground_truth[i].squeeze().cpu().numpy()
        pred_mask = predictions[i].squeeze().cpu().numpy()
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot prediction heatmap
        axes[i, 2].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction Heatmap')
        axes[i, 2].axis('off')
        
        # Plot binary prediction
        axes[i, 3].imshow(pred_binary, cmap='gray')
        axes[i, 3].set_title(f'Binary Prediction (t={threshold})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
    
    plt.show()


def create_overlay_visualization(image: np.ndarray,
                               mask: np.ndarray,
                               alpha: float = 0.5,
                               color: Tuple[int, int, int] = (255, 0, 0)):
    """
    Create overlay visualization of mask on image
    
    Args:
        image: Input image (RGB)
        mask: Binary mask
        alpha: Transparency factor
        color: Overlay color (RGB)
        
    Returns:
        np.ndarray: Overlay image
    """
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure mask is binary
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color
    
    # Blend image and overlay
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


def plot_confusion_matrix(confusion_matrix: dict,
                         title: str = 'Confusion Matrix',
                         save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Dictionary with TP, TN, FP, FN values
        title: Plot title
        save_path: Path to save the plot
    """
    # Create confusion matrix array
    cm = np.array([[confusion_matrix['true_negative'], confusion_matrix['false_positive']],
                   [confusion_matrix['false_negative'], confusion_matrix['true_positive']]])
    
    # Calculate percentages
    cm_percent = cm / cm.sum() * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                   ha='center', va='center', fontsize=10, color='gray')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: dict,
                          title: str = 'Metrics Comparison',
                          save_path: str = None):
    """
    Plot bar chart comparing different metrics
    
    Args:
        metrics_dict: Dictionary of metrics {metric_name: value}
        title: Plot title
        save_path: Path to save the plot
    """
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                         '#9467bd', '#8c564b', '#e377c2'])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
    
    plt.show()


def visualize_noise_patterns(image: np.ndarray,
                           filter_types: List[str] = ['high_pass', 'laplacian'],
                           save_path: str = None):
    """
    Visualize noise patterns extracted by different filters
    
    Args:
        image: Input image
        filter_types: List of filter types to apply
        save_path: Path to save the plot
    """
    num_filters = len(filter_types)
    fig, axes = plt.subplots(1, num_filters + 1, figsize=(4 * (num_filters + 1), 4))
    
    if num_filters == 0:
        axes = [axes]
    
    # Plot original image
    if len(image.shape) == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Apply and plot different filters
    for i, filter_type in enumerate(filter_types):
        if filter_type == 'high_pass':
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=np.float32)
            filtered = cv2.filter2D(image, -1, kernel)
        
        elif filter_type == 'laplacian':
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                filtered = cv2.Laplacian(gray, cv2.CV_64F)
            else:
                filtered = cv2.Laplacian(image, cv2.CV_64F)
        
        # Normalize for visualization
        filtered_norm = ((filtered - filtered.min()) / 
                        (filtered.max() - filtered.min() + 1e-8))
        
        axes[i + 1].imshow(filtered_norm, cmap='gray')
        axes[i + 1].set_title(f'{filter_type.replace("_", " ").title()} Filter')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Noise patterns visualization saved to: {save_path}")
    
    plt.show()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array for visualization
    
    Args:
        tensor: Input tensor (C, H, W) or (H, W)
        
    Returns:
        np.ndarray: Numpy array ready for visualization
    """
    # Move to CPU and detach
    tensor = tensor.cpu().detach()
    
    # Convert to numpy
    if len(tensor.shape) == 3:
        # (C, H, W) -> (H, W, C)
        array = tensor.permute(1, 2, 0).numpy()
        
        # Denormalize if needed (assuming ImageNet normalization)
        if array.min() < 0:  # Likely normalized
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            array = array * std + mean
        
        # Clip to [0, 1] range
        array = np.clip(array, 0, 1)
        
    else:
        # (H, W)
        array = tensor.numpy()
        # Clip to [0, 1] range
        array = np.clip(array, 0, 1)
    
    return array


def save_prediction_results(image_path: str,
                          image: np.ndarray,
                          prediction: np.ndarray,
                          ground_truth: np.ndarray = None,
                          save_dir: str = 'results',
                          threshold: float = 0.5):
    """
    Save prediction results including overlay and separate mask
    
    Args:
        image_path: Original image path
        image: Input image
        prediction: Predicted mask
        ground_truth: Ground truth mask (optional)
        save_dir: Directory to save results
        threshold: Threshold for binary prediction
    """
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create binary prediction
    pred_binary = (prediction > threshold).astype(np.uint8) * 255
    
    # Save binary mask
    mask_path = os.path.join(save_dir, f'{base_name}_mask.png')
    cv2.imwrite(mask_path, pred_binary)
    
    # Create and save overlay
    if len(image.shape) == 3:
        overlay = create_overlay_visualization(image, pred_binary, alpha=0.3)
        overlay_path = os.path.join(save_dir, f'{base_name}_overlay.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Save heatmap
    heatmap = (prediction * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap_path = os.path.join(save_dir, f'{base_name}_heatmap.png')
    cv2.imwrite(heatmap_path, heatmap_colored)
    
    print(f"Results saved to {save_dir}:")
    print(f"  - Mask: {mask_path}")
    if len(image.shape) == 3:
        print(f"  - Overlay: {overlay_path}")
    print(f"  - Heatmap: {heatmap_path}")