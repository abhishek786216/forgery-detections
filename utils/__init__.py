"""
Utils package initialization
"""

from .dataset import ForgeryDetectionDataset, ForgeryDataModule, create_sample_data
from .preprocessing import ImagePreprocessor, apply_noise_filters, augment_image
from .metrics import (SegmentationMetrics, DiceLoss, FocalLoss, CombinedLoss,
                     pixel_accuracy, iou_score, dice_coefficient, f1_score_segmentation,
                     auc_score, calculate_confusion_matrix)
from .visualization import (plot_training_history, visualize_predictions, 
                           create_overlay_visualization, plot_confusion_matrix,
                           plot_metrics_comparison, visualize_noise_patterns,
                           save_prediction_results)

__all__ = [
    # Dataset
    'ForgeryDetectionDataset', 'ForgeryDataModule', 'create_sample_data',
    
    # Preprocessing
    'ImagePreprocessor', 'apply_noise_filters', 'augment_image',
    
    # Metrics
    'SegmentationMetrics', 'DiceLoss', 'FocalLoss', 'CombinedLoss',
    'pixel_accuracy', 'iou_score', 'dice_coefficient', 'f1_score_segmentation',
    'auc_score', 'calculate_confusion_matrix',
    
    # Visualization
    'plot_training_history', 'visualize_predictions', 'create_overlay_visualization',
    'plot_confusion_matrix', 'plot_metrics_comparison', 'visualize_noise_patterns',
    'save_prediction_results'
]