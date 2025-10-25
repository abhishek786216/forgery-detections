"""
Evaluation metrics for image forgery detection
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import cv2


def pixel_accuracy(pred, target, threshold=0.5):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: Pixel accuracy
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        correct = (pred_binary == target).float()
        accuracy = correct.sum() / correct.numel()
        return accuracy.item()


def iou_score(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) score
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: IoU score
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        
        # Calculate intersection and union
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        
        # Avoid division by zero
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return iou.item()


def dice_coefficient(pred, target, threshold=0.5):
    """
    Calculate Dice coefficient (F1 score for segmentation)
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binary prediction
        
    Returns:
        float: Dice coefficient
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        
        # Calculate intersection
        intersection = (pred_binary * target).sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)
        return dice.item()


def f1_score_segmentation(pred, target, threshold=0.5):
    """
    Calculate F1 score for segmentation
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binary prediction
        
    Returns:
        dict: Precision, recall, and F1 score
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()
        
        # Calculate metrics
        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


def auc_score(pred, target):
    """
    Calculate Area Under Curve (AUC) score
    
    Args:
        pred (torch.Tensor): Predicted probabilities
        target (torch.Tensor): Ground truth mask
        
    Returns:
        float: AUC score
    """
    with torch.no_grad():
        # Flatten tensors
        pred_flat = pred.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()
        
        # Calculate AUC
        try:
            auc = roc_auc_score(target_flat, pred_flat)
            return auc
        except ValueError:
            # Return 0.5 if all labels are the same class
            return 0.5


class SegmentationMetrics:
    """
    Class to calculate and store segmentation metrics
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.pixel_accs = []
        self.ious = []
        self.dices = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.aucs = []
    
    def update(self, pred, target):
        """
        Update metrics with new predictions
        
        Args:
            pred (torch.Tensor): Predicted mask
            target (torch.Tensor): Ground truth mask
        """
        # Calculate metrics
        pixel_acc = pixel_accuracy(pred, target, self.threshold)
        iou = iou_score(pred, target, self.threshold)
        dice = dice_coefficient(pred, target, self.threshold)
        f1_metrics = f1_score_segmentation(pred, target, self.threshold)
        auc = auc_score(pred, target)
        
        # Store metrics
        self.pixel_accs.append(pixel_acc)
        self.ious.append(iou)
        self.dices.append(dice)
        self.precisions.append(f1_metrics['precision'])
        self.recalls.append(f1_metrics['recall'])
        self.f1s.append(f1_metrics['f1_score'])
        self.aucs.append(auc)
    
    def get_metrics(self):
        """
        Get average metrics
        
        Returns:
            dict: Average metrics
        """
        return {
            'pixel_accuracy': np.mean(self.pixel_accs) if self.pixel_accs else 0.0,
            'iou': np.mean(self.ious) if self.ious else 0.0,
            'dice': np.mean(self.dices) if self.dices else 0.0,
            'precision': np.mean(self.precisions) if self.precisions else 0.0,
            'recall': np.mean(self.recalls) if self.recalls else 0.0,
            'f1_score': np.mean(self.f1s) if self.f1s else 0.0,
            'auc': np.mean(self.aucs) if self.aucs else 0.0,
            'num_samples': len(self.pixel_accs)
        }
    
    def print_metrics(self):
        """Print formatted metrics"""
        metrics = self.get_metrics()
        
        print("\n" + "="*50)
        print("SEGMENTATION METRICS")
        print("="*50)
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"IoU Score:      {metrics['iou']:.4f}")
        print(f"Dice Coeff:     {metrics['dice']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1 Score:       {metrics['f1_score']:.4f}")
        print(f"AUC Score:      {metrics['auc']:.4f}")
        print(f"Samples:        {metrics['num_samples']}")
        print("="*50)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Calculate Dice loss
        
        Args:
            pred (torch.Tensor): Predicted logits or probabilities
            target (torch.Tensor): Ground truth mask
            
        Returns:
            torch.Tensor: Dice loss
        """
        # Convert logits to probabilities if needed
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection
        intersection = (pred_flat * target_flat).sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Calculate Focal loss
        
        Args:
            pred (torch.Tensor): Predicted logits or probabilities
            target (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Focal loss
        """
        # For mixed precision compatibility, use BCE with logits if pred is not sigmoid-ed
        if pred.max() > 1.0 or pred.min() < 0.0:
            # Pred contains logits, use BCEWithLogitsLoss
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pred_prob = torch.sigmoid(pred)
        else:
            # Pred contains probabilities, use regular BCE  
            bce_loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
            pred_prob = pred
        
        # Calculate p_t
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Calculate focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Calculate focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function (BCE + Dice + Focal)
    """
    
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=0.5, 
                 focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()  # Mixed precision safe
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, pred, target):
        """
        Calculate combined loss
        
        Args:
            pred (torch.Tensor): Predicted logits or probabilities 
            target (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss and loss dictionary
        """
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)  # DiceLoss handles logits internally
        focal = self.focal_loss(pred, target)  # FocalLoss handles logits internally
        
        total_loss = (self.bce_weight * bce + 
                     self.dice_weight * dice + 
                     self.focal_weight * focal)
        
        return total_loss, {
            'bce_loss': bce.item(),
            'dice_loss': dice.item(),
            'focal_loss': focal.item(),
            'total_loss': total_loss.item()
        }


def calculate_confusion_matrix(pred, target, threshold=0.5):
    """
    Calculate confusion matrix for binary segmentation
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binary prediction
        
    Returns:
        dict: Confusion matrix components
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Calculate confusion matrix components
        tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
        tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
        fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
        
        return {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'total': tp + tn + fp + fn
        }


# Alias for compatibility
dice_score = dice_coefficient