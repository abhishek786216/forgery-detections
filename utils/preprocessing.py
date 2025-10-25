"""
Preprocessing utilities for image forgery detection
"""

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class ImagePreprocessor:
    """
    Image preprocessing utilities for forgery detection
    """
    
    def __init__(self, target_size=(256, 256), normalize=True):
        """
        Initialize preprocessor
        
        Args:
            target_size (tuple): Target image size (height, width)
            normalize (bool): Whether to normalize pixel values
        """
        self.target_size = target_size
        self.normalize = normalize
        
        # Define transforms
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            # ImageNet normalization values
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
        
        # Transform for masks (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess input image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path  # Assume PIL Image
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def preprocess_mask(self, mask_path):
        """
        Preprocess ground truth mask
        
        Args:
            mask_path (str): Path to mask file
            
        Returns:
            torch.Tensor: Preprocessed mask tensor
        """
        try:
            # Load mask
            if isinstance(mask_path, str):
                mask = Image.open(mask_path).convert('L')  # Convert to grayscale
            else:
                mask = mask_path  # Assume PIL Image
            
            # Apply transforms
            mask_tensor = self.mask_transform(mask)
            
            # Ensure binary values (0 or 1)
            mask_tensor = (mask_tensor > 0.5).float()
            
            return mask_tensor
            
        except Exception as e:
            print(f"Error preprocessing mask {mask_path}: {e}")
            return None
    
    def denormalize_image(self, tensor):
        """
        Denormalize image tensor for visualization
        
        Args:
            tensor (torch.Tensor): Normalized image tensor
            
        Returns:
            torch.Tensor: Denormalized image tensor
        """
        if not self.normalize:
            return tensor
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Move to same device as tensor
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
        
        # Denormalize
        denormalized = tensor * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized


def apply_noise_filters(image, filter_type='srm'):
    """
    Apply noise extraction filters to image
    
    Args:
        image (np.ndarray): Input image
        filter_type (str): Type of filter to apply ('srm', 'high_pass', 'laplacian')
        
    Returns:
        np.ndarray: Filtered image
    """
    if filter_type == 'srm':
        # Spatial Rich Model (SRM) filter
        kernel = np.array([[-1, 2, -2, 2, -1],
                          [2, -6, 8, -6, 2],
                          [-2, 8, -12, 8, -2],
                          [2, -6, 8, -6, 2],
                          [-1, 2, -2, 2, -1]], dtype=np.float32) / 12
        
        filtered = cv2.filter2D(image, -1, kernel)
        
    elif filter_type == 'high_pass':
        # Simple high-pass filter
        kernel = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]], dtype=np.float32)
        
        filtered = cv2.filter2D(image, -1, kernel)
        
    elif filter_type == 'laplacian':
        # Laplacian filter
        filtered = cv2.Laplacian(image, cv2.CV_64F)
        
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return filtered


def augment_image(image, mask=None, augmentation_type='random'):
    """
    Apply data augmentation to image and mask
    
    Args:
        image (PIL.Image or np.ndarray): Input image
        mask (PIL.Image or np.ndarray): Ground truth mask
        augmentation_type (str): Type of augmentation
        
    Returns:
        tuple: Augmented image and mask
    """
    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    if augmentation_type == 'horizontal_flip':
        image = cv2.flip(image, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)
            
    elif augmentation_type == 'vertical_flip':
        image = cv2.flip(image, 0)
        if mask is not None:
            mask = cv2.flip(mask, 0)
            
    elif augmentation_type == 'rotation':
        angle = np.random.uniform(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (w, h))
            
    elif augmentation_type == 'brightness':
        # Adjust brightness
        factor = np.random.uniform(0.7, 1.3)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
    elif augmentation_type == 'contrast':
        # Adjust contrast
        factor = np.random.uniform(0.7, 1.3)
        image = np.clip((image - 128) * factor + 128, 0, 255).astype(np.uint8)
        
    elif augmentation_type == 'random':
        # Apply random augmentation
        augmentations = ['horizontal_flip', 'brightness', 'contrast']
        chosen_aug = np.random.choice(augmentations)
        return augment_image(image, mask, chosen_aug)
    
    return image, mask


def calculate_noise_statistics(image):
    """
    Calculate noise statistics for an image
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        dict: Noise statistics
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply high-pass filter
    kernel = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]], dtype=np.float32)
    
    noise = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    
    # Calculate statistics
    stats = {
        'mean': np.mean(noise),
        'std': np.std(noise),
        'var': np.var(noise),
        'energy': np.sum(noise ** 2),
        'entropy': calculate_entropy(noise)
    }
    
    return stats


def calculate_entropy(image):
    """
    Calculate entropy of an image
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        float: Image entropy
    """
    # Normalize to 0-255 range
    image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Calculate histogram
    hist, _ = np.histogram(image_norm, bins=256, range=(0, 256))
    
    # Calculate probabilities
    hist = hist / hist.sum()
    
    # Remove zero probabilities
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy


def resize_with_padding(image, target_size, fill_value=0):
    """
    Resize image while maintaining aspect ratio with padding
    
    Args:
        image (PIL.Image or np.ndarray): Input image
        target_size (tuple): Target size (height, width)
        fill_value (int): Padding fill value
        
    Returns:
        PIL.Image or np.ndarray: Resized image with padding
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), fill_value, dtype=image.dtype)
        else:
            padded = np.full((target_h, target_w), fill_value, dtype=image.dtype)
        
        # Calculate padding offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        if len(image.shape) == 3:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized
        else:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    else:  # PIL Image
        w, h = image.size
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Create padded image
        padded = Image.new(image.mode, (target_w, target_h), fill_value)
        
        # Calculate padding offsets
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Paste resized image in center
        padded.paste(resized, (x_offset, y_offset))
        
        return padded