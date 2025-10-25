"""
Custom dataset class for image forgery detection
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import random
from typing import List, Tuple, Optional, Callable
import torchvision.transforms as transforms

from .preprocessing import ImagePreprocessor, augment_image


class ForgeryDetectionDataset(Dataset):
    """
    Custom Dataset for Image Forgery Detection and Localization
    """
    
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 image_size: Tuple[int, int] = (256, 256),
                 transforms: Optional[Callable] = None,
                 augment: bool = True,
                 normalize: bool = True,
                 transform_type: str = 'basic'):
        """
        Initialize the dataset
        
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing ground truth masks
            image_size: Target image size (height, width)
            transforms: Custom transforms to apply
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images
            transform_type: Type of transforms ('basic', 'enhanced_train', 'val')
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.augment = augment
        self.custom_transforms = transforms
        self.transform_type = transform_type
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            target_size=image_size,
            normalize=normalize
        )
        
        # Setup transforms based on type
        self._setup_transforms()
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        # Verify that corresponding mask files exist
        self._verify_mask_files()
        
        print(f"Dataset initialized with {len(self.image_files)} samples")
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = []
        for file in os.listdir(self.images_dir):
            if os.path.splitext(file.lower())[1] in valid_extensions:
                image_files.append(file)
        
        image_files.sort()
        return image_files
    
    def _verify_mask_files(self):
        """Verify that corresponding mask files exist"""
        valid_files = []
        
        for image_file in self.image_files:
            # Try different mask naming conventions
            base_name = os.path.splitext(image_file)[0]
            
            possible_mask_names = [
                f"{base_name}_mask.png",
                f"{base_name}_mask.jpg",
                f"{base_name}.png",
                f"{base_name}.jpg"
            ]
            
            mask_found = False
            for mask_name in possible_mask_names:
                mask_path = os.path.join(self.masks_dir, mask_name)
                if os.path.exists(mask_path):
                    valid_files.append(image_file)
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"Warning: No mask found for image {image_file}")
        
        self.image_files = valid_files
        print(f"Found {len(valid_files)} valid image-mask pairs")
    
    def _setup_transforms(self):
        """Setup transforms based on transform type"""
        if self.transform_type == 'enhanced_train':
            # Enhanced augmentations for training
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.mask_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
            
        elif self.transform_type == 'val':
            # Validation transforms (no augmentation)
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.mask_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
            
        else:  # basic
            # Basic transforms (Windows-compatible, no lambda)
            if self.augment:
                self.image_transforms = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.image_transforms = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            self.mask_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
    
    def _get_mask_path(self, image_file: str) -> str:
        """Get corresponding mask path for an image file"""
        base_name = os.path.splitext(image_file)[0]
        
        possible_mask_names = [
            f"{base_name}_mask.png",
            f"{base_name}_mask.jpg", 
            f"{base_name}.png",
            f"{base_name}.jpg"
        ]
        
        for mask_name in possible_mask_names:
            mask_path = os.path.join(self.masks_dir, mask_name)
            if os.path.exists(mask_path):
                return mask_path
        
        raise FileNotFoundError(f"No mask found for image {image_file}")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, mask) tensors
        """
        # Get file paths
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        mask_path = self._get_mask_path(image_file)
        
        try:
            # Load image and mask
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Apply data augmentation if enabled
            if self.augment and random.random() > 0.5:
                image_np = np.array(image)
                mask_np = np.array(mask)
                
                # Apply random augmentation
                augmented_image, augmented_mask = augment_image(
                    image_np, mask_np, 'random'
                )
                
                image = Image.fromarray(augmented_image.astype(np.uint8))
                mask = Image.fromarray(augmented_mask.astype(np.uint8))
            
            # Apply custom transforms if provided
            if self.custom_transforms:
                image = self.custom_transforms(image)
                mask = self.custom_transforms(mask)
            
            # Preprocess image and mask
            image_tensor = self.preprocessor.preprocess_image(image)
            mask_tensor = self.preprocessor.preprocess_mask(mask)
            
            return image_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error loading item {idx} ({image_file}): {e}")
            # Return a default tensor in case of error
            return torch.zeros(3, *self.image_size), torch.zeros(1, *self.image_size)


class ForgeryDataModule:
    """
    Data module for managing train/validation/test datasets
    """
    
    def __init__(self,
                 train_images_dir: str,
                 train_masks_dir: str,
                 val_images_dir: Optional[str] = None,
                 val_masks_dir: Optional[str] = None,
                 test_images_dir: Optional[str] = None,
                 test_masks_dir: Optional[str] = None,
                 image_size: Tuple[int, int] = (256, 256),
                 batch_size: int = 16,
                 num_workers: int = 4,
                 val_split: float = 0.2):
        """
        Initialize data module
        
        Args:
            train_images_dir: Training images directory
            train_masks_dir: Training masks directory
            val_images_dir: Validation images directory (optional)
            val_masks_dir: Validation masks directory (optional)
            test_images_dir: Test images directory (optional)
            test_masks_dir: Test masks directory (optional)
            image_size: Target image size
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            val_split: Validation split ratio if val dirs not provided
        """
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.val_images_dir = val_images_dir
        self.val_masks_dir = val_masks_dir
        self.test_images_dir = test_images_dir
        self.test_masks_dir = test_masks_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        
        # Create training dataset
        if os.path.exists(self.train_images_dir) and os.path.exists(self.train_masks_dir):
            full_dataset = ForgeryDetectionDataset(
                images_dir=self.train_images_dir,
                masks_dir=self.train_masks_dir,
                image_size=self.image_size,
                augment=True
            )
            
            # Split into train and validation if validation dirs not provided
            if self.val_images_dir is None or self.val_masks_dir is None:
                dataset_size = len(full_dataset)
                val_size = int(self.val_split * dataset_size)
                train_size = dataset_size - val_size
                
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size]
                )
                
                # Disable augmentation for validation set
                if hasattr(self.val_dataset, 'dataset'):
                    self.val_dataset.dataset.augment = False
            else:
                self.train_dataset = full_dataset
        
        # Create validation dataset if separate directories provided
        if (self.val_images_dir and self.val_masks_dir and 
            os.path.exists(self.val_images_dir) and os.path.exists(self.val_masks_dir)):
            
            self.val_dataset = ForgeryDetectionDataset(
                images_dir=self.val_images_dir,
                masks_dir=self.val_masks_dir,
                image_size=self.image_size,
                augment=False  # No augmentation for validation
            )
        
        # Create test dataset if directories provided
        if (self.test_images_dir and self.test_masks_dir and
            os.path.exists(self.test_images_dir) and os.path.exists(self.test_masks_dir)):
            
            self.test_dataset = ForgeryDetectionDataset(
                images_dir=self.test_images_dir,
                masks_dir=self.test_masks_dir,
                image_size=self.image_size,
                augment=False  # No augmentation for testing
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation data loader"""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test data loader"""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_dataset_info(self) -> dict:
        """Get information about the datasets"""
        info = {}
        
        if self.train_dataset:
            info['train_size'] = len(self.train_dataset)
        
        if self.val_dataset:
            info['val_size'] = len(self.val_dataset)
        
        if self.test_dataset:
            info['test_size'] = len(self.test_dataset)
        
        info['image_size'] = self.image_size
        info['batch_size'] = self.batch_size
        
        return info


def create_sample_data(images_dir: str, masks_dir: str, num_samples: int = 5):
    """
    Create sample data for testing (generates random images and masks)
    
    Args:
        images_dir: Directory to save sample images
        masks_dir: Directory to save sample masks
        num_samples: Number of samples to create
    """
    import os
    
    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create random image
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Create random mask with some structure
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Add some random forgery regions
        num_regions = np.random.randint(1, 4)
        for _ in range(num_regions):
            x = np.random.randint(0, 200)
            y = np.random.randint(0, 200)
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 80)
            mask[y:y+h, x:x+w] = 255
        
        # Save image and mask
        image_path = os.path.join(images_dir, f"sample_{i:03d}.png")
        mask_path = os.path.join(masks_dir, f"sample_{i:03d}_mask.png")
        
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
    
    print(f"Created {num_samples} sample images in {images_dir}")
    print(f"Created {num_samples} sample masks in {masks_dir}")


if __name__ == "__main__":
    # Test the dataset
    
    # Create sample data for testing
    sample_images_dir = "dataset/images"
    sample_masks_dir = "dataset/masks"
    
    create_sample_data(sample_images_dir, sample_masks_dir, num_samples=10)
    
    # Test dataset
    try:
        dataset = ForgeryDetectionDataset(
            images_dir=sample_images_dir,
            masks_dir=sample_masks_dir,
            image_size=(256, 256),
            augment=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Test data module
        data_module = ForgeryDataModule(
            train_images_dir=sample_images_dir,
            train_masks_dir=sample_masks_dir,
            batch_size=4,
            val_split=0.3
        )
        
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Validation batches: {len(val_loader)}")
        
        # Test loading a batch
        for batch_idx, (images, masks) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images {images.shape}, Masks {masks.shape}")
            if batch_idx == 0:  # Only test first batch
                break
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()


# Alias for compatibility
ImageForgeryDataset = ForgeryDetectionDataset