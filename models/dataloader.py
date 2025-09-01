"""
Data Loading Module for GAN Training

This module contains the TIFImageDataset class for loading and preprocessing
TIF images for GAN training.
"""

import torch
import numpy as np
import os
import glob
from PIL import Image
import tifffile
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TIFImageDataset(Dataset):
    """
    Dataset class for loading TIF images for GAN training
    """
    
    def __init__(self, data_dir, image_size=256, transform=None):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Directory containing TIF files
            image_size (int): Target size for images (assumed square)
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        
        # Find all TIF files in the directory
        self.image_paths = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
            # Also search in subdirectories
            self.image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No TIF files found in {data_dir}")
        
        print(f"Found {len(self.image_paths)} TIF images in {data_dir}")
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and process a TIF image
        
        Args:
            idx (int): Index of image to load
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load TIF image
            if image_path.lower().endswith(('.tif', '.tiff')):
                # Use tifffile for better TIF support
                image = tifffile.imread(image_path)
            else:
                # Fallback to PIL
                image = Image.open(image_path)
                image = np.array(image)
            
            # Handle different image formats
            if len(image.shape) == 3:
                # If RGB, convert to grayscale
                if image.shape[2] == 3:
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                elif image.shape[2] == 1:
                    image = image.squeeze(2)
            
            # Normalize to 0-255 range if needed
            if image.dtype != np.uint8:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image for transforms
            image = Image.fromarray(image.astype(np.uint8))
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            blank_image = torch.zeros(1, self.image_size, self.image_size)
            return blank_image
    
    def get_sample_images(self, num_samples=8):
        """
        Get a few sample images for visualization
        
        Args:
            num_samples (int): Number of sample images to return
            
        Returns:
            list: List of sample image tensors
        """
        samples = []
        for i in range(min(num_samples, len(self.image_paths))):
            sample = self[i]
            samples.append(sample.unsqueeze(0))
        return samples
    
    def get_image_info(self):
        """
        Get information about the loaded images
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        info = {
            'num_images': len(self.image_paths),
            'data_directory': self.data_dir,
            'target_size': (self.image_size, self.image_size),
            'sample_paths': self.image_paths[:5] if len(self.image_paths) >= 5 else self.image_paths
        }
        return info
