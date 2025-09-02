"""
Conditional Data Loading Module for GAN Training

This module contains dataset classes for loading paired fluorescent images and binary masks
for conditional GAN training.
"""

import torch
import numpy as np
import os
import glob
from PIL import Image
import tifffile
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ConditionalImageDataset(Dataset):
    """
    Dataset class for loading paired fluorescent images and binary masks for conditional GAN training
    """
    
    def __init__(self, fluorescent_dir, mask_dir, image_size=256, transform=None):
        """
        Initialize the conditional dataset
        
        Args:
            fluorescent_dir (str): Directory containing fluorescent TIF files
            mask_dir (str): Directory containing binary mask TIF files
            image_size (int): Target size for images (assumed square)
            transform: Optional transforms to apply
        """
        self.fluorescent_dir = fluorescent_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        
        # Find all TIF files in fluorescent directory
        self.fluorescent_paths = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            self.fluorescent_paths.extend(glob.glob(os.path.join(fluorescent_dir, ext)))
            # Also search in subdirectories
            self.fluorescent_paths.extend(glob.glob(os.path.join(fluorescent_dir, '**', ext), recursive=True))
        
        # Find corresponding mask files
        self.mask_paths = []
        self.valid_pairs = []
        
        for fluor_path in self.fluorescent_paths:
            # Extract filename without extension
            fluor_basename = os.path.splitext(os.path.basename(fluor_path))[0]
            
            # Look for corresponding mask file
            mask_candidates = []
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                # Try exact filename match
                mask_pattern = os.path.join(mask_dir, fluor_basename + ext[1:])  # Remove * from pattern
                if os.path.exists(mask_pattern):
                    mask_candidates.append(mask_pattern)
                
                # Try with common mask suffixes
                for suffix in ['_mask', '_binary', '_seg', '_segmentation']:
                    mask_pattern = os.path.join(mask_dir, fluor_basename + suffix + ext[1:])
                    if os.path.exists(mask_pattern):
                        mask_candidates.append(mask_pattern)
            
            # Use first found mask
            if mask_candidates:
                self.valid_pairs.append((fluor_path, mask_candidates[0]))
        
        if len(self.valid_pairs) == 0:
            raise ValueError(f"No paired fluorescent/mask files found in {fluorescent_dir} and {mask_dir}")
        
        print(f"Found {len(self.valid_pairs)} paired fluorescent/mask images")
        
        # Default transforms if none provided
        if self.transform is None:
            self.fluorescent_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
            
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] like fluorescent images
            ])
        else:
            self.fluorescent_transform = transform
            self.mask_transform = transform
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        """
        Load and process a paired fluorescent image and mask
        
        Args:
            idx (int): Index of image pair to load
            
        Returns:
            tuple: (fluorescent_image, binary_mask) both as torch tensors
        """
        fluorescent_path, mask_path = self.valid_pairs[idx]
        
        try:
            # Load fluorescent image
            fluorescent_image = self._load_image(fluorescent_path)
            fluorescent_tensor = self.fluorescent_transform(fluorescent_image)
            
            # Load mask image
            mask_image = self._load_image(mask_path)
            mask_tensor = self.mask_transform(mask_image)
            
            # Ensure mask is binary (-1 or 1) since we normalized to [-1, 1]
            # Convert back to [0,1], threshold, then back to [-1,1]
            mask_binary = ((mask_tensor + 1.0) / 2.0 > 0.5).float()  # Convert to [0,1], threshold
            mask_tensor = mask_binary * 2.0 - 1.0  # Convert back to [-1,1]
            
            return fluorescent_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error loading image pair {fluorescent_path}, {mask_path}: {e}")
            # Return blank images if loading fails
            blank_fluorescent = torch.zeros(1, self.image_size, self.image_size)
            blank_mask = torch.full((1, self.image_size, self.image_size), -1.0)  # -1 for background in [-1,1] range
            return blank_fluorescent, blank_mask
    
    def _load_image(self, image_path):
        """
        Helper method to load an image from file
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            PIL.Image: Loaded image
        """
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
            if image.max() > image.min():
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(image.astype(np.uint8))
        
        return image
    
    def get_sample_pairs(self, num_samples=8):
        """
        Get a few sample image pairs for visualization
        
        Args:
            num_samples (int): Number of sample pairs to return
            
        Returns:
            tuple: (fluorescent_samples, mask_samples) as lists of tensors
        """
        fluorescent_samples = []
        mask_samples = []
        
        for i in range(min(num_samples, len(self.valid_pairs))):
            fluorescent, mask = self[i]
            fluorescent_samples.append(fluorescent.unsqueeze(0))
            mask_samples.append(mask.unsqueeze(0))
        
        return fluorescent_samples, mask_samples
    
    def get_dataset_info(self):
        """
        Get information about the loaded dataset
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        info = {
            'num_pairs': len(self.valid_pairs),
            'fluorescent_directory': self.fluorescent_dir,
            'mask_directory': self.mask_dir,
            'target_size': (self.image_size, self.image_size),
            'sample_pairs': self.valid_pairs[:5] if len(self.valid_pairs) >= 5 else self.valid_pairs
        }
        return info


class SingleDirectoryConditionalDataset(Dataset):
    """
    Dataset for when fluorescent images and masks are in the same directory
    with different naming conventions
    """
    
    def __init__(self, data_dir, fluorescent_pattern='*_fluor*', mask_pattern='*_mask*', image_size=256):
        """
        Initialize dataset from single directory with naming patterns
        
        Args:
            data_dir (str): Directory containing both fluorescent and mask files
            fluorescent_pattern (str): Glob pattern for fluorescent files
            mask_pattern (str): Glob pattern for mask files
            image_size (int): Target image size
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Find fluorescent files
        fluorescent_files = []
        for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
            pattern = fluorescent_pattern.replace('*', '*') + ext
            fluorescent_files.extend(glob.glob(os.path.join(data_dir, pattern)))
        
        # Find mask files
        mask_files = []
        for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
            pattern = mask_pattern.replace('*', '*') + ext
            mask_files.extend(glob.glob(os.path.join(data_dir, pattern)))
        
        # Create pairs based on common base names
        self.valid_pairs = []
        for fluor_path in fluorescent_files:
            fluor_base = self._extract_base_name(fluor_path, fluorescent_pattern)
            
            for mask_path in mask_files:
                mask_base = self._extract_base_name(mask_path, mask_pattern)
                
                if fluor_base == mask_base:
                    self.valid_pairs.append((fluor_path, mask_path))
                    break
        
        print(f"Found {len(self.valid_pairs)} paired images in {data_dir}")
        
        # Set up transforms
        self.fluorescent_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] like fluorescent images
        ])
    
    def _extract_base_name(self, filepath, pattern):
        """Extract base name from filepath given a pattern"""
        basename = os.path.basename(filepath)
        # Simple extraction - remove extension and pattern-specific parts
        base = os.path.splitext(basename)[0]
        # Remove common suffixes
        for suffix in ['_fluor', '_fluorescent', '_mask', '_binary', '_seg']:
            base = base.replace(suffix, '')
        return base
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        fluorescent_path, mask_path = self.valid_pairs[idx]
        
        try:
            # Load and transform images (reuse logic from ConditionalImageDataset)
            dataset = ConditionalImageDataset('.', '.', image_size=self.image_size)
            fluorescent_image = dataset._load_image(fluorescent_path)
            mask_image = dataset._load_image(mask_path)
            
            fluorescent_tensor = self.fluorescent_transform(fluorescent_image)
            mask_tensor = self.mask_transform(mask_image)
            
            # Ensure mask is binary (-1 or 1) since we normalized to [-1, 1]
            # Convert back to [0,1], threshold, then back to [-1,1]
            mask_binary = ((mask_tensor + 1.0) / 2.0 > 0.5).float()  # Convert to [0,1], threshold
            mask_tensor = mask_binary * 2.0 - 1.0  # Convert back to [-1,1]
            
            return fluorescent_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error loading pair {fluorescent_path}, {mask_path}: {e}")
            blank_fluorescent = torch.zeros(1, self.image_size, self.image_size)
            blank_mask = torch.full((1, self.image_size, self.image_size), -1.0)  # -1 for background in [-1,1] range
            return blank_fluorescent, blank_mask
