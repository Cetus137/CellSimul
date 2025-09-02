"""
Unpaired Conditional Data Loading Module for GAN Training

This module contains dataset classes for loading unpaired fluorescent images and distance masks
for conditional GAN training where the conditioning masks are synthetic and unpaired with real data.
"""

import torch
import numpy as np
import os
import glob
from PIL import Image
import tifffile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class UnpairedConditionalImageDataset(Dataset):
    """
    Dataset class for loading unpaired fluorescent images and distance masks for conditional GAN training.
    This is useful when you have real fluorescent images but synthetic conditioning masks that are not
    paired with specific fluorescent images.
    """
    
    def __init__(self, fluorescent_dir, mask_dir, image_size=256, transform=None, max_images=None):
        """
        Initialize the unpaired conditional dataset
        
        Args:
            fluorescent_dir (str): Directory containing real fluorescent TIF images
            mask_dir (str): Directory containing synthetic distance mask TIF images (unpaired)
            image_size (int): Target size for images (assumed square)
            transform: Optional transforms to apply
            max_images (int, optional): Maximum number of images to use for training. If None, use all available images.
        """
        self.fluorescent_dir = fluorescent_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        self.max_images = max_images
        
        # Find all TIF files in fluorescent directory
        self.fluorescent_paths = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            self.fluorescent_paths.extend(glob.glob(os.path.join(fluorescent_dir, ext)))
            self.fluorescent_paths.extend(glob.glob(os.path.join(fluorescent_dir, '**', ext), recursive=True))
        
        # Find all TIF files in mask directory
        self.mask_paths = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            self.mask_paths.extend(glob.glob(os.path.join(mask_dir, ext)))
            self.mask_paths.extend(glob.glob(os.path.join(mask_dir, '**', ext), recursive=True))
        
        if not self.fluorescent_paths:
            raise ValueError(f"No fluorescent TIF files found in {fluorescent_dir}")
        
        if not self.mask_paths:
            raise ValueError(f"No mask TIF files found in {mask_dir}")
        
        # Sort paths for consistency
        self.fluorescent_paths.sort()
        self.mask_paths.sort()
        
        # Limit the number of images if max_images is specified
        if self.max_images is not None:
            self.fluorescent_paths = self.fluorescent_paths[:self.max_images]
            self.mask_paths = self.mask_paths[:self.max_images]
        
        print(f"Found {len(self.fluorescent_paths)} fluorescent images")
        print(f"Found {len(self.mask_paths)} distance masks")
        if self.max_images is not None:
            print(f"Limited to {self.max_images} images as requested")
        print("Note: Images and masks are unpaired - will be randomly combined during training")
        
        # Set up transforms
        if transform is None:
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
        # Use the larger of the two datasets as the epoch length
        return max(len(self.fluorescent_paths), len(self.mask_paths))
    
    def __getitem__(self, idx):
        """
        Load and process an unpaired fluorescent image and distance mask
        
        Args:
            idx (int): Index for sampling
            
        Returns:
            tuple: (fluorescent_image, distance_mask) both as torch tensors
        """
        # Sample fluorescent image (cycle through if necessary)
        fluorescent_idx = idx % len(self.fluorescent_paths)
        fluorescent_path = self.fluorescent_paths[fluorescent_idx]
        
        # Sample distance mask randomly (unpaired)
        mask_idx = random.randint(0, len(self.mask_paths) - 1)
        mask_path = self.mask_paths[mask_idx]
        
        try:
            # Load fluorescent image
            fluorescent_image = self._load_image(fluorescent_path)
            fluorescent_tensor = self.fluorescent_transform(fluorescent_image)
            
            # Load distance mask
            mask_image = self._load_image(mask_path)
            mask_tensor = self.mask_transform(mask_image)
            
            # Masks are now normalized to [-1, 1] range like fluorescent images
            # Keep them in this range for consistency
            
            return fluorescent_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error loading images {fluorescent_path}, {mask_path}: {e}")
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
            image = image[:, :, 0]  # Take first channel
        
        # Convert to PIL Image for transforms
        # Handle different data types and ranges
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume already normalized, convert to 0-255 range for PIL
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            # Integer types, ensure uint8
            image = image.astype(np.uint8)
        
        return Image.fromarray(image)
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        return {
            'fluorescent_dir': self.fluorescent_dir,
            'mask_dir': self.mask_dir,
            'num_fluorescent': len(self.fluorescent_paths),
            'num_masks': len(self.mask_paths),
            'image_size': self.image_size,
            'dataset_length': len(self),
            'pairing': 'unpaired'
        }
    
    def get_sample_pairs(self, num_samples=4):
        """
        Get sample pairs for visualization (randomly paired)
        
        Args:
            num_samples (int): Number of sample pairs to return
            
        Returns:
            tuple: (fluorescent_samples, mask_samples)
        """
        fluorescent_samples = []
        mask_samples = []
        
        for i in range(num_samples):
            fluorescent, mask = self[i]
            fluorescent_samples.append(fluorescent.unsqueeze(0))
            mask_samples.append(mask.unsqueeze(0))
        
        return fluorescent_samples, mask_samples
    
    def visualize_unpaired_samples(self, num_samples=8, save_path=None):
        """
        Visualize unpaired fluorescent images and distance masks
        
        Args:
            num_samples (int): Number of samples to visualize
            save_path (str): Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        
        for i in range(num_samples):
            fluorescent, mask = self[i]
            
            # Convert tensors back to numpy for visualization
            fluorescent_img = fluorescent.squeeze().numpy()
            fluorescent_img = (fluorescent_img + 1) / 2  # Denormalize from [-1,1] to [0,1]
            
            mask_img = mask.squeeze().numpy()
            
            # Plot fluorescent image
            axes[0, i].imshow(fluorescent_img, cmap='green')
            axes[0, i].set_title(f'Fluorescent {i+1}')
            axes[0, i].axis('off')
            
            # Plot distance mask
            axes[1, i].imshow(mask_img, cmap='hot')
            axes[1, i].set_title(f'Distance Mask {i+1}')
            axes[1, i].axis('off')
            
            # Plot overlay
            overlay = np.stack([mask_img, fluorescent_img, np.zeros_like(mask_img)], axis=-1)
            axes[2, i].imshow(overlay)
            axes[2, i].set_title(f'Overlay {i+1}')
            axes[2, i].axis('off')
        
        plt.suptitle('Unpaired Fluorescent Images and Distance Masks', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Unpaired samples visualization saved to {save_path}")
        
        plt.show()


def test_unpaired_dataset():
    """Test function for unpaired conditional dataset"""
    print("Testing Unpaired Conditional Dataset...")
    
    # Test with the actual data directories
    fluorescent_dir = "/Users/edwheeler/cond_GAN/CellSimul/CellSimul/data/fluorescence_rescaled"
    mask_dir = "/Users/edwheeler/cond_GAN/CellSimul/CellSimul/data/distance_masks_rescaled"
    
    try:
        dataset = UnpairedConditionalImageDataset(
            fluorescent_dir=fluorescent_dir,
            mask_dir=mask_dir,
            image_size=256
        )
        
        # Get dataset info
        info = dataset.get_dataset_info()
        print("Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test loading a few samples
        print(f"\nTesting sample loading...")
        for i in range(3):
            fluorescent, mask = dataset[i]
            print(f"Sample {i+1}: Fluorescent shape {fluorescent.shape}, Mask shape {mask.shape}")
            print(f"  Fluorescent range: [{fluorescent.min():.3f}, {fluorescent.max():.3f}]")
            print(f"  Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Test visualization
        print(f"\nGenerating visualization...")
        dataset.visualize_unpaired_samples(num_samples=4, save_path='unpaired_samples_preview.png')
        
        print("✓ Unpaired dataset test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error testing unpaired dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_unpaired_dataset()
