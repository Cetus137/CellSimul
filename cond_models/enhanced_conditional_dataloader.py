"""
Enhanced Conditional Data Loading Module with Distance Transform Support

This module contains dataset classes for loading paired fluorescent images and masks
with support for distance transform conditioning.
"""

import torch
import numpy as np
import os
import glob
from PIL import Image
import tifffile
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from distance_transform_utils import DistanceTransformProcessor


class EnhancedConditionalImageDataset(Dataset):
    """
    Enhanced dataset class with distance transform support for conditional GAN training
    """
    
    def __init__(self, fluorescent_dir, mask_dir, image_size=256, transform=None, 
                 conditioning_type='binary', distance_config=None):
        """
        Initialize the enhanced conditional dataset
        
        Args:
            fluorescent_dir (str): Directory containing fluorescent TIF files
            mask_dir (str): Directory containing binary mask TIF files
            image_size (int): Target size for images (assumed square)
            transform: Optional transforms to apply
            conditioning_type (str): Type of conditioning:
                - 'binary': Use binary masks (0/1)
                - 'distance': Use distance transform
                - 'signed_distance': Use signed distance transform
                - 'gradient_distance': Use gradient distance (fluorescent-like)
                - 'multi_scale': Use multi-scale distance features
            distance_config (dict): Configuration for distance transforms
        """
        self.fluorescent_dir = fluorescent_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.conditioning_type = conditioning_type
        
        # Default distance transform configuration
        default_distance_config = {
            'max_distance': 50,
            'normalize_method': 'sigmoid',
            'scales': [1, 2, 4]  # For multi-scale
        }
        self.distance_config = distance_config or default_distance_config
        
        # Initialize distance transform processor if needed
        if conditioning_type != 'binary':
            self.distance_processor = DistanceTransformProcessor(
                max_distance=self.distance_config['max_distance'],
                normalize_method=self.distance_config['normalize_method']
            )
        
        # Find all TIF files in fluorescent directory
        self.fluorescent_paths = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            self.fluorescent_paths.extend(glob.glob(os.path.join(fluorescent_dir, ext)))
            self.fluorescent_paths.extend(glob.glob(os.path.join(fluorescent_dir, '**', ext), recursive=True))
        
        # Find corresponding mask files and create valid pairs
        self.mask_paths = []
        self.valid_pairs = []
        
        for fluor_path in self.fluorescent_paths:
            # Extract filename without extension
            fluor_filename = os.path.splitext(os.path.basename(fluor_path))[0]
            
            # Try different naming conventions for mask files
            possible_mask_names = [
                f"{fluor_filename}_mask",
                f"{fluor_filename}_seg",
                f"{fluor_filename}_binary",
                f"mask_{fluor_filename}",
                f"seg_{fluor_filename}",
                fluor_filename  # Same name, different directory
            ]
            
            mask_found = False
            for mask_name in possible_mask_names:
                for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                    mask_path = os.path.join(mask_dir, mask_name + ext)
                    if os.path.exists(mask_path):
                        self.valid_pairs.append((fluor_path, mask_path))
                        mask_found = True
                        break
                if mask_found:
                    break
        
        if not self.valid_pairs:
            raise ValueError(f"No valid fluorescent-mask pairs found in {fluorescent_dir} and {mask_dir}")
        
        print(f"Found {len(self.valid_pairs)} valid fluorescent-mask pairs")
        print(f"Conditioning type: {conditioning_type}")
        
        # Set up transforms
        if transform is None:
            self.fluorescent_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.fluorescent_transform = transform
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        """
        Load and process a paired fluorescent image and conditioning input
        
        Args:
            idx (int): Index of image pair to load
            
        Returns:
            tuple: (fluorescent_image, conditioning_input) both as torch tensors
        """
        fluorescent_path, mask_path = self.valid_pairs[idx]
        
        try:
            # Load fluorescent image
            fluorescent_image = self._load_image(fluorescent_path)
            fluorescent_tensor = self.fluorescent_transform(fluorescent_image)
            
            # Load and process mask based on conditioning type
            mask_image = self._load_image(mask_path)
            conditioning_tensor = self._process_conditioning_input(mask_image)
            
            return fluorescent_tensor, conditioning_tensor
            
        except Exception as e:
            print(f"Error loading image pair {fluorescent_path}, {mask_path}: {e}")
            # Return blank images if loading fails
            blank_fluorescent = torch.zeros(1, self.image_size, self.image_size)
            
            if self.conditioning_type == 'multi_scale':
                # Multi-scale has multiple channels
                scales = self.distance_config.get('scales', [1, 2, 4])
                blank_conditioning = torch.zeros(len(scales), self.image_size, self.image_size)
            else:
                blank_conditioning = torch.zeros(1, self.image_size, self.image_size)
            
            return blank_fluorescent, blank_conditioning
    
    def _process_conditioning_input(self, mask_image):
        """
        Process mask image based on conditioning type
        
        Args:
            mask_image (PIL.Image): Raw mask image
            
        Returns:
            torch.Tensor: Processed conditioning input
        """
        # Convert to numpy and resize
        mask_resized = mask_image.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_array = np.array(mask_resized)
        
        # Handle different image formats
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]  # Take first channel
        
        # Ensure binary (0 or 1)
        binary_mask = (mask_array > 128).astype(np.uint8)
        
        if self.conditioning_type == 'binary':
            # Standard binary conditioning
            conditioning_array = binary_mask.astype(np.float32)
            conditioning_tensor = torch.from_numpy(conditioning_array).unsqueeze(0)
            
        elif self.conditioning_type == 'distance':
            # Distance transform conditioning
            conditioning_array = self.distance_processor.binary_to_distance_transform(binary_mask)
            conditioning_tensor = torch.from_numpy(conditioning_array).unsqueeze(0)
            
        elif self.conditioning_type == 'signed_distance':
            # Signed distance transform conditioning
            conditioning_array = self.distance_processor.binary_to_signed_distance_transform(binary_mask)
            conditioning_tensor = torch.from_numpy(conditioning_array).unsqueeze(0)
            
        elif self.conditioning_type == 'gradient_distance':
            # Gradient distance transform (fluorescent-like)
            conditioning_array = self.distance_processor.binary_to_gradient_distance_transform(binary_mask)
            conditioning_tensor = torch.from_numpy(conditioning_array).unsqueeze(0)
            
        elif self.conditioning_type == 'multi_scale':
            # Multi-scale distance features
            scales = self.distance_config.get('scales', [1, 2, 4])
            conditioning_array = self.distance_processor.binary_to_multi_scale_distance(binary_mask, scales)
            # conditioning_array is (H, W, C), need (C, H, W)
            conditioning_tensor = torch.from_numpy(conditioning_array).permute(2, 0, 1)
            
        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning_type}")
        
        return conditioning_tensor
    
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
            image_array = tifffile.imread(image_path)
            # Convert numpy array to PIL Image
            if len(image_array.shape) == 3:
                image_array = image_array[:, :, 0]  # Take first channel
            image = Image.fromarray(image_array)
        else:
            image = Image.open(image_path)
        
        return image
    
    def get_conditioning_info(self):
        """Get information about the conditioning setup"""
        info = {
            'conditioning_type': self.conditioning_type,
            'image_size': self.image_size,
            'num_pairs': len(self.valid_pairs),
            'distance_config': self.distance_config if self.conditioning_type != 'binary' else None
        }
        
        # Get conditioning tensor shape
        sample_conditioning = self[0][1]
        info['conditioning_shape'] = list(sample_conditioning.shape)
        info['conditioning_channels'] = sample_conditioning.shape[0]
        
        return info
    
    def get_sample_pairs(self, num_samples=4):
        """
        Get sample pairs for visualization
        
        Args:
            num_samples (int): Number of sample pairs to return
            
        Returns:
            tuple: (fluorescent_samples, conditioning_samples)
        """
        fluorescent_samples = []
        conditioning_samples = []
        
        indices = np.linspace(0, len(self) - 1, num_samples, dtype=int)
        
        for idx in indices:
            fluorescent, conditioning = self[idx]
            fluorescent_samples.append(fluorescent.unsqueeze(0))
            conditioning_samples.append(conditioning.unsqueeze(0))
        
        return fluorescent_samples, conditioning_samples
    
    def visualize_conditioning_types(self, idx=0, save_path=None):
        """
        Visualize different conditioning types for comparison
        
        Args:
            idx (int): Index of sample to visualize
            save_path (str): Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Get the raw mask for processing
        fluorescent_path, mask_path = self.valid_pairs[idx]
        mask_image = self._load_image(mask_path)
        mask_resized = mask_image.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_array = np.array(mask_resized)
        
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        binary_mask = (mask_array > 128).astype(np.uint8)
        
        # Create distance processor for visualization
        processor = DistanceTransformProcessor(
            max_distance=self.distance_config['max_distance'],
            normalize_method=self.distance_config['normalize_method']
        )
        
        # Generate all conditioning types
        binary_cond = binary_mask.astype(np.float32)
        distance_cond = processor.binary_to_distance_transform(binary_mask)
        signed_distance_cond = processor.binary_to_signed_distance_transform(binary_mask)
        gradient_distance_cond = processor.binary_to_gradient_distance_transform(binary_mask)
        
        # Load fluorescent image for comparison
        fluorescent_image = self._load_image(fluorescent_path)
        fluorescent_resized = fluorescent_image.resize((self.image_size, self.image_size))
        fluorescent_array = np.array(fluorescent_resized)
        if len(fluorescent_array.shape) == 3:
            fluorescent_array = fluorescent_array[:, :, 0]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Fluorescent image
        axes[0, 0].imshow(fluorescent_array, cmap='green')
        axes[0, 0].set_title('Original Fluorescent Image')
        axes[0, 0].axis('off')
        
        # Binary conditioning
        axes[0, 1].imshow(binary_cond, cmap='gray')
        axes[0, 1].set_title('Binary Conditioning')
        axes[0, 1].axis('off')
        
        # Distance conditioning
        im1 = axes[0, 2].imshow(distance_cond, cmap='viridis')
        axes[0, 2].set_title('Distance Transform Conditioning')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
        
        # Signed distance conditioning
        im2 = axes[1, 0].imshow(signed_distance_cond, cmap='RdBu_r')
        axes[1, 0].set_title('Signed Distance Conditioning')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
        
        # Gradient distance conditioning
        im3 = axes[1, 1].imshow(gradient_distance_cond, cmap='hot')
        axes[1, 1].set_title('Gradient Distance Conditioning\n(Fluorescent-like)')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
        
        # Overlay comparison
        axes[1, 2].imshow(fluorescent_array, cmap='green', alpha=0.7)
        axes[1, 2].imshow(gradient_distance_cond, cmap='hot', alpha=0.4)
        axes[1, 2].set_title('Fluorescent + Gradient Overlay')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Conditioning Type Comparison - Sample {idx}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Conditioning comparison saved to {save_path}")
        
        plt.show()


def test_enhanced_dataset():
    """Test function for enhanced conditional dataset"""
    print("Testing Enhanced Conditional Dataset...")
    
    # Test different conditioning types
    conditioning_types = ['binary', 'distance', 'signed_distance', 'gradient_distance', 'multi_scale']
    
    # Mock directories (would need actual data in practice)
    fluorescent_dir = "../data/fluorescent"
    mask_dir = "../data/masks"
    
    for cond_type in conditioning_types:
        print(f"\nTesting conditioning type: {cond_type}")
        
        try:
            # Different config for multi-scale
            if cond_type == 'multi_scale':
                distance_config = {'max_distance': 30, 'normalize_method': 'sigmoid', 'scales': [1, 3, 5]}
            else:
                distance_config = {'max_distance': 40, 'normalize_method': 'sigmoid'}
            
            dataset = EnhancedConditionalImageDataset(
                fluorescent_dir=fluorescent_dir,
                mask_dir=mask_dir,
                image_size=128,  # Smaller for testing
                conditioning_type=cond_type,
                distance_config=distance_config
            )
            
            # Get conditioning info
            info = dataset.get_conditioning_info()
            print(f"  Conditioning shape: {info['conditioning_shape']}")
            print(f"  Conditioning channels: {info['conditioning_channels']}")
            
            # Test loading a sample (would work with real data)
            # fluorescent, conditioning = dataset[0]
            # print(f"  Sample shapes: Fluorescent {fluorescent.shape}, Conditioning {conditioning.shape}")
            
            print(f"  ✓ {cond_type} conditioning configured successfully")
            
        except Exception as e:
            print(f"  ✗ Error with {cond_type}: {e}")
    
    print("\n✓ Enhanced dataset testing completed!")


if __name__ == "__main__":
    test_enhanced_dataset()
