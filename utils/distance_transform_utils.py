"""
Distance Transform Utilities for Conditional GAN

This module provides utilities for converting binary masks to distance transforms
and other preprocessing operations for enhanced conditional GAN training.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class DistanceTransformProcessor:
    """
    Processor for converting binary masks to various distance transform representations
    """
    
    def __init__(self, max_distance=50, normalize_method='minmax'):
        """
        Initialize the distance transform processor
        
        Args:
            max_distance (float): Maximum distance to consider (clips beyond this)
            normalize_method (str): Method for normalization ('minmax', 'sigmoid', 'tanh')
        """
        self.max_distance = max_distance
        self.normalize_method = normalize_method
    
    def binary_to_distance_transform(self, binary_mask):
        """
        Convert binary mask to distance transform
        
        Args:
            binary_mask (np.ndarray): Binary mask (0s and 1s)
            
        Returns:
            np.ndarray: Distance transform
        """
        # Ensure binary mask is boolean
        binary_mask = binary_mask.astype(bool)
        
        # Compute distance transform (distance to nearest boundary)
        distance_transform = ndimage.distance_transform_edt(~binary_mask)
        
        # Clip to max distance
        distance_transform = np.clip(distance_transform, 0, self.max_distance)
        
        # Normalize
        distance_transform = self._normalize_distance(distance_transform)
        
        return distance_transform.astype(np.float32)
    
    def binary_to_signed_distance_transform(self, binary_mask):
        """
        Convert binary mask to signed distance transform
        (positive inside, negative outside)
        
        Args:
            binary_mask (np.ndarray): Binary mask (0s and 1s)
            
        Returns:
            np.ndarray: Signed distance transform
        """
        binary_mask = binary_mask.astype(bool)
        
        # Distance inside (positive)
        dist_inside = ndimage.distance_transform_edt(binary_mask)
        
        # Distance outside (negative)
        dist_outside = ndimage.distance_transform_edt(~binary_mask)
        
        # Combine into signed distance
        signed_distance = dist_inside - dist_outside
        
        # Clip to max distance
        signed_distance = np.clip(signed_distance, -self.max_distance, self.max_distance)
        
        # Normalize to [-1, 1]
        signed_distance = signed_distance / self.max_distance
        
        return signed_distance.astype(np.float32)
    
    def binary_to_gradient_distance_transform(self, binary_mask):
        """
        Convert binary mask to gradient-based distance transform
        (smoother falloff, good for fluorescent intensity modeling)
        
        Args:
            binary_mask (np.ndarray): Binary mask (0s and 1s)
            
        Returns:
            np.ndarray: Gradient distance transform
        """
        binary_mask = binary_mask.astype(bool)
        
        # Standard distance transform
        distance = ndimage.distance_transform_edt(~binary_mask)
        
        # Apply exponential decay for fluorescent-like falloff
        # Intensity = exp(-distance / decay_factor)
        decay_factor = self.max_distance / 3  # Adjustable decay rate
        gradient_distance = np.exp(-distance / decay_factor)
        
        # Ensure membrane pixels have maximum intensity
        gradient_distance[binary_mask] = 1.0
        
        return gradient_distance.astype(np.float32)
    
    def binary_to_multi_scale_distance(self, binary_mask, scales=[1, 2, 4]):
        """
        Convert binary mask to multi-scale distance features
        
        Args:
            binary_mask (np.ndarray): Binary mask (0s and 1s)
            scales (list): Different scales for distance computation
            
        Returns:
            np.ndarray: Multi-scale distance features (H, W, len(scales))
        """
        binary_mask = binary_mask.astype(bool)
        
        multi_scale_features = []
        
        for scale in scales:
            # Dilate mask for different scales
            dilated_mask = ndimage.binary_dilation(binary_mask, iterations=scale)
            
            # Compute distance transform
            distance = ndimage.distance_transform_edt(~dilated_mask)
            distance = np.clip(distance, 0, self.max_distance)
            distance = self._normalize_distance(distance)
            
            multi_scale_features.append(distance)
        
        # Stack along channel dimension
        return np.stack(multi_scale_features, axis=-1).astype(np.float32)
    
    def _normalize_distance(self, distance):
        """Normalize distance transform based on chosen method"""
        if self.normalize_method == 'minmax':
            # Normalize to [0, 1]
            if distance.max() > distance.min():
                return (distance - distance.min()) / (distance.max() - distance.min())
            else:
                return distance
        
        elif self.normalize_method == 'sigmoid':
            # Sigmoid normalization for smooth falloff
            return 1 / (1 + np.exp(-distance + self.max_distance/2))
        
        elif self.normalize_method == 'tanh':
            # Tanh normalization
            return np.tanh(distance / (self.max_distance/2))
        
        else:
            return distance / self.max_distance  # Simple division
    
    def visualize_transforms(self, binary_mask, save_path=None):
        """
        Visualize different distance transform representations
        
        Args:
            binary_mask (np.ndarray): Input binary mask
            save_path (str): Optional path to save the visualization
        """
        # Compute different transforms
        dist_transform = self.binary_to_distance_transform(binary_mask)
        signed_dist = self.binary_to_signed_distance_transform(binary_mask)
        gradient_dist = self.binary_to_gradient_distance_transform(binary_mask)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original binary mask
        axes[0, 0].imshow(binary_mask, cmap='gray')
        axes[0, 0].set_title('Original Binary Mask')
        axes[0, 0].axis('off')
        
        # Distance transform
        im1 = axes[0, 1].imshow(dist_transform, cmap='viridis')
        axes[0, 1].set_title('Distance Transform')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Signed distance transform
        im2 = axes[0, 2].imshow(signed_dist, cmap='RdBu_r')
        axes[0, 2].set_title('Signed Distance Transform')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Gradient distance transform
        im3 = axes[1, 0].imshow(gradient_dist, cmap='hot')
        axes[1, 0].set_title('Gradient Distance Transform\n(Fluorescent-like)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Comparison: Binary vs Gradient
        axes[1, 1].imshow(binary_mask, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(gradient_dist, cmap='hot', alpha=0.5)
        axes[1, 1].set_title('Binary + Gradient Overlay')
        axes[1, 1].axis('off')
        
        # Cross-section comparison
        center_row = binary_mask.shape[0] // 2
        axes[1, 2].plot(binary_mask[center_row, :], 'k-', label='Binary', linewidth=2)
        axes[1, 2].plot(dist_transform[center_row, :], 'b-', label='Distance', linewidth=2)
        axes[1, 2].plot(gradient_dist[center_row, :], 'r-', label='Gradient', linewidth=2)
        axes[1, 2].set_title('Cross-section Comparison')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def test_distance_transforms():
    """Test function for distance transform utilities"""
    print("Testing Distance Transform Utilities...")
    
    # Create a synthetic binary mask (cell membrane-like)
    size = 128
    binary_mask = np.zeros((size, size))
    
    # Create a cell-like shape
    center = size // 2
    radius = 30
    y, x = np.ogrid[:size, :size]
    mask_circle = (x - center)**2 + (y - center)**2 <= radius**2
    
    # Create membrane (boundary)
    inner_radius = radius - 3
    inner_circle = (x - center)**2 + (y - center)**2 <= inner_radius**2
    membrane = mask_circle & ~inner_circle
    binary_mask[membrane] = 1
    
    # Add some noise/complexity
    noise = np.random.random((size, size)) > 0.95
    binary_mask[noise] = 1
    
    # Test distance transforms
    processor = DistanceTransformProcessor(max_distance=30, normalize_method='sigmoid')
    
    print("Computing different distance transforms...")
    
    # Test each transform type
    dist_transform = processor.binary_to_distance_transform(binary_mask)
    signed_dist = processor.binary_to_signed_distance_transform(binary_mask)
    gradient_dist = processor.binary_to_gradient_distance_transform(binary_mask)
    multi_scale = processor.binary_to_multi_scale_distance(binary_mask)
    
    print(f"Distance transform shape: {dist_transform.shape}, range: [{dist_transform.min():.3f}, {dist_transform.max():.3f}]")
    print(f"Signed distance shape: {signed_dist.shape}, range: [{signed_dist.min():.3f}, {signed_dist.max():.3f}]")
    print(f"Gradient distance shape: {gradient_dist.shape}, range: [{gradient_dist.min():.3f}, {gradient_dist.max():.3f}]")
    print(f"Multi-scale shape: {multi_scale.shape}")
    
    # Visualize
    processor.visualize_transforms(binary_mask, 'distance_transform_comparison.png')
    
    print("✓ Distance transform utilities tested successfully!")
    
    return {
        'binary_mask': binary_mask,
        'distance_transform': dist_transform,
        'signed_distance': signed_dist,
        'gradient_distance': gradient_dist,
        'multi_scale': multi_scale
    }


if __name__ == "__main__":
    results = test_distance_transforms()
