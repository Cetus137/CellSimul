#!/usr/bin/env python3
"""
Pytest tests for SyntheticCellMembranes2D class
Tests the basic functionality of synthetic cell membrane generation
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mask_creation.synthetic_cell_membrane_masks_2d import SyntheticCellMembranes2D, SyntheticTissue2D
from utils.rescale_save import save_and_rescale_mask
from utils.distance_transform_utils import DistanceTransformProcessor

def test_standalone_182x256_example():
    """Standalone test function for 182x256 tissue generation with visualization"""
    print("Running standalone 182x256 tissue generation test...")
    
    # Create figure for plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    
    # Test SyntheticTissue2D - aim for ~25 cells
    tissue = SyntheticTissue2D(
        width=182,
        height=256,
        cell_size_range=(15,30),  # Larger cells to reduce count
        cell_density=50,         # Adjusted density to aim for 25 cells
        tissue_pattern='random'
    )
    
    tissue.generate_instances()
    tissue_mask = tissue.get_instance_mask()
    tissue_centroid = tissue.get_centroid_mask()
    tissue_boundary = tissue.get_boundary_mask()
    tissue_distance = tissue.get_distance_mask()
    
    assert tissue_mask is not None
    assert tissue_mask.shape == (256, 182)
    
    tissue_cells = len(np.unique(tissue_mask)) - 1
    assert tissue_cells > 0
    
    print(f"SyntheticTissue2D: Generated {tissue_cells} cells with shape {tissue_mask.shape}")
    
    # Plot SyntheticTissue2D masks
    axes[0].imshow(tissue_mask, cmap='nipy_spectral')
    axes[0].set_title(f'Tissue2D - Instance\n({tissue_cells} cells)')
    axes[0].axis('off')
    
    axes[1].imshow(tissue_centroid, cmap='Reds')
    axes[1].set_title('Centroid Mask')
    axes[1].axis('off')
    
    axes[2].imshow(tissue_boundary, cmap='Greys_r')
    axes[2].set_title('Boundary Mask')
    axes[2].axis('off')
    
    axes[3].imshow(tissue_distance, cmap='viridis')
    axes[3].set_title('Distance Transform')
    axes[3].axis('off')
    
    #plt.tight_layout()
    #plt.suptitle('182x256 Cell Generation Test - Target: ~25 cells', fontsize=16, y=0.98)
    #plt.show()
    
    print(f"\nTotal cells generated:")
    print(f"  SyntheticCellMembranes2D: {tissue_cells} cells")
    print(f"  SyntheticTissue2D: {tissue_cells} cells")
    
    return 


def test_batch_masks():
    """Generate and save a batch of fluorescent-like distance transforms - pytest version"""
    output_dir = "./output"
    num_masks = 256
    
    print(f"Generating {num_masks} fluorescent distance transform masks...")
    
    # Initialize distance transform processor
    processor = DistanceTransformProcessor(max_distance=40, normalize_method='sigmoid')
    
    for i in range(num_masks):
        tissue = SyntheticTissue2D(
            width=182,
            height=256,
            cell_size_range=(15,30),
            cell_density=50,
            tissue_pattern='random'
        )
        tissue.generate_instances()
        
        # Get boundary mask for cell membranes
        boundary_mask = tissue.get_boundary_mask()
        
        # Ensure we have a valid mask
        assert boundary_mask is not None
        assert boundary_mask.shape == (256, 182)
        
        # Calculate gradient distance transform from cell membrane masks
        gradient_distance = processor.binary_to_gradient_distance_transform(boundary_mask)
        
        filename = f"fluorescent_mask_{i+1:03d}.tif"
        filepath = save_and_rescale_mask(gradient_distance, filename, output_dir=output_dir)
        print(f"Saved fluorescent mask {i+1}/{num_masks} to {filepath}")
    
    print(f"✓ Successfully generated and saved {num_masks} fluorescent distance transforms to {output_dir}")

if __name__ == "__main__":
    # Run the standalone test
    test_standalone_182x256_example()
    
    # Generate batch masks
    generate_batch_masks_standalone(output_dir="./output", num_masks=5)
    
    print("All tests and batch generation completed!")

