#!/usr/bin/env python3
"""
Pytest tests for prepare_data.py functions
Tests the data preparation and file processing functionality
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from PIL import Image
import sys

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.prepare_data import load_and_rescale_tif_files


def test_load_and_rescale_tif_files():
    """Simple test that takes all TIF files in an input directory and applies prepare_data.load_and_rescale_tif_files"""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, 'input')
    output_dir = os.path.join(temp_dir, 'output')
    
    try:
        # Create input directory
        os.makedirs(input_dir, exist_ok=True)
        
        # Create a few sample TIF files
        # Sample 1: 256x182 binary mask
        binary_mask = np.zeros((256, 182), dtype=np.uint8)
        binary_mask[50:100, 50:100] = 255
        binary_mask[150:200, 100:150] = 255
        
        sample1_path = os.path.join(input_dir, 'mask_001.tif')
        Image.fromarray(binary_mask).save(sample1_path)
        
        # Sample 2: 512x512 grayscale image  
        grayscale_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        sample2_path = os.path.join(input_dir, 'image_002.tiff')
        Image.fromarray(grayscale_img).save(sample2_path)
        
        # Sample 3: 128x128 image
        small_img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        sample3_path = os.path.join(input_dir, 'small_003.tif')
        Image.fromarray(small_img).save(sample3_path)
        
        # Apply the function under test
        load_and_rescale_tif_files(input_dir, output_dir)
        
        # Check that output files were created
        assert os.path.exists(output_dir)
        output_files = os.listdir(output_dir)
        assert len(output_files) == 3  # Should have processed 3 TIF files
        
        # Verify each output file exists and has correct properties
        for output_file in output_files:
            output_path = os.path.join(output_dir, output_file)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Load and check that image was rescaled to 256x256
            processed_image = np.array(Image.open(output_path))
            assert processed_image.shape == (256, 256), f"Image {output_file} should be rescaled to 256x256"
        
        print(f"✓ Successfully processed {len(output_files)} TIF files")
        print(f"✓ All output images rescaled to 256x256")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run the simple test when script is executed directly
    test_load_and_rescale_tif_files()
    print("Simple prepare_data test completed!")
