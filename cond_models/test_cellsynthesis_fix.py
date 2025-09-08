#!/usr/bin/env python3
"""
Quick test script for the CellSynthesis-inspired fixes
"""

import sys
import os

# Test the updated training script with conservative settings
if __name__ == "__main__":
    print("Testing CellSynthesis-inspired fixes...")
    print("Key improvements:")
    print("1. Balanced RAdam optimizer with conservative LR (0.0002)")
    print("2. Reduced structural loss weight (0.1x instead of 1.0x)")
    print("3. Label smoothing for discriminator stability (0.9 instead of 1.0)")
    print("4. CellSynthesis-style alternating training")
    print("5. Proper identity loss for structure preservation")
    print()
    
    # Run a short test
    cmd = """python conditional_train_ada_structural.py \
    --fluorescent_dir /Users/edwheeler/cond_GAN/CellSimul/CellSimul/data/fluorescence_rescaled \
    --mask_dir /Users/edwheeler/cond_GAN/CellSimul/CellSimul/data/distance_masks_rescaled \
    --epochs 3 \
    --batch_size 4 \
    --output_dir ./outputs_cellsynthesis_conservative"""
    
    print("Running test command:")
    print(cmd)
    print()
    
    os.system(cmd)
