"""
CellSimul - Synthetic Cell Membrane Mask Generation

This package provides tools for generating synthetic 2D cell membrane masks
with various patterns and configurations.
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .synthetic_cell_membrane_masks_2d import (
    SyntheticCellMembranes2D,
    SyntheticTissue2D,
    SyntheticEpithelium2D,
    generate_data_2d,
    h5_writer
)

# Import mask creation utilities if available

__all__ = [
    'SyntheticCellMembranes2D',
    'SyntheticTissue2D', 
    'SyntheticEpithelium2D',
    'generate_data_2d',
    'h5_writer'
]