import os
import numpy as np
from PIL import Image
from scipy import ndimage


def save_and_rescale_mask(mask, filename, output_dir="./output"):
    """
    Utility to crop, rescale mask and save as TIF
    
    Args:
        mask: Input mask (binary or instance mask)
        filename: Output filename
        output_dir: Output directory
        save_fluorescent: If True, convert to fluorescent-like distance transform
        max_distance: Maximum distance for fluorescent transform
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Crop from (256, 182) to (182, 182) - crop height to match width
    crop_size = min(mask.shape)  # 182
    h, w = mask.shape
    
    # Center crop
    start_h = (h - crop_size) // 2
    cropped_mask = mask[start_h:start_h + crop_size, :]
    
    # Rescale to (256, 256)
    img = Image.fromarray(cropped_mask.astype(np.float32), mode='F')
    img_resized = img.resize((256, 256), Image.NEAREST)
    
    # Save as TIF
    filepath = os.path.join(output_dir, filename)
    if not filepath.endswith('.tif'):
        filepath += '.tif'
    img_resized.save(filepath)
    
    print(f"Saved to {filepath} (cropped to {crop_size}x{crop_size}, rescaled to 256x256)")
    return filepath
