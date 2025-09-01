"""
Model Application Script for CellSimul Conditional GAN

This script provides functions to apply trained conditional GAN models
to generate synthetic fluorescent images from distance mask inputs.
"""

import torch
import torch.nn as nn
import numpy as np
import tifffile
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cond_models'))

try:
    from conditional_generator import ConditionalGenerator, SimpleConditionalGenerator
except ImportError:
    print("Warning: Could not import generator models. Make sure you're running from the correct directory.")
    print("Expected path: /path/to/CellSimul/CellSimul/tests/")


def apply_generator_to_mask(distance_mask, generator_model_path, device=None, noise_vector=None, 
                           model_type='complex', normalize_output=True, return_tensor=False):
    """
    Apply a trained conditional generator to a distance mask to produce a synthetic fluorescent image.
    
    Args:
        distance_mask (numpy.ndarray or torch.Tensor or str): 
            Input distance mask image. Can be:
            - NumPy array (H, W) or (1, H, W) 
            - PyTorch tensor (1, H, W) or (1, 1, H, W)
            - String path to image file (TIF, PNG, etc.)
        generator_model_path (str): Path to the saved generator model (.pth file)
        device (str or torch.device, optional): Device to run inference on. Auto-detects if None.
        noise_vector (torch.Tensor, optional): Specific noise vector to use. Random if None.
        model_type (str): Type of model architecture ('simple' or 'complex')
        normalize_output (bool): Whether to normalize output to [0, 255] range
        return_tensor (bool): Whether to return PyTorch tensor (True) or NumPy array (False)
        
    Returns:
        numpy.ndarray or torch.Tensor: Generated synthetic fluorescent image
        
    Example:
        >>> # Generate from file path
        >>> synthetic_img = apply_generator_to_mask(
        ...     distance_mask="path/to/mask.tif",
        ...     generator_model_path="models/generator.pth"
        ... )
        >>> 
        >>> # Generate from numpy array
        >>> mask_array = np.random.rand(256, 256)
        >>> synthetic_img = apply_generator_to_mask(
        ...     distance_mask=mask_array,
        ...     generator_model_path="models/generator.pth",
        ...     model_type='simple'
        ... )
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load and preprocess input mask
    mask_tensor = _prepare_mask_input(distance_mask, device)
    print(f"Input mask shape: {mask_tensor.shape}")
    print(f"Input mask range: [{mask_tensor.min():.3f}, {mask_tensor.max():.3f}]")
    
    # Load the generator model
    generator = _load_generator_model(generator_model_path, model_type, device)
    
    # Generate noise vector if not provided
    if noise_vector is None:
        noise_vector = torch.randn(1, 100, device=device)
    else:
        noise_vector = noise_vector.to(device)
        if noise_vector.dim() == 1:
            noise_vector = noise_vector.unsqueeze(0)  # Add batch dimension
    
    print(f"Noise vector shape: {noise_vector.shape}")
    
    # Generate synthetic image
    generator.eval()
    with torch.no_grad():
        try:
            synthetic_tensor = generator(noise_vector, mask_tensor)
            print(f"Generated image shape: {synthetic_tensor.shape}")
            print(f"Generated image range: [{synthetic_tensor.min():.3f}, {synthetic_tensor.max():.3f}]")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            print("This might indicate model architecture mismatch or corrupted weights.")
            raise
    
    # Post-process output
    if return_tensor:
        return synthetic_tensor
    else:
        return _tensor_to_numpy(synthetic_tensor, normalize_output)


def _prepare_mask_input(distance_mask, device):
    """
    Prepare mask input for the generator.
    
    Args:
        distance_mask: Input mask (various formats)
        device: PyTorch device
        
    Returns:
        torch.Tensor: Preprocessed mask tensor (1, 1, H, W)
    """
    
    # Handle string path input
    if isinstance(distance_mask, str):
        if not os.path.exists(distance_mask):
            raise FileNotFoundError(f"Mask file not found: {distance_mask}")
        
        # Try to load as TIF first, then as regular image
        try:
            mask_array = tifffile.imread(distance_mask)
        except:
            mask_array = np.array(Image.open(distance_mask))
        
        print(f"Loaded mask from file: {distance_mask}")
        print(f"Original mask shape: {mask_array.shape}")
        
    # Handle numpy array input
    elif isinstance(distance_mask, np.ndarray):
        mask_array = distance_mask.copy()
        
    # Handle tensor input
    elif isinstance(distance_mask, torch.Tensor):
        mask_array = distance_mask.cpu().numpy()
        
    else:
        raise ValueError(f"Unsupported mask input type: {type(distance_mask)}")
    
    # Ensure 2D mask
    if mask_array.ndim == 3:
        if mask_array.shape[0] == 1:
            mask_array = mask_array.squeeze(0)
        elif mask_array.shape[2] == 1:
            mask_array = mask_array.squeeze(2)
        else:
            # Take first channel if multi-channel
            mask_array = mask_array[:, :, 0]
    
    # Resize to 256x256 if needed
    if mask_array.shape != (256, 256):
        mask_array = np.array(Image.fromarray(mask_array).resize((256, 256)))
        print(f"Resized mask to 256x256")
    
    # Normalize to [0, 1] range
    if mask_array.max() > 1.0:
        mask_array = mask_array.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch and channel dimensions
    mask_tensor = torch.from_numpy(mask_array).float()
    
    # Ensure shape is (1, 1, H, W)
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
    
    # Normalize to [-1, 1] range as expected by the model
    mask_tensor = mask_tensor * 2.0 - 1.0
    
    return mask_tensor.to(device)


def _load_generator_model(model_path, model_type, device):
    """
    Load the generator model from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ('simple' or 'complex')
        device: PyTorch device
        
    Returns:
        torch.nn.Module: Loaded generator model
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize the appropriate model architecture
    if model_type.lower() == 'simple':
        generator = SimpleConditionalGenerator()
        print("Initialized simple conditional generator")
    else:
        generator = ConditionalGenerator()
        print("Initialized complex conditional generator")
    
    # Load the saved weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict)
        print(f"Loaded model weights from: {model_path}")
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("This might indicate:")
        print("1. Wrong model_type specified (try 'simple' instead of 'complex' or vice versa)")
        print("2. Corrupted model file")
        print("3. Model was saved with different PyTorch version")
        raise
    
    generator.to(device)
    generator.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {total_params:,}")
    
    return generator


def _tensor_to_numpy(tensor, normalize_output=True):
    """
    Convert PyTorch tensor to NumPy array.
    
    Args:
        tensor: PyTorch tensor (1, 1, H, W)
        normalize_output: Whether to normalize to [0, 255]
        
    Returns:
        numpy.ndarray: Output image (H, W)
    """
    
    # Move to CPU and remove batch/channel dimensions
    output = tensor.cpu().squeeze().numpy()
    
    if normalize_output:
        # Convert from [-1, 1] to [0, 1]
        output = (output + 1.0) / 2.0
        # Convert to [0, 255] and uint8
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
    
    return output


def generate_multiple_variations(distance_mask, generator_model_path, num_variations=4, 
                                device=None, model_type='complex', normalize_output=True):
    """
    Generate multiple variations of synthetic images from the same distance mask.
    
    Args:
        distance_mask: Input distance mask (various formats)
        generator_model_path: Path to the saved generator model
        num_variations: Number of different variations to generate
        device: PyTorch device
        model_type: Type of model architecture
        normalize_output: Whether to normalize output
        
    Returns:
        list: List of generated images (NumPy arrays)
        
    Example:
        >>> variations = generate_multiple_variations(
        ...     distance_mask="mask.tif",
        ...     generator_model_path="generator.pth",
        ...     num_variations=6
        ... )
        >>> print(f"Generated {len(variations)} variations")
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and prepare mask (do this once)
    mask_tensor = _prepare_mask_input(distance_mask, device)
    generator = _load_generator_model(generator_model_path, model_type, device)
    
    variations = []
    
    print(f"Generating {num_variations} variations...")
    
    for i in range(num_variations):
        # Generate different noise for each variation
        noise_vector = torch.randn(1, 100, device=device)
        
        generator.eval()
        with torch.no_grad():
            synthetic_tensor = generator(noise_vector, mask_tensor)
        
        # Convert to numpy
        synthetic_image = _tensor_to_numpy(synthetic_tensor, normalize_output)
        variations.append(synthetic_image)
        
        print(f"Generated variation {i+1}/{num_variations}")
    
    return variations


def visualize_generation(distance_mask, generator_model_path, num_variations=4, 
                        save_path=None, model_type='complex'):
    """
    Visualize the generation process by showing the input mask, generated variations, and overlays.
    
    Args:
        distance_mask: Input distance mask
        generator_model_path: Path to the saved generator model
        num_variations: Number of variations to show
        save_path: Path to save the visualization (optional)
        model_type: Type of model architecture
        
    Example:
        >>> visualize_generation(
        ...     distance_mask="mask.tif",
        ...     generator_model_path="generator.pth",
        ...     save_path="generation_result.png"
        ... )
    """
    
    # Generate variations
    variations = generate_multiple_variations(
        distance_mask, generator_model_path, num_variations, model_type=model_type
    )
    
    # Load original mask for display
    if isinstance(distance_mask, str):
        try:
            mask_display = tifffile.imread(distance_mask)
        except:
            mask_display = np.array(Image.open(distance_mask))
    else:
        mask_display = distance_mask.copy()
    
    # Ensure mask is 2D
    if mask_display.ndim == 3:
        if mask_display.shape[0] == 1:
            mask_display = mask_display.squeeze(0)
        elif mask_display.shape[2] == 1:
            mask_display = mask_display.squeeze(2)
        else:
            mask_display = mask_display[:, :, 0]
    
    # Resize mask to match generated images if needed
    if mask_display.shape != (256, 256):
        mask_display = np.array(Image.fromarray(mask_display).resize((256, 256)))
    
    # Create binary mask for overlay (threshold at mean or median)
    mask_normalized = (mask_display - mask_display.min()) / (mask_display.max() - mask_display.min() + 1e-8)
    binary_mask = mask_normalized > np.median(mask_normalized)  # Binary threshold
    
    # Create visualization with 3 rows: mask, generated, overlay
    num_cols = max(num_variations, 4)  # At least 4 columns
    fig, axes = plt.subplots(3, num_cols, figsize=(4*num_cols, 12))
    
    # Row 1: Original mask repeated for comparison
    for col in range(num_cols):
        if col == 0:
            axes[0, col].imshow(mask_display, cmap='hot')
            axes[0, col].set_title('Input Distance Mask', fontsize=12, fontweight='bold')
        elif col < num_variations + 1:
            axes[0, col].imshow(mask_display, cmap='hot')
            axes[0, col].set_title(f'Mask for Var {col}', fontsize=10)
        else:
            axes[0, col].axis('off')
        axes[0, col].axis('off')
    
    # Row 2: Generated variations
    for col in range(num_cols):
        if col < len(variations):
            axes[1, col].imshow(variations[col])
            axes[1, col].set_title(f'Generated Variation {col+1}', fontsize=10)
        else:
            axes[1, col].axis('off')
        axes[1, col].axis('off')
    
    # Row 3: Overlays (binary mask + generated image)
    for col in range(num_cols):
        if col < len(variations):
            overlay = _create_overlay(binary_mask, variations[col])
            axes[2, col].imshow(overlay)
            axes[2, col].set_title(f'Overlay {col+1}\n(Red=Mask, Green=Generated)', fontsize=10)
        else:
            axes[2, col].axis('off')
        axes[2, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Conditional GAN: Distance Mask → Synthetic Fluorescent Images with Overlays', 
                 fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def _create_overlay(binary_mask, generated_image):
    """
    Create an RGB overlay of binary mask and generated image.
    
    Args:
        binary_mask: Binary mask (True/False or 1/0)
        generated_image: Generated fluorescent image (0-255)
        
    Returns:
        numpy.ndarray: RGB overlay image
    """
    
    # Ensure inputs are the right type and range
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)
    
    # Normalize generated image to [0, 1]
    if generated_image.max() > 1.0:
        gen_norm = generated_image.astype(np.float32) / 255.0
    else:
        gen_norm = generated_image.astype(np.float32)
    
    # Create RGB overlay
    overlay = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3))
    
    # Red channel: binary mask
    overlay[:, :, 0] = binary_mask.astype(np.float32) * 0.7  # Semi-transparent red
    
    # Green channel: generated image
    overlay[:, :, 1] = gen_norm
    
    # Blue channel: slight background for better visualization
    overlay[:, :, 2] = gen_norm * 0.3
    
    # Clip to valid range
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_single_generation_with_overlay(distance_mask, generator_model_path, 
                                           save_path=None, model_type='complex', noise_vector=None):
    """
    Create a simple 4-panel visualization: mask, generated, binary mask, overlay.
    
    Args:
        distance_mask: Input distance mask
        generator_model_path: Path to the saved generator model
        save_path: Path to save the visualization (optional)
        model_type: Type of model architecture
        noise_vector: Optional specific noise vector for reproducible results
        
    Example:
        >>> visualize_single_generation_with_overlay(
        ...     distance_mask="mask.tif",
        ...     generator_model_path="generator.pth",
        ...     save_path="single_result.png"
        ... )
    """
    
    # Generate single image
    generated_img = apply_generator_to_mask(
        distance_mask=distance_mask,
        generator_model_path=generator_model_path,
        model_type=model_type,
        noise_vector=noise_vector,
        normalize_output=True
    )
    
    # Load original mask for display
    if isinstance(distance_mask, str):
        try:
            mask_display = tifffile.imread(distance_mask)
        except:
            mask_display = np.array(Image.open(distance_mask))
    else:
        mask_display = distance_mask.copy()
    
    # Ensure mask is 2D and correct size
    if mask_display.ndim == 3:
        if mask_display.shape[0] == 1:
            mask_display = mask_display.squeeze(0)
        elif mask_display.shape[2] == 1:
            mask_display = mask_display.squeeze(2)
        else:
            mask_display = mask_display[:, :, 0]
    
    if mask_display.shape != (256, 256):
        mask_display = np.array(Image.fromarray(mask_display).resize((256, 256)))
    
    # Create binary mask
    mask_normalized = (mask_display - mask_display.min()) / (mask_display.max() - mask_display.min() + 1e-8)
    binary_mask = mask_normalized > np.median(mask_normalized)
    
    # Create overlay
    overlay = _create_overlay(binary_mask, generated_img)
    
    # Create 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top left: Original distance mask
    axes[0, 0].imshow(mask_display, cmap='hot')
    axes[0, 0].set_title('Input Distance Mask', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top right: Generated fluorescent image
    axes[0, 1].imshow(generated_img, cmap='green')
    axes[0, 1].set_title('Generated Fluorescent Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom left: Binary mask
    axes[1, 0].imshow(binary_mask, cmap='Reds')
    axes[1, 0].set_title('Binary Mask (Threshold)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Bottom right: Overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay\n(Red=Mask, Green=Generated)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Conditional GAN Generation with Mask Overlay Analysis', 
                 fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Single generation visualization saved to: {save_path}")
    
    plt.show()
    
    return generated_img, overlay


def batch_apply_generator(mask_dir, generator_model_path, output_dir, 
                         model_type='complex', file_pattern='*.tif'):
    """
    Apply the generator to all masks in a directory.
    
    Args:
        mask_dir: Directory containing distance mask files
        generator_model_path: Path to the saved generator model
        output_dir: Directory to save generated images
        model_type: Type of model architecture
        file_pattern: Pattern to match mask files
        
    Example:
        >>> batch_apply_generator(
        ...     mask_dir="distance_masks/",
        ...     generator_model_path="generator.pth",
        ...     output_dir="generated_images/"
        ... )
    """
    
    import glob
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mask files
    mask_files = glob.glob(os.path.join(mask_dir, file_pattern))
    
    if not mask_files:
        print(f"No mask files found in {mask_dir} with pattern {file_pattern}")
        return
    
    print(f"Found {len(mask_files)} mask files to process")
    
    # Process each mask
    for i, mask_path in enumerate(mask_files):
        try:
            # Generate synthetic image
            synthetic_img = apply_generator_to_mask(
                distance_mask=mask_path,
                generator_model_path=generator_model_path,
                model_type=model_type,
                normalize_output=True
            )
            
            # Save result
            filename = os.path.basename(mask_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_synthetic{ext}")
            
            tifffile.imwrite(output_path, synthetic_img)
            
            print(f"Processed {i+1}/{len(mask_files)}: {filename} → {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            continue
    
    print(f"Batch processing completed! Results saved to: {output_dir}")


def visualize_multiple_masks_generation(mask_folder, generator_model_path, 
                                       num_masks=4, save_folder=None, model_type='complex', 
                                       mask_indices=None):
    """
    Generate synthetic images from 4 different masks and create visualization with overlays.
    
    Args:
        mask_folder (str): Path to folder containing distance masks
        generator_model_path (str): Path to the saved generator model
        num_masks (int): Number of different masks to use (default 4)
        save_folder (str): Folder to save results (default: outputs/predictions)
        model_type (str): Type of model architecture
        mask_indices (list): Specific mask indices to use, or None for random selection
        
    Example:
        >>> visualize_multiple_masks_generation(
        ...     mask_folder="data/distance_masks_rescaled",
        ...     generator_model_path="outputs/unpaired_generator_final.pth",
        ...     save_folder="outputs/predictions"
        ... )
    """
    
    # Set default save folder if not provided
    if save_folder is None:
        save_folder = "/Users/edwheeler/cond_GAN/CellSimul/CellSimul/outputs/predictions"
    
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Get list of available mask files
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.tif')]
    mask_files.sort()  # Sort for consistent ordering
    
    if len(mask_files) < num_masks:
        print(f"Warning: Only {len(mask_files)} masks available, using all of them")
        num_masks = len(mask_files)
    
    # Select masks to use
    if mask_indices is None:
        # Random selection
        import random
        selected_indices = random.sample(range(len(mask_files)), num_masks)
        selected_indices.sort()  # Sort for better visualization order
    else:
        selected_indices = mask_indices[:num_masks]
    
    selected_masks = [mask_files[i] for i in selected_indices]
    
    print(f"Selected masks: {selected_masks}")
    
    # Generate images from each mask
    mask_images = []
    generated_images = []
    mask_names = []
    
    for i, mask_file in enumerate(selected_masks):
        mask_path = os.path.join(mask_folder, mask_file)
        print(f"Processing mask {i+1}/{num_masks}: {mask_file}")
        
        # Generate synthetic image
        synthetic_img = apply_generator_to_mask(
            distance_mask=mask_path,
            generator_model_path=generator_model_path,
            model_type=model_type,
            normalize_output=True
        )
        
        # Load original mask for display
        mask_display = tifffile.imread(mask_path)
        
        # Ensure mask is 2D and correct size
        if mask_display.ndim == 3:
            if mask_display.shape[0] == 1:
                mask_display = mask_display.squeeze(0)
            elif mask_display.shape[2] == 1:
                mask_display = mask_display.squeeze(2)
            else:
                mask_display = mask_display[:, :, 0]
        
        if mask_display.shape != (256, 256):
            mask_display = np.array(Image.fromarray(mask_display).resize((256, 256)))
        
        mask_images.append(mask_display)
        generated_images.append(synthetic_img)
        mask_names.append(mask_file.replace('.tif', '').replace('fluorescent_mask_', 'Mask_'))
        
        # Save individual generated image as greyscale TIF
        individual_save_path = os.path.join(save_folder, f"generated_{mask_file}")
        tifffile.imwrite(individual_save_path, synthetic_img)
        print(f"  Saved individual result: {individual_save_path}")
        
        # Save individual input mask as TIF for reference
        mask_save_path = os.path.join(save_folder, f"input_{mask_file}")
        tifffile.imwrite(mask_save_path, mask_display.astype(np.uint16))  # Save as 16-bit for distance values
        print(f"  Saved input mask: {mask_save_path}")
    
    # Create 3-row visualization: masks, generated images, overlays
    fig, axes = plt.subplots(3, num_masks, figsize=(4*num_masks, 12))
    
    if num_masks == 1:
        axes = axes.reshape(-1, 1)  # Ensure 2D array for single mask
    
    # Row 1: Original distance masks
    for col in range(num_masks):
        axes[0, col].imshow(mask_images[col], cmap='hot')
        axes[0, col].set_title(f'{mask_names[col]}\n(Distance Transform)', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
    
    # Row 2: Generated fluorescent images
    for col in range(num_masks):
        axes[1, col].imshow(generated_images[col])
        axes[1, col].set_title(f'Generated from {mask_names[col]}', fontsize=12, fontweight='bold')
        axes[1, col].axis('off')
    
    # Row 3: Overlays (binary mask + generated image)
    for col in range(num_masks):
        # Create binary mask for overlay
        mask_normalized = (mask_images[col] - mask_images[col].min()) / (mask_images[col].max() - mask_images[col].min() + 1e-8)
        binary_mask = mask_normalized > np.median(mask_normalized)
        
        overlay = _create_overlay(binary_mask, generated_images[col])
        axes[2, col].imshow(overlay)
        axes[2, col].set_title(f'Overlay {mask_names[col]}\n(Red=Mask, Green=Generated)', fontsize=10)
        axes[2, col].axis('off')
        
        # Save individual overlay as greyscale TIF
        # Convert overlay to greyscale (weighted average of RGB channels)
        overlay_grey = 0.299 * overlay[:, :, 0] + 0.587 * overlay[:, :, 1] + 0.114 * overlay[:, :, 2]
        overlay_grey = (overlay_grey * 255).astype(np.uint8)
        
        overlay_save_path = os.path.join(save_folder, f"overlay_{mask_names[col].lower()}.tif")
        tifffile.imwrite(overlay_save_path, overlay_grey)
        print(f"  Saved overlay: {overlay_save_path}")
    
    plt.tight_layout()
    plt.suptitle('Conditional GAN: Multiple Distance Masks → Synthetic Fluorescent Images', 
                 fontsize=16, y=0.98)
    
    # Save main visualization as TIF
    main_save_path = os.path.join(save_folder, f"multi_mask_generation_{num_masks}_masks.tif")
    plt.savefig(main_save_path, dpi=150, bbox_inches='tight')
    print(f"\nMain visualization saved to: {main_save_path}")
    
    plt.show()
    
    # Save summary info
    summary_path = os.path.join(save_folder, f"generation_summary_{num_masks}_masks.txt")
    with open(summary_path, 'w') as f:
        f.write("Multiple Mask Generation Summary\n")
        f.write("=" * 35 + "\n\n")
        f.write(f"Generator model: {generator_model_path}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Number of masks used: {num_masks}\n")
        f.write(f"Mask folder: {mask_folder}\n\n")
        f.write("Selected masks:\n")
        for i, (idx, mask_file) in enumerate(zip(selected_indices, selected_masks)):
            f.write(f"  {i+1}. Index {idx:03d}: {mask_file}\n")
        f.write(f"\nResults saved to: {save_folder}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return {
        'mask_images': mask_images,
        'generated_images': generated_images,
        'mask_names': mask_names,
        'save_folder': save_folder,
        'selected_indices': selected_indices
    }


if __name__ == "__main__":
    """
    Example usage and testing
    """
    
    # Example paths (update these to match your setup)
    example_mask_path =  r"/Users/edwheeler/cond_GAN/CellSimul/CellSimul/data/distance_masks_rescaled/fluorescent_mask_001.tif"
    example_model_path = r"/Users/edwheeler/cond_GAN/CellSimul/CellSimul/outputs/models/unpaired_generator_final.pth"
    
    print("CellSimul Conditional GAN Model Application")
    print("=" * 50)
    
    # Test if files exist
    if os.path.exists(example_mask_path) and os.path.exists(example_model_path):
        print("Testing single image generation...")
        
        try:
            # Generate single image
            synthetic_img = apply_generator_to_mask(
                distance_mask=example_mask_path,
                generator_model_path=example_model_path,
                model_type='complex'
            )
            
            print(f"✅ Single generation successful! Output shape: {synthetic_img.shape}")
            
            # Test multiple variations
            print("\nTesting multiple variations...")
            variations = generate_multiple_variations(
                distance_mask=example_mask_path,
                generator_model_path=example_model_path,
                num_variations=3
            )
            
            print(f"✅ Multiple variations successful! Generated {len(variations)} images")
            
            # Test visualization
            print("\nTesting visualization...")
            visualize_generation(
                distance_mask=example_mask_path,
                generator_model_path=example_model_path,
                num_variations=4,
                save_path="test_generation.png"
            )
            
            print("✅ All tests passed!")
            
            # Test new multiple masks function
            print("\nTesting multiple masks generation...")
            mask_folder = "/Users/edwheeler/cond_GAN/CellSimul/CellSimul/data/distance_masks_rescaled"
            if os.path.exists(mask_folder):
                try:
                    results = visualize_multiple_masks_generation(
                        mask_folder=mask_folder,
                        generator_model_path=example_model_path,
                        num_masks=4,
                        save_folder="/Users/edwheeler/cond_GAN/CellSimul/CellSimul/outputs/predictions",
                        mask_indices=[0, 10, 20, 30]  # Use specific masks for consistent results
                    )
                    print("✅ Multiple masks generation successful!")
                except Exception as e:
                    print(f"❌ Multiple masks test failed: {e}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("Example files not found. Please update the paths in the script.")
        print(f"Looking for mask: {example_mask_path}")
        print(f"Looking for model: {example_model_path}")
        print("\nYou can still use the functions with your own paths:")
        print("\n# Example usage:")
        print("synthetic_img = apply_generator_to_mask(")
        print("    distance_mask='path/to/your/mask.tif',")
        print("    generator_model_path='path/to/your/model.pth'")
        print(")")
        print("\n# For multiple different masks:")
        print("visualize_multiple_masks_generation(")
        print("    mask_folder='data/distance_masks_rescaled',")
        print("    generator_model_path='outputs/unpaired_generator_final.pth',")
        print("    num_masks=4,")
        print("    save_folder='outputs/predictions'")
        print(")")
