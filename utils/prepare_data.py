import os
import glob
from skimage import io
from .rescale_save import save_and_rescale_mask  # Use the actual function name

def load_and_rescale_tif_files(input_dir, output_dir):
    """
    Load all .tif files from input directory and save rescaled versions to output directory.
    
    Args:
        input_dir (str): Path to directory containing .tif files
        output_dir (str): Path to directory where rescaled files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .tif files from input directory
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(input_dir, "*.tiff")))
    
    for tif_file in tif_files:
        # Load the image
        image = io.imread(tif_file)
        
        # Get filename without path
        filename = os.path.basename(tif_file)
        
        # Apply save_and_rescale_mask function
        # Extract filename without extension for proper naming
        filename_without_ext = os.path.splitext(filename)[0]
        save_and_rescale_mask(image, filename, output_dir=output_dir)
        
        print(f"Processed: {filename}")
    
    print(f"Finished processing {len(tif_files)} files")
