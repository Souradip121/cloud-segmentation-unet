import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.windows import Window
from torch.nn.functional import softmax
from unet import UNET
import os

def preprocess_image(image_path, target_size=(384, 384)):
    """
    Load a GeoTIFF image, convert from float32 to uint16, and resize for inference
    
    Args:
        image_path: Path to the SWIR GeoTIFF image
        target_size: Tuple of (height, width) for resizing
    
    Returns:
        Preprocessed tensor ready for model input
    """
    # Open the GeoTIFF file
    with rasterio.open(image_path) as src:
        # Read all bands
        img_data = src.read()
        
        # Get metadata for later use
        metadata = src.meta.copy()
        
        # Handle single-band SWIR image
        if img_data.shape[0] == 1:
            print(f"Single-band image detected. Creating a 4-band image by duplicating.")
            # Duplicate the band to create a 4-band image (using the same band for R,G,B and NIR)
            img_data = np.repeat(img_data, 4, axis=0)
        # Check if we have 4 bands (assuming R,G,B,NIR)
        elif img_data.shape[0] < 4:
            raise ValueError(f"Expected at least 4 bands, got {img_data.shape[0]}")
        
        # Extract the first 4 bands (R,G,B,NIR)
        img_data = img_data[:4]
        
        # Normalize the data (assuming float32 input)
        if img_data.dtype == np.float32:
            # Scale to uint16 range
            img_data = (img_data * 65535.0).astype(np.uint16)
        
        # Convert to float32 normalized between 0-1 for model input
        img_float = img_data.astype(np.float32) / 65535.0
    
    # Resize the image to target size
    resized_img = np.zeros((4, target_size[0], target_size[1]), dtype=np.float32)
    for i in range(4):
        band_img = Image.fromarray(img_float[i])
        resized_band = band_img.resize(target_size, Image.BILINEAR)
        resized_img[i] = np.array(resized_band)
    
    # Convert to tensor with shape [1, C, H, W]
    tensor_img = torch.from_numpy(resized_img).unsqueeze(0)
    
    return tensor_img, metadata

def load_model(model_path):
    """
    Load the trained U-Net model
    
    Args:
        model_path: Path to the saved model weights
        
    Returns:
        Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET(in_channels=4, out_channels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def perform_inference(model, image_tensor, device):
    """
    Run the model inference on the preprocessed image
    
    Args:
        model: The loaded U-Net model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        
    Returns:
        Prediction mask
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        
        # Apply softmax to get probabilities
        probs = softmax(prediction, dim=1)
        
        # Get the predicted class (0 = no cloud, 1 = cloud)
        mask = probs.argmax(dim=1).cpu().numpy()[0]
        confidence = probs.max(dim=1)[0].cpu().numpy()[0]
    
    return mask, confidence

def process_large_image(image_path, model_path, tile_size=384, overlap=32, output_dir=None):
    """
    Process a large image by dividing it into tiles, performing inference on each tile,
    and then assembling the final prediction mask
    
    Args:
        image_path: Path to the large image
        model_path: Path to the model weights
        tile_size: Size of tiles to process
        overlap: Overlap between tiles
        output_dir: Directory to save output
        
    Returns:
        Full prediction mask
    """
    # Load the model
    model, device = load_model(model_path)
    
    # Get image dimensions
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        meta = src.meta.copy()
        band_count = src.count
    
    # Calculate the effective stride (tile_size - overlap)
    stride = tile_size - overlap
    
    # Calculate number of tiles in each dimension
    n_tiles_width = (width + stride - 1) // stride
    n_tiles_height = (height + stride - 1) // stride
    
    # Initialize the full prediction mask
    full_prediction = np.zeros((height, width), dtype=np.uint8)
    full_confidence = np.zeros((height, width), dtype=np.float32)
    
    # Process each tile
    for i in range(n_tiles_height):
        for j in range(n_tiles_width):
            # Calculate tile coordinates
            x_start = j * stride
            y_start = i * stride
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            
            # Adjust start positions for edge tiles
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            # Read the tile
            with rasterio.open(image_path) as src:
                window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
                tile_data = src.read(window=window)
                
                # Handle single-band image
                if tile_data.shape[0] == 1:
                    tile_data = np.repeat(tile_data, 4, axis=0)
                # Handle if there are fewer than 4 bands
                elif tile_data.shape[0] < 4:
                    print(f"Warning: Expected at least 4 bands, got {tile_data.shape[0]}. Padding with zeros.")
                    padding = np.zeros((4 - tile_data.shape[0], tile_data.shape[1], tile_data.shape[2]), 
                                      dtype=tile_data.dtype)
                    tile_data = np.vstack((tile_data, padding))
                
                # Extract the first 4 bands (R,G,B,NIR)
                tile_data = tile_data[:4]
                
                # Normalize
                if tile_data.dtype == np.float32:
                    tile_data = (tile_data * 65535.0).astype(np.uint16)
                
                # Convert to float32 normalized between 0-1
                tile_float = tile_data.astype(np.float32) / 65535.0
            
            # Resize if the tile is smaller than the target size
            if tile_float.shape[1] < tile_size or tile_float.shape[2] < tile_size:
                resized_tile = np.zeros((4, tile_size, tile_size), dtype=np.float32)
                for b in range(4):
                    band_img = Image.fromarray(tile_float[b])
                    resized_band = band_img.resize((tile_size, tile_size), Image.BILINEAR)
                    resized_tile[b] = np.array(resized_band)
                tile_tensor = torch.from_numpy(resized_tile).unsqueeze(0)
            else:
                tile_tensor = torch.from_numpy(tile_float).unsqueeze(0)
            
            # Perform inference
            mask, confidence = perform_inference(model, tile_tensor, device)
            
            # If we had to resize, resize back to the original tile size
            if tile_float.shape[1] < tile_size or tile_float.shape[2] < tile_size:
                mask_img = Image.fromarray(mask.astype(np.uint8))
                conf_img = Image.fromarray(confidence)
                
                mask = np.array(mask_img.resize((x_end - x_start, y_end - y_start), Image.NEAREST))
                confidence = np.array(conf_img.resize((x_end - x_start, y_end - y_start), Image.BILINEAR))
            
            # Update the full prediction mask
            full_prediction[y_start:y_end, x_start:x_end] = mask
            full_confidence[y_start:y_end, x_start:x_end] = confidence
            
            print(f"Processed tile {i+1}/{n_tiles_height}, {j+1}/{n_tiles_width}")
    
    # Save the full prediction mask if output directory is specified
    if output_dir:
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_prediction.tif")
        
        # Update metadata for the output raster
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(full_prediction, 1)
        
        # Save confidence map
        conf_path = os.path.join(output_dir, f"{Path(image_path).stem}_confidence.tif")
        meta.update({'dtype': 'float32'})
        
        with rasterio.open(conf_path, 'w', **meta) as dst:
            dst.write(full_confidence, 1)
        
        # Create a visualization
        plt.figure(figsize=(12, 12))
        plt.imshow(full_prediction, cmap='viridis')
        plt.title("Cloud Prediction Mask")
        plt.colorbar(label="Class (0=No Cloud, 1=Cloud)")
        plt.savefig(os.path.join(output_dir, f"{Path(image_path).stem}_prediction.png"))
        plt.close()
    
    return full_prediction, full_confidence

def visualize_results(image_path, prediction_mask, confidence_map, output_path=None):
    """
    Visualize the original image, prediction mask, and confidence map
    
    Args:
        image_path: Path to the original image
        prediction_mask: The prediction mask
        confidence_map: The confidence map
        output_path: Path to save the visualization
    """
    # Create a figure with 3 subplots
    plt.figure(figsize=(18, 6))
    
    # Plot the original image
    with rasterio.open(image_path) as src:
        bands = src.count
        
        # Single-band SWIR
        if bands == 1:
            img = src.read(1)
            
            # Normalize for visualization
            img_min = np.min(img)
            img_max = np.max(img)
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = img - img_min
                
            plt.subplot(1, 3, 1)
            plt.imshow(img_norm, cmap='gray')
            plt.title("Original SWIR Image")
            plt.axis("off")
        # RGB image
        elif bands >= 3:
            rgb = src.read([1, 2, 3])  # Assuming R=1, G=2, B=3
            rgb = np.transpose(rgb, (1, 2, 0))
            
            # Normalize for visualization
            rgb_min = np.min(rgb)
            rgb_max = np.max(rgb)
            if rgb_max > rgb_min:
                rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
            else:
                rgb_norm = rgb - rgb_min
            
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_norm)
            plt.title("Original RGB Image")
            plt.axis("off")
    
    # Plot the prediction mask
    plt.subplot(1, 3, 2)
    plt.imshow(prediction_mask, cmap="gray")
    plt.title("Cloud Prediction Mask")
    plt.axis("off")
    
    # Plot the confidence map
    plt.subplot(1, 3, 3)
    plt.imshow(confidence_map, cmap="plasma")
    plt.colorbar(label="Confidence")
    plt.title("Prediction Confidence")
    plt.axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    
    plt.close()

def main():
    """Main function to run inference"""
    # Set paths
    image_path = "red_patch_100_5_by_12_LC08_L1TP_064015_20160420_20170223_01_T1.tiff"
    model_path = "unet_cloud_segmentation.pth"
    output_dir = "results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting inference process...")
    
    try:
        # Option 2: Process large image by tiles (this is more robust)
        print(f"Processing large image: {image_path}")
        full_mask, full_confidence = process_large_image(
            image_path, 
            model_path,
            tile_size=384,
            overlap=32,
            output_dir=output_dir
        )
        
        # Visualize results
        print("Generating visualization...")
        vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_visualization.png")
        visualize_results(image_path, full_mask, full_confidence, vis_path)
        
        print(f"Inference completed. Results saved in {output_dir}")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
