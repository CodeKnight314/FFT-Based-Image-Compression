import numpy as np
import argparse
from batch_compression import compress
from PIL import Image

def psnr_calc(img_gt: np.array, img_comp: np.array):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img_gt (np.array): Ground truth image (original image).
        img_comp (np.array): Compressed or reconstructed image.

    Returns:
        float: PSNR value in decibels (dB).
    """
    mse_loss = np.mean((img_gt - img_comp) ** 2)
    if mse_loss == 0:
        return float('inf')
    
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse_loss))
    
    return psnr

def compression_evaluation(img_path: str, threshold_value: int):
    """
    Evaluate the compression performance using PSNR.

    Args:
        img_path (str): Path to the image to be compressed.
        threshold_value (int): Threshold value for compression.
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    img_compressed = compress(img, threshold_value)
    psnr = psnr_calc(img, img_compressed)
    print(f"[INFO] Compression Performance at {threshold_value}: {psnr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the compression performance using PSNR.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image to be compressed.")
    parser.add_argument("--threshold_value", type=int, required=True, help="Threshold value for compression.")
    
    args = parser.parse_args()
    compression_evaluation(args.img_path, args.threshold_value)