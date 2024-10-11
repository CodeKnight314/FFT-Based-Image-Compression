import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from compression import compress
import os

def visualize_compression(img_path: str, threshold_value: int):
    """
    Visualize the original and compressed images side by side.

    Args:
        img_path (str): Path to the image to be compressed.
        threshold_value (int): Threshold value for compression.
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    
    img_compressed = compress(img_path, threshold_value)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(img_compressed.astype(np.uint8))
    ax[1].set_title(f"Compressed Image (Threshold: {threshold_value})")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the original and compressed images.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image to be compressed.")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value for compression.")
    args = parser.parse_args()
    visualize_compression(args.img_path, args.threshold)