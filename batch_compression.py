import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from fft_ops import FFT_2D, IFFT_2D

def compress(image_path: str, threshold_value: float, save_path=None):
    """
    Compression of image with FFT based compression.

    Args:
        image_path (str): Path of the image to be compressed.
        threshold_value (float): Threshold value to set selected positions to zero.
        save_path (str, optional): Path to save the compressed image. Defaults to None.

    Returns:
        img_dcomp (np.array): numpy array of the compressed image.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    img_comp = FFT_2D(img)

    compressed_fft = np.where(np.abs(img_comp) > threshold_value, img_comp, 0)
    img_dcomp = np.abs(IFFT_2D(compressed_fft))

    if save_path:
        Image.fromarray(np.uint8(img_dcomp)).save(save_path)
    
    return img_dcomp

def compress_batch(img_dir: str, threshold_value: float, output_path: str):
    img_paths = glob(os.path.join(img_dir, "*"))

    for img in tqdm(img_paths, desc="[Image Compression]"):
        output_file = os.path.join(output_path, os.path.basename(img).split('.')[0] + "_compressed.png")
        compress(img, threshold_value, save_path=output_file)

    print("[INFO] Image compression completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Image for compression")
    parser.add_argument("--img_dir", type=str, help="Directory of images for compression")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value for FFT-based compression")
    parser.add_argument("--output_path", type=str, help="Output directory for compressed images", default="./compressed_images")

    args = parser.parse_args()

    if args.img:
        output_file = os.path.join(args.output_path, os.path.basename(args.img).split('.')[0] + "_compressed.png")
        os.makedirs(args.output_path, exist_ok=True)
        compress(args.img, args.threshold, save_path=output_file)
    elif args.img_dir:
        os.makedirs(args.output_path, exist_ok=True)
        compress_batch(args.img_dir, args.threshold, args.output_path)
    else:
        print("[ERROR] Either --img or --img_dir should be provided.")