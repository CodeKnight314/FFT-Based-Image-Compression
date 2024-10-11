# FFT-Based Image Compression

This educational repository provides a Python implementation of image compression using the Fast Fourier Transform (FFT). The code utilizes the Discrete Fourier Transform (DFT) to transform image data into the frequency domain, where less significant frequencies can be discarded to achieve compression.

## Results:
<p align="center">
  <img src="visuals/Figure_1.png" alt="Your image alt text">
  <br>
  <em>PSNR:  129.52 with 12% storage reduction</em>
</p>

## Usage

### Prerequisites

  * Python 3.6 or higher
  * NumPy
  * Pillow (PIL Fork)

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/CodeKnight314/FFT-Based-Image-Compression.git
    ```

2.  Install the required packages:

    ```bash
    pip install -r FFT-Based-Image-Compression/requirement.txt
    ```

### Running the Code

```bash
python main.py --img_dir <path_to_image_directory> --threshold <threshold_value> --output_path <output_directory>
