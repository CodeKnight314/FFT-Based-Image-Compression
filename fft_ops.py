import numpy as np

## Cooley-Tukey FFT Implementation (O(n log n))
def FFT_1D_optimized(array: np.array):
    n = array.size
    if n <= 1:
        return array
    if n % 2 > 0:
        raise ValueError("Array length must be a power of 2")
    even = FFT_1D_optimized(array[0::2])
    odd = FFT_1D_optimized(array[1::2])
    terms = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + terms[:n // 2] * odd, even + terms[n // 2:] * odd])

## Cooley-Tukey Inverse FFT Implementation (O(n log n))
def IFFT_1D_optimized(array: np.array):
    n = array.size
    if n <= 1:
        return array
    if n % 2 > 0:
        raise ValueError("Array length must be a power of 2")
    even = IFFT_1D_optimized(array[0::2])
    odd = IFFT_1D_optimized(array[1::2])
    terms = np.exp(2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + terms[:n // 2] * odd, even + terms[n // 2:] * odd]) 

## Pad array to the next power of 2
def pad_to_power_of_2(array: np.array):
    cols, rows = array.shape[:2]  # Switch height and width
    new_cols = 2**int(np.ceil(np.log2(cols)))
    new_rows = 2**int(np.ceil(np.log2(rows)))
    
    if len(array.shape) == 3:
        padded_array = np.zeros((new_cols, new_rows, array.shape[2]), dtype=array.dtype)
        padded_array[:cols, :rows, :] = array
    else:
        padded_array = np.zeros((new_cols, new_rows), dtype=array.dtype)
        padded_array[:cols, :rows] = array
    
    return padded_array, cols, rows

## FFT Transformation of 2D matrices in O(m * n * log(m * n))
def FFT_2D_optimized(matrix: np.array):
    # Ensure matrix dimensions are powers of 2
    matrix = pad_to_power_of_2(matrix)[0]
    
    # FFT Matrix
    fft_matrix = np.zeros(matrix.shape, dtype=complex)
    
    # Processing rows
    for i in range(matrix.shape[0]):
        fft_matrix[i] = FFT_1D_optimized(matrix[i])
    
    # Processing columns
    for i in range(matrix.shape[1]):
        fft_matrix[:, i] = FFT_1D_optimized(fft_matrix[:, i])
        
    return fft_matrix

## Inverse FFT Transformation of 2D matrices in O(m * n * log(m * n))
def IFFT_2D_optimized(matrix: np.array):
    # Ensure matrix dimensions are powers of 2
    matrix = pad_to_power_of_2(matrix)[0]
    
    # IFFT Matrix
    ifft_matrix = np.zeros(matrix.shape, dtype=complex)
    
    # Processing columns
    for i in range(matrix.shape[0]):
        ifft_matrix[i] = IFFT_1D_optimized(matrix[:, i])
        
    # Processing rows
    for i in range(matrix.shape[1]):
        ifft_matrix[:, i] = IFFT_1D_optimized(ifft_matrix[:, i])
        
    ifft_matrix = ifft_matrix / np.prod(matrix.shape)  # Normalize by matrix size
    return ifft_matrix.real.astype(matrix.dtype)  # Convert to real and original data type

## FFT Transformation of 2D matrices for RGB images
def FFT_2D_RGB(matrix: np.array):
    fft_matrix = np.zeros(matrix.shape, dtype=complex)
    for channel in range(3):
        fft_matrix[:, :, channel] = FFT_2D_optimized(matrix[:, :, channel])
    return fft_matrix

## Inverse FFT Transformation of 2D matrices for RGB images
def IFFT_2D_RGB(matrix: np.array):
    ifft_matrix = np.zeros(matrix.shape, dtype=complex)
    for channel in range(3):
        ifft_matrix[:, :, channel] = IFFT_2D_optimized(matrix[:, :, channel])
    return ifft_matrix

# Testing the optimized FFT and IFFT functions
if __name__ == "__main__":
    # 1D FFT Test
    test_array = np.random.rand(8)
    fft_result = FFT_1D_optimized(test_array)
    ifft_result = IFFT_1D_optimized(fft_result)
    print("Original array:", test_array)
    print("After FFT and IFFT:", ifft_result.real)

    # 2D FFT Test
    test_matrix = np.random.rand(4, 4, 3)
    fft_matrix_result = FFT_2D_RGB(test_matrix)
    ifft_matrix_result = IFFT_2D_RGB(fft_matrix_result)
    ifft_matrix_result = ifft_matrix_result[:4, :4, :]  # Unpad the result to original dimensions
    print("\nOriginal matrix:")
    print(test_matrix)
    print("\nAfter FFT and IFFT:")
    print(ifft_matrix_result.real)