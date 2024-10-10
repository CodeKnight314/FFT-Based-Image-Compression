import numpy as np

## Cooley-Tukey FFT Implementation (O(n log n))
def FFT_1D_optimized(array: np.array):
    n = array.size
    if n <= 1:
        return array
    even = FFT_1D_optimized(array[0::2])
    odd = FFT_1D_optimized(array[1::2])
    terms = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.concatenate([even + terms[:n // 2] * odd, even + terms[n // 2:] * odd])

## Cooley-Tukey Inverse FFT Implementation (O(n log n))
def IFFT_1D_optimized(array: np.array):
    n = array.size
    if n <= 1:
        return array
    even = IFFT_1D_optimized(array[0::2])
    odd = IFFT_1D_optimized(array[1::2])
    terms = np.exp(2j * np.pi * np.arange(n) / n)
    result = np.concatenate([even + terms[:n // 2] * odd, even + terms[n // 2:] * odd])
    return result / 2

## FFT Transformation of 2D matrices in O(m * n * log(m * n))
def FFT_2D_optimized(matrix: np.array):
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
    # IFFT Matrix
    ifft_matrix = np.zeros(matrix.shape, dtype=complex)
    
    # Processing columns
    for i in range(matrix.shape[0]):
        ifft_matrix[:, i] = IFFT_1D_optimized(matrix[:, i])
        
    # Processing rows
    for i in range(matrix.shape[1]):
        ifft_matrix[i] = IFFT_1D_optimized(ifft_matrix[i])
        
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
    test_matrix = np.random.rand(4, 4)
    fft_matrix_result = FFT_2D_optimized(test_matrix)
    ifft_matrix_result = IFFT_2D_optimized(fft_matrix_result)
    print("\nOriginal matrix:")
    print(test_matrix)
    print("\nAfter FFT and IFFT:")
    print(ifft_matrix_result.real)