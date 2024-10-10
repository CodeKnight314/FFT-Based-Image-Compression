import numpy as np 
import random
import time

## FFT Transformation of 1D array in O(n^2)
def FFT_1D(array: np.array):
    fft_arr = np.zeros((len(array)), dtype=complex)

    # Processing each element of array
    for i in range(array.size):
        val = 0
        for j in range(array.size):
            val += array[j] * np.exp(-2j * np.pi * i * j / array.size)
        fft_arr[i] = val
    return fft_arr

## Inverse FFT Transformation of 1D array in O(n^2)
def IFFT_1D(array: np.array):
    ifft_arr = np.zeros((len(array)), dtype=complex)
    
    # Processing each element of array
    for i in range(array.size):
        val = 0
        for j in range(array.size):
            val += array[j] * np.exp(2j * np.pi * i * j / array.size) / array.size
        ifft_arr[i] = val

    return ifft_arr

## FFT Transformation of 2D matrices in O(m^2 * n^2)
def FFT_2D(matrix: np.array):
    # FFT Matrix
    fft_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=complex)
    
    # Processing rows
    for i in range(matrix.shape[0]):
        fft_matrix[i] = FFT_1D(matrix[i])
    
    # Processing columns
    for i in range(matrix.shape[1]):
        fft_matrix[:, i] = FFT_1D(fft_matrix[:, i])
        
    return fft_matrix

## Inverse Transformation of 2D matrices in O(m^2 * n^2)
def IFFT_2D(matrix: np.array):
    # IFFT Matrix
    ifft_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=complex)
    
    # Processing columns
    for i in range(matrix.shape[0]):
        ifft_matrix[i] = IFFT_1D(matrix[:, i])
        
    # Processing rows 
    for i in range(matrix.shape[1]):
        ifft_matrix[:, i] = IFFT_1D(ifft_matrix[i])
        
    return ifft_matrix

