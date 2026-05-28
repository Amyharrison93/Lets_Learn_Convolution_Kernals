import asyncio
import tracemalloc
from multiprocessing import Pool

import cv2
import numpy as np
import pathlib
import time

class convolution_kernals():
    """
    A class that implements various convolution kernels for image processing. 
    This class provides both custom implementations and wrapper functions that utilize OpenCV's optimized filter2D function for comparison.
    it also includes helper functions for importing images, generating kernels, and applying kernels to images.
    """
    def __init__(self):
        tracemalloc.start()

    def import_image(self, image_path: str = None) -> np.ndarray:
        """
        A helper function to import an image from the specified file path.
        Parameters:
        image_path (str): The file path to the image to be imported.
        Returns:
        numpy.ndarray: The imported image as a NumPy array.
        """
        script_dir = pathlib.Path(__file__).parent.resolve()
        if image_path is None:
            image_path = script_dir / ".." / "__documents__" / "image.png" 
        print(f"Loading image from: {image_path}")
        input_image = cv2.imread(str(image_path))

        if input_image is None:
            raise ValueError("Could not load the image. Please check the file path.")
        return input_image
    
    def unstitch_image(self, image1: np.ndarray, num_pieces: int = 4) -> list:
        """
        A helper function that splits an image into several smaller images and returns them as a list.
        this function is used to create smaller images for running in parallel for faster processing.
        Parameters:
        image1 (numpy.ndarray): The input image to be unstitched.
        num_pieces (int): The number of pieces to split the image into (default is 4).
        Returns:
        list: A list of smaller images.
        """
        height, width, channels = image1.shape
        piece_height = height // num_pieces
        pieces = []
        for i in range(num_pieces):
            start_row = i * piece_height
            end_row = (i + 1) * piece_height if i < num_pieces - 1 else height
            pieces.append(image1[start_row:end_row, :, :])
        return pieces
    
    def stitch_image(self, pieces: list) -> np.ndarray:
        """
        A helper function that stitches a list of smaller images back into a single image.
        this function is used to combine the smaller images processed in parallel back into a single image.
        Parameters:
        pieces (list): A list of smaller images to be stitched together.
        Returns:
        numpy.ndarray: The stitched image.
        """
        return np.vstack(pieces)
    
    def generate_identity_kernel(self, kernel_size: int=3) -> np.ndarray:
        """
        A helper function to generate an identity kernel of the specified size.
        Parameters:
        kernel_size (int): The size of the identity kernel (default is 3).
        Returns:
        numpy.ndarray: The generated identity kernel, which is a matrix with a single 1 in the center and 0s elsewhere.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be an odd number.")
        
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
        return kernel
    
    def apply_kernel(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        A helper function to apply a given kernel to the input image using a custom implementation of convolution.
        Parameters:
        image (numpy.ndarray): The input image to which the kernel will be applied.
        kernel (numpy.ndarray): The convolution kernel to be applied to the image.
        is_async (bool): Whether to apply the kernel asynchronously.
        Returns:
        numpy.ndarray: The output image after applying the convolution kernel.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        kernel_size = kernel.shape[0]
        output_image = np.zeros_like(image)

        for i in range(len(image)):
            for j in range(len(image[0])):
                for k in range(3):  # For each color channel
                    output_pixel_value = 0.0
                    for m in range(kernel_size):
                        for n in range(kernel_size):
                            x = i + m - kernel_size // 2
                            y = j + n - kernel_size // 2
                            if 0 <= x < len(image) and 0 <= y < len(image[0]):
                                output_pixel_value += image[x, y, k] * kernel[m, n]
                    output_image[i, j, k] = np.clip(output_pixel_value, 0, 255)

        return output_image
    
    def apply_kernel_multiprocessing(self, image: np.ndarray, kernel: np.ndarray, num_pieces: int = 4) -> np.ndarray:
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")

        pieces = self.unstitch_image(image, num_pieces=num_pieces)

        with Pool(processes=num_pieces) as pool:
            results = pool.starmap(self.apply_kernel, [(piece, kernel) for piece in pieces])

        return self.stitch_image(results)
    
    def identity_kernal_cv2(self, image: np.ndarray, kernel_size: int=3) -> np.ndarray:
        """
        A simple wrapper function for the identity kernel that uses OpenCV's filter2D function.
        Parameters:
        image (numpy.ndarray): The input image to which the identity kernel will be applied.
        kernel_size (int): The size of the identity kernel (default is 3).
        Returns:
        numpy.ndarray: The output image after applying the identity kernel, which should be the same as the input image.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be an odd number.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        # Define the identity kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, kernel_size // 2] = 1.0

        # Apply the kernel to the image using cv2.filter2D
        try:
            output_image = cv2.filter2D(image, -1, kernel)
        except Exception as e:
            raise ValueError(f"Error occurred while applying the identity kernel: {e}")

        return output_image

if __name__ == "__main__":
    # Load the input image
    kernals = convolution_kernals()
    input_image = kernals.import_image()

    start_time = time.time()
    # Apply the identity kernel to the input image
    kernal = kernals.generate_identity_kernel(kernel_size=3)

    start_time = time.time()
    output_image_cv2 = kernals.identity_kernal_cv2(input_image)
    end_time = time.time()
    print(f"Time taken (OpenCV implementation): {end_time - start_time:.4f} seconds")

    start_time = time.time()
    output_image_multiprocessing = kernals.apply_kernel_multiprocessing(input_image, kernal, num_pieces=8)
    end_time = time.time()
    print(f"Time taken (multiprocessing implementation): {end_time - start_time:.4f} seconds")

    for i in range(3):
        if not np.array_equal(input_image[:, :, i], output_image_cv2[:, :, i]):
            print(f"Channel {i} is not identical between input and OpenCV output images.")
        else:
            print(f"Channel {i} is identical between input and OpenCV output images.")

    for i in range(3):
        if not np.array_equal(input_image[:, :, i], output_image_multiprocessing[:, :, i]):
            print(f"Channel {i} is not identical between input and multiprocessing output images.")
        else:
            print(f"Channel {i} is identical between input and multiprocessing output images.")

    # Display the original and output images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Output Image (Identity Kernel - OpenCV)', output_image_cv2)
    cv2.imshow('Output Image (Identity Kernel - Multiprocessing)', output_image_multiprocessing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()