import cv2
import numpy as np
import pathlib

from identity_kernal import convolution_kernals

class blur_kernals(convolution_kernals):
    """
    A class that implements blur kernels for image processing. The blur kernel is a convolution kernel that smooths the image by averaging the pixel values 
    in a neighborhood. This class provides both a custom implementation of the blur kernel 
    and a wrapper function that utilizes OpenCV's optimized filter2D function for comparison.
    """

    def generate_blur_kernel(self, kernel_size: int=3) -> np.ndarray:
        """
        A helper function to generate a blur kernel of the specified size. The blur kernel is a matrix that averages the pixel values in a neighborhood, 
        resulting in a smoothed version of the image when applied.
        Parameters:
        kernel_size (int): The size of the blur kernel (default is 3).
        Returns:
        numpy.ndarray: The generated blur kernel, which is a matrix where each element is equal to 1/(kernel_size*kernel_size), 
                       effectively averaging the pixel values in the neighborhood when applied to an image.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be an odd number.")
        
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        return kernel

    def blur_kernal(image: np.ndarray, kernel_size: int=3) -> np.ndarray:
        """
        Apply a blur kernel to the input image. The blur kernel is a matrix that averages the pixel values in a neighborhood, 
        resulting in a smoothed version of the image when applied.
        warning: this is very slow and just an example of how to implement a convolution kernel from scratch.
        Parameters:
        image (numpy.ndarray): The input image to which the blur kernel will be applied.
        kernel_size (int): The size of the blur kernel (default is 3).
        Returns:
        numpy.ndarray: The output image after applying the blur kernel, which is a smoothed version of the input image.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be an odd number.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        blur_kernal = blur_kernals()
        kernel = blur_kernal.generate_blur_kernel(kernel_size)

        return blur_kernal.apply_kernel(image, kernel)

    def blur_kernal_cv2(image: np.ndarray, kernel_size: int=3) -> np.ndarray:
        """
        A simple wrapper function for the blur kernel that uses OpenCV's filter2D function.
        Parameters:
        image (numpy.ndarray): The input image to which the blur kernel will be applied.
        kernel_size (int): The size of the blur kernel (default is 3).
        Returns:
        numpy.ndarray: The output image after applying the blur kernel, which is a smoothed version of the input image.
        """
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be an odd number.")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        # Define the blur kernel
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

        # Apply the kernel to the image using cv2.filter2D
        try:
            output_image = cv2.filter2D(image, -1, kernel)
        except Exception as e:
            raise ValueError(f"Error occurred while applying the blur kernel: {e}")

        return output_image
    
def main():
    input_image = convolution_kernals.import_image()

    blurred_image_cv2 = blur_kernals.blur_kernal_cv2(input_image.copy())
    blurred_image = blur_kernals.blur_kernal(input_image.copy())

    cv2.imshow("Blurred Image (OpenCV)", blurred_image_cv2)
    cv2.imshow("Blurred Image (Custom)", blurred_image)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()