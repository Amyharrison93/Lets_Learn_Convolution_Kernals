import cv2
import numpy as np
import pathlib
import time

class convolution_kernals():
    """
    A class that implements various convolution kernels for image processing. 
    This class provides both custom implementations and wrapper functions that utilize OpenCV's optimized filter2D function for comparison.
    """

    def import_image(image_path: str = None) -> np.ndarray:
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
            
        input_image = cv2.imread(str(image_path))

        if input_image is None:
            raise ValueError("Could not load the image. Please check the file path.")
        return input_image
    
    def generate_identity_kernel(kernel_size: int=3) -> np.ndarray:
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
    
    def identity_kernal_cv2(image: np.ndarray, kernel_size: int=3) -> np.ndarray:
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
    input_image = convolution_kernals.import_image()
    
    print(f"Input image shape: {input_image.shape}")

    start_time = time.time()
    # Apply the identity kernel to the input image
    kernal = convolution_kernals.generate_identity_kernel(kernel_size=3)
    print(f"Using identity kernel:\n{kernal}")

    output_image = convolution_kernals.apply_kernel(input_image, kernal)
    end_time = time.time()
    print(f"Output image shape: {output_image.shape}")
    print(f"Time taken (custom implementation): {end_time - start_time:.4f} seconds")

    start_time = time.time()
    output_image_cv2 = convolution_kernals.identity_kernal_cv2(input_image)
    end_time = time.time()
    print(f"Output image (cv2) shape: {output_image_cv2.shape}")
    print(f"Time taken (OpenCV implementation): {end_time - start_time:.4f} seconds")


    for i in range(3):
        if not np.array_equal(input_image[:, :, i], output_image[:, :, i]):
            print(f"Channel {i} is not identical between input and output images.")
        else:
            print(f"Channel {i} is identical between input and output images.")

    for i in range(3):
        if not np.array_equal(input_image[:, :, i], output_image_cv2[:, :, i]):
            print(f"Channel {i} is not identical between input and OpenCV output images.")
        else:
            print(f"Channel {i} is identical between input and OpenCV output images.")

    # Display the original and output images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Output Image (Identity Kernel)', output_image)
    cv2.imshow('Output Image (Identity Kernel - OpenCV)', output_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()