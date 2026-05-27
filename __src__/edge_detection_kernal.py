import cv2
import numpy as np
import time

from identity_kernal import convolution_kernals

class edge_detection_kernal(convolution_kernals):
    """
    A class that implements edge detection kernels for image processing. 
    This class provides both custom implementations of vertical and horizontal edge detection kernels 
    and wrapper functions that utilize OpenCV's optimized filter2D function for comparison.
    """
    def generate_vertical_edge_detection_kernel(self) -> np.ndarray:
        """
        A helper function to generate a vertical edge detection kernel.
        Returns:
        numpy.ndarray: The generated vertical edge detection kernel, which is a 3x3 matrix that highlights vertical edges in an image.
        """
        kernel = np.array([[ -1, 0, 1],
                        [ -2, 0, 2],
                        [ -1, 0, 1]], dtype=np.float32)
        return kernel
    
    def generate_horizontal_edge_detection_kernel(self) -> np.ndarray:
        """
        A helper function to generate a horizontal edge detection kernel.
        Returns:
        numpy.ndarray: The generated horizontal edge detection kernel, which is a 3x3 matrix that highlights horizontal edges in an image.
        """
        kernel = np.array([[ -1, -2, -1],
                        [  0,  0,  0],
                        [  1,  2,  1]], dtype=np.float32)
        return kernel

    def vertical_edge_detection_kernal_cv2(self, image: np.ndarray) -> np.ndarray:  
        """
        A simple wrapper function for the vertical edge detection kernel that uses OpenCV's filter2D function.
        Parameters:
        image (numpy.ndarray): The input image to which the vertical edge detection kernel will be applied.
        Returns:
        numpy.ndarray: The output image after applying the vertical edge detection kernel, which highlights vertical edges in the input image.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        # Define the vertical edge detection kernel
        kernel = self.generate_vertical_edge_detection_kernel()

        # Apply the kernel to the image using cv2.filter2D
        try:
            output_image = cv2.filter2D(image, -1, kernel)
        except Exception as e:
            raise ValueError(f"Error occurred while applying the vertical edge detection kernel: {e}")

        return output_image

    def vertical_edge_detection_kernal(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a vertical edge detection kernel to the input image. The vertical edge detection kernel is a 3x3 matrix that highlights vertical edges in the image when applied.
        warning: this is very slow and just an example of how to implement a convolution kernel from scratch. For practical use, consider using the vertical_edge_detection_kernal_cv2 function which utilizes OpenCV's optimized filter2D function.
        Parameters:
        image (numpy.ndarray): The input image to which the vertical edge detection kernel will be applied.
        Returns:
        numpy.ndarray: The output image after applying the vertical edge detection kernel, which highlights vertical edges in the input image.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        kernel = self.generate_vertical_edge_detection_kernel()

        return self.apply_kernel(image, kernel)

    def horizontal_edge_detection_kernal_cv2(self, image: np.ndarray) -> np.ndarray:
        """
        A simple wrapper function for the horizontal edge detection kernel that uses OpenCV's filter2D function.
        Parameters:
        image (numpy.ndarray): The input image to which the horizontal edge detection kernel will be applied.
        Returns:
        numpy.ndarray: The output image after applying the horizontal edge detection kernel, which highlights horizontal edges in the input image.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        # Define the horizontal edge detection kernel
        kernel = self.generate_horizontal_edge_detection_kernel()

        # Apply the kernel to the image using cv2.filter2D
        try:
            output_image = cv2.filter2D(image, -1, kernel)
        except Exception as e:
            raise ValueError(f"Error occurred while applying the horizontal edge detection kernel: {e}")

        return output_image

    def horizontal_edge_detection_kernal(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a horizontal edge detection kernel to the input image. The horizontal edge detection kernel is a 3x3 matrix that highlights horizontal edges in the image when applied.
        warning: this is very slow and just an example of how to implement a convolution kernel from scratch. For practical use, consider using the horizontal_edge_detection_kernal_cv2 function which utilizes OpenCV's optimized filter2D function.
        Parameters:
        image (numpy.ndarray): The input image to which the horizontal edge detection kernel will be applied.
        Returns:
        numpy.ndarray: The output image after applying the horizontal edge detection kernel, which highlights horizontal edges in the input image.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color image (3 channels).")
        
        kernel = self.generate_horizontal_edge_detection_kernel()
        #average_pixel_value = np.mean(image, axis=2)  # Average across color channels to get a single channel for edge detection

        return self.apply_kernel(image, kernel)

    async def channelwise_convolution_kernal(self, channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply a convolution kernel to a single color channel of an image. This function is designed to be used with asyncio for concurrent processing of multiple channels.
        Parameters:
        channel (numpy.ndarray): A 2D array representing a single color channel of the image.
        kernel (numpy.ndarray): A 2D array representing the convolution kernel to be applied to the channel.
        Returns:
        numpy.ndarray: The output channel after applying the convolution kernel, which contains the convolved values.
        """
        output_channel = np.zeros_like(channel)
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2

        # Pad the input channel to handle borders
        padded_channel = np.pad(channel, pad_size, mode='constant', constant_values=0)

        for i in range(len(channel)):
            for j in range(len(channel[0])):
                output_value = 0
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        x = i + m
                        y = j + n
                        output_value += padded_channel[x, y] * kernel[m, n]
                output_channel[i, j] = np.clip(output_value, 0, 255)

        return output_channel

def main():
    input_image = convolution_kernals.import_image()
    detectionKernal = edge_detection_kernal()
    vertical_edges_cv2 = detectionKernal.vertical_edge_detection_kernal_cv2(input_image.copy())
    horizontal_edges_cv2 = detectionKernal.horizontal_edge_detection_kernal_cv2(input_image.copy())

    vertical_edges = detectionKernal.vertical_edge_detection_kernal(input_image.copy())
    horizontal_edges = detectionKernal.horizontal_edge_detection_kernal(input_image.copy())

    cv2.imshow("Horizontal Edges (OpenCV)", horizontal_edges_cv2)
    cv2.imshow("Horizontal Edges (Custom)", horizontal_edges)

    cv2.imshow("Vertical Edges (OpenCV)", vertical_edges_cv2)
    cv2.imshow("Vertical Edges (Custom)", vertical_edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()