import cv2
import numpy as np
import pathlib

def vertical_edge_detection_kernal_cv2(image: np.ndarray) -> np.ndarray:
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
    kernel = np.array([[ -1, 0, 1],
                       [ -2, 0, 2],
                       [ -1, 0, 1]], dtype=np.float32)

    # Apply the kernel to the image using cv2.filter2D
    try:
        output_image = cv2.filter2D(image, -1, kernel)
    except Exception as e:
        raise ValueError(f"Error occurred while applying the vertical edge detection kernel: {e}")

    return output_image

def vertical_edge_detection_kernal(image: np.ndarray) -> np.ndarray:
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
    
    kernel = np.array([[ -1, 0, 1],
                       [ -2, 0, 2],
                       [ -1, 0, 1]], dtype=np.float32)

    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(3):  # For each color channel
                output_pixel_value = 0.0
                for m in range(3):
                    for n in range(3):
                        x = i + m - 1
                        y = j + n - 1
                        if 0 <= x < len(image) and 0 <= y < len(image[0]):
                            output_pixel_value += image[x, y, k] * kernel[m, n]
                image[i, j, k] = np.clip(output_pixel_value, 0, 255)

    return image

def horizontal_edge_detection_kernal_cv2(image: np.ndarray) -> np.ndarray:
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
    kernel = np.array([[ -1, -2, -1],
                       [  0,  0,  0],
                       [  1,  2,  1]], dtype=np.float32)

    # Apply the kernel to the image using cv2.filter2D
    try:
        output_image = cv2.filter2D(image, -1, kernel)
    except Exception as e:
        raise ValueError(f"Error occurred while applying the horizontal edge detection kernel: {e}")

    return output_image

def horizontal_edge_detection_kernal(image: np.ndarray) -> np.ndarray:
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
    
    kernel = np.array([[ -1, -2, -1],
                       [  0,  0,  0],
                       [  1,  2,  1]], dtype=np.float32)

    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(3):  # For each color channel

                output_pixel_value = 0
                for m in range(3):
                    for n in range(3):
                        x = i + m - 1
                        y = j + n - 1
                        if 0 <= x < len(image) and 0 <= y < len(image[0]):
                            output_pixel_value += image[x, y, k] * kernel[m, n]
                image[i, j, k] = np.clip(output_pixel_value, 0, 255)

    return image

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    image_path = script_dir / ".." / "__documents__" / "image.png"  # Update this path to your image location
    input_image = cv2.imread(str(image_path))

    if input_image is None:
        raise ValueError("Could not load the image. Please check the file path.")
    
    vertical_edges_cv2 = vertical_edge_detection_kernal_cv2(input_image.copy())
    horizontal_edges_cv2 = horizontal_edge_detection_kernal_cv2(input_image.copy())

    vertical_edges = vertical_edge_detection_kernal(input_image.copy())
    horizontal_edges = horizontal_edge_detection_kernal(input_image.copy())

    cv2.imshow("Horizontal Edges (OpenCV)", horizontal_edges_cv2)
    cv2.imshow("Horizontal Edges (Custom)", horizontal_edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()