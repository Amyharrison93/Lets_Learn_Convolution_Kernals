import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "__src__"))
from identity_kernal import identity_kernal

@pytest.mark.identity_kernal

def test_identity_kernal():
    # Create a sample image (3x3 RGB)
    sample_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                              [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
                              [[128, 128, 128], [64, 64, 64], [32, 32, 32]]], dtype=np.uint8)

    # Apply the identity kernel
    output_image = identity_kernal(sample_image)

    # Assert that the output image is the same as the input image
    assert np.array_equal(sample_image, output_image), "The output image should be the same as the input image when using the identity kernel."

@pytest.mark.identity_kernal
def test_invalid_kernel_size():
    sample_image = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        identity_kernal(sample_image, kernel_size=4)  # Even kernel size should raise ValueError

@pytest.mark.identity_kernal
def test_invalid_image_shape():
    invalid_image = np.zeros((3, 3), dtype=np.uint8)  # Grayscale image (2D)
    with pytest.raises(ValueError):
        identity_kernal(invalid_image)  # Should raise ValueError for non-color image

    invalid_image = np.zeros((3, 3, 4), dtype=np.uint8)  # Image with 4 channels
    with pytest.raises(ValueError):
        identity_kernal(invalid_image)  # Should raise ValueError for non-color image
