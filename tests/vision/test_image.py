from pathlib import Path

import numpy as np

import vision
from vision.image import Image


def test_load_image(test_directory: Path) -> None:
    """Test loading of an image stored in 8-bit binary raw format (.bin)."""

    image = vision.image.load(
        test_directory / "sample_files" / "plane_640x480.bin", shape=(480, 640)
    )

    assert image.data.shape == (480, 640)


def test_compute_gaussian_kernel() -> None:
    """Test computation of a Gaussian kernel of size 3x3 with sigma 1.0."""

    kernel = vision.image.compute_gaussian_kernel(kernel_size=3, sigma=1.0)

    kernel_expected = (1 / 4.8976) * np.array(
        [[0.3679, 0.6065, 0.3679], [0.6065, 1.0000, 0.6065], [0.3679, 0.6065, 0.3679]],
        dtype=np.float64,
    )

    assert np.allclose(kernel, kernel_expected, atol=1e-4)


def test_gaussian_blur_image() -> None:
    """Test that filtering an image with a Gaussian kernel produces a blurred image"""

    image = Image(data=np.indices((4, 4)).sum(axis=0) % 2 * 10)
    image.blur(kernel_size=5, sigma=1.0)

    image_expected = np.array(
        [[3, 4, 5, 6], [4, 4, 5, 5], [5, 5, 4, 4], [6, 5, 4, 3]], dtype=np.uint8
    )

    assert np.array_equal(image.data, image_expected)


def test_segment_image() -> None:
    """Test that segmenting an image produces two distinct regions"""

    image = Image(
        data=np.array([[240, 245, 5], [150, 230, 210], [250, 250, 250]], dtype=np.uint8)
    )
    image = image.segment()

    image_expected = np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

    assert np.array_equal(image.data, image_expected)


def test_center_of_gravity() -> None:
    """Test computation of the center of gravity of the foreground region in an image"""

    image = Image(
        data=np.array([[255, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    )

    assert np.allclose(image.center_of_gravity, (1.0, 1.5))


def test_principal_axes() -> None:
    """Test computation of the principal axes of the foreground region in an image"""

    image = Image(
        data=np.array([[255, 255, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
    )

    assert np.allclose(image.principal_axes[0], (0, 1))
    assert np.allclose(image.principal_axes[1], (1, 0))
