from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from PIL import Image as PillowImage

TImage = TypeVar("TImage", bound="Image")


@dataclass
class Image:
    """A container class for performing various image related manipulations."""

    data: npt.NDArray[np.uint8]

    def blur(self: TImage, kernel_size: int, sigma: float) -> TImage:
        """Perform a Gaussian blurring of the image.

        Keyword arguments:
        kernel_size -- the size of the Gaussian kernel
        sigma -- the standard deviation for the Gaussian kernel
        """

        kernel = compute_gaussian_kernel(kernel_size, sigma)

        output = np.zeros_like(self.data)
        self.data = np.pad(self.data, pad_width=int((kernel_size - 1) / 2), mode="edge")
        for y in range(self.data.shape[1] - kernel_size + 1):
            for x in range(self.data.shape[0] - kernel_size + 1):
                output[x, y] = np.sum(
                    kernel
                    * self.data[
                        x : x + kernel_size,
                        y : y + kernel_size,
                    ]
                )

        self.data = output.astype(np.uint8)
        return self

    def segment(self: TImage, threshold: int = 240) -> TImage:
        """Perform a basic image segmentation of the image.

        The image is segmented into two regions, where the foreground is defined as the parts of the image with lower intensities.
        The foreground elements are mapped to 1 while the background elements are mapped to 0.

        Keyword arguments:
        threshold -- intensities smaller than this value will be included in the foreground region of the segmentation
        """

        output = np.zeros_like(self.data)
        output[self.data < threshold] = 1
        output[self.data >= threshold] = 0
        return type(self)(data=output)

    @property
    def center_of_gravity(self) -> tuple[float, float]:
        """Compute the center of gravity of the foreground region of an image."""

        image_segmented = self.segment()
        return tuple(np.average(np.nonzero(image_segmented.data), axis=1))  # type: ignore

    @property
    def principal_axes(self) -> npt.NDArray[np.float64]:
        """Compute the principal axes of the foreground region of an image.

        The eigenvectors constituting the principal axes are returned in descending order
        based on their corresponding eigenvalues.
        """

        image_segmented = self.segment()
        region_coordinates = np.array(np.nonzero(image_segmented.data)).T
        num_points = np.sum(image_segmented.data)
        mean_vector = 1 / num_points * np.sum(region_coordinates, axis=0)

        covariance_matrix = (
            (1 / (num_points - 1))
            * (region_coordinates - mean_vector).T
            @ (region_coordinates - mean_vector)
        )

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvectors  # type: ignore

    def get_image_with_normalized_foreground_region(self: TImage) -> TImage:
        """Return a boundary image with the foreground transformed such that it has a normalized translation and rotation."""

        image_segmented = self.segment()
        region_coordinates = np.array(np.nonzero(image_segmented.data)).T
        coordinates_transformed = (
            (region_coordinates - self.center_of_gravity) @ self.principal_axes
            + tuple(dim / 2 for dim in self.data.shape)
        ).astype(np.int64)

        output = np.zeros_like(image_segmented.data)
        output[coordinates_transformed[:, 0], coordinates_transformed[:, 1]] = 255

        return type(self)(data=output)


def load(path: Path, shape: tuple[int, int]) -> Image:
    """Load an image stored in 8-bit binary raw format (.bin) with a known shape."""
    data = np.fromfile(path, dtype=np.uint8).reshape(shape)
    return Image(data=data)


def save(path: Path, image: Image) -> None:
    """Save an image to disk.

    The image format is inferred from the specified path.
    """
    image_pillow = PillowImage.fromarray(image.data)
    image_pillow.save(path)


def compute_gaussian_kernel(kernel_size: int, sigma: float) -> npt.NDArray[np.float64]:
    """Compute the normalized kernel used for Gaussian blurring.

    Keyword arguments:
    kernel_size -- the size of the Gaussian kernel
    sigma -- the standard deviation for the Gaussian kernel
    """
    kernel_range = np.arange(kernel_size)
    s, t = np.meshgrid(kernel_range, kernel_range)
    center = (kernel_size - 1) / 2
    r = np.sqrt((s - center) ** 2 + (t - center) ** 2)

    kernel = np.exp(-0.5 * (r / sigma) ** 2).astype(np.float64)
    scale = 1.0 / np.sum(kernel)

    return scale * kernel
