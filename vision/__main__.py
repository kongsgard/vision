from pathlib import Path

import image
import numpy as np
import point_cloud
import pyvista as pv
from image import Image
from point_cloud import get_plane_transform


def process_image() -> None:
    """Apply some various image processing algorithms to an image of a plane"""

    image_path = (
        Path(__file__).parent.parent / "tests" / "sample_files" / "plane_640x480.bin"
    )
    plane_image = image.load(image_path, shape=(480, 640))
    image.save(image_path.with_name("plane.bmp"), plane_image)

    plane_image.blur(kernel_size=5, sigma=1.0)
    image.save(image_path.with_name("plane_blurred.bmp"), plane_image)

    plane_segmented = plane_image.segment(threshold=240)
    plane_segmented.data *= 255
    image.save(image_path.with_name("plane_segmented.bmp"), plane_segmented)

    print(f"Center of gravity: {plane_image.center_of_gravity}")
    plane_image_with_center_marked = Image(data=np.zeros_like(plane_image.data))
    plane_image_with_center_marked.data[
        int(plane_image.center_of_gravity[0]), int(plane_image.center_of_gravity[1])
    ] = 255
    image.save(image_path.with_name("plane_center.bmp"), plane_image_with_center_marked)

    print(
        f"Principal axes: {plane_image.principal_axes[0]} and {plane_image.principal_axes[1]}"
    )
    image_transformed = plane_image.get_image_with_normalized_foreground_region()
    image.save(image_path.with_name("plane_transformed.bmp"), image_transformed)


def process_point_cloud() -> None:
    """Transform and segment a point cloud of an object lying on a table"""
    point_cloud_path = (
        Path(__file__).parent.parent / "tests" / "sample_files" / "pineapple.bin"
    )
    pineapple = point_cloud.load(point_cloud_path, shape=(1200 * 1920, 3))

    plane_model, inliers = pineapple.segment_plane(
        distance_threshold=10, num_iterations=20
    )
    mask = np.ones(pineapple.points.shape[0], dtype=bool)
    mask[inliers] = False
    pineapple.points = pineapple.points[mask]
    transform = get_plane_transform(plane_model)
    pineapple.transform(transform)

    point_cloud.save(point_cloud_path.with_name("pineapple_segmented.xyz"), pineapple)


if __name__ == "__main__":
    process_image()
    process_point_cloud()
