from pathlib import Path

import numpy as np

import vision
from vision.point_cloud import PointCloud


def test_load_and_save_point_cloud(
    test_directory: Path, output_directory: Path
) -> None:
    """Test loading of an image stored in 8-bit binary raw format (.bin)."""

    point_cloud = vision.point_cloud.load(
        test_directory / "sample_files" / "pineapple.bin", shape=(1200 * 1920, 3)
    )
    assert point_cloud.points.shape[-1] == 3

    point_cloud_simplified = PointCloud(points=point_cloud.points[0:10, :])
    vision.point_cloud.save(output_directory / "pineapple.xyz", point_cloud_simplified)
    point_cloud_copy = vision.point_cloud.load(output_directory / "pineapple.xyz")
    assert point_cloud_simplified == point_cloud_copy


def test_multiplying_by_a_translation_matrix() -> None:
    """Test applying a translation"""

    point_cloud = PointCloud(points=np.array([[0, 0, 0], [-3, 4, 5]]))
    point_cloud.transform(
        np.array([[1, 0, 0, 5], [0, 1, 0, -3], [0, 0, 1, 2], [0, 0, 0, 1]])
    )

    assert np.allclose(point_cloud.points, np.array([[5, -3, 2], [2, 1, 7]]))


def test_segment_plane() -> None:
    """Test that the biggest plane is correctly identified"""

    point_cloud = PointCloud(
        # Points sampled from the plane x + y + z + 1 = 0
        points=np.array(
            [
                [2.0, 1.0, -4.0],
                [1.0, 3.0, -5.0],
                [-2.0, -1.0, 2.0],
                [-2.0, -2.0, 3.0],
                [10.0, 10.0, -21.0],
            ]
        )
    )

    _, inliers = point_cloud.segment_plane(distance_threshold=1, num_iterations=10)
    assert len(inliers) == 5


def test_rotate_plane_to_xy_plane() -> None:
    """Test that a transformation matrix can be computed to transform any plane the xy plane"""

    plane_model = np.array([0, 0, 1, 0])
    assert np.allclose(vision.point_cloud.get_plane_transform(plane_model), np.eye(4))
