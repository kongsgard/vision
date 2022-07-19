from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import numpy.typing as npt

TPointCloud = TypeVar("TPointCloud", bound="PointCloud")


@dataclass
class PointCloud:
    """A container class for point clouds and associated operations."""

    points: npt.NDArray[np.float32]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointCloud):
            return NotImplemented
        return np.allclose(self.points, other.points, atol=1e-6)

    def transform(
        self: TPointCloud, transformation: npt.NDArray[np.float64]
    ) -> TPointCloud:
        """Transform the points by a homogeneous 4x4 transformation matrix."""
        assert transformation.shape == (4, 4)

        self.points = (
            np.hstack((self.points, np.ones((self.points.shape[0], 1))))
            @ transformation.T
        )[:, :3]
        return self

    def segment_plane(
        self, distance_threshold: float = 0.01, num_iterations: int = 1000
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Identify and segment the plane consisting of the most points in the point cloud."""

        def compute_triangle_plane(
            point_0: npt.NDArray[np.float64],
            point_1: npt.NDArray[np.float64],
            point_2: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Compute the plane equation ax + by + cz + d = 0 from three points."""

            plane_normal = np.cross(point_1 - point_0, point_2 - point_0)
            plane_normal /= np.linalg.norm(plane_normal)
            d = -np.sum(np.multiply(plane_normal, point_0))
            return np.append(plane_normal, d)

        best_plane_model = np.zeros(
            4
        )  # Initialize the best plane model ax + by + cz + d = 0
        best_inliers = np.zeros(1, dtype=np.int64)  # Initialize the consensus set

        for _ in range(num_iterations):
            point_samples = self.points[
                np.random.choice(self.points.shape[0], 3, replace=False)
            ]
            plane_model = compute_triangle_plane(
                point_samples[0], point_samples[1], point_samples[2]
            )
            if np.isnan(plane_model).any():
                continue

            points_to_plane_distances = (
                plane_model[0] * self.points[:, 0]
                + plane_model[1] * self.points[:, 1]
                + plane_model[2] * self.points[:, 2]
                + plane_model[3]
            ) / np.sqrt(plane_model[0] ** 2 + plane_model[1] ** 2 + plane_model[2] ** 2)

            inliers = np.where(np.abs(points_to_plane_distances) <= distance_threshold)[
                0
            ]
            if len(inliers) > len(best_inliers):
                best_plane_model = plane_model
                best_inliers = inliers

        return best_plane_model, best_inliers


def load(path: Path, shape: tuple[int, int] | None = None) -> PointCloud:
    """Load 32-bit single precision float point cloud.

    Any points with NaN as one or more of their coordinates will be removed.

    Supported formats:
    - binary raw format (.bin) with a known shape
    - row-by-row text file format (.xyz)
    """
    match path.suffix:
        case ".bin":
            assert (
                shape is not None
            ), "A shape has to be defined when loading the .bin format"
            points = np.fromfile(path, dtype=np.float32).reshape(shape)
        case ".xyz":
            points = np.loadtxt(path, dtype=np.float32)
        case _:
            raise ImportError(
                f"The provided file format {path.suffix} is not supported"
            )

    return PointCloud(points=points[~np.isnan(points).any(axis=1)])


def save(path: Path, point_cloud: PointCloud) -> None:
    """Save a point cloud to disk in the .xyz format.

    Each line contains [x, y, z], where x, y, z are the 3D coordinates.
    """
    assert path.suffix == ".xyz", f"The {path.suffix} format is not supported"

    np.savetxt(path, point_cloud.points)


def get_plane_transform(
    plane_model: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Get the 4x4 homogeneous transform which transforms the plane to be in the XY plane"""

    transform = align_vectors(plane_model[:3], np.array([0, 0, 1]))
    transform[2, 3] = (
        -plane_model[3] / plane_model[2]
    )  # TODO: Improve numerical stability
    return transform


def align_vectors(
    vector_0: npt.NDArray[np.float64], vector_1: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the 4x4 homogeneous transformation matrix which rotates vector_0 to vector_1"""
    assert vector_0.shape == (3,)
    assert vector_1.shape == (3,)

    vector_0_unitary = np.linalg.svd(vector_0.reshape((-1, 1)))[0]
    vector_1_unitary = np.linalg.svd(vector_1.reshape((-1, 1)))[0]

    if np.linalg.det(vector_0_unitary) < 0:
        vector_0_unitary[:, -1] *= -1.0
    if np.linalg.det(vector_1_unitary) < 0:
        vector_1_unitary[:, -1] *= -1.0

    transform = np.eye(4)
    transform[:3, :3] = vector_1_unitary.dot(vector_0_unitary.T)

    return transform
