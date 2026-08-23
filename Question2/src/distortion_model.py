"""Camera projection and two-coefficient radial distortion model.

This module only evaluates the forward imaging model. It does not estimate
intrinsics, pose, distortion, or reprojection error.
"""

from __future__ import annotations

import numpy as np


def _points_array(points: np.ndarray, dimensions: int, name: str) -> np.ndarray:
	values = np.asarray(points, dtype=float)
	if values.ndim != 2 or values.shape[1] != dimensions:
		raise ValueError(f"{name} must have shape (N, {dimensions})")
	if not np.all(np.isfinite(values)):
		raise ValueError(f"{name} must contain only finite values")
	return values


def _rotation_matrix(rotation: np.ndarray) -> np.ndarray:
	values = np.asarray(rotation, dtype=float)
	if values.shape != (3, 3):
		raise ValueError("rotation must have shape (3, 3)")
	if not np.all(np.isfinite(values)):
		raise ValueError("rotation must contain only finite values")
	if not np.allclose(values.T @ values, np.eye(3), atol=1e-8):
		raise ValueError("rotation must be an orthonormal matrix")
	if not np.isclose(np.linalg.det(values), 1.0, atol=1e-8):
		raise ValueError("rotation must have determinant +1")
	return values


def world_to_camera(
	world_points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
	"""Transform grid points from world coordinates into camera coordinates.

	The world frame is attached to the planar grid. A grid corner is
	``(X, Y, 0)`` because every corner lies on the same physical plane; the
	third coordinate is not discarded, it is known to be zero. The camera
	frame has its origin at the camera center and its depth axis is ``Zc``.
	"""
	points = _points_array(world_points, 3, "world_points")
	if not np.allclose(points[:, 2], 0.0):
		raise ValueError("world_points must lie on the planar grid Z=0")
	rotation_values = _rotation_matrix(rotation)
	translation_values = np.asarray(translation, dtype=float).reshape(-1)
	if translation_values.shape != (3,):
		raise ValueError("translation must contain exactly 3 values")
	if not np.all(np.isfinite(translation_values)):
		raise ValueError("translation must contain only finite values")

	# Row-vector form of Pc = R P + t. This is equivalent to P @ R.T + t.
	return points @ rotation_values.T + translation_values


def camera_to_normalized(camera_points: np.ndarray) -> np.ndarray:
	"""Apply perspective division and return undistorted normalized points."""
	points = _points_array(camera_points, 3, "camera_points")
	depths = points[:, 2]
	invalid = np.flatnonzero(depths <= 0)
	if invalid.size:
		raise ValueError(
			"camera_points contains non-positive depth at indices "
			f"{invalid.tolist()}"
		)

	# Perspective projection divides transverse coordinates by depth. Without
	# this step, points at different distances would not change apparent size.
	normalized = points[:, :2] / depths[:, np.newaxis]
	if not np.all(np.isfinite(normalized)):
		raise ValueError("perspective normalization produced non-finite values")
	return normalized


def apply_radial_distortion(
	normalized_points: np.ndarray, k1: float, k2: float
) -> np.ndarray:
	"""Apply Brown-Conrady radial distortion to normalized coordinates."""
	points = _points_array(normalized_points, 2, "normalized_points")
	coefficients = np.asarray([k1, k2], dtype=float)
	if not np.all(np.isfinite(coefficients)):
		raise ValueError("k1 and k2 must be finite")

	radius_squared = np.sum(points * points, axis=1)
	radial_scale = 1.0 + k1 * radius_squared + k2 * radius_squared**2
	# Radial distortion enters after perspective normalization and before K
	# converts normalized coordinates into pixel coordinates.
	distorted = points * radial_scale[:, np.newaxis]
	if not np.all(np.isfinite(distorted)):
		raise ValueError("radial distortion produced non-finite values")
	return distorted


def normalized_to_pixel(
	distorted_points: np.ndarray, intrinsic_matrix: np.ndarray
) -> np.ndarray:
	"""Convert distorted normalized coordinates to pixel coordinates."""
	points = _points_array(distorted_points, 2, "distorted_points")
	intrinsic = np.asarray(intrinsic_matrix, dtype=float)
	if intrinsic.shape != (3, 3):
		raise ValueError("intrinsic_matrix must have shape (3, 3)")
	if not np.all(np.isfinite(intrinsic)):
		raise ValueError("intrinsic_matrix must contain only finite values")
	if not np.allclose(intrinsic[2], [0.0, 0.0, 1.0]):
		raise ValueError("intrinsic_matrix must have bottom row [0, 0, 1]")
	if not np.isclose(intrinsic[0, 1], 0.0) or not np.isclose(intrinsic[1, 0], 0.0):
		raise ValueError("intrinsic_matrix must have zero skew and no axis coupling")

	fx, fy = intrinsic[0, 0], intrinsic[1, 1]
	cx, cy = intrinsic[0, 2], intrinsic[1, 2]
	if fx <= 0 or fy <= 0:
		raise ValueError("focal lengths fx and fy must be positive")
	pixels = np.column_stack((fx * points[:, 0] + cx, fy * points[:, 1] + cy))
	if not np.all(np.isfinite(pixels)):
		raise ValueError("pixel conversion produced non-finite values")
	return pixels


def project_points(
	world_points: np.ndarray,
	intrinsic_matrix: np.ndarray,
	rotation: np.ndarray,
	translation: np.ndarray,
	k1: float,
	k2: float,
) -> np.ndarray:
	"""Run the complete grid-to-pixel forward model.

	The returned array has shape ``(N, 2)`` and contains predicted ``(u, v)``
	image coordinates. Invalid camera depths raise before distortion is
	applied, so a point behind or on the camera plane cannot enter the result.
	"""
	camera_points = world_to_camera(world_points, rotation, translation)
	normalized_points = camera_to_normalized(camera_points)
	distorted_points = apply_radial_distortion(normalized_points, k1, k2)
	return normalized_to_pixel(distorted_points, intrinsic_matrix)


def synthetic_example() -> np.ndarray:
	"""Project a few known grid points as a small smoke test."""
	intrinsic = np.array(
		[[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
	)
	rotation = np.eye(3)
	translation = np.array([0.0, 0.0, 5.0])
	grid_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	pixels = project_points(
		grid_points,
		intrinsic,
		rotation,
		translation,
		k1=0.01,
		k2=-0.001,
	)
	expected_center = np.array([320.0, 240.0])
	if not np.allclose(pixels[0], expected_center):
		raise AssertionError("The grid origin should project to the principal point")
	radius_squared = 0.2**2
	radial_scale = 1.0 + 0.01 * radius_squared - 0.001 * radius_squared**2
	expected_pixels = np.array(
		[
			[320.0, 240.0],
			[800.0 * 0.2 * radial_scale + 320.0, 240.0],
			[320.0, 800.0 * 0.2 * radial_scale + 240.0],
		]
	)
	if not np.allclose(pixels, expected_pixels):
		raise AssertionError("Synthetic projection does not match the forward model")
	if not np.all(np.isfinite(pixels)):
		raise AssertionError("Synthetic projection returned non-finite pixels")
	return pixels


if __name__ == "__main__":
	print("Synthetic projected pixel coordinates:")
	print(synthetic_example())
