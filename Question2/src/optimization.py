"""Robust reprojection cost for the camera model.

This module evaluates an objective only. It deliberately does not choose
parameters, run RANSAC, undistort images, or calculate final validation
statistics.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .distortion_model import project_points


def _array(values: np.ndarray, columns: int, name: str) -> np.ndarray:
	array = np.asarray(values, dtype=float)
	if array.ndim != 2 or array.shape[1] != columns:
		raise ValueError(f"{name} must have shape (N, {columns})")
	if not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must contain only finite values")
	return array


def predicted_image_points(
	world_points: np.ndarray,
	intrinsic_matrix: np.ndarray,
	rotation: np.ndarray,
	translation: np.ndarray,
	k1: float,
	k2: float,
) -> np.ndarray:
	"""Return predicted pixels using the existing forward projection model."""
	world = _array(world_points, 3, "world_points")
	if not np.allclose(world[:, 2], 0.0):
		raise ValueError("world_points must lie on the planar grid Z=0")
	return project_points(world, intrinsic_matrix, rotation, translation, k1, k2)


def reprojection_residuals(
	world_points: np.ndarray,
	observed_points: np.ndarray,
	intrinsic_matrix: np.ndarray,
	rotation: np.ndarray,
	translation: np.ndarray,
	k1: float,
	k2: float,
) -> np.ndarray:
	"""Return ``observed_points - predicted_points`` for every correspondence."""
	world = _array(world_points, 3, "world_points")
	observed = _array(observed_points, 2, "observed_points")
	if len(world) != len(observed):
		raise ValueError("world_points and observed_points must have equal length")
	if not np.allclose(world[:, 2], 0.0):
		raise ValueError("world_points must lie on the planar grid Z=0")

	predicted = predicted_image_points(
		world, intrinsic_matrix, rotation, translation, k1, k2
	)
	residuals = observed - predicted
	if not np.all(np.isfinite(residuals)):
		raise ValueError("reprojection residuals must be finite")
	return residuals


def residual_magnitudes(residuals: np.ndarray) -> np.ndarray:
	"""Return the Euclidean magnitude of each 2D residual vector."""
	values = _array(residuals, 2, "residuals")
	magnitudes = np.linalg.norm(values, axis=1)
	if not np.all(np.isfinite(magnitudes)):
		raise ValueError("residual magnitudes must be finite")
	return magnitudes


def huber_loss(residual_magnitudes_values: np.ndarray, delta: float) -> np.ndarray:
	"""Return the Huber penalty for each non-negative residual magnitude.

	The quadratic branch preserves sensitivity to small detector noise. The
	linear branch limits the influence of a noisy or incorrect grid point, so a
	single bad detection cannot dominate the whole parameter objective.
	"""
	values = np.asarray(residual_magnitudes_values, dtype=float)
	if values.ndim != 1 or not np.all(np.isfinite(values)):
		raise ValueError("residual magnitudes must be a finite 1D array")
	if np.any(values < 0):
		raise ValueError("residual magnitudes must be non-negative")
	if not np.isfinite(delta) or delta <= 0:
		raise ValueError("delta must be finite and positive")

	return np.where(
		values <= delta,
		0.5 * values**2,
		delta * (values - 0.5 * delta),
	)


def robust_reprojection_cost(
	world_points: np.ndarray,
	observed_points: np.ndarray,
	intrinsic_matrix: np.ndarray,
	rotation: np.ndarray,
	translation: np.ndarray,
	k1: float,
	k2: float,
	delta: float,
) -> float:
	"""Return ``sum_i rho(||observed_i - predicted_i||_2)``."""
	residuals = reprojection_residuals(
		world_points,
		observed_points,
		intrinsic_matrix,
		rotation,
		translation,
		k1,
		k2,
	)
	cost = float(np.sum(huber_loss(residual_magnitudes(residuals), delta)))
	if not np.isfinite(cost):
		raise ValueError("robust reprojection cost is not finite")
	return cost


def rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
	"""Convert a 3-vector to a valid rotation matrix with Rodrigues' formula."""
	vector = np.asarray(rotation_vector, dtype=float).reshape(-1)
	if vector.shape != (3,) or not np.all(np.isfinite(vector)):
		raise ValueError("rotation_vector must contain 3 finite values")
	angle = np.linalg.norm(vector)
	if angle < 1e-12:
		return np.eye(3)
	axis = vector / angle
	axis_matrix = np.array(
		[[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
	)
	return (
		np.eye(3)
		+ np.sin(angle) * axis_matrix
		+ (1.0 - np.cos(angle)) * (axis_matrix @ axis_matrix)
	)


def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
	"""Unpack ``[fx, fy, cx, cy, k1, k2, rx, ry, rz, tx, ty, tz]``.

	The rotation vector is converted to a matrix, so every optimizer trial uses
	a valid rotation rather than nine unconstrained matrix entries.
	"""
	values = np.asarray(theta, dtype=float).reshape(-1)
	if values.shape != (12,) or not np.all(np.isfinite(values)):
		raise ValueError("theta must be a finite vector with 12 values")
	intrinsic = np.array(
		[[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]]
	)
	rotation = rotation_vector_to_matrix(values[6:9])
	translation = values[9:12]
	return intrinsic, rotation, translation, values[4], values[5]


def cost_from_theta(
	theta: np.ndarray,
	world_points: np.ndarray,
	observed_points: np.ndarray,
	delta: float,
) -> float:
	"""Evaluate the robust objective for an optimizer parameter vector."""
	intrinsic, rotation, translation, k1, k2 = unpack_theta(theta)
	return robust_reprojection_cost(
		world_points,
		observed_points,
		intrinsic,
		rotation,
		translation,
		k1,
		k2,
		delta,
	)


def initialize_theta(
	world_points: np.ndarray,
	observed_points: np.ndarray,
	image_size: tuple[int, int] | None = None,
) -> np.ndarray:
	"""Create a practical initial vector from RANSAC inlier correspondences."""
	world = _array(world_points, 3, "world_points")
	observed = _array(observed_points, 2, "observed_points")
	if len(world) != len(observed) or len(world) < 4:
		raise ValueError("at least four matched inlier points are required")
	if image_size is not None:
		width, height = image_size
		if width <= 0 or height <= 0:
			raise ValueError("image_size must be positive")
		focal_length = float(max(width, height))
		principal_point = np.array([width / 2.0, height / 2.0])
	else:
		focal_length = float(max(np.ptp(observed[:, 0]), np.ptp(observed[:, 1]), 1.0))
		principal_point = np.mean(observed, axis=0)
	depth = max(float(np.ptp(world[:, :2])), 1.0)
	return np.array(
		[
			focal_length,
			focal_length,
			principal_point[0],
			principal_point[1],
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			0.0,
			depth,
		],
		dtype=float,
	)


def optimize_parameters(
	world_points: np.ndarray,
	observed_points: np.ndarray,
	initial_theta: np.ndarray | None = None,
	delta: float = 3.0,
	max_iterations: int = 1000,
	image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, object, float, float]:
	"""Minimize the existing robust cost from RANSAC inlier correspondences."""
	world = _array(world_points, 3, "world_points")
	observed = _array(observed_points, 2, "observed_points")
	if len(world) != len(observed) or len(world) < 4:
		raise ValueError("at least four matched inlier points are required")
	initial = initialize_theta(world, observed, image_size) if initial_theta is None else np.asarray(initial_theta, dtype=float).reshape(-1)
	if initial.shape != (12,) or not np.all(np.isfinite(initial)):
		raise ValueError("initial_theta must be a finite vector with 12 values")
	initial_cost = cost_from_theta(initial, world, observed, delta)

	def objective(theta: np.ndarray) -> float:
		try:
			return cost_from_theta(theta, world, observed, delta)
		except (ValueError, FloatingPointError):
			return np.finfo(float).max / 1000.0

	result = minimize(
		objective,
		initial,
		method="Powell",
		options={"maxiter": max_iterations, "xtol": 1e-8, "ftol": 1e-8},
	)
	final_cost = objective(result.x)
	return result.x, result, float(initial_cost), float(final_cost)


def synthetic_example() -> tuple[float, float, float]:
	"""Check quadratic and linear Huber branches with synthetic observations."""
	intrinsic = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
	rotation = np.eye(3)
	translation = np.array([0.0, 0.0, 5.0])
	world = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
	predicted = predicted_image_points(world, intrinsic, rotation, translation, 0.0, 0.0)
	observed = predicted + np.array([[0.5, 0.0], [10.0, 0.0]])
	residuals = reprojection_residuals(
		world, observed, intrinsic, rotation, translation, 0.0, 0.0
	)
	if not np.allclose(residuals, observed - predicted):
		raise AssertionError("Residuals do not equal observed minus predicted points")
	magnitudes = residual_magnitudes(residuals)
	delta = 2.0
	losses = huber_loss(magnitudes, delta)
	if not np.isclose(losses[0], 0.5 * 0.5**2):
		raise AssertionError("Small residual did not use the quadratic Huber branch")
	if not np.isclose(losses[1], delta * (10.0 - 0.5 * delta)):
		raise AssertionError("Large residual did not use the linear Huber branch")
	cost = robust_reprojection_cost(
		world, observed, intrinsic, rotation, translation, 0.0, 0.0, delta
	)
	if not np.isfinite(cost):
		raise AssertionError("Synthetic robust cost is not finite")
	return float(losses[0]), float(losses[1]), cost


def synthetic_optimization_example() -> tuple[np.ndarray, object, float, float]:
	"""Fit perturbed parameters to synthetic inlier observations."""
	world = np.array([[x, y, 0.0] for y in range(4) for x in range(5)], dtype=float)
	true_theta = np.array(
		[720.0, 735.0, 320.0, 240.0, 0.001, -0.0001, 0.02, -0.03, 0.01, 0.1, -0.1, 6.0]
	)
	intrinsic, rotation, translation, k1, k2 = unpack_theta(true_theta)
	observed = predicted_image_points(world, intrinsic, rotation, translation, k1, k2)
	observed += np.array([0.15, -0.1])
	initial_theta = true_theta + np.array(
		[-45.0, 35.0, 12.0, -10.0, 0.002, -0.0002, 0.01, 0.01, -0.01, 0.2, -0.2, 0.5]
	)
	return optimize_parameters(world, observed, initial_theta, delta=3.0, max_iterations=300)


if __name__ == "__main__":
	optimized_theta, result, initial_cost, final_cost = synthetic_optimization_example()
	print(f"Initial robust cost: {initial_cost:.6f}")
	print(f"Final robust cost: {final_cost:.6f}")
	print(f"Optimization success: {result.success}")
	print(f"Estimated fx: {optimized_theta[0]:.6f}")
	print(f"Estimated fy: {optimized_theta[1]:.6f}")
	print(f"Estimated cx: {optimized_theta[2]:.6f}")
	print(f"Estimated cy: {optimized_theta[3]:.6f}")
	print(f"Estimated k1: {optimized_theta[4]:.9f}")
	print(f"Estimated k2: {optimized_theta[5]:.9f}")
	print(f"Estimated rotation: {optimized_theta[6:9]}")
	print(f"Estimated translation: {optimized_theta[9:12]}")
