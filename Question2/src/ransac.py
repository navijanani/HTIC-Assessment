"""RANSAC rejection of incorrect planar grid correspondences.

The temporary model is a planar homography estimated with normalized DLT.
This is appropriate for one view of a plane and uses only the current
correspondence geometry. It is not a camera calibration and does not estimate
the Brown-Conrady parameters; later optimization can refine those parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .optimization import residual_magnitudes


def _points(values: np.ndarray, columns: int, name: str) -> np.ndarray:
	array = np.asarray(values, dtype=float)
	if array.ndim != 2 or array.shape[1] != columns:
		raise ValueError(f"{name} must have shape (N, {columns})")
	if not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must contain only finite values")
	return array


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	centroid = np.mean(points, axis=0)
	centered = points - centroid
	mean_distance = np.mean(np.linalg.norm(centered, axis=1))
	if mean_distance <= np.finfo(float).eps:
		raise ValueError("sample points are degenerate")
	scale = np.sqrt(2.0) / mean_distance
	transform = np.array(
		[[scale, 0.0, -scale * centroid[0]],
		 [0.0, scale, -scale * centroid[1]],
		 [0.0, 0.0, 1.0]]
	)
	homogeneous = np.column_stack((points, np.ones(len(points))))
	return (transform @ homogeneous.T).T, transform


def estimate_homography(world_points: np.ndarray, image_points: np.ndarray) -> np.ndarray:
	"""Estimate ``image ~ H [X,Y,1]`` using normalized DLT."""
	world = _points(world_points, 3, "world_points")
	image = _points(image_points, 2, "image_points")
	if len(world) != len(image) or len(world) < 4:
		raise ValueError("at least four matched points are required")
	if not np.allclose(world[:, 2], 0.0):
		raise ValueError("world_points must lie on Z=0")

	world_normalized, world_transform = _normalize_points(world[:, :2])
	image_normalized, image_transform = _normalize_points(image)
	matrix = []
	for source, target in zip(world_normalized, image_normalized):
		x, y, _ = source
		u, v, _ = target
		matrix.extend(
			[
				[-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u],
				[0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v],
			]
		)
	_, singular_values, vh = np.linalg.svd(np.asarray(matrix))
	# A valid DLT system has rank 8. A near-zero second-smallest singular
	# value indicates a degenerate sample, such as nearly collinear points.
	if singular_values[-2] <= singular_values[0] * np.finfo(float).eps:
		raise ValueError("sample points do not define a stable homography")
	normalized_homography = vh[-1].reshape(3, 3)
	homography = np.linalg.inv(image_transform) @ normalized_homography @ world_transform
	if abs(homography[2, 2]) <= np.finfo(float).eps:
		raise ValueError("estimated homography has invalid scale")
	return homography / homography[2, 2]


def project_with_homography(world_points: np.ndarray, homography: np.ndarray) -> np.ndarray:
	"""Project planar points with a homography and reject zero depth."""
	world = _points(world_points, 3, "world_points")
	if not np.allclose(world[:, 2], 0.0):
		raise ValueError("world_points must lie on Z=0")
	matrix = np.asarray(homography, dtype=float)
	if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
		raise ValueError("homography must be a finite 3x3 matrix")
	homogeneous = np.column_stack((world[:, :2], np.ones(len(world))))
	projected_homogeneous = (matrix @ homogeneous.T).T
	depths = projected_homogeneous[:, 2]
	if np.any(depths <= 0):
		raise ValueError("homography projected a point to non-positive depth")
	projected = projected_homogeneous[:, :2] / depths[:, np.newaxis]
	if not np.all(np.isfinite(projected)):
		raise ValueError("homography projection produced non-finite points")
	return projected


@dataclass
class RansacResult:
	image_points: np.ndarray
	inlier_mask: np.ndarray
	outlier_mask: np.ndarray
	inlier_image_points: np.ndarray
	inlier_grid_points: np.ndarray
	best_model: np.ndarray
	residuals: np.ndarray
	iterations: int
	threshold: float


def ransac_correspondences(
	world_points: np.ndarray,
	image_points: np.ndarray,
	threshold: float = 3.0,
	iterations: int = 500,
	seed: int | None = 0,
) -> RansacResult:
	"""Reject correspondences whose homography error exceeds ``threshold``."""
	world = _points(world_points, 3, "world_points")
	image = _points(image_points, 2, "image_points")
	if len(world) != len(image):
		raise ValueError("world_points and image_points must have equal length")
	if len(world) < 4:
		raise ValueError("RANSAC needs at least four correspondences")
	if not np.isfinite(threshold) or threshold <= 0:
		raise ValueError("threshold must be finite and positive")
	if iterations < 1:
		raise ValueError("iterations must be positive")

	random = np.random.default_rng(seed)
	best_model = None
	best_mask = None
	best_score = (-1, -np.inf)
	for _ in range(iterations):
		sample_indices = random.choice(len(world), size=4, replace=False)
		try:
			model = estimate_homography(world[sample_indices], image[sample_indices])
			predicted = project_with_homography(world, model)
			residuals = image - predicted
			magnitudes = residual_magnitudes(residuals)
		except ValueError:
			continue
		mask = magnitudes <= threshold
		inlier_median = np.median(magnitudes[mask]) if np.any(mask) else np.inf
		score = (int(np.count_nonzero(mask)), -float(inlier_median))
		if score > best_score:
			best_score = score
			best_model = model
			best_mask = mask

	if best_model is None or best_mask is None:
		raise RuntimeError("RANSAC could not estimate a valid homography")
	if np.count_nonzero(best_mask) >= 4:
		best_model = estimate_homography(world[best_mask], image[best_mask])
	predicted = project_with_homography(world, best_model)
	residuals = image - predicted
	magnitudes = residual_magnitudes(residuals)
	inlier_mask = magnitudes <= threshold
	return RansacResult(
		image_points=image,
		inlier_mask=inlier_mask,
		outlier_mask=~inlier_mask,
		inlier_image_points=image[inlier_mask],
		inlier_grid_points=world[inlier_mask],
		best_model=best_model,
		residuals=residuals,
		iterations=iterations,
		threshold=threshold,
	)


def save_visualization(
	image_path: Path, result: RansacResult, output_path: Path
) -> None:
	import cv2

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	for point, is_inlier in zip(result.image_points, result.inlier_mask):
		color = (0, 180, 0) if is_inlier else (0, 0, 255)
		cv2.circle(image, (round(point[0]), round(point[1])), 5, color, -1)
	if not cv2.imwrite(str(output_path), image):
		raise OSError(f"Could not save visualization: {output_path}")


def synthetic_example() -> tuple[int, int]:
	"""Demonstrate recovery of eight inliers from four corrupted matches."""
	world = np.array([[x, y, 0.0] for y in range(3) for x in range(4)], dtype=float)
	homography = np.array([[80.0, 5.0, 120.0], [2.0, 75.0, 90.0], [0.001, 0.002, 1.0]])
	image = project_with_homography(world, homography)
	image[:8] += np.array([[0.2, -0.1]])
	image[8:] += np.array([[300.0, -250.0]])
	result = ransac_correspondences(world, image, threshold=2.0, iterations=300, seed=4)
	if np.count_nonzero(result.inlier_mask) < 8:
		raise AssertionError("Synthetic RANSAC did not recover the inlier majority")
	return int(np.count_nonzero(result.inlier_mask)), int(np.count_nonzero(result.outlier_mask))


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--image", type=Path, default=Path("data/raw/grid_image.jpg"))
	parser.add_argument("--correspondences", type=Path, default=Path("results/grid_correspondences.npz"))
	parser.add_argument("--threshold", type=float, default=3.0)
	parser.add_argument("--iterations", type=int, default=500)
	parser.add_argument("--output", type=Path, default=Path("results/ransac_inliers_overlay.png"))
	parser.add_argument("--synthetic", action="store_true")
	args = parser.parse_args()
	if args.synthetic:
		print("Synthetic RANSAC result (inliers, outliers):", synthetic_example())
		return
	data = np.load(args.correspondences)
	world = _points(data["grid_points"], 3, "grid_points")
	image = _points(data["image_points"], 2, "image_points")
	result = ransac_correspondences(world, image, args.threshold, args.iterations)
	save_visualization(args.image, result, args.output)
	total = len(image)
	inliers = int(np.count_nonzero(result.inlier_mask))
	print(f"Total valid correspondences: {total}")
	print(f"Inliers: {inliers}")
	print(f"Outliers: {total - inliers}")
	print(f"Inlier percentage: {100.0 * inliers / total:.2f}%")
	print(f"RANSAC threshold: {result.threshold:.2f} pixels")
	print(f"RANSAC iterations: {result.iterations}")


if __name__ == "__main__":
	main()
