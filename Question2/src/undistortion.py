"""Remove radial lens distortion using fixed Step 6 parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .optimization import unpack_theta


def _array(values: np.ndarray, columns: int, name: str) -> np.ndarray:
	array = np.asarray(values, dtype=float)
	if array.ndim != 2 or array.shape[1] != columns:
		raise ValueError(f"{name} must have shape (N, {columns})")
	if not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must contain only finite values")
	return array


def undistorted_normalized_points(
	distorted_points: np.ndarray, k1: float, k2: float, iterations: int = 8
) -> np.ndarray:
	"""Invert Brown-Conrady radial distortion by fixed-point iteration."""
	points = _array(distorted_points, 2, "distorted_points")
	if not np.all(np.isfinite([k1, k2])) or iterations < 1:
		raise ValueError("k1 and k2 must be finite and iterations must be positive")

	estimate = points.copy()
	for _ in range(iterations):
		radius_squared = np.sum(estimate * estimate, axis=1)
		scale = 1.0 + k1 * radius_squared + k2 * radius_squared**2
		if np.any(np.abs(scale) <= np.finfo(float).eps):
			raise ValueError("radial inversion encountered a zero scale")
		estimate = points / scale[:, np.newaxis]
	if not np.all(np.isfinite(estimate)):
		raise ValueError("radial inversion produced non-finite values")
	return estimate


def undistort_grid_corners(
	image_points: np.ndarray,
	intrinsic_matrix: np.ndarray,
	k1: float,
	k2: float,
) -> np.ndarray:
	"""Recover ideal pixel locations while preserving perspective geometry."""
	points = _array(image_points, 2, "image_points")
	intrinsic = np.asarray(intrinsic_matrix, dtype=float)
	if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
		raise ValueError("intrinsic_matrix must be a finite 3x3 matrix")
	fx, fy = intrinsic[0, 0], intrinsic[1, 1]
	if fx <= 0 or fy <= 0:
		raise ValueError("focal lengths must be positive")
	normalized_distorted = np.column_stack(
		((points[:, 0] - intrinsic[0, 2]) / fx,
		 (points[:, 1] - intrinsic[1, 2]) / fy)
	)
	normalized_ideal = undistorted_normalized_points(normalized_distorted, k1, k2)
	return np.column_stack(
		(fx * normalized_ideal[:, 0] + intrinsic[0, 2],
		 fy * normalized_ideal[:, 1] + intrinsic[1, 2])
	)


def undistort_image(
	image: np.ndarray, intrinsic_matrix: np.ndarray, k1: float, k2: float,
	return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
	"""Remap pixels with the supplied model; no parameters are estimated."""
	import cv2

	intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
	if intrinsic.shape != (3, 3):
		raise ValueError("intrinsic_matrix must have shape (3, 3)")
	if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
		raise ValueError("image must be a non-empty 2D or 3D array")
	distortion = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
	map_x, map_y = cv2.initUndistortRectifyMap(
		intrinsic, distortion, None, intrinsic,
		(image.shape[1], image.shape[0]), cv2.CV_32FC1
	)
	finite = np.isfinite(map_x) & np.isfinite(map_y)
	inside = (
		finite
		& (map_x >= 0)
		& (map_x < image.shape[1])
		& (map_y >= 0)
		& (map_y < image.shape[0])
	)
	print(f"Intrinsic matrix:\n{intrinsic}")
	print(f"fx={intrinsic[0, 0]:.6f}, fy={intrinsic[1, 1]:.6f}, cx={intrinsic[0, 2]:.6f}, cy={intrinsic[1, 2]:.6f}")
	print(f"k1={k1:.9f}, k2={k2:.9f}")
	print(f"map_x range: {np.nanmin(map_x):.6f} to {np.nanmax(map_x):.6f}")
	print(f"map_y range: {np.nanmin(map_y):.6f} to {np.nanmax(map_y):.6f}")
	print(f"Finite map coordinates: {100.0 * finite.mean():.4f}%")
	print(f"Map coordinates inside source image: {100.0 * inside.mean():.4f}%")
	corrected = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
	finite_pixels = corrected[np.isfinite(corrected)]
	non_black = np.any(corrected != 0, axis=2) if corrected.ndim == 3 else corrected != 0
	diagnostics = {
		"map_x_min": float(np.nanmin(map_x)), "map_x_max": float(np.nanmax(map_x)),
		"map_y_min": float(np.nanmin(map_y)), "map_y_max": float(np.nanmax(map_y)),
		"finite_map_percentage": float(100.0 * finite.mean()),
		"inside_source_percentage": float(100.0 * inside.mean()),
		"corrected_min": float(np.min(finite_pixels)), "corrected_max": float(np.max(finite_pixels)),
		"corrected_mean": float(np.mean(finite_pixels)),
		"non_black_percentage": float(100.0 * non_black.mean()),
	}
	print(f"Corrected image min/max/mean: {diagnostics['corrected_min']:.6f}/{diagnostics['corrected_max']:.6f}/{diagnostics['corrected_mean']:.6f}")
	print(f"Corrected image non-black pixels: {diagnostics['non_black_percentage']:.4f}%")
	return (corrected, diagnostics) if return_diagnostics else corrected


def draw_grid_visualization(
	original: np.ndarray,
	corrected_image: np.ndarray,
	original_points: np.ndarray,
	corrected_points: np.ndarray,
	row_column: np.ndarray,
	output_path: Path,
) -> None:
	"""Save side-by-side overlays with horizontal and vertical grid lines."""
	import cv2

	points = _array(original_points, 2, "original_points")
	corrected = _array(corrected_points, 2, "corrected_points")
	indices = np.asarray(row_column, dtype=int)
	if len(points) != len(corrected) or indices.shape != (len(points), 2):
		raise ValueError("corner and row_column arrays must have matching lengths")

	def draw(canvas: np.ndarray, locations: np.ndarray, color: tuple[int, int, int]) -> None:
		for point in locations:
			cv2.circle(canvas, tuple(np.round(point).astype(int)), 5, color, -1)
		for axis in (0, 1):
			for value in np.unique(indices[:, axis]):
				mask = indices[:, axis] == value
				order = np.argsort(indices[mask, 1 - axis])
				ordered = locations[mask][order]
				for first, second in zip(ordered[:-1], ordered[1:]):
					cv2.line(canvas, tuple(np.round(first).astype(int)),
							 tuple(np.round(second).astype(int)), color, 1)

	left, right = original.copy(), corrected_image.copy()
	draw(left, points, (0, 0, 255))
	draw(right, corrected, (0, 180, 0))
	if not cv2.imwrite(str(output_path), np.hstack((left, right))):
		raise OSError(f"Could not save visualization: {output_path}")


def load_theta(path: Path) -> np.ndarray:
	"""Load the Step 6 theta vector from .npy or a JSON list/dict."""
	if path.suffix.lower() == ".json":
		value = json.loads(path.read_text(encoding="utf-8"))
		return np.asarray(value.get("theta", value) if isinstance(value, dict) else value, dtype=float)
	return np.asarray(np.load(path), dtype=float)


def run_undistortion(
	image_path: Path, correspondence_path: Path, theta_path: Path, output_dir: Path
) -> dict[str, object]:
	import cv2

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	data = np.load(correspondence_path)
	image_points = _array(data["image_points"], 2, "image_points")
	grid_points = _array(data["grid_points"], 3, "grid_points")
	row_column = np.asarray(data["row_column"], dtype=int)
	if len(image_points) != len(grid_points) or row_column.shape != (len(image_points), 2):
		raise ValueError("correspondence arrays must have matching lengths")
	if not np.allclose(grid_points[:, 2], 0.0):
		raise ValueError("grid_points must lie on Z=0")

	intrinsic, _, _, k1, k2 = unpack_theta(load_theta(theta_path))
	corrected_points = undistort_grid_corners(image_points, intrinsic, k1, k2)
	corrected_image, image_diagnostics = undistort_image(
		image, intrinsic, k1, k2, return_diagnostics=True
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	image_output = output_dir / "grid_image_undistorted.png"
	overlay_output = output_dir / "undistorted_grid_visualization.png"
	if not cv2.imwrite(str(image_output), corrected_image):
		raise OSError(f"Could not save undistorted image: {image_output}")
	draw_grid_visualization(image, corrected_image, image_points, corrected_points, row_column, overlay_output)
	report = {
		"original_dimensions": [int(image.shape[1]), int(image.shape[0])],
		"undistorted_dimensions": [int(corrected_image.shape[1]), int(corrected_image.shape[0])],
		"valid_grid_corners": len(image_points),
		"k1": float(k1), "k2": float(k2), "success": True,
		**image_diagnostics,
		"undistorted_image": str(image_output), "visualization": str(overlay_output),
	}
	(output_dir / "undistortion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
	return report


def synthetic_example() -> tuple[np.ndarray, np.ndarray]:
	"""Verify that known distorted normalized points return to ideal points."""
	ideal = np.array([[0.2, 0.1], [-0.25, 0.15]], dtype=float)
	k1, k2 = 0.2, 0.01
	radius_squared = np.sum(ideal * ideal, axis=1)
	distorted = ideal * (1.0 + k1 * radius_squared + k2 * radius_squared**2)[:, None]
	recovered = undistorted_normalized_points(distorted, k1, k2)
	if not np.allclose(recovered, ideal, atol=1e-8):
		raise AssertionError("Synthetic radial inversion did not recover ideal points")
	return distorted, recovered


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--image", type=Path, default=Path("data/raw/grid_image.jpg"))
	parser.add_argument("--correspondences", type=Path, default=Path("results/grid_correspondences.npz"))
	parser.add_argument("--theta", type=Path, default=Path("results/optimized_theta.npy"))
	parser.add_argument("--output-dir", type=Path, default=Path("results"))
	parser.add_argument("--synthetic", action="store_true")
	args = parser.parse_args()
	if args.synthetic:
		distorted, recovered = synthetic_example()
		print("Synthetic distorted normalized points:")
		print(distorted)
		print("Synthetic recovered ideal normalized points:")
		print(recovered)
		print("Undistortion completed successfully: True")
		return
	report = run_undistortion(args.image, args.correspondences, args.theta, args.output_dir)
	print(f"Original image dimensions: {report['original_dimensions']}")
	print(f"Undistorted image dimensions: {report['undistorted_dimensions']}")
	print(f"Valid grid corners used: {report['valid_grid_corners']}")
	print(f"Estimated k1: {report['k1']}")
	print(f"Estimated k2: {report['k2']}")
	print(f"Undistortion completed successfully: {report['success']}")


if __name__ == "__main__":
	main()
