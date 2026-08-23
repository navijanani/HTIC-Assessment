"""Reproject an estimated grid model into the original distorted image.

This stage evaluates fixed Step 6 parameters only. It does not optimize,
calibrate, run RANSAC, undistort, or compute the final error statistics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .distortion_model import project_points
from .optimization import unpack_theta


def _array(values: np.ndarray, columns: int, name: str) -> np.ndarray:
	array = np.asarray(values, dtype=float)
	if array.ndim != 2 or array.shape[1] != columns:
		raise ValueError(f"{name} must have shape (N, {columns})")
	if not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must contain only finite values")
	return array


@dataclass
class ReprojectionResult:
	observed_points: np.ndarray
	predicted_points: np.ndarray
	residuals: np.ndarray
	residual_magnitudes: np.ndarray
	mean_absolute_x: float
	mean_absolute_y: float
	minimum_magnitude: float
	maximum_magnitude: float


def reproject_points(
	world_points: np.ndarray,
	theta: np.ndarray,
) -> np.ndarray:
	"""Predict distorted pixels using the fixed optimized theta vector."""
	world = _array(world_points, 3, "world_points")
	if not np.allclose(world[:, 2], 0.0):
		raise ValueError("world_points must lie on Z=0")
	intrinsic, rotation, translation, k1, k2 = unpack_theta(theta)
	return project_points(world, intrinsic, rotation, translation, k1, k2)


def compute_reprojection(
	world_points: np.ndarray,	observed_points: np.ndarray,	theta: np.ndarray
) -> ReprojectionResult:
	"""Return each predicted point, residual vector, and intermediate diagnostics."""
	world = _array(world_points, 3, "world_points")
	observed = _array(observed_points, 2, "observed_points")
	if len(world) != len(observed):
		raise ValueError("world_points and observed_points must have equal length")
	predicted = reproject_points(world, theta)
	residuals = observed - predicted
	magnitudes = np.linalg.norm(residuals, axis=1)
	if not np.all(np.isfinite(residuals)):
		raise ValueError("reprojection residuals must be finite")
	return ReprojectionResult(
		observed_points=observed,
		predicted_points=predicted,
		residuals=residuals,
		residual_magnitudes=magnitudes,
		mean_absolute_x=float(np.mean(np.abs(residuals[:, 0]))) if len(residuals) else 0.0,
		mean_absolute_y=float(np.mean(np.abs(residuals[:, 1]))) if len(residuals) else 0.0,
		minimum_magnitude=float(np.min(magnitudes)) if len(magnitudes) else 0.0,
		maximum_magnitude=float(np.max(magnitudes)) if len(magnitudes) else 0.0,
	)


def draw_reprojection(
	image_path: Path,
	result: ReprojectionResult,
	row_column: np.ndarray,
	output_path: Path,
) -> None:
	"""Draw observed/predicted points, vectors, and predicted grid lines."""
	import cv2

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	indices = np.asarray(row_column, dtype=int)
	if indices.shape != (len(result.observed_points), 2):
		raise ValueError("row_column must match the number of points")
	for observed, predicted in zip(result.observed_points, result.predicted_points):
		observed_pixel = tuple(np.round(observed).astype(int))
		predicted_pixel = tuple(np.round(predicted).astype(int))
		cv2.circle(image, observed_pixel, 5, (0, 0, 255), -1)
		cv2.circle(image, predicted_pixel, 5, (0, 180, 0), -1)
		cv2.arrowedLine(image, observed_pixel, predicted_pixel, (255, 0, 0), 1, tipLength=0.25)
	for axis in (0, 1):
		for value in np.unique(indices[:, axis]):
			mask = indices[:, axis] == value
			order = np.argsort(indices[mask, 1 - axis])
			points = result.predicted_points[mask][order]
			for first, second in zip(points[:-1], points[1:]):
				cv2.line(image, tuple(np.round(first).astype(int)), tuple(np.round(second).astype(int)), (0, 180, 0), 1)
	if not cv2.imwrite(str(output_path), image):
		raise OSError(f"Could not save visualization: {output_path}")


def load_theta(path: Path) -> np.ndarray:
	if path.suffix.lower() == ".json":
		value = json.loads(path.read_text(encoding="utf-8"))
		return np.asarray(value.get("theta", value) if isinstance(value, dict) else value, dtype=float)
	return np.asarray(np.load(path), dtype=float)


def residual_metrics(residuals: np.ndarray) -> dict[str, float | int]:
	"""Calculate final error statistics from already-produced residual vectors."""
	values = _array(residuals, 2, "residuals")
	if len(values) == 0:
		raise ValueError("at least one residual is required")
	magnitudes = np.linalg.norm(values, axis=1)
	return {
		"number_of_correspondences": int(len(values)),
		"mean_residual_x": float(np.mean(values[:, 0])),
		"mean_residual_y": float(np.mean(values[:, 1])),
		"mean_absolute_residual_x": float(np.mean(np.abs(values[:, 0]))),
		"mean_absolute_residual_y": float(np.mean(np.abs(values[:, 1]))),
		"mean_residual_magnitude": float(np.mean(magnitudes)),
		"median_residual_magnitude": float(np.median(magnitudes)),
		"rmse_reprojection_error": float(np.sqrt(np.mean(np.sum(values**2, axis=1)))),
		"maximum_residual_magnitude": float(np.max(magnitudes)),
		"minimum_residual_magnitude": float(np.min(magnitudes)),
	}


def analyze_residuals(
	observed_points: np.ndarray,
	predicted_points: np.ndarray,
	inlier_mask: np.ndarray | None = None,
) -> dict[str, object]:
	"""Return final metrics for all points and, when available, RANSAC inliers."""
	observed = _array(observed_points, 2, "observed_points")
	predicted = _array(predicted_points, 2, "predicted_points")
	if len(observed) != len(predicted):
		raise ValueError("observed_points and predicted_points must have equal length")
	residuals = observed - predicted
	metrics: dict[str, object] = {
		"all_valid_correspondences": residual_metrics(residuals),
		"main_final_reprojection_error": "rmse_reprojection_error on RANSAC inliers",
	}
	if inlier_mask is not None:
		mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
		if mask.shape != (len(residuals),):
			raise ValueError("inlier_mask must match the number of residuals")
		if not np.any(mask):
			raise ValueError("inlier_mask must contain at least one inlier")
		metrics["ransac_inliers"] = residual_metrics(residuals[mask])
		metrics["ransac_outliers"] = residual_metrics(residuals[~mask]) if np.any(~mask) else None
		metrics["main_final_reprojection_error"] = "rmse_reprojection_error on RANSAC inliers"
	else:
		metrics["main_final_reprojection_error"] = "rmse_reprojection_error on all valid correspondences"
	return {"residuals": residuals, "metrics": metrics}


def draw_residual_visualization(
	image_path: Path,
	observed_points: np.ndarray,
	predicted_points: np.ndarray,
	residuals: np.ndarray,
	row_column: np.ndarray,
	output_path: Path,
) -> None:
	"""Save original-image vectors beside a residual-magnitude chart."""
	import cv2

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	indices = np.asarray(row_column, dtype=int)
	if indices.shape != (len(residuals), 2):
		raise ValueError("row_column must match residual count")
	for observed, predicted in zip(observed_points, predicted_points):
		start = tuple(np.round(observed).astype(int))
		end = tuple(np.round(predicted).astype(int))
		cv2.circle(image, start, 4, (0, 0, 255), -1)
		cv2.circle(image, end, 4, (0, 180, 0), -1)
		cv2.arrowedLine(image, start, end, (255, 0, 0), 1, tipLength=0.25)

	chart_width, chart_height = max(700, image.shape[1]), 360
	chart = np.full((chart_height, chart_width, 3), 255, dtype=np.uint8)
	magnitudes = np.linalg.norm(residuals, axis=1)
	maximum = max(float(np.max(magnitudes)), 1e-12)
	margin = 45
	for index, magnitude in enumerate(magnitudes):
		x = margin + int(index * (chart_width - 2 * margin) / max(len(magnitudes) - 1, 1))
		y = chart_height - margin - int((magnitude / maximum) * (chart_height - 2 * margin))
		cv2.line(chart, (x, chart_height - margin), (x, y), (190, 80, 30), 2)
		cv2.circle(chart, (x, y), 3, (190, 80, 30), -1)
	cv2.putText(chart, "Residual magnitude per grid point (pixels)", (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
	combined = np.vstack((image, chart))
	if not cv2.imwrite(str(output_path), combined):
		raise OSError(f"Could not save residual visualization: {output_path}")


def run_error_analysis(
	image_path: Path,
	residual_path: Path,
	output_dir: Path,
	inlier_mask_path: Path | None = None,
) -> dict[str, object]:
	"""Load Step 8 residuals and save Step 9 metrics and visualization."""
	data = np.load(residual_path)
	observed = _array(data["observed_points"], 2, "observed_points")
	predicted = _array(data["predicted_points"], 2, "predicted_points")
	row_column = np.asarray(data.get("row_column", np.zeros((len(observed), 2))), dtype=int)
	inlier_mask = None
	if inlier_mask_path is not None:
		inlier_mask = np.asarray(np.load(inlier_mask_path), dtype=bool)
	analysis = analyze_residuals(observed, predicted, inlier_mask)
	output_dir.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		output_dir / "final_reprojection_residuals.npz",
		residuals=analysis["residuals"],
		residual_magnitudes=np.linalg.norm(analysis["residuals"], axis=1),
	)
	visualization = output_dir / "final_reprojection_residuals.png"
	draw_residual_visualization(image_path, observed, predicted, analysis["residuals"], row_column, visualization)
	metrics_path = output_dir / "final_reprojection_metrics.json"
	metrics_path.write_text(json.dumps(analysis["metrics"], indent=2), encoding="utf-8")
	return {"metrics": analysis["metrics"], "visualization": str(visualization), "metrics_path": str(metrics_path)}


def synthetic_error_analysis() -> dict[str, object]:
	"""Verify zero residual metrics for identical known observations/predictions."""
	points = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
	analysis = analyze_residuals(points, points.copy())
	metrics = analysis["metrics"]["all_valid_correspondences"]
	for key in ("mean_residual_x", "mean_residual_y", "median_residual_magnitude", "rmse_reprojection_error", "maximum_residual_magnitude"):
		if not np.isclose(metrics[key], 0.0):
			raise AssertionError(f"Synthetic metric {key} was not zero")
	return metrics


def run_reprojection(
	image_path: Path,	correspondence_path: Path,	theta_path: Path,	output_dir: Path
) -> ReprojectionResult:
	import cv2

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	data = np.load(correspondence_path)
	observed = _array(data["image_points"], 2, "image_points")
	world = _array(data["grid_points"], 3, "grid_points")
	row_column = np.asarray(data["row_column"], dtype=int)
	if len(observed) != len(world) or row_column.shape != (len(observed), 2):
		raise ValueError("correspondence arrays must have matching lengths")
	result = compute_reprojection(world, observed, load_theta(theta_path))
	output_dir.mkdir(parents=True, exist_ok=True)
	visualization = output_dir / "reprojection_overlay.png"
	draw_reprojection(image_path, result, row_column, visualization)
	residual_data = output_dir / "reprojection_residuals.npz"
	np.savez_compressed(
		residual_data,
		observed_points=result.observed_points,
		predicted_points=result.predicted_points,
		residuals=result.residuals,
		residual_magnitudes=result.residual_magnitudes,
		row_column=row_column,
	)
	return result


def synthetic_example() -> ReprojectionResult:
	"""Confirm reprojection matches known forward-model observations exactly."""
	world = np.array([[x, y, 0.0] for y in range(3) for x in range(4)], dtype=float)
	theta = np.array([800.0, 810.0, 320.0, 240.0, 0.01, -0.001, 0.02, -0.01, 0.01, 0.1, -0.1, 5.0])
	intrinsic, rotation, translation, k1, k2 = unpack_theta(theta)
	observed = project_points(world, intrinsic, rotation, translation, k1, k2)
	result = compute_reprojection(world, observed, theta)
	if not np.allclose(result.predicted_points, observed):
		raise AssertionError("Reprojection does not agree with the forward model")
	if not np.allclose(result.residuals, 0.0):
		raise AssertionError("Exact synthetic observations should have zero residuals")
	return result


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--image", type=Path, default=Path("data/raw/grid_image.jpg"))
	parser.add_argument("--correspondences", type=Path, default=Path("results/grid_correspondences.npz"))
	parser.add_argument("--residuals", type=Path, default=Path("results/reprojection_residuals.npz"))
	parser.add_argument("--theta", type=Path, default=Path("results/optimized_theta.npy"))
	parser.add_argument("--output-dir", type=Path, default=Path("results"))
	parser.add_argument("--synthetic", action="store_true")
	parser.add_argument("--analyze", action="store_true")
	parser.add_argument("--synthetic-analysis", action="store_true")
	parser.add_argument("--inlier-mask", type=Path, default=None)
	args = parser.parse_args()
	if args.synthetic_analysis:
		metrics = synthetic_error_analysis()
		print(f"Synthetic correspondences evaluated: {metrics['number_of_correspondences']}")
		print(f"Synthetic mean residual x: {metrics['mean_residual_x']:.6f}")
		print(f"Synthetic mean residual y: {metrics['mean_residual_y']:.6f}")
		print(f"Synthetic median residual magnitude: {metrics['median_residual_magnitude']:.6f}")
		print(f"Synthetic RMSE reprojection error: {metrics['rmse_reprojection_error']:.6f}")
		print(f"Synthetic maximum residual magnitude: {metrics['maximum_residual_magnitude']:.6f}")
		return
	if args.analyze:
		result = run_error_analysis(args.image, args.residuals, args.output_dir, args.inlier_mask)
		print(json.dumps(result["metrics"], indent=2))
		print(f"Saved residual visualization: {result['visualization']}")
		print(f"Saved final metrics: {result['metrics_path']}")
		return
	if args.synthetic:
		result = synthetic_example()
		print(f"Synthetic observed points: {len(result.observed_points)}")
		print(f"Synthetic successfully reprojected points: {len(result.predicted_points)}")
		print(f"Synthetic mean absolute residual x: {result.mean_absolute_x:.6f}")
		print(f"Synthetic mean absolute residual y: {result.mean_absolute_y:.6f}")
		print(f"Synthetic minimum residual magnitude: {result.minimum_magnitude:.6f}")
		print(f"Synthetic maximum residual magnitude: {result.maximum_magnitude:.6f}")
		return
	result = run_reprojection(args.image, args.correspondences, args.theta, args.output_dir)
	print(f"Observed points: {len(result.observed_points)}")
	print(f"Successfully reprojected points: {len(result.predicted_points)}")
	print(f"Mean absolute residual x: {result.mean_absolute_x:.6f}")
	print(f"Mean absolute residual y: {result.mean_absolute_y:.6f}")
	print(f"Minimum residual magnitude: {result.minimum_magnitude:.6f}")
	print(f"Maximum residual magnitude: {result.maximum_magnitude:.6f}")
	print(f"Saved visualization: {args.output_dir / 'reprojection_overlay.png'}")


if __name__ == "__main__":
	main()
