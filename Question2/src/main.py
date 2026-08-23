"""Reproducible entry point for the complete Question 2 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detection import detect_checkerboard, draw_result, load_image, make_correspondences
from .ransac import ransac_correspondences, save_visualization
from .optimization import optimize_parameters
from .undistortion import run_undistortion
from .reprojection import run_error_analysis, run_reprojection


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} is missing: {path}")


def run_pipeline(
    image_path: Path,
    columns: int = 10,
    rows: int = 7,
    spacing: float = 1.0,
    output_dir: Path = Path("results"),
) -> dict[str, object]:
    """Run all six stages without estimating anything twice."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== Question 2 Pipeline ===")

    print("\n[1/6] Detection")
    _require_file(image_path, "input image")
    image, gray = load_image(image_path)
    points, method = detect_checkerboard(gray, (columns, rows))
    if points is None:
        raise RuntimeError(f"Checkerboard detection failed for pattern {columns}x{rows}")
    correspondences = make_correspondences(points, (columns, rows), method, spacing)
    image_points = points.astype(float)
    grid_points = np.array([[item["X"], item["Y"], item["Z"]] for item in correspondences])
    row_column = np.array([[item["row"], item["column"]] for item in correspondences], dtype=int)
    np.savez_compressed(
        output_dir / "grid_correspondences.npz",
        image_points=image_points,
        grid_points=grid_points,
        row_column=row_column,
        detected_image_points=image_points,
    )
    (output_dir / "grid_correspondences.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "pattern_columns": columns,
                "pattern_rows": rows,
                "grid_spacing": spacing,
                "detection_method": method,
                "detected_corners": len(points),
                "valid_correspondences": len(correspondences),
                "correspondences": correspondences,
            },
            indent=2,
        )
    )
    draw_result(image, correspondences, output_dir / "grid_corners_overlay.png")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Detected corners: {len(points)}; valid correspondences: {len(correspondences)}")
    print(f"Detection method: {method}")

    print("\n[2/6] RANSAC")
    ransac = ransac_correspondences(grid_points, image_points, threshold=3.0, iterations=500, seed=0)
    np.save(output_dir / "ransac_inlier_mask.npy", ransac.inlier_mask)
    save_visualization(image_path, ransac, output_dir / "ransac_inliers_overlay.png")
    homography_magnitudes = np.linalg.norm(ransac.residuals, axis=1)
    print(f"Inliers: {ransac.inlier_mask.sum()}; outliers: {(~ransac.inlier_mask).sum()}")
    print(f"Inlier percentage: {100.0 * ransac.inlier_mask.mean():.2f}%")
    print(f"Median homography residual: {np.median(homography_magnitudes):.6f} px")
    print(f"Maximum homography residual: {np.max(homography_magnitudes):.6f} px")

    print("\n[3/6] Robust optimization")
    optimized_theta, result, initial_cost, final_cost = optimize_parameters(
        ransac.inlier_grid_points,
        ransac.inlier_image_points,
        delta=3.0,
        max_iterations=1000,
        image_size=(image.shape[1], image.shape[0]),
        homography=ransac.best_model,
    )
    np.save(output_dir / "optimized_theta.npy", optimized_theta)
    (output_dir / "optimization_report.json").write_text(
        json.dumps(
            {
                "initial_robust_cost": initial_cost,
                "final_robust_cost": final_cost,
                "optimization_success": bool(result.success),
                "message": str(result.message),
                "theta": optimized_theta.tolist(),
            },
            indent=2,
        )
    )
    print(f"Initial robust cost: {initial_cost:.6f}; final robust cost: {final_cost:.6f}")
    print(f"Optimization success: {result.success}")

    print("\n[4/6] Undistortion")
    undistortion_report = run_undistortion(
        image_path,
        output_dir / "grid_correspondences.npz",
        output_dir / "optimized_theta.npy",
        output_dir,
    )
    print(f"Undistorted image: {undistortion_report['undistorted_image']}")
    print(f"Corrected grid points: {undistortion_report['valid_grid_corners']}")
    print(f"Map finite: {undistortion_report['finite_map_percentage']:.4f}%")
    print(f"Map inside source: {undistortion_report['inside_source_percentage']:.4f}%")
    print(f"Corrected image min/max/mean: {undistortion_report['corrected_min']:.6f}/{undistortion_report['corrected_max']:.6f}/{undistortion_report['corrected_mean']:.6f}")
    print(f"Corrected image non-black: {undistortion_report['non_black_percentage']:.4f}%")

    print("\n[5/6] Reprojection")
    reprojection = run_reprojection(
        image_path,
        output_dir / "grid_correspondences.npz",
        output_dir / "optimized_theta.npy",
        output_dir,
    )
    print(f"Observed points: {len(reprojection.observed_points)}")
    print(f"Predicted points: {len(reprojection.predicted_points)}")

    print("\n[6/6] Final error analysis")
    final_report = run_error_analysis(
        image_path,
        output_dir / "reprojection_residuals.npz",
        output_dir,
        output_dir / "ransac_inlier_mask.npy",
    )
    print(f"Main final error: {final_report['metrics']['main_final_reprojection_error']}")

    print("\n=== Final Summary ===")
    print(f"Detection: {len(points)} corners via {method}")
    print(f"RANSAC: {ransac.inlier_mask.sum()} inliers / {len(points)} correspondences")
    print(f"Optimization: {result.success}, cost {final_cost:.6f}")
    print(f"Final metrics: {final_report['metrics_path']}")
    return {"detection_method": method, "ransac": ransac, "optimization": result, "final_report": final_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("data/raw/grid_image.jpg"))
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    image_path = args.image
    if image_path == Path("data/raw/grid_image.jpg") and not image_path.exists():
        fallback = Path("data/raw/grid_image.jpeg")
        if fallback.exists():
            print(f"Input warning: {image_path} is missing; using available real image {fallback}")
            image_path = fallback
    run_pipeline(image_path, args.columns, args.rows, args.spacing, args.output_dir)


if __name__ == "__main__":
    main()
