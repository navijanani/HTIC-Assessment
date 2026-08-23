"""Detect and export visible corners from one planar rectangular grid image.

This stage deliberately stops before distortion estimation, RANSAC, and camera
calibration. The grid dimensions are the number of inner corners, not the
number of squares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def preprocess(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_checkerboard(
    gray: np.ndarray, pattern_size: tuple[int, int]
) -> tuple[np.ndarray | None, str]:
    columns, rows = pattern_size
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    attempts = [gray, preprocess(gray)]

    for candidate in attempts:
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                candidate, (columns, rows), flags=sb_flags
            )
            if found:
                return corners.reshape(-1, 2).astype(np.float32), "findChessboardCornersSB"

        classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(candidate, (columns, rows), classic_flags)
        if found:
            corners = cv2.cornerSubPix(
                candidate,
                corners,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
            )
            return corners.reshape(-1, 2).astype(np.float32), "findChessboardCorners"

    return None, "none"


def detect_visible_features(gray: np.ndarray, max_corners: int) -> np.ndarray:
    enhanced = preprocess(gray)
    points = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=5,
        useHarrisDetector=True,
        k=0.04,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    refined = cv2.cornerSubPix(
        enhanced,
        points,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    return refined.reshape(-1, 2).astype(np.float32)


def make_correspondences(
    points: np.ndarray,
    pattern_size: tuple[int, int],
    ordering: str,
    spacing: float,
) -> list[dict[str, object]]:
    columns, rows = pattern_size
    expected = columns * rows
    if len(points) != expected:
        raise ValueError(f"Expected {expected} checkerboard corners, received {len(points)}")
    correspondences = []
    for index, (u, v) in enumerate(points):
        row, column = divmod(index, columns)
        correspondences.append(
            {
                "row": int(row),
                "column": int(column),
                "u": float(u),
                "v": float(v),
                "X": float(column * spacing),
                "Y": float(row * spacing),
                "Z": 0.0,
                "ordering": ordering,
            }
        )
    return correspondences


def make_uncertain_detections(points: np.ndarray) -> list[dict[str, object]]:
    """Record fallback image points without assigning planar coordinates."""
    return [
        {
            "row": None,
            "column": None,
            "u": float(u),
            "v": float(v),
            "X": None,
            "Y": None,
            "Z": None,
            "ordering": None,
            "status": "uncertain_feature_only",
        }
        for u, v in points
    ]


def draw_result(
    image: np.ndarray, correspondences: list[dict[str, object]], output_path: Path
) -> None:
    display = image.copy()
    for item in correspondences:
        center = (round(float(item["u"])), round(float(item["v"])))
        if item.get("status") == "uncertain_feature_only":
            label = "uncertain"
            color = (0, 165, 255)
        else:
            label = f"{item['row']},{item['column']}"
            color = (0, 0, 255)
        cv2.circle(display, center, 5, color, -1)
        cv2.putText(
            display,
            label,
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), display):
        raise OSError(f"Could not save visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the single grid photograph")
    parser.add_argument("--columns", type=int, required=True, help="Number of inner corners")
    parser.add_argument("--rows", type=int, required=True, help="Number of inner corners")
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="Physical spacing between corners"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.columns < 2 or args.rows < 2 or args.spacing <= 0:
        parser.error("columns and rows must be at least 2; spacing must be positive")

    image, gray = load_image(args.image)
    pattern_size = (args.columns, args.rows)
    points, method = detect_checkerboard(gray, pattern_size)
    if points is not None:
        correspondences = make_correspondences(points, pattern_size, method, args.spacing)
        detected_points = points
        uncertain_detections: list[dict[str, object]] = []
    else:
        detected_points = detect_visible_features(gray, args.columns * args.rows * 2)
        correspondences = []
        uncertain_detections = make_uncertain_detections(detected_points)
        method = "goodFeaturesToTrack_fallback"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = correspondences + uncertain_detections
    image_points = np.array(
        [[item["u"], item["v"]] for item in correspondences], dtype=np.float32
    ).reshape(-1, 2)
    grid_points = np.array(
        [[item["X"], item["Y"], item["Z"]] for item in correspondences], dtype=np.float32
    ).reshape(-1, 3)
    row_column = np.array(
        [[item["row"], item["column"]] for item in correspondences], dtype=np.int32
    ).reshape(-1, 2)
    detected_image_points = detected_points.reshape(-1, 2).astype(np.float32)
    np.savez_compressed(
        args.output_dir / "grid_correspondences.npz",
        image_points=image_points,
        grid_points=grid_points,
        row_column=row_column,
        detected_image_points=detected_image_points,
    )
    metadata = {
        "image": str(args.image),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "image_channels": int(image.shape[2]),
        "pattern_columns": args.columns,
        "pattern_rows": args.rows,
        "grid_spacing": args.spacing,
        "detection_method": method,
        "detected_corners": len(detected_points),
        "valid_correspondences": len(correspondences),
        "correspondences": records,
    }
    (args.output_dir / "grid_correspondences.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    draw_result(image, records, args.output_dir / "grid_corners_overlay.png")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")
    print(f"Detection method: {method}")
    print(f"Detected corners: {len(detected_points)}")
    print(f"Valid correspondences: {len(correspondences)}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()