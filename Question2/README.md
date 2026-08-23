# Question 2: Grid Corner Detection

This stage loads one planar grid photograph, detects visible inner corners,
assigns row/column ordering, creates an overlay, and saves reusable image/grid
correspondences. It does not estimate camera intrinsics, pose, or radial
distortion, and it does not run RANSAC or undistortion.

Install dependencies with `python3 -m pip install -r requirements.txt`.

Run the detector by supplying the number of visible-pattern inner corners:

```text
python3 -m src.main data/raw/grid_image.jpg --columns 9 --rows 6 --spacing 25 --output-dir results
```

The dimensions are the number of inner corners, not the number of checkerboard
squares. `--spacing` is the physical distance between neighboring corners in
the chosen grid units; use `1` when only normalized planar coordinates are
needed. The script saves:

- `results/grid_corners_overlay.png`: original image with corner markers and `(row,column)` labels. Fallback-only points are marked `uncertain`.
- `results/grid_correspondences.npz`: matched `image_points`, `grid_points`, and `row_column` arrays, plus all raw `detected_image_points`.
- `results/grid_correspondences.json`: image dimensions, method, counts, and readable records.

The primary detector is `findChessboardCornersSB`, with adaptive preprocessing
and a classic checkerboard fallback. If the full checkerboard cannot be found,
the script uses visible feature points only for visualization and review. These
points are marked `uncertain_feature_only`, have no row/column or planar
coordinates, and are excluded from the calibration-ready arrays. This avoids
inventing correspondences for occluded corners.