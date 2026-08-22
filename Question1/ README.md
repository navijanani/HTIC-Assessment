# Question 1 — Imaging Science

## Overview

This project addresses an image-processing problem involving **non-uniform illumination**.

The goal is to take a single grayscale photograph of a textured surface that is unevenly illuminated and estimate the underlying texture/reflectance as if the surface had been uniformly illuminated.

The problem is modeled as:

[
I(x,y) = R(x,y) \times L(x,y)
]

where:

* (I(x,y)) — observed image
* (R(x,y)) — reflectance or underlying texture
* (L(x,y)) — illumination field

The main assumption is that illumination is a **smooth, slowly varying function**, while the underlying texture contains relatively higher-frequency information.

---

## Objectives

The implementation is divided into three parts.

### Part 1 — Model Formulation

* Explain the image formation model.
* Explain why simple histogram equalization cannot recover the underlying reflectance.
* Discuss why separating (R(x,y)) and (L(x,y)) from a single image is an ill-posed problem.
* Introduce the assumption that illumination varies slowly.
* Propose an algorithm for estimating illumination and reflectance.

### Part 2 — Grayscale Image Processing

The grayscale implementation will:

1. Load the input image.
2. Normalize the image intensity.
3. Convert the image into the logarithmic domain.
4. Manually separate low- and high-frequency components.
5. Estimate the illumination component.
6. Estimate the reflectance component.
7. Convert the result back from the log domain.
8. Save and visualize intermediate and final results.

The logarithmic transformation is based on:

[
\log I = \log R + \log L
]

The estimated illumination is obtained from the low-frequency component:

[
\widehat{\log L} \approx LowPass(\log I)
]

The reflectance estimate is then:

[
\widehat{\log R}
================

\log I-\widehat{\log L}
]

and:

[
\hat R =
\exp(\widehat{\log R})
]

The filtering operation will be implemented manually rather than using a built-in image-filtering function, as required by the problem statement.

### Part 3 — Color Image Extension

The grayscale approach has been extended to RGB images in `src/color.py`.

For each color channel `c` in `{R, G, B}`, the image formation model is:

```text
I_c(x, y) = R_c(x, y) × L_c(x, y)
```

Taking logarithms gives:

```text
log(I_c) = log(R_c) + log(L_c)
```

The implementation applies the existing manually generated Gaussian kernel and manual convolution independently to the red, green, and blue log channels. The low-frequency result is the channel illumination estimate.

### Joint illumination model

The three channel estimates can contain two different effects:

* A channel-wide offset describes the spectral color of the light.
* A spatially varying component describes the shared brightness or shading pattern.

The mean low-frequency log illumination is calculated for each channel and treated as its spectral offset. These offsets are removed before averaging the channels to form one shared spatial illumination field:

```text
log(L_shared) = mean_c(log(L_c) - spectral_offset_c)
```

The final corrected RGB image is calculated from the original normalized RGB values:

```text
corrected_c(x, y) = I_c(x, y) / L_shared(x, y)
```

Using the same divisor for all three channels preserves the true color ratios exactly:

```text
corrected_R / corrected_G = I_R / I_G
corrected_G / corrected_B = I_G / I_B
corrected_R / corrected_B = I_R / I_B
```

The original values are used for reconstruction instead of exponentiating the epsilon-stabilized log image. This prevents the small logarithm stabilizer from changing ratios at dark pixels.

### RGB assumptions

The assignment does not prescribe a unique RGB decomposition. This implementation assumes that illumination has a channel-dependent global spectral component and a common spatial component. It therefore removes uneven spatial brightness while preserving color ratios. It does not remove arbitrary spatially varying color casts, because changing those casts would also change the original RGB ratios.

---

## Project Structure

```text
question1/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── main.py
│   ├── filters.py
│   ├── grayscale.py
│   └── color.py
│
├── data/
│   ├── raw/
│   └── processed/
│
└── results/
    ├── grayscale/
    └── color/
```

### `data/raw/`

Contains the original input photograph used for the experiment.

### `data/processed/`

Contains any intermediate data generated during preprocessing.

### `src/`

Contains the implementation of the image-processing pipeline.

### `results/`

Contains generated visualizations and reconstructed images.

---

## Current Pipeline

The planned grayscale pipeline is:

```text
Input Image
     │
     ▼
Normalize
     │
     ▼
Log Transform
     │
     ▼
Manual Low-Pass Filtering
     │
     ├──────────────► Estimated Illumination
     │
     ▼
Subtract Illumination
     │
     ▼
Estimated Log Reflectance
     │
     ▼
Exponentiation
     │
     ▼
Estimated Reflectance
```

For RGB processing, the channel-specific low-frequency estimates are combined after spectral-offset removal:

```text
RGB Input
     │
     ▼
Normalize each channel
     │
     ▼
Log transform each channel
     │
     ▼
Manual Gaussian convolution per channel
     │
     ├──────────────► Channel illumination estimates
     │                         │
     │                         ▼
     │                  Remove spectral offsets
     │                         │
     │                         ▼
     │                  Shared illumination field
     │
     ▼
Divide every original channel by the shared field
     │
     ▼
Corrected RGB image with preserved color ratios
```

---

## Dependencies

The project uses Python and the following packages:

```text
numpy
Pillow
matplotlib
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Place the input image inside:

```text
data/raw/
```

For example:

```text
data/raw/uneven_texture.png
```

The grayscale pipeline uses `uneven_texture.png`. The RGB pipeline uses the separate color image:

```text
data/raw/uneven_texture_color.png
```

Then run:

```bash
python src/main.py
```

---

## Important Assumption

The decomposition

[
I=R\times L
]

is not uniquely determined from a single observed image.

For example, the same observed intensity can be explained by different combinations of reflectance and illumination.

Therefore, the approach relies on the problem's assumption that the illumination field is **smooth and slowly varying**.

This allows the low-frequency component of the log-transformed image to be used as an estimate of illumination.

---

## Results

The experiment will produce and compare:

* Original image
* Log-transformed image
* Estimated illumination
* Estimated log-reflectance
* Reconstructed reflectance

The RGB experiment also saves the following files under `results/color/`:

* `original_rgb.png` — original RGB input.
* `log_image_red.png`, `log_image_green.png`, `log_image_blue.png` — log-domain channel views.
* `channel_illumination_red.png`, `channel_illumination_green.png`, `channel_illumination_blue.png` — per-channel illumination views.
* `joint-illumination.png` — shared spatial illumination estimate.
* `log_reflectance_red.png`, `log_reflectance_green.png`, `log_reflectance_blue.png` — log-reflectance channel views.
* `corrected_rgb.png` — final corrected RGB image.
* `ratio_error.png` — visual diagnostic of RGB ratio error.
* `.npy` files — full-precision arrays for the log image, illumination estimates, reflectance, corrected image, and spectral offsets.

### Observed RGB result

The method was tested using `data/raw/uneven_texture_color.png`, a textured color image with uneven illumination. With the default 51 × 51 Gaussian kernel and sigma 10.0, the saved joint illumination ranged from approximately `0.188` to `5.479`, showing substantial spatial brightness variation. Dividing by this field reduces the uneven illumination while retaining the texture.

The maximum measured normalized RGB-ratio error was approximately `1.7e-16`. This is floating-point roundoff, so no visible change to the true per-pixel RGB color ratios was introduced.

The results will be used to evaluate whether uneven illumination has been reduced while preserving the underlying texture.

---

## Notes

This repository is being developed incrementally. The implementation and experiments will be documented as the project progresses.

The solution is intended to demonstrate the reasoning behind the image-processing approach rather than relying on a single black-box image-enhancement operation. Both the Gaussian kernel and convolution are implemented manually; OpenCV, SciPy, and built-in blur/filter functions are not used.
