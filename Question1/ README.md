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

The grayscale approach will subsequently be extended to RGB images.

The color implementation will consider illumination variation across the red, green, and blue channels while attempting to preserve the true color relationships between channels.

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

The results will be used to evaluate whether uneven illumination has been reduced while preserving the underlying texture.

---

## Notes

This repository is being developed incrementally. The implementation and experiments will be documented as the project progresses.

The solution is intended to demonstrate the reasoning behind the image-processing approach rather than relying on a single black-box image-enhancement operation.
