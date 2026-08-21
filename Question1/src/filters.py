import math

import numpy as np


def create_gaussian_kernel(size=51, sigma=10.0):
    """Build a normalized square Gaussian kernel using scalar operations."""

    if size < 3 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd and at least 3.")

    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    kernel_sum = 0.0

    for y in range(size):
        for x in range(size):
            distance_squared = (
                (x - center) ** 2
                + (y - center) ** 2
            )
            value = math.exp(
                -distance_squared / (2.0 * sigma * sigma)
            )
            kernel[y, x] = value
            kernel_sum += value

    kernel /= kernel_sum

    return kernel


def gaussian_kernel(size, sigma):
    """Preserve the original filter API."""

    return create_gaussian_kernel(
        size=size,
        sigma=sigma
    )


def manual_convolution(image, kernel):
    """Apply a 2D filter with explicit reflected-boundary convolution."""

    if image.ndim != 2:
        raise ValueError("manual_convolution expects a 2D image.")

    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array.")

    kernel_height, kernel_width = kernel.shape

    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(
        image,
        (
            (pad_y, pad_y),
            (pad_x, pad_x)
        ),
        mode="reflect"
    )

    height, width = image.shape

    output = np.zeros(
        (height, width),
        dtype=np.float64
    )

    for y in range(height):
        for x in range(width):

            window = padded[
                y:y + kernel_height,
                x:x + kernel_width
            ]

            output[y, x] = np.sum(
                window * kernel
            )

    return output