import numpy as np


def gaussian_kernel(size, sigma):
    """Create a normalized 2D Gaussian kernel."""

    if size < 3 or size % 2 == 0:
        raise ValueError("Kernel size must be an odd number >= 3.")

    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    radius = size // 2

    axis = np.arange(-radius, radius + 1)

    x, y = np.meshgrid(axis, axis)

    kernel = np.exp(
        -(x * x + y * y) / (2.0 * sigma * sigma)
    )

    kernel /= kernel.sum()

    return kernel


def manual_convolution(image, kernel):
    """Apply a 2D filter using explicit convolution."""

    kernel_height, kernel_width = kernel.shape

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

    result = np.zeros(
        (height, width),
        dtype=np.float64
    )

    for y in range(height):
        for x in range(width):

            window = padded[
                y:y + kernel_height,
                x:x + kernel_width
            ]

            result[y, x] = np.sum(
                window * kernel
            )

    return result