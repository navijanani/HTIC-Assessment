from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from filters import gaussian_kernel, manual_convolution

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "uneven_texture.png"
)

RESULT_DIR = (
    PROJECT_DIR
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



EPSILON = 1e-6

KERNEL_SIZE = 51
SIGMA = 12.0



def load_grayscale(path):
    """Load image and convert it to a floating-point grayscale array."""

    image = Image.open(path).convert("L")

    image = np.asarray(
        image,
        dtype=np.float64
    )

    return image


def normalize_image(image):
    """Convert 8-bit image values to approximately [0, 1]."""

    return image / 255.0


def to_log_domain(image):
    """Convert image to logarithmic domain."""

    return np.log(
        image + EPSILON
    )


def normalize_for_display(image):
    """Scale an array to [0, 1] for visualization."""

    minimum = image.min()
    maximum = image.max()

    return (
        image - minimum
    ) / (
        maximum - minimum + EPSILON
    )



def main():

    # 1. Load image
    image = load_grayscale(
        IMAGE_PATH
    )

    # 2. Normalize
    image = normalize_image(
        image
    )

    # 3. Log transformation
    log_image = to_log_domain(
        image
    )

    # 4. Construct low-pass filter
    kernel = gaussian_kernel(
        KERNEL_SIZE,
        SIGMA
    )

    # 5. Estimate low-frequency illumination
    log_illumination = manual_convolution(
        log_image,
        kernel
    )

    # 6. Remove estimated illumination
    log_reflectance = (
        log_image
        - log_illumination
    )

    # 7. Return to original domain
    reflectance = np.exp(
        log_reflectance
    )

 
    illumination = np.exp(
        log_illumination
    )

   
    illumination_display = normalize_for_display(
        illumination
    )

    reflectance_display = normalize_for_display(
        reflectance
    )

   
    print("Image shape:", image.shape)

    print(
        "Image range:",
        image.min(),
        "to",
        image.max()
    )

    print(
        "Log illumination range:",
        log_illumination.min(),
        "to",
        log_illumination.max()
    )

    print(
        "Reflectance range:",
        reflectance.min(),
        "to",
        reflectance.max()
    )

    print(
        "Kernel size:",
        KERNEL_SIZE
    )

    print(
        "Sigma:",
        SIGMA
    )

  
    plt.figure(
        figsize=(16, 4)
    )

    plt.subplot(1, 4, 1)
    plt.imshow(
        image,
        cmap="gray"
    )
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(
        log_image,
        cmap="gray"
    )
    plt.title("Log Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(
        illumination_display,
        cmap="gray"
    )
    plt.title("Estimated Illumination")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(
        reflectance_display,
        cmap="gray"
    )
    plt.title("Estimated Reflectance")
    plt.axis("off")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()