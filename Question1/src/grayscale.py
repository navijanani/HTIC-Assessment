import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from filters import (
    create_gaussian_kernel,
    manual_convolution
)


EPSILON = 1e-6
def load_image(path, max_width=500):
    """
    Load grayscale image and resize for manual convolution.
    """

    image = Image.open(path).convert("L")

    if image.width > max_width:

        ratio = max_width / image.width

        new_height = int(
            image.height * ratio
        )

        image = image.resize(
            (max_width, new_height)
        )

    return np.asarray(
        image,
        dtype=np.float64
    )

def normalize_image(image):
    """Normalize image from [0, 255] to [0, 1]."""

    return image / 255.0


def log_transform(image):
    """Convert image to logarithmic domain."""

    return np.log(
        image + EPSILON
    )


def normalize_for_display(image):
    """Normalize image to [0, 1]."""

    minimum = image.min()
    maximum = image.max()

    return (
        (image - minimum)
        /
        (maximum - minimum + EPSILON)
    )


def process_grayscale(
    input_path,
    output_dir,
    kernel_size=51,
    sigma=10.0
):
    """
    Complete grayscale illumination correction pipeline.
    """

    
    # Load image
   

    image = load_image(
        input_path
    )

    print(
        "Original image shape:",
        image.shape
    )


   
    # Normalize the image
   

    image_normalized = normalize_image(
        image
    )


    
    # Log transformation in normalized image
    

    log_image = log_transform(
        image_normalized
    )



    kernel = create_gaussian_kernel(
        size=kernel_size,
        sigma=sigma
    )

    print(
        "Gaussian kernel:",
        kernel.shape
    )

    print(
        "Kernel sum:",
        kernel.sum()
    )


    # Estimate illumination -low frequency
    

    print(
        "Estimating illumination..."
    )

    log_illumination = manual_convolution(
        log_image,
        kernel
    )


  
    # Estimate reflectance-high frequency
    

    log_reflectance = (
        log_image
        -
        log_illumination
    )


    illumination = np.exp(
        log_illumination
    )

    reflectance = np.exp(
        log_reflectance
    )


   
    # Normalize results
  

    illumination_display = (
        normalize_for_display(
            illumination
        )
    )

    reflectance_display = (
        normalize_for_display(
            reflectance
        )
    )


   

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.imsave(
        output_dir / "log_image.png",
        log_image,
        cmap="gray"
    )


    plt.imsave(
        output_dir / "illumination.png",
        illumination_display,
        cmap="gray"
    )


    plt.imsave(
        output_dir / "reflectance.png",
        reflectance_display,
        cmap="gray"
    )


   
    # Comparison figure
  

    plt.figure(
        figsize=(16, 4)
    )


    plt.subplot(1, 4, 1)

    plt.imshow(
        image_normalized,
        cmap="gray"
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")


    plt.subplot(1, 4, 2)

    plt.imshow(
        log_image,
        cmap="gray"
    )

    plt.title(
        "Log Image"
    )

    plt.axis("off")


    plt.subplot(1, 4, 3)

    plt.imshow(
        illumination_display,
        cmap="gray"
    )

    plt.title(
        "Estimated Illumination"
    )

    plt.axis("off")


    plt.subplot(1, 4, 4)

    plt.imshow(
        reflectance_display,
        cmap="gray"
    )

    plt.title(
        "Estimated Reflectance"
    )

    plt.axis("off")


    plt.tight_layout()


    plt.savefig(
        output_dir / "comparison.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


    return reflectance_display