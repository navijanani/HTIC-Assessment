from pathlib import Path

from color import process_color
from grayscale import process_grayscale



# PROJECT PATHS


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RAW_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)



# Input images


GRAYSCALE_INPUT = (
    RAW_DIR
    / "uneven_texture.png"
)

COLOR_INPUT = (
    RAW_DIR
    / "uneven_texture.png"
)



# OUTPUT DIRECTORIES


GRAYSCALE_RESULTS = (
    RESULTS_DIR
    / "grayscale"
)

COLOR_RESULTS = (
    RESULTS_DIR
    / "color"
)



# MAIN


def main():

    print(
        "Project root:",
        PROJECT_ROOT
    )

    print(
        "Input image:",
        GRAYSCALE_INPUT
    )

    if not GRAYSCALE_INPUT.exists():

        raise FileNotFoundError(
            f"Input image not found: "
            f"{GRAYSCALE_INPUT}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GRAYSCALE_RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    COLOR_RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    process_grayscale(
        input_path=GRAYSCALE_INPUT,
        output_dir=GRAYSCALE_RESULTS,
        kernel_size=51,
        sigma=10.0
    )

    print(
        "Grayscale processing completed."
    )

    if not COLOR_INPUT.exists():
        raise FileNotFoundError(
            f"Color image not found: "
            f"{COLOR_INPUT}"
        )

    process_color(
        input_path=COLOR_INPUT,
        output_dir=COLOR_RESULTS,
        kernel_size=51,
        sigma=10.0
    )

    print(
        "Color processing completed."
    )


if __name__ == "__main__":
    main()