"""Joint RGB illumination correction built on the grayscale log-domain method."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from filters import create_gaussian_kernel, manual_convolution


EPSILON = 1e-6
CHANNEL_NAMES = ("red", "green", "blue")


def load_color_image(path, max_width=500):
	"""Load an RGB image and resize it before manual convolution."""

	image = Image.open(path).convert("RGB")

	if image.width > max_width:
		ratio = max_width / image.width
		new_height = int(image.height * ratio)
		image = image.resize((max_width, new_height))

	return np.asarray(image, dtype=np.float64) / 255.0


def normalize_for_display(image):
	"""Apply one scalar normalization so channel ratios are not changed."""

	scale = image.max()
	if scale <= EPSILON:
		return np.zeros_like(image)
	return np.clip(image / scale, 0.0, 1.0)


def save_channel_images(image, output_dir, name):
	"""Save RGB channels as grayscale views for inspecting intermediate data."""

	for channel_index, channel_name in enumerate(CHANNEL_NAMES):
		channel = image[:, :, channel_index]
		channel_min = channel.min()
		channel_range = channel.max() - channel_min
		display = (channel - channel_min) / (channel_range + EPSILON)
		plt.imsave(
			output_dir / f"{name}_{channel_name}.png",
			display,
			cmap="gray"
		)


def process_color(input_path, output_dir, kernel_size=51, sigma=10.0):
	"""Correct spatial illumination while preserving each pixel's RGB ratios."""

	image = load_color_image(input_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	log_image = np.log(image + EPSILON)
	kernel = create_gaussian_kernel(size=kernel_size, sigma=sigma)

	log_channel_illumination = np.empty_like(log_image)
	for channel_index in range(3):
		log_channel_illumination[:, :, channel_index] = manual_convolution(
			log_image[:, :, channel_index],
			kernel
		)

	# Channel means provide channel-dependent offsets for the assumed
	# spectral component; the centered part represents shared spatial shading.
	spectral_offsets = log_channel_illumination.mean(axis=(0, 1))
	centered_log_illumination = (
		log_channel_illumination - spectral_offsets
	)
	log_joint_illumination = centered_log_illumination.mean(axis=2)
	joint_illumination = np.exp(log_joint_illumination)
	log_reflectance = log_image - log_joint_illumination[:, :, None]
	corrected = image / joint_illumination[:, :, None]

	np.save(output_dir / "joint_illumination.npy", joint_illumination)
	np.save(output_dir / "log_reflectance.npy", log_reflectance)
	np.save(output_dir / "corrected_rgb.npy", corrected)
	np.save(output_dir / "spectral_offsets.npy", spectral_offsets)

	
	plt.imsave(
		output_dir / "joint-illumination.png",
		normalize_for_display(joint_illumination),
		cmap="gray"
	)
	plt.imsave(
		output_dir / "corrected_rgb.png",
		normalize_for_display(corrected)
	)
	plt.imsave(
		output_dir / "original_rgb.png",
		image
	)

	channel_sum = image.sum(axis=2)
	valid_pixels = channel_sum > EPSILON

	input_ratios = np.zeros_like(image)
	input_ratios[valid_pixels] = (
		image[valid_pixels]
		/ channel_sum[valid_pixels, None]
	)

	corrected_sum = corrected.sum(axis=2)
	corrected_ratios = np.zeros_like(corrected)
	corrected_ratios[valid_pixels] = (
		corrected[valid_pixels]
		/ corrected_sum[valid_pixels, None]
	)

	ratio_error = np.zeros_like(channel_sum)
	ratio_error[valid_pixels] = np.max(
		np.abs(
			input_ratios[valid_pixels]
			- corrected_ratios[valid_pixels]
		),
		axis=1
	)
	plt.imsave(
		output_dir / "ratio_error.png",
		np.clip(ratio_error * 100000.0, 0.0, 1.0),
		cmap="gray"
	)

	print("Color image shape:", image.shape)
	print("Joint illumination range:", joint_illumination.min(), joint_illumination.max())
	if np.any(valid_pixels):
		print(
			"Maximum RGB ratio error:",
			ratio_error[valid_pixels].max()
		)
	else:
		print("Maximum RGB ratio error: no valid color pixels")

	return {
		"image": image,
		
		"corrected": corrected,
		"ratio_error": ratio_error
	}
