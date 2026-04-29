# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Image-level degradation utilities.

This module provides a collection of stochastic degradations that are used to
synthesize training pairs for image-restoration style datasets (denoising,
deblurring, inpainting, super-resolution, rain/fog, low-light, ...).

All functions operate on 3-channel BGR images in numpy format.
"""

import numpy as np
import cv2
import random
import os


# -------------------------------------------------------------
# 1. Helper functions for individual noises / degradations.
#    These helpers assume the input is a float32 image in [0.0, 1.0].
# -------------------------------------------------------------

def add_gaussian_noise(image_org):
    """Add additive white Gaussian noise (AWGN).

    sigma is sampled uniformly from [5, 50] on the 0-255 scale.
    """
    # Sample sigma on the 0-255 scale.
    sigma = random.uniform(5, 50)

    # Convert sigma to the [0.0, 1.0] scale.
    sigma_norm = sigma / 255.0

    # Generate Gaussian noise with np.random.normal(mean, std, shape).
    image = image_org.copy()
    noise = np.random.normal(0.0, sigma_norm, image.shape).astype(np.float32)

    # Add the noise to the image.
    noisy_image = image + noise
    return noisy_image


def add_poisson_noise(image_org):
    """Add Poisson (shot) noise, which is signal-dependent.

    The [0, 1] image is scaled to a reasonable "photon count" range to
    simulate the Poisson process.
    """
    # Scale the image so that the brightest pixel corresponds to roughly
    # ``photon_scale`` photons.
    photon_scale = random.uniform(10, 50)
    image = image_org.copy()
    # np.random.poisson(lam) -- ``lam`` is the expected event rate.
    image_non_negative = np.abs(image)
    image_counts = np.random.poisson(image_non_negative * photon_scale)

    # Scale back into the approximate [0, 1] range.
    noisy_image = image_counts / photon_scale

    # Make sure the output dtype is float32.
    return noisy_image.astype(np.float32)


def add_jpeg_compression(image_org):
    """Add JPEG compression artifacts. ``quality`` is sampled from [30, 50]."""
    # Sample a random JPEG quality.
    quality = random.randint(30, 50)
    image = image_org.copy()
    # cv2's JPEG codec needs uint8 images in [0, 255].
    image_uint8 = (image * 255.0).astype(np.uint8)

    # Encode (compress).
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_image = cv2.imencode('.jpg', image_uint8, encode_param)

    if not result:
        # If encoding fails, return the original image.
        return image

    # Decode (decompress). IMREAD_COLOR always returns a 3-channel BGR image.
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    # Convert back to float32 in [0.0, 1.0].
    noisy_image = decoded_image.astype(np.float32) / 255.0
    return noisy_image


def add_salt_pepper_noise(image):
    """Add salt-and-pepper noise with a random intensity in [0.5%, 5.0%]."""
    # Sample the noise amount.
    amount = random.uniform(0.005, 0.05)

    # Copy the image to avoid mutating the input.
    noisy_image = np.copy(image)

    # Salt pixels (set to 1.0, i.e. white).
    num_salt = int(np.ceil(amount * image.shape[0] * image.shape[1] * 0.5))
    salt_coords = (np.random.randint(0, image.shape[0], num_salt),
                   np.random.randint(0, image.shape[1], num_salt))
    noisy_image[salt_coords] = 1.0

    # Pepper pixels (set to 0.0, i.e. black).
    num_pepper = int(np.ceil(amount * image.shape[0] * image.shape[1] * 0.5))
    pepper_coords = (np.random.randint(0, image.shape[0], num_pepper),
                     np.random.randint(0, image.shape[1], num_pepper))
    noisy_image[pepper_coords] = 0.0

    return noisy_image


def apply_stochastic_degradation(image):
    """Apply a randomized degradation pipeline to a uint8 image.

    :param image: a clean image of shape (H, W, 3), np.uint8 in [0, 255].
    :return: a degraded image of the same shape and dtype.
    """

    # 1. Convert to float32 in [0.0, 1.0]. We assume BGR input (cv2.imread).
    clean_image_uint8 = image.copy()
    image_float = clean_image_uint8.astype(np.float32) / 255.0

    # 2. Define the list of degradation operations and their probabilities.
    #    (operation_fn, probability_of_being_applied)
    degradations = [
        (add_gaussian_noise, 0.8),
        (add_jpeg_compression, 0.5),
        (add_poisson_noise, 0.5),
        (add_salt_pepper_noise, 0.2)
    ]

    # 3. Randomly shuffle the order of operations (important for diversity!).
    random.shuffle(degradations)

    noisy_image = image_float

    # 4. Apply each degradation in the random order with its own probability.
    for op_func, op_prob in degradations:
        if random.random() < op_prob:
            noisy_image = op_func(noisy_image)

    # 5. Clip values back into [0.0, 1.0] to simulate sensor saturation.
    final_image_float = np.clip(noisy_image, 0.0, 1.0)

    # 6. Convert back to uint8 in [0, 255].
    final_image_uint8 = (final_image_float * 255.0).astype(np.uint8)

    return final_image_uint8


def create_sr_pair(image, scale_factor_list=[2, 4, 6, 8]):
    """Create a (low-resolution, high-resolution) pair from an HR image.

    This is the standard "Bicubic" degradation model used for super resolution.

    :param image: a clean HR image as a NumPy BGR uint8 array.
    :param scale_factor_list: candidate integer scale factors, one is picked
        at random (e.g. picking 4 means x4 SR).
    :return: a tuple (lr_image, lr_upsampled, hr_image_target) -- low-res
        image, low-res image bicubic-upsampled back to HR size, and the HR
        target image (cropped to be divisible by the scale factor).
    """
    scale_factor = random.choice(scale_factor_list)
    # 1. Read the original HR size.
    hr_image = image.copy()
    h_hr, w_hr = hr_image.shape[:2]

    # 2. [important] crop HR so the spatial dims are divisible by the scale
    #    factor -- this guarantees a perfect pixel-level correspondence
    #    between LR and HR.
    h_hr_target = h_hr - (h_hr % scale_factor)
    w_hr_target = w_hr - (w_hr % scale_factor)

    hr_image_target = hr_image[:h_hr_target, :w_hr_target, :]

    # 3. Compute the LR target size.
    h_lr = h_hr_target // scale_factor
    w_lr = w_hr_target // scale_factor

    # 4. Bicubic down-sampling via cv2.resize.
    #    - dsize is (width, height).
    #    - INTER_CUBIC is the standard choice for SR benchmarks.
    lr_image = cv2.resize(hr_image_target,
                          (w_lr, h_lr),
                          interpolation=cv2.INTER_CUBIC)
    lr_upsampled = cv2.resize(lr_image,
                              (hr_image_target.shape[1], hr_image_target.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
    # 5. Return (input, baseline, target).
    return lr_image, lr_upsampled, hr_image_target  # low-res | bicubic-up | high-res


def apply_inpainting_corruption(image, mode='mixed',
                                mask_color_list=[(255, 255, 255), (0, 0, 0)],
                                intensity_list=[0.3, 0.5, 0.6, 0.8]):
    """Corrupt an image by drawing random masks, with controllable intensity.

    Args:
        image (np.ndarray): input image (H, W, 3), uint8.
        mode (str): 'rect', 'stroke' or 'mixed'.
        mask_color_list (list[tuple]): candidate mask colors.
        intensity_list (list[float]): candidate intensities in [0.0, 1.0].
            - 0.1 : very small / subtle corruption
            - 0.5 : moderate corruption (default level)
            - 0.9+: heavy corruption, large occluded area

    Returns:
        np.ndarray: the corrupted image.
    """
    intensity = random.choice(intensity_list)
    mask_color = random.choice(mask_color_list)  # Randomly pick a mask color.

    corrupted_img = image.copy()
    h, w, _ = corrupted_img.shape
    min_dim = min(h, w)

    # Clamp intensity into a sensible range.
    intensity = np.clip(intensity, 0.1, 1.0)

    # -------------------------------------------------
    # Helper: draw random rectangles (scaled by intensity).
    # -------------------------------------------------
    def _draw_random_rects(img):
        # Number of rectangles grows with intensity:
        #   intensity=0.5 -> 1~6 rects; intensity=1.0 -> 1~11 rects.
        max_rects = int(5 + 10 * intensity)
        num_rects = np.random.randint(1, max_rects)

        for _ in range(num_rects):
            # Size also grows with intensity.
            #   min size: 5% of the shortest side.
            #   max size: (10% + 40% * intensity) of the shortest side.
            min_s = int(min_dim * 0.05)
            max_s = int(min_dim * (0.1 + 0.4 * intensity))

            # Make sure max > min.
            max_s = max(max_s, min_s + 1)

            rh = np.random.randint(min_s, max_s)
            rw = np.random.randint(min_s, max_s)

            rx = np.random.randint(0, w - rw)
            ry = np.random.randint(0, h - rh)

            cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), mask_color, -1)

    # -------------------------------------------------
    # Helper: draw irregular strokes (scaled by intensity).
    # -------------------------------------------------
    def _draw_irregular_strokes(img):
        # Number of strokes grows with intensity.
        max_strokes = int(3 + 12 * intensity)
        num_strokes = np.random.randint(1, max_strokes)

        for _ in range(num_strokes):
            # Vertex count -> how long / convoluted the stroke is.
            max_vertex = int(4 + 12 * intensity)
            num_vertex = np.random.randint(2, max_vertex)

            start_x = np.random.randint(0, w)
            start_y = np.random.randint(0, h)
            prev_pt = (start_x, start_y)

            for i in range(num_vertex):
                offset_x = np.random.randint(-50, 50)
                offset_y = np.random.randint(-50, 50)
                curr_x = np.clip(prev_pt[0] + offset_x, 0, w)
                curr_y = np.clip(prev_pt[1] + offset_y, 0, h)
                curr_pt = (int(curr_x), int(curr_y))

                # Stroke thickness grows with intensity.
                # Minimum thickness is 2; maximum grows with intensity.
                max_thick = int(min_dim * (0.02 + 0.1 * intensity))
                thickness = np.random.randint(2, max(3, max_thick))

                cv2.line(img, prev_pt, curr_pt, mask_color, thickness)
                cv2.circle(img, prev_pt, thickness // 2, mask_color, -1)
                prev_pt = curr_pt

    # -------------------------------------------------
    # Main dispatch.
    # -------------------------------------------------
    if mode == 'rect':
        _draw_random_rects(corrupted_img)
    elif mode == 'stroke':
        _draw_irregular_strokes(corrupted_img)
    elif mode == 'mixed':
        # In "mixed" mode, intensity decides whether to stack rects on top
        # of strokes. Higher intensity -> more likely to stack.
        _draw_irregular_strokes(corrupted_img)
        if np.random.rand() < (0.3 + 0.5 * intensity):
            _draw_random_rects(corrupted_img)

    return corrupted_img


def apply_motion_blur(image, size_list=[10, 20, 30], angle=None):
    """Apply linear motion blur to an image.

    Args:
        image (np.ndarray): input image (H, W, 3), uint8, BGR.
        size_list (list[int]): candidate kernel sizes (the motion trajectory
            length). A larger value means a heavier blur. Typical range is
            [3, 30].
        angle (float | None): motion angle in [0, 360] degrees. If None, a
            random angle is sampled.

    Returns:
        np.ndarray: the blurred image (H, W, 3), uint8, BGR.
    """
    # 1. Resolve random parameters.
    size = random.choice(size_list)
    if size is None:
        # Odd sizes make center handling a bit cleaner.
        size = np.random.randint(5, 21)
        if size % 2 == 0:
            size += 1

    if angle is None:
        angle = np.random.uniform(0, 360)

    # 2. Build the motion blur kernel.
    # A (size, size) matrix with a horizontal line of 1s through the center.
    kernel = np.zeros((size, size))
    center = (size - 1) // 2
    kernel[center, :] = 1

    # 3. Rotate the kernel line to simulate motion at arbitrary angles.
    # cv2.getRotationMatrix2D(rotation_center, angle, scale)
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))

    # 4. Normalize so the kernel sums to 1 (preserve image brightness).
    kernel = kernel / np.sum(kernel)

    # 5. Apply the convolution.
    # cv2.filter2D handles all 3 BGR channels. ddepth=-1 keeps the input
    # dtype (uint8). BORDER_REFLECT_101 avoids black borders.
    blurred_img = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    return blurred_img


def apply_low_light_corruption(image_org, brightness_factor=None, noise_sigma=None, beta_gamma=1.0):
    """Simulate a low-light environment and add sensor noise.

    Args:
        image_org (np.ndarray): input image (H, W, 3), uint8, BGR.
        brightness_factor (float | None): brightness scaling factor in
            (0.0, 1.0]. Smaller is darker, typical range [0.1, 0.5]. If
            None a random value is sampled.
        noise_sigma (float | None): stddev of the additive Gaussian noise on
            the normalized [0, 1] scale, typical range [0.01, 0.05]. If None
            a random value is sampled.
        beta_gamma (float): gamma correction for non-linear darkening.
            Defaults to 1.0 (linear). Values > 1.0 push shadows darker.

    Returns:
        np.ndarray: the low-light image (H, W, 3), uint8, BGR.
    """
    # 1. Convert to float32 in [0, 1] for physically meaningful math.
    image = image_org.copy()
    img_f = image.astype(np.float32) / 255.0

    # 2. Resolve parameters.
    if brightness_factor is None:
        # Sample brightness: 0.1 (very dark) to 0.5 (dim).
        brightness_factor = np.random.uniform(0.1, 0.5)

    if noise_sigma is None:
        # Sample noise stddev: 0.01 (light) to 0.04 (noticeable).
        # Low light often comes with high-ISO sensor noise.
        noise_sigma = np.random.uniform(0.01, 0.04)

    # 3. Signal attenuation: fewer photons -> linear multiplication.
    dark_img = img_f * brightness_factor

    # Optionally apply a gamma curve for non-linear darkening.
    if beta_gamma != 1.0:
        dark_img = np.power(dark_img, beta_gamma)

    # 4. Inject noise with the same shape as the image.
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=dark_img.shape)
    dark_img_noisy = dark_img + noise

    # 5. Clip and quantize back to uint8 BGR.
    dark_img_noisy = np.clip(dark_img_noisy, 0.0, 1.0)
    return (dark_img_noisy * 255).astype(np.uint8)


def apply_rain_fog_corruption(image_org, mode_list=['rain', 'fog'], intensity_list=[0.4, 0.6, 0.8]):
    """Apply rain or fog corruption to an image.

    Args:
        image_org (np.ndarray): input image (H, W, 3), uint8, BGR.
        mode_list (list[str]): candidate modes, 'rain' or 'fog'.
        intensity_list (list[float]): candidate intensities in [0, 1].
            - For rain: controls droplet density and streak length.
            - For fog: controls the fog density (beta).

    Returns:
        np.ndarray: the corrupted image (H, W, 3), uint8, BGR.
    """

    # ==========================================================
    # Helper: generate rain streaks.
    # Idea: random noise -> motion blur -> threshold -> alpha blend.
    # ==========================================================
    def _generate_rain(img, strength):
        h, w = img.shape[:2]

        # 1. Generate a random noise layer; more noise means denser rain.
        noise = np.random.randint(0, 255, (h, w), dtype=np.uint8)

        # Keep only the very brightest pixels as "rain drop seeds".
        # The threshold is very tight: only a handful of pixels are white.
        # Higher strength -> slightly lower threshold -> more seeds.
        threshold = 254 - int(strength * 2)
        _, rain_mask = cv2.threshold(noise, threshold, 255, cv2.THRESH_BINARY)

        # 2. Motion-blur the drops into streaks.
        # Longer streaks with higher strength.
        length = int(10 + strength * 30)
        angle = 105.0  # Slightly tilted rain (vertical is 90 degrees).

        M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(length))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (length, length))
        motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)

        # Convolve the seed mask to produce streaks.
        rain_streaks = cv2.filter2D(rain_mask, -1, motion_blur_kernel)

        # 3. Blend the rain layer onto the input image.
        # Expand the single-channel rain streak mask to 3-channel BGR.
        rain_layer = cv2.cvtColor(rain_streaks, cv2.COLOR_GRAY2BGR)

        # Alpha-blend the rain layer on top of the image; directly adding
        # would easily saturate, so use cv2.addWeighted which clips for us.
        alpha = 0.8                        # Weight of the input image.
        beta = 0.4 + (strength * 0.2)      # Weight of the rain layer.

        rainy_img = cv2.addWeighted(img, alpha, rain_layer, beta, 0)

        return rainy_img

    # ==========================================================
    # Helper: generate fog / haze.
    # Atmospheric scattering model: I = J * t + A * (1 - t).
    # ==========================================================
    def _generate_fog(img, strength):
        h, w = img.shape[:2]
        img_f = img.astype(np.float32) / 255.0

        # 1. Atmospheric light A (typically bright, grayish-white).
        A = 0.95

        # 2. Transmission map t(x).
        # Physically t = exp(-beta * depth). Since we don't have a depth map,
        # we approximate it with a simple vertical gradient + random noise.
        beta = 0.5 + strength * 3.0    # Higher beta -> thicker fog.

        # Assume the top of the image is farther (larger depth).
        row_idx = np.linspace(0.8, 0.2, h)
        depth_map = np.tile(row_idx[:, None], (1, w))

        # Add a little noise so the fog doesn't look like a uniform sheet.
        noise = np.random.normal(0, 0.05, (h, w))
        depth_map = np.clip(depth_map + noise, 0.1, 1.0)

        # Compute transmission.
        t = np.exp(-beta * depth_map)

        # Expand t to 3 channels for broadcasting.
        t = np.dstack([t] * 3)

        # 3. Apply the atmospheric scattering formula.
        foggy_img = img_f * t + A * (1 - t)

        return (np.clip(foggy_img, 0, 1) * 255).astype(np.uint8)

    # ==========================================================
    # Main dispatch.
    # ==========================================================
    mode = random.choice(mode_list)
    intensity = random.choice(intensity_list)

    intensity = np.clip(intensity, 0.1, 1.0)
    image = image_org.copy()
    if mode == 'rain':
        return _generate_rain(image, intensity)
    elif mode == 'fog':
        return _generate_fog(image, intensity)
    else:
        print(f"Unknown mode: {mode}, returning the original image.")
        return image


# ==========================================================
# Minimal smoke test. Run with:
#     python -m data.add_noise
# to quickly sanity-check that every degradation function runs.
# ==========================================================
if __name__ == "__main__":
    # Use a deterministic synthetic image so this smoke test does not depend
    # on any external data.
    input_img = np.random.randint(100, 255, (512, 512, 3), dtype=np.uint8)
    filename = "test_dummy.png"

    output_dir = "noise_demo"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Start processing. Input image shape: {input_img.shape}")

    # --- Rain ---
    rain_weak = apply_rain_fog_corruption(input_img, mode_list=['rain'], intensity_list=[0.2])
    cv2.imwrite(os.path.join(output_dir, f"rain_weak_{filename}"), rain_weak)

    rain_heavy = apply_rain_fog_corruption(input_img, mode_list=['rain'], intensity_list=[0.8])
    cv2.imwrite(os.path.join(output_dir, f"rain_heavy_{filename}"), rain_heavy)

    # --- Fog ---
    fog_light = apply_rain_fog_corruption(input_img, mode_list=['fog'], intensity_list=[0.3])
    cv2.imwrite(os.path.join(output_dir, f"fog_light_{filename}"), fog_light)

    fog_dense = apply_rain_fog_corruption(input_img, mode_list=['fog'], intensity_list=[0.9])
    cv2.imwrite(os.path.join(output_dir, f"fog_dense_{filename}"), fog_dense)

    print(f"Done. See output directory: {output_dir}")
    print(f"Output format: dtype={fog_dense.dtype}, shape={fog_dense.shape}")
