import jax
import jax.numpy as jnp

from absl import flags, logging

from idnerf import base, math


def __random(rgbdm_img, pixel_count, rng):
    mask = (rgbdm_img[:, :, -1]).astype(dtype=bool)
    indices = jnp.indices(mask.shape)
    indices_yx = jnp.column_stack([indices[0].flatten(), indices[1].flatten()])[mask.flatten()]
    pixel_coords_yx = jax.random.choice(rng, indices_yx, (pixel_count,), replace=False)
    return pixel_coords_yx


def __fast(rgbdm_img, pixel_count, rng):
    import numpy as np
    import cv2
    detector = cv2.FastFeatureDetector_create()  # TODO save detector object for reuse
    gray = np.uint8(rgbdm_img[:, :, :3] * 255)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    key_points = detector.detect(gray, None)
    pixel_coords_yx = np.array([key_point.pt for key_point in key_points], np.int32)[:, [1, 0]]
    weights = np.array([key_point.response for key_point in key_points], np.int32)
    mask = rgbdm_img[pixel_coords_yx[:, 0], pixel_coords_yx[:, 1], -1].astype(bool)
    pixel_coords_yx = pixel_coords_yx[mask]
    weights = weights[mask]
    probs = weights / weights.sum()
    if pixel_coords_yx.shape[0] > pixel_count:
        pixel_coords_yx = jax.random.choice(rng, pixel_coords_yx, (pixel_count,), replace=False, p=probs)
    return pixel_coords_yx


def sample_img(rgbdm_img, rng, method=None, pixel_count=None) -> jnp.ndarray:
    if method is None:
        method = flags.FLAGS.pixel_sampling
    if pixel_count is None:
        pixel_count = flags.FLAGS.pixel_count
    if method == 'total':
        pixel_coords_yx = math.total_pixel_coords_xy(rgbdm_img.shape[1], rgbdm_img.shape[0])
    elif method == 'random':
        pixel_coords_yx = __random(rgbdm_img, pixel_count, rng)
    elif method == 'fast':
        pixel_coords_yx = __fast(rgbdm_img, pixel_count, rng)
    elif method == 'fast_random':
        random_pixels = int(pixel_count * 0.8)
        fast_pixels = pixel_count - random_pixels
        pixel_coords_yx = jnp.concatenate([__random(rgbdm_img, random_pixels, rng),
                                           __random(rgbdm_img, fast_pixels, rng)])
    else:
        raise NotImplementedError(f"Sampling method not implemented: {method}")
    return pixel_coords_yx


def sample_imgs(data: base.Data, rng) -> None:
    img_count = len(data.frames)
    pixel_count = flags.FLAGS.pixel_count
    assert pixel_count % img_count == 0
    pixels_per_img = pixel_count // img_count

    for frame in data.frames:
        rng, subkey = jax.random.split(rng, 2)
        pixel_coords_yx = sample_img(frame.rgbdm_img, subkey, pixel_count=pixels_per_img)
        frame.pixel_coords_yx = pixel_coords_yx

    logging.info("Sampling pixels finished")
