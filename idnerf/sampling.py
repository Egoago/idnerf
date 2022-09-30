import jax
import jax.numpy as jnp

from absl import flags


def img_pixel_coords(width, height):
    pixel_center = 0.5 if flags.FLAGS.use_pixel_centers else 0.0
    xx, yy = jnp.meshgrid(jnp.arange(width, dtype=jnp.float32) + pixel_center,
                          jnp.arange(height, dtype=jnp.float32) + pixel_center,
                          indexing="ij")
    pixel_coords = jnp.column_stack([xx.flatten(), yy.flatten()])
    return pixel_coords


def __random(rgbdm_img, pixel_count, rng):
    mask = (rgbdm_img[:, :, -1]).astype(dtype=bool)
    indices = jnp.indices(mask.shape)
    indices_yx = jnp.column_stack([indices[0].flatten(), indices[1].flatten()])[mask.flatten()]
    pixel_coords = jax.random.choice(rng, indices_yx, (pixel_count,), replace=False)
    return pixel_coords


def __fast(rgbdm_img, pixel_count, rng):
    import numpy as np
    import cv2
    detector = cv2.FastFeatureDetector_create()  # TODO save detector object for reuse
    gray = np.uint8(rgbdm_img[:, :, :3] * 255)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    key_points = detector.detect(gray, None)
    pixel_coords = np.array([key_point.pt for key_point in key_points], np.int32)[:, [1, 0]]
    weights = np.array([key_point.response for key_point in key_points], np.int32)
    mask = rgbdm_img[pixel_coords[:, 0], pixel_coords[:, 1], -1].astype(bool)
    pixel_coords = pixel_coords[mask]
    weights = weights[mask]
    probs = weights / weights.sum()
    if pixel_coords.shape[0] > pixel_count:
        pixel_coords = jax.random.choice(rng, pixel_coords, (pixel_count,), replace=False, p=probs)
    return pixel_coords


def sample_pixels(rgbdm_img, rng, method=None, pixel_count=None):
    if method is None:
        method = flags.FLAGS.pixel_sampling
    if pixel_count is None:
        pixel_count = flags.FLAGS.pixel_count
    if method == 'total':
        pixel_coords = img_pixel_coords()
    elif method == 'random':
        pixel_coords = __random(rgbdm_img, pixel_count, rng)
    elif method == 'fast':
        pixel_coords = __fast(rgbdm_img, pixel_count, rng)
    elif method == 'fast_random':
        random_pixels = int(pixel_count*0.8)
        fast_pixels = pixel_count-random_pixels
        pixel_coords = jnp.concatenate([__random(rgbdm_img, random_pixels, rng),
                                        __random(rgbdm_img, fast_pixels, rng)])
    else:
        raise NotImplementedError(f"Sampling method not implemented: {method}")
    return rgbdm_img[pixel_coords[:, 0], pixel_coords[:, 1], :4], pixel_coords


def generate_rays(pixel_coords, T, width, height, focal):
    if jnp.ndim(pixel_coords) != 2:
        pixel_coords = pixel_coords.reshape((1, 2))
    pixel_count = pixel_coords.shape[0]
    camera_dirs = jnp.stack([(pixel_coords[:, 1] - width * 0.5) / focal,
                             -(pixel_coords[:, 0] - height * 0.5) / focal,
                             -jnp.ones(pixel_count),
                             jnp.zeros(pixel_count)], axis=-1)
    directions = camera_dirs.dot(T.T)[:, :-1]
    origins = jnp.tile(T[:3, -1], (pixel_count, 1))
    viewdirs = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    return jnp.stack([origins, directions, viewdirs], axis=-1)
