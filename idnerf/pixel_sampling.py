import jax
import jax.numpy as jnp

from absl import flags

from jaxnerf.nerf import utils


def generate_pixel_coords(img_shape):
    pixel_center = 0.5 if flags.FLAGS.use_pixel_centers else 0.0
    xx, yy = jnp.meshgrid(jnp.arange(img_shape[1], dtype=jnp.float32) + pixel_center,
                          jnp.arange(img_shape[0], dtype=jnp.float32) + pixel_center,
                          indexing="ij")
    pixel_coords = jnp.column_stack([xx.flatten(), yy.flatten()])
    return pixel_coords


def random(rgbdm_img, pixel_count, rng):
    mask = (rgbdm_img[:, :, -1]).astype(dtype=bool)
    indices = jnp.indices(mask.shape)
    indices_yx = jnp.column_stack([indices[0].flatten(), indices[1].flatten()])[mask.flatten()]
    pixel_coords = jax.random.choice(rng, indices_yx, (pixel_count,), replace=False)
    return pixel_coords

def fast(rgbdm_img, pixel_count, rng):
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


def sample_pixels(rgbdm_img, rng, method=None):
    pixel_count = flags.FLAGS.pixel_count
    if method is None:
        method = flags.FLAGS.pixel_sampling
    if method == 'total':
        pixel_coords = generate_pixel_coords(rgbdm_img.shape[:2])
    elif method == 'random':
        pixel_coords = random(rgbdm_img, pixel_count, rng)
    elif method == 'fast':
        pixel_coords = fast(rgbdm_img, pixel_count, rng)
    elif method == 'fast_random':
        random_pixels = int(pixel_count*0.8)
        fast_pixels = pixel_count-random_pixels
        pixel_coords = jnp.concatenate([random(rgbdm_img, random_pixels, rng),
                                        fast(rgbdm_img, fast_pixels, rng)])
    else:
        raise NotImplementedError(f"Sampling method not implemented: {method}")
    return rgbdm_img[pixel_coords[:, 0], pixel_coords[:, 1], :4], pixel_coords


def generate_ray(pixel_coords, img_shape, focal, T):
    camera_dir = jnp.array([(pixel_coords[1] - img_shape[1] * 0.5) / focal,
                            -(pixel_coords[0] - img_shape[0] * 0.5) / focal,
                            -1.,
                            0.])
    direction = camera_dir.dot(T.T)[:-1]
    origin = T[:3, -1]
    viewdir = direction / jnp.linalg.norm(direction)
    return utils.Rays(origins=origin[None, :], directions=direction[None, :], viewdirs=viewdir[None, :])


def generate_rays(pixel_coords, img_shape, focal, T):
    pixel_count = pixel_coords.shape[0]
    camera_dirs = jnp.stack([(pixel_coords[:, 1] - img_shape[1] * 0.5) / focal,
                             -(pixel_coords[:, 0] - img_shape[0] * 0.5) / focal,
                             -jnp.ones(pixel_count),
                             jnp.zeros(pixel_count)], axis=-1)
    directions = camera_dirs.dot(T.T)[:, :-1]
    origins = jnp.tile(T[:3, -1], (pixel_count, 1))
    viewdirs = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    return utils.Rays(origins=origins, directions=directions, viewdirs=viewdirs)
