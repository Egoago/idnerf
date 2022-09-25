import jax
import jax.numpy as jnp
from absl import flags

from jaxnerf.nerf import utils


def generate_pixel_coords(img_shape):
    pixel_center = 0.5 if flags.FLAGS.use_pixel_centers else 0.0
    xx, yy = jnp.meshgrid(jnp.arange(img_shape[1], dtype=jnp.float32) + pixel_center,
                          jnp.arange(img_shape[0], dtype=jnp.float32) + pixel_center,
                          indexing="xy")
    pixel_coords = jnp.column_stack([xx.flatten(), yy.flatten()])
    return pixel_coords


def sample_pixels(rgbd_img, rng, method='None'):
    if method is None:
        method = flags.FLAGS.pixel_sampling
    if method == 'total':
        pixel_coords = generate_pixel_coords(rgbd_img.shape)
    elif method == 'random':
        rng, key1, key2 = jax.random.split(rng, 3)
        pixel_coords = jnp.column_stack([
            jax.random.choice(key1, jnp.arange(rgbd_img.shape[1], dtype=jnp.float32), (flags.FLAGS.pixel_count,)),
            jax.random.choice(key2, jnp.arange(rgbd_img.shape[0], dtype=jnp.float32), (flags.FLAGS.pixel_count,))
        ])
    elif method == 'random_no_white':
        mask = jnp.mean(rgbd_img[:, :, :3], axis=-1) < 0.99
        indices = jnp.indices(mask.shape)
        indices_yx = jnp.column_stack([indices[0].flatten(), indices[1].flatten()])[mask.flatten()]
        rng, key1 = jax.random.split(rng, 2)
        pixel_coords = jax.random.choice(key1, indices_yx, (flags.FLAGS.pixel_count,))
    else:
        raise NotImplementedError(f"Sampling method not implemented: {flags.FLAGS.sampling}")
    return rgbd_img[pixel_coords[:, 0], pixel_coords[:, 1]], pixel_coords


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
