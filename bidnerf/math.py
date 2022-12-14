from typing import Tuple, List

import jax
import jaxlie
from jax import numpy as jnp

from bidnerf import base
from jaxnerf.nerf import utils


def coords_to_local_rays(pixel_coords_yx: jnp.ndarray, cam_params: base.CameraParameters) -> utils.Rays:
    if jnp.ndim(pixel_coords_yx) != 2:
        pixel_coords_yx = pixel_coords_yx.reshape((1, 2))
    pixel_count = pixel_coords_yx.shape[0]
    directions = jnp.stack([(pixel_coords_yx[:, 1] - cam_params.width * 0.5) / cam_params.focal,
                            -(pixel_coords_yx[:, 0] - cam_params.height * 0.5) / cam_params.focal,
                            -jnp.ones(pixel_count)], axis=-1)
    origins = jnp.zeros_like(directions)
    viewdirs = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    return utils.Rays(directions=directions, viewdirs=viewdirs, origins=origins)


# noinspection PyUnresolvedReferences
def transform_rays(rays: utils.Rays, T: jaxlie.SE3) -> utils.Rays:
    R, t = T.rotation().as_matrix(), T.translation()
    directions = rays.directions @ R.T
    origins = rays.origins @ R.T + t[None, :]
    viewdirs = rays.viewdirs @ R.T
    return utils.Rays(directions=directions, viewdirs=viewdirs, origins=origins)


def relative_transformations(T_cam2worlds: List[jaxlie.SE3]) -> Tuple[List[jaxlie.SE3], jaxlie.SE3]:
    assert len(T_cam2worlds) > 0
    T_cam2lasts = []
    T_true = T_cam2worlds[-1]
    T_true_inv = T_true.inverse()
    for T_cam2world in T_cam2worlds:
        T_cam2last = T_true_inv @ T_cam2world
        assert jnp.allclose(T_cam2world @ jnp.array([1, 2, 3]),
                            T_true @ T_cam2last @ jnp.array([1, 2, 3]), 1e-4)
        T_cam2lasts.append(T_cam2last)
    return T_cam2lasts, T_true


def twist_transformation(T_true: jaxlie.SE3, rng):
    """Applies a random noise on the given transofrmation."""
    rng, subkey = jax.random.split(rng, 2)

    rotation_error = jaxlie.SO3.sample_uniform(rng).log()
    rotation_error = rotation_error / jnp.linalg.norm(rotation_error)
    rotation_error = rotation_error * base.FLAGS.perturbation_R

    translation_error = jax.random.uniform(key=subkey,
                                           shape=(3,),
                                           minval=-1.0,
                                           maxval=1.0)
    translation_error = translation_error / jnp.linalg.norm(translation_error)
    translation_error = translation_error * base.FLAGS.perturbation_t

    error = jaxlie.SE3.from_rotation_and_translation(rotation=jaxlie.SO3.exp(rotation_error),
                                                     translation=translation_error)
    T_init = error @ T_true
    return T_init


def total_pixel_coords_xy(width: int, height: int) -> jnp.ndarray:
    pixel_center = 0.5 if base.FLAGS.use_pixel_centers else 0.0
    xx, yy = jnp.meshgrid(jnp.arange(height, dtype=jnp.float32) + pixel_center,
                          jnp.arange(width, dtype=jnp.float32) + pixel_center,
                          indexing="ij")
    pixel_coords_yx = jnp.column_stack([xx.flatten(), yy.flatten()])
    return pixel_coords_yx


def compute_errors(T_true: jaxlie.SE3, T_pred: jaxlie.SE3):
    T_diff = T_pred.inverse() @ T_true
    translation_error = jnp.linalg.norm(T_diff.translation()).tolist()
    rotation_error = jnp.linalg.norm(T_diff.rotation().log()).tolist()
    return translation_error, rotation_error


def convert_to_ndc(rays: utils.Rays, cam_params: base.CameraParameters) -> utils.Rays:
    """Convert a set of rays to NDC coordinates."""
    origins = rays.origins
    directions = rays.directions
    focal = cam_params.focal
    w = cam_params.width
    h = cam_params.height
    near = 1.
    # Shift ray origins to near plane
    t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = tuple(jnp.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(jnp.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = jnp.stack([o0, o1, o2], -1)
    directions = jnp.stack([d0, d1, d2], -1)
    return utils.Rays(directions=directions, viewdirs=rays.viewdirs, origins=origins)
