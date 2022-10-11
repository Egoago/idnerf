from typing import Tuple, List

import jax
import jaxlie
from jax import numpy as jnp

from absl.flags import FLAGS

from idnerf import base
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
                            T_true @ T_cam2last @ jnp.array([1, 2, 3]))
        T_cam2lasts.append(T_cam2last)
    return T_cam2lasts, T_true


def twist_transformation(T_true: jaxlie.SE3, rng):
    rng, subkey = jax.random.split(rng, 2)

    rotation_error = jaxlie.SO3.sample_uniform(rng).log() * FLAGS.perturbation_R
    translation_error = jax.random.uniform(key=subkey,
                                           shape=(3,),
                                           minval=-1.0,
                                           maxval=1.0) * FLAGS.perturbation_t
    error = jaxlie.SE3.from_rotation_and_translation(rotation=jaxlie.SO3.exp(rotation_error),
                                                     translation=translation_error)
    T_init = error @ T_true
    return T_init


def total_pixel_coords_xy(width: int, height: int) -> jnp.ndarray:
    pixel_center = 0.5 if FLAGS.use_pixel_centers else 0.0
    xx, yy = jnp.meshgrid(jnp.arange(width, dtype=jnp.float32) + pixel_center,
                          jnp.arange(height, dtype=jnp.float32) + pixel_center,
                          indexing="ij")
    pixel_coords_yx = jnp.column_stack([xx.flatten(), yy.flatten()])
    return pixel_coords_yx


def compute_errors(T_true: jaxlie.SE3, T_pred: jaxlie.SE3):
    t_gt, t_pred = T_true.translation(), T_pred.translation()
    R_gt, R_pred = T_true.rotation(), T_pred.rotation()
    translation_error = jnp.linalg.norm(t_gt - t_pred).tolist()
    rotation_error = jnp.linalg.norm((R_gt @ R_pred.inverse()).log()).tolist()
    return translation_error, rotation_error
