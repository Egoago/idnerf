from typing import Tuple, List

import jax
import jaxlie
from jax import numpy as jnp

from absl.flags import FLAGS

from idnerf import base


def __coords2directions(pixel_coords_yx: jnp.ndarray, cam_params: base.CameraParameters) -> jnp.ndarray:
    if jnp.ndim(pixel_coords_yx) != 2:
        pixel_coords_yx = pixel_coords_yx.reshape((1, 2))
    pixel_count = pixel_coords_yx.shape[0]
    directions = jnp.stack([(pixel_coords_yx[:, 1] - cam_params.width * 0.5) / cam_params.focal,
                            -(pixel_coords_yx[:, 0] - cam_params.height * 0.5) / cam_params.focal,
                            -jnp.ones(pixel_count)], axis=-1)
    return directions


def __directions2rays(directions: jnp.ndarray, T: jaxlie.SE3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    R, t = T.rotation().as_matrix(), T.translation()
    directions = directions @ R.T
    origins = jnp.tile(t, (directions.shape[0], 1))
    viewdirs = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    return origins, directions, viewdirs


def coords2rays(pixel_coords_yx: jnp.ndarray, cam_params: base.CameraParameters, T: jaxlie.SE3):
    directions = __coords2directions(pixel_coords_yx, cam_params)
    return __directions2rays(directions, T)


def calculate_T_rels(Ts: List[jaxlie.SE3]) -> Tuple[List[jaxlie.SE3], jaxlie.SE3]:
    assert len(Ts) > 0
    T_rels = []
    T_true = Ts[-1]
    T_rels.append(jaxlie.SE3.identity())
    T_true_inv = T_true.inverse()
    for T in reversed(Ts[:-1]):
        T_rel = T @ T_true_inv
        assert jnp.allclose(T.apply(jnp.array([1, 2, 3])),
                            T_rel.apply(T_true.apply(jnp.array([1, 2, 3]))))
        T_rels.append(T_rel)
    T_rels.reverse()
    return T_rels, T_true


def update_pose(T_base: jaxlie.SE3, T_rel: jaxlie.SE3, epsilon: jnp.ndarray) -> jaxlie.SE3:
    T_base = jaxlie.manifold.rplus(T_base, epsilon)
    T = T_rel @ T_base
    return T


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
