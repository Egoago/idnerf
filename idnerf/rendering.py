from typing import Tuple

import jax
import jaxlie
from jax import numpy as jnp

from idnerf import base, math
from jaxnerf.nerf import utils


def render_img(render_fun, T: jaxlie.SE3, cam_params: base.CameraParameters, rng) -> Tuple[jnp.ndarray, jnp.ndarray]:
    pixel_coords_yx = math.total_pixel_coords_xy(cam_params.width, cam_params.height)
    rays = math.coords2rays(pixel_coords_yx, cam_params, T)
    rgb, depth = render_rays(render_fun, rays, rng)
    rgb = rgb.reshape((cam_params.height, cam_params.width, 3))
    depth = depth.reshape((cam_params.height, cam_params.width))
    return rgb, depth


def render_rays(render_fun, rays: utils.Rays, rng, parallel=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_0, key_1 = jax.random.split(rng, 2)
    if parallel:
        host_id = jax.host_id()
        chunk_size = rays[0].shape[0]
        rays_remaining = chunk_size % jax.device_count()
        if rays_remaining != 0:
            padding = jax.device_count() - rays_remaining
            rays = utils.namedtuple_map(lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), rays)
        else:
            padding = 0
        rays_per_host = rays[0].shape[0] // jax.host_count()
        start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
        rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]), rays)
        chunk_results = render_fun(key_0, key_1, rays)[-1]
        rgb, disp, acc = [utils.unshard(x[0], padding) for x in chunk_results]
    else:
        rgb, disp, acc = render_fun(key_0, key_1, rays)[-1]
    depth = acc / disp
    return rgb, depth
