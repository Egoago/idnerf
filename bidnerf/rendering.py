from typing import Tuple

import jax
import jaxlie
from jax import numpy as jnp
from tqdm import tqdm

from bidnerf import base, math
from jaxnerf.nerf import utils


def render_img(render_fun, T: jaxlie.SE3, cam_params: base.CameraParameters, rng, coarse=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Render an image from the given pose."""
    pixel_coords_yx = math.total_pixel_coords_xy(cam_params.width, cam_params.height)
    rays_cam = math.coords_to_local_rays(pixel_coords_yx, cam_params)
    rays_world = math.transform_rays(rays_cam, T)
    if base.FLAGS.dataset == 'llff2':
        rays_world = math.convert_to_ndc(rays_world, cam_params)
    rgb, depth = render_rays(render_fun, rays_world, rng, coarse)
    rgb = rgb.reshape((cam_params.height, cam_params.width, 3))
    depth = depth.reshape((cam_params.height, cam_params.width))
    return rgb, depth


def render_rays(render_fun, rays: utils.Rays, rng, coarse=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rgb_list, depth_list = [], []
    for i in tqdm(range(0, rays.directions.shape[0], base.FLAGS.chunk), desc='Rendering', unit='chunk', leave=False):
        chunk_rays = utils.namedtuple_map(lambda r: r[i:i + base.FLAGS.chunk], rays)
        rgb, depth = __render_chunk_rays(render_fun, chunk_rays, rng, coarse)
        rgb_list.append(rgb)
        depth_list.append(depth)
    return jnp.concatenate(rgb_list), jnp.concatenate(depth_list)


def __render_chunk_rays(render_fun, rays: utils.Rays, rng, coarse):
    key_0, key_1 = jax.random.split(rng, 2)
    if base.FLAGS.distributed_render:
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
        chunk_results = render_fun(key_0, key_1, rays)
        chunk_results = chunk_results[0] if coarse else chunk_results[1]
        chunk_results = [utils.unshard(x[0], padding) for x in chunk_results]
    else:
        chunk_results = render_fun(key_0, key_1, rays)
        chunk_results = chunk_results[0] if coarse else chunk_results[1]
    return chunk_results
