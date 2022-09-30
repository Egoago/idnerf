import jax
from jax import numpy as jnp

from idnerf.sampling import img_pixel_coords, generate_rays
from jaxnerf.nerf import utils


def render_img(render_fun, T, width, height, focal, rng):
    pixel_coords = img_pixel_coords(width, height)
    rays = generate_rays(pixel_coords, T, width, height, focal)
    rgb, depth = render_rays(render_fun, rays, rng)
    rgb = rgb.reshape((height, width, 3))
    depth = depth.reshape((height, width))
    return rgb, depth


def render_rays(render_fun, rays: utils.Rays, rng):
    key_0, key_1 = jax.random.split(rng, 2)
    rgb, disp, acc = render_fun(key_0, key_1, rays)[-1]
    depth = acc / disp
    return rgb, depth


def render_chunk_parallel(render_fun, chunk_rays: utils.Rays, rng):
    key_0, key_1 = jax.random.split(rng, 2)
    host_id = jax.host_id()
    chunk_size = chunk_rays[0].shape[0]
    rays_remaining = chunk_size % jax.device_count()
    if rays_remaining != 0:
        padding = jax.device_count() - rays_remaining
        chunk_rays = utils.namedtuple_map(lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), chunk_rays)
    else:
        padding = 0
    rays_per_host = chunk_rays[0].shape[0] // jax.host_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]), chunk_rays)
    chunk_results = render_fun(key_0, key_1, chunk_rays)[-1]
    return [utils.unshard(x[0], padding) for x in chunk_results]
