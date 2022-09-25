import jax
from absl import flags
from jax import numpy as jnp

from idnerf import pixel_sampling
from idnerf.pixel_sampling import sample_pixels
from jaxnerf.nerf import utils

class Renderer:
    def __init__(self, rng, render_fun, img_shape, focal):
        if flags.FLAGS.distributed_render:
            render_fun = jax.pmap(
                lambda variables, key_0, key_1, rays: jax.lax.all_gather(render_fun(key_0, key_1, rays),
                                                                         axis_name="batch"),
                in_axes=(None, None, 0),
                donate_argnums=2,
                axis_name="batch")
        self.render_fun = render_fun
        self.img_shape = img_shape
        self.focal = focal
        self.rng = rng
        self.sample_pixels = None
        self.sample_pixel_coords = None

    def render(self, T):
        assert self.sample_pixel_coords is not None, "renderer.resample_pixels() must be called before render"
        rays = pixel_sampling.generate_rays(self.sample_pixel_coords, self.img_shape, self.focal, T)
        return self.render_rays(rays)

    def render_img(self, T):
        pixel_coords = pixel_sampling.generate_pixel_coords(self.img_shape)
        rays = pixel_sampling.generate_rays(pixel_coords, self.img_shape, self.focal, T)
        return self.render_rays(rays)

    def render_rays(self, rays: utils.Rays):
        self.rng, key_0, key_1 = jax.random.split(self.rng, 3)
        results = []
        for i in range(0, rays[0].shape[0], flags.FLAGS.chunk):
            chunk_rays = utils.namedtuple_map(lambda r: r[i:i + flags.FLAGS.chunk], rays)
            if flags.FLAGS.distributed_render:
                result = self.render_chunk_parallel(chunk_rays)
            else:
                result = self.render_fun(key_0, key_1, chunk_rays)[-1]
            results.append(result)
        rgb, disp, acc = [jnp.concatenate(r, axis=0) for r in zip(*results)]
        return rgb, disp

    def resample_pixels(self, rgbd_img, method=None):
        self.rng, key_0 = jax.random.split(self.rng, 2)
        self.sample_pixels, self.sample_pixel_coords = sample_pixels(rgbd_img, key_0, method)
        return self.sample_pixels, self.sample_pixel_coords

    def render_chunk_parallel(self, chunk_rays):
        self.rng, key_0, key_1 = jax.random.split(self.rng, 3)
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
        chunk_results = self.render_fun(key_0, key_1, chunk_rays)[-1]
        return [utils.unshard(x[0], padding) for x in chunk_results]
