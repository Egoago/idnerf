from typing import List, Tuple

import jax
import jax.numpy as jnp

from idnerf import base, math
from jaxnerf.nerf import utils


def __random(rgbdm_img, pixel_count, rng):
    mask = (rgbdm_img[:, :, -1]).astype(dtype=bool)
    indices = jnp.indices(mask.shape)
    indices_yx = jnp.column_stack([indices[0].flatten(), indices[1].flatten()])[mask.flatten()]
    pixel_coords_yx = jax.random.choice(rng, indices_yx, (pixel_count,), replace=False)
    return pixel_coords_yx


def __fast(rgbdm_img, pixel_count, rng):
    import numpy as np
    import cv2
    detector = cv2.FastFeatureDetector_create()  # TODO save detector object for reuse
    gray = np.uint8(rgbdm_img[:, :, :3] * 255)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    key_points = detector.detect(gray, None)
    pixel_coords_yx = np.array([key_point.pt for key_point in key_points], np.int32)[:, [1, 0]]
    weights = np.array([key_point.response for key_point in key_points], np.int32)
    mask = rgbdm_img[pixel_coords_yx[:, 0], pixel_coords_yx[:, 1], -1].astype(bool)
    pixel_coords_yx = pixel_coords_yx[mask]
    weights = weights[mask]
    probs = weights / weights.sum()
    if pixel_coords_yx.shape[0] > pixel_count:
        pixel_coords_yx = jax.random.choice(rng, pixel_coords_yx, (pixel_count,), replace=False, p=probs)
    return pixel_coords_yx


def sample_img(rgbdm_img, rng, method=None, pixel_count=None) -> jnp.ndarray:
    if method is None:
        method = base.FLAGS.pixel_sampling
    if pixel_count is None:
        pixel_count = base.FLAGS.pixel_count
    if method == 'total':
        pixel_coords_yx = math.total_pixel_coords_xy(rgbdm_img.shape[1], rgbdm_img.shape[0])
    elif method == 'random':
        pixel_coords_yx = __random(rgbdm_img, pixel_count, rng)
    elif method == 'fast':
        pixel_coords_yx = __fast(rgbdm_img, pixel_count, rng)
    elif method == 'fast_random':
        random_pixels = int(pixel_count * 0.8)
        fast_pixels = pixel_count - random_pixels
        pixel_coords_yx = jnp.concatenate([__random(rgbdm_img, random_pixels, rng),
                                           __random(rgbdm_img, fast_pixels, rng)])
    else:
        raise NotImplementedError(f"Sampling method not implemented: {method}")
    return pixel_coords_yx


class Sampler:
    def __init__(self, data: base.Data):
        self.data = data
        self.method = base.FLAGS.pixel_sampling
        self.pixels_per_img = base.FLAGS.pixel_count // len(data.frames)

    @staticmethod
    def __concatenate_rays(rays_list: List[utils.Rays]) -> utils.Rays:
        return utils.Rays(directions=jnp.concatenate([rays.directions for rays in rays_list]),
                          viewdirs=jnp.concatenate([rays.viewdirs for rays in rays_list]),
                          origins=jnp.concatenate([rays.origins for rays in rays_list]))

    def sample(self, rng) -> Tuple[utils.Rays, jnp.ndarray]:
        """Samples rays from each frame, with relative transformation to the base frame.

        Returns: rays_relative_to_base(utils.Rays): sampled rays relative to rays
                 rgbd_pixels(jnp.ndarray): rgbd pixels corresponding to the sampled rays

        """
        rays_list = []
        rgbd_pixels = []
        for frame in self.data.frames:
            rng, subkey = jax.random.split(rng, 2)
            pixel_coords_yx = sample_img(frame.rgbdm_img, subkey, pixel_count=self.pixels_per_img)
            rays_cam = math.coords_to_local_rays(pixel_coords_yx, self.data.cam_params)
            rays_relative_to_base = math.transform_rays(rays_cam, frame.T_cam2base)
            rays_list.append(rays_relative_to_base)
            rgbd_pixels.append(frame.rgbdm_img[pixel_coords_yx[:, 0], pixel_coords_yx[:, 1], :4])
        rays_relative_to_base = self.__concatenate_rays(rays_list)
        rgbd_pixels = jnp.concatenate(rgbd_pixels)
        return rays_relative_to_base, rgbd_pixels


