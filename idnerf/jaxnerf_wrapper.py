import functools
import logging
from os import path

import flax
import jax
from absl import flags

import jax.numpy as jnp
from flax.training import checkpoints

from jaxnerf.nerf import utils, models


def load_model(rng, ray_count=None):
    if flags.FLAGS.pixel_sampling == "total":
        assert ray_count is not None
    elif ray_count is None:
        ray_count = flags.FLAGS.pixel_count
    input_shape = (1, min(ray_count, flags.FLAGS.chunk), 3)
    sample_rays = utils.Rays(origins=jnp.zeros(input_shape),
                             directions=jnp.zeros(input_shape),
                             viewdirs=jnp.zeros(input_shape))
    sample_rays = {"rays": utils.to_device(sample_rays)}
    model, variables = models.get_model(rng, sample_rays, flags.FLAGS)
    del variables
    checkpoint_path = path.join(flags.FLAGS.train_dir, flags.FLAGS.subset)
    variables = checkpoints.restore_checkpoint(checkpoint_path, None)['optimizer']['target']
    variables = jax.device_put(flax.core.freeze(variables))

    def _render_fn(_variables, key_0, key_1, rays):
        return model.apply(_variables, key_0, key_1, rays, flags.FLAGS.randomized)

    render_fn = _render_fn
    if flags.FLAGS.distributed_render:
        render_fn = jax.pmap(
            lambda _variables, key_0, key_1, rays: jax.lax.all_gather(_render_fn(_variables, key_0, key_1, rays),
                                                                      axis_name="batch"),
            in_axes=(None, None, None, 0),
            donate_argnums=3,
            axis_name="batch")
    render_fn = functools.partial(render_fn, variables)
    render_fn = jax.jit(render_fn)
    logging.info("Loading model finished")
    return render_fn
