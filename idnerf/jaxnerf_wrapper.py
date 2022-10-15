import functools
import logging
from os import path

import flax
import jax

import jax.numpy as jnp
from flax.training import checkpoints

from idnerf import base
from jaxnerf.nerf import utils, models


def load_model(rng, ray_count=None):
    if base.FLAGS.pixel_sampling == "total":
        assert ray_count is not None
    elif ray_count is None:
        ray_count = base.FLAGS.pixel_count
    input_shape = (1, min(ray_count, base.FLAGS.chunk), 3)
    sample_rays = utils.Rays(origins=jnp.zeros(input_shape),
                             directions=jnp.zeros(input_shape),
                             viewdirs=jnp.zeros(input_shape))
    sample_rays = {"rays": utils.to_device(sample_rays)}
    model, variables = models.get_model(rng, sample_rays, base.FLAGS)
    del variables
    checkpoint_path = path.join(base.FLAGS.train_dir, base.FLAGS.subset)
    variables = checkpoints.restore_checkpoint(checkpoint_path, None)['optimizer']['target']
    variables = jax.device_put(flax.core.freeze(variables))

    def _render_fn(_variables, key_0, key_1, rays):
        return model.apply(_variables, key_0, key_1, rays, base.FLAGS.randomized)

    render_fn = _render_fn
    if base.FLAGS.distributed_render:
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
