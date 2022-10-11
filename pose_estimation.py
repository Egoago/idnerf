
from absl import app, flags
import jax
from jax import numpy as jnp

import idnerf
from jaxnerf.nerf import utils

FLAGS = flags.FLAGS
utils.define_flags()


def init():
    import warnings
    warnings.filterwarnings("ignore")

    jnp.set_printoptions(precision=4)

    flags.DEFINE_string("result_dir", "results/", "")
    flags.DEFINE_enum("pixel_sampling", "total", ["total", "random", "patch", "fast", "fast_random"], "")
    flags.DEFINE_enum("subset", "lego", ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], "")
    flags.DEFINE_enum("optimizer", "adam", ["adam", "sgd"], "")
    flags.DEFINE_float("perturbation_R", 1.0, "", 0., 1e8)
    flags.DEFINE_float("perturbation_t", 1.0, "", 0., 1e8)
    flags.DEFINE_float("decay_rate", 0.6, "", 0., 1.)
    flags.DEFINE_float("huber_delta", 1.0, "", 0., 1e6)
    flags.DEFINE_float("clip_grad", 0., "", 0., 10.)
    flags.DEFINE_integer("pixel_count", 128, "", 1, 16384)
    flags.DEFINE_integer("decay_steps", 100, "", 1, 16384)
    flags.DEFINE_integer("frame_idx", 0, "", 0, 16384)
    flags.DEFINE_bool("distributed_render", False, "")
    flags.DEFINE_bool("per_sample_gradient", False, "")
    flags.DEFINE_bool("use_original_img", True, "")
    flags.DEFINE_list("frame_ids", [0], "")

    if FLAGS.config is not None:
        utils.update_flags(FLAGS)
    if FLAGS.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if FLAGS.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")
    if FLAGS.result_dir is None:
        raise ValueError("data_dir must be set. None set now.")

    rng = jax.random.PRNGKey(20220905)
    return rng


def main(_):
    rng = init()
    rng_keys = jax.random.split(rng, 5)

    data = idnerf.load_data(rng_keys[0])

    render_fn = idnerf.load_model(rng_keys[2])
    rgb, depth = idnerf.render_img(render_fn, data.T_init, data.cam_params, rng)
    if flags.FLAGS.per_sample_gradient:
        raise NotImplementedError()  # TODO implement
    #idnerf.fit(data, render_fn, rng_keys[3])

    #idnerf.save(data, render_fn, rng_keys[4], "gd")


if __name__ == "__main__":
    app.run(main)
