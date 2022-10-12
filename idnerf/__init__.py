from .base import Frame, History, CameraParameters, Data
from .dataset import load_data
from .log import save
from .jaxnerf_wrapper import load_model
from .optimization import fit
from .rendering import render_img

from jaxnerf.nerf import utils
utils.define_flags()


def init():
    import warnings
    import jax
    import jax.numpy as jnp
    from absl import flags

    warnings.filterwarnings("ignore")

    jnp.set_printoptions(precision=4)

    flags.DEFINE_string("result_dir", "results/", "")
    flags.DEFINE_string("test_name", None, "")
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
    flags.DEFINE_bool("resample_rays", True, "")
    flags.DEFINE_bool("coarse_opt", False, "")
    flags.DEFINE_list("frame_ids", [0], "")

    if flags.FLAGS.config is not None:
        from jaxnerf.nerf import utils
        utils.update_flags(flags.FLAGS)
    if flags.FLAGS.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if flags.FLAGS.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")
    if flags.FLAGS.result_dir is None:
        raise ValueError("data_dir must be set. None set now.")

    rng = jax.random.PRNGKey(20221012)
    return rng
