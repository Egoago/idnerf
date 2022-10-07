import functools

import optax
from absl import app, flags, logging
import jax
from jax import numpy as jnp
import jaxlie

import idnerf
import idnerf.math
from jaxnerf.nerf import utils
from tqdm import tqdm

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


def get_optimizer(scheduler):
    optimizer = flags.FLAGS.optimizer
    if optimizer == "sgd":
        return optax.sgd(scheduler, 0.4, False)
    elif optimizer == "adam":
        return optax.adam(scheduler)
    else:
        raise NotImplementedError()


def fit(data: idnerf.Data, render_fn, rng):
    params = {'epsilon': jaxlie.manifold.zero_tangents(data.T_init)}
    scheduler = optax.exponential_decay(FLAGS.lr_init, FLAGS.decay_steps, FLAGS.decay_rate)
    optimizer = get_optimizer(scheduler)
    opt_state = optimizer.init(params)

    history = idnerf.History()

    def _generate_rays(epsilon, pixel_coords_yx, T_rel):
        T = idnerf.update_pose(data.T_init, jaxlie.SE3.from_matrix(T_rel), epsilon)
        return idnerf.coords2rays(pixel_coords_yx, data.cam_params, T)

    generate_rays = jax.vmap(functools.partial(_generate_rays), in_axes=(None, 0, 0))

    def loss(_params, pixel_coords_yx: jnp.ndarray, rgbd_pixels: jnp.ndarray, T_rels: jnp.ndarray, rng):
        origins, directions, viewdirs = generate_rays(_params['epsilon'], pixel_coords_yx, T_rels)
        rays = utils.Rays(origins=origins.reshape(-1, 3),
                          directions=directions.reshape(-1, 3),
                          viewdirs=viewdirs.reshape(-1, 3))

        rgb, depth = idnerf.render_rays(render_fn, rays, rng, flags.FLAGS.distributed_render)
        mask = (depth > flags.FLAGS.near) * (depth < flags.FLAGS.far)
        #   TODO add depth
        loss_rgb = (optax.huber_loss(rgb, rgbd_pixels[:, :3], FLAGS.huber_delta) * mask[:, None]).mean()
        return loss_rgb

    loss_grad = jax.jit(jax.value_and_grad(loss))

    pixel_coords = jnp.array([frame.pixel_coords_yx for frame in data.frames])
    rgbd_pixels = jnp.array([frame.rgbdm_img[frame.pixel_coords_yx[:, 0], frame.pixel_coords_yx[:, 1]]
                             for frame in data.frames]).reshape(-1, 5)
    T_rels = jnp.array([frame.T_rel.as_matrix() for frame in data.frames])

    @jax.jit
    def step(_params, _opt_state, _rng):
        loss, grads = loss_grad(_params, pixel_coords, rgbd_pixels, T_rels, rng)
        if flags.FLAGS.clip_grad > 0:
            grads['epsilon'] = jnp.clip(grads, -flags.FLAGS.clip_grad, flags.FLAGS.clip_grad)
        updates, _opt_state = optimizer.update(grads, _opt_state, _params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, loss, grads

    print(f'|{"step":^5s}|{"loss":^7s}|{"t error":^7s}|{"R error":^7s}|{"grads":^49s}|')
    for i in tqdm(range(FLAGS.max_steps)):
        rng, key = jax.random.split(rng, 2)
        params, opt_state, loss, grads = step(params, opt_state, key)

        T_pred = jaxlie.manifold.rplus(data.T_init, params['epsilon'])
        t_error, R_error = idnerf.math.compute_errors(data.T_true, T_pred)

        history.loss.append(loss.tolist())
        history.grads.append(grads["epsilon"].tolist())
        history.epsilon.append(params["epsilon"].tolist())
        history.t_error.append(t_error)
        history.R_error.append(R_error)
        if i % FLAGS.print_every == 0:
            tqdm.write(f'|{i:5d}|{loss:7.4f}|{t_error:7.4f}|{R_error:7.4f}|{grads["epsilon"]}|')
    data.T_final = jaxlie.manifold.rplus(data.T_init, params['epsilon'])
    data.history = history


def main(_):
    rng = init()
    rng_keys = jax.random.split(rng, 5)

    data = idnerf.load_data(rng_keys[0])
    idnerf.sample_imgs(data, rng_keys[1])

    render_fn = idnerf.load_model(rng_keys[2])

    if flags.FLAGS.per_sample_gradient:
        raise NotImplementedError()  # TODO implement
        # history = fit2(T_init, rgbdm_img, renderer, rng, T_true)
    else:
        fit(data, render_fn, rng_keys[3])

    idnerf.save(data, render_fn, rng_keys[4], "gd")


if __name__ == "__main__":
    app.run(main)
