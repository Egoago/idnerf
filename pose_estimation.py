import functools
from os import path

import flax
import optax
from absl import app, flags, logging
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import jaxlie

from idnerf.dataset import load_dataset, get_frame
from idnerf.log import save_history, compute_errors
from idnerf.pixel_sampling import sample_pixels, generate_ray
from idnerf.renderer import Renderer
from jaxnerf.nerf import models, utils
from tqdm import tqdm

FLAGS = flags.FLAGS
utils.define_flags()


def init():
    import warnings
    warnings.filterwarnings("ignore")

    jnp.set_printoptions(precision=4)

    flags.DEFINE_string("result_dir", "results/", "")
    flags.DEFINE_enum("pixel_sampling", "total", ["total", "random", "patch", "fast", "random_no_white"], "")
    flags.DEFINE_enum("subset", "lego", ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], "")
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


def load_model(rng, dataset):
    if FLAGS.pixel_sampling == "total":
        ray_count = dataset['img_shape'][0] * dataset['img_shape'][1]
    else:
        ray_count = FLAGS.pixel_count
    input_shape = (1, min(ray_count, FLAGS.chunk), 3)
    sample_rays = utils.Rays(origins=jnp.zeros(input_shape),
                             directions=jnp.zeros(input_shape),
                             viewdirs=jnp.zeros(input_shape))
    sample_rays = {"rays": utils.to_device(sample_rays)}
    model, variables = models.get_model(rng, sample_rays, FLAGS)
    del variables
    variables = checkpoints.restore_checkpoint(path.join(FLAGS.train_dir, FLAGS.subset), None)['optimizer']['target']
    variables = jax.device_put(flax.core.freeze(variables))

    def _render_fn(variables, key_0, key_1, rays):
        return model.apply(variables, key_0, key_1, rays, FLAGS.randomized)

    render_fn = _render_fn
    if FLAGS.distributed_render:
        render_fn = jax.pmap(
            lambda variables, key_0, key_1, rays: jax.lax.all_gather(_render_fn(variables, key_0, key_1, rays),
                                                                     axis_name="batch"),
            in_axes=(None, None, None, 0),
            donate_argnums=3,
            axis_name="batch")
    render_fn = functools.partial(render_fn, variables)
    render_fn = jax.jit(render_fn)
    return render_fn


def twist_transformation(T_matrix, rng):
    rng, key0, key1 = jax.random.split(rng, 3)
    T_true = jaxlie.SE3.from_matrix(T_matrix)

    rotation_error = jaxlie.SO3.sample_uniform(key0).log() * FLAGS.perturbation_R
    translation_error = jax.random.uniform(key=key1,
                                           shape=(3,),
                                           minval=-1.0,
                                           maxval=1.0) * FLAGS.perturbation_t
    error = jaxlie.SE3.from_rotation_and_translation(rotation=jaxlie.SO3.exp(rotation_error),
                                                     translation=translation_error)
    T_init = error @ T_true
    return T_true, T_init


def fit2(T_init, rgbdm_img, renderer: Renderer, rng, T_true):
    params = {'epsilon': jaxlie.manifold.zero_tangents(T_init)}
    scheduler = optax.exponential_decay(FLAGS.lr_init, FLAGS.decay_steps, FLAGS.decay_rate)
    optimizer = optax.adam(scheduler)
    opt_state = optimizer.init(params)
    rgbd_pixels, pixel_coords = sample_pixels(rgbdm_img, rng)

    history = {'loss': [],
               't error': [],
               'R error': [],
               'grad': [],
               'epsilon': [],
               'T_init': T_init,
               'T_true': T_true,
               'pixel_coords': pixel_coords.tolist()}

    def compute_loss2(params,
                      rgbd_pixel,
                      pixel_coord,
                      rng):
        T = jaxlie.manifold.rplus(T_init, params['epsilon']).as_matrix()
        ray = generate_ray(pixel_coord, renderer.img_shape, renderer.focal, T)
        rgb, depth = renderer.render_ray(ray, rng)
        mask = depth > 0.001
        loss_rgb = (optax.huber_loss(rgb, rgbd_pixel[:3], FLAGS.huber_delta) * mask).mean()
        # loss_disp = optax.huber_loss(disp, rgbd_pixel[-1], FLAGS.huber_delta) * mask[:, None]
        return loss_rgb  # + loss_disp

    loss_per_sample = jax.jit(jax.vmap(jax.value_and_grad(compute_loss2), in_axes=(None, 0, 0, 0)))

    @jax.jit
    def step(params, opt_state, rng):
        keys = jax.random.split(rng, rgbd_pixels.shape[0])
        loss = []
        grads = []
        for i in range(0, rgbd_pixels.shape[0], flags.FLAGS.chunk):
            chunk_loss, chunk_grads = loss_per_sample(params,
                                                      rgbd_pixels[i:i+flags.FLAGS.chunk],
                                                      pixel_coords[i:i+flags.FLAGS.chunk],
                                                      keys[i:i+flags.FLAGS.chunk])
            loss.append(chunk_loss)
            grads.append(chunk_grads['epsilon'])
        grads = jnp.concatenate(grads)
        grads = grads.mean(axis=0)
        loss = jnp.concatenate(loss)
        loss = loss.mean()
        updates, opt_state = optimizer.update({'epsilon': grads}, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    print(f'|{"step":^5s}|{"loss":^7s}|{"grads":^49s}|{"t error":^7s}|{"R error":^7s}|')
    for i in tqdm(range(FLAGS.max_steps)):
        rng, key = jax.random.split(rng, 2)
        params, opt_state, loss, grads = step(params, opt_state, key)

        history['loss'].append(loss.tolist())
        history['grad'].append(grads.tolist())
        history['epsilon'].append(params['epsilon'].tolist())
        t_error, R_error = compute_errors(T_true, jaxlie.manifold.rplus(T_init, params['epsilon']))
        history['t error'].append(t_error)
        history['R error'].append(R_error)
        if i % FLAGS.print_every == 0:
            tqdm.write(f'|{i:5d}|{loss:7.4f}|{grads}|{t_error:7.4f}|{R_error:7.4f}|')

    history['T_final'] = jaxlie.manifold.rplus(T_init, params['epsilon'])
    return history

def get_loss_fn():
    if flags.FLAGS.loss == "huber":
        return lambda y, x: optax.huber_loss(y, x, flags.FLAGS.huber_delta)
    elif flags.FLAGS.loss == "mse":
        return lambda y, x: (y-x)**2
    else:
        raise NotImplementedError()


def fit(T_init, rgbdm_img, renderer: Renderer, rng, T_true):
    params = {'epsilon': jaxlie.manifold.zero_tangents(T_init)}
    scheduler = optax.exponential_decay(FLAGS.lr_init, FLAGS.decay_steps, FLAGS.decay_rate)
    optimizer = optax.adam(scheduler)
    opt_state = optimizer.init(params)
    rgbd_pixels, pixel_coords = renderer.resample_pixels(rgbdm_img, rng)

    history = {'loss': [],
               't error': [],
               'R error': [],
               'grad': [],
               'epsilon': [],
               'T_init': T_init,
               'T_true': T_true,
               'pixel_coords': pixel_coords.tolist()}

    def compute_loss(params,
                     rgbd_pixels,
                     rng):
        T = jaxlie.manifold.rplus(T_init, params['epsilon']).as_matrix()
        rgb, depth = renderer.render(T, rng)
        mask = (depth > flags.FLAGS.near) * (depth < flags.FLAGS.far)
        loss_rgb = (optax.huber_loss(rgb, rgbd_pixels[:, :3], FLAGS.huber_delta) * mask[:, None]).mean()
        #loss_disp = (optax.huber_loss(depth, 1./rgbd_pixels[:, -1]*2., FLAGS.huber_delta) * mask[:, None]).mean()
        return loss_rgb# + loss_disp

    @jax.jit
    def step(params, opt_state, rng):
        loss, grads = jax.value_and_grad(compute_loss)(params,
                                                       rgbd_pixels,
                                                       rng)
        if flags.FLAGS.clip_grad != 0:
            grads['epsilon'] = jnp.clip(grads['epsilon'], -flags.FLAGS.clip_grad, flags.FLAGS.clip_grad)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    print(f'|{"step":^5s}|{"loss":^7s}|{"grads":^49s}|{"t error":^7s}|{"R error":^7s}|')
    for i in tqdm(range(FLAGS.max_steps)):
        rng, key = jax.random.split(rng, 2)
        params, opt_state, loss, grads = step(params, opt_state, key)

        history['loss'].append(loss.tolist())
        history['grad'].append(grads["epsilon"].tolist())
        history['epsilon'].append(params["epsilon"].tolist())
        t_error, R_error = compute_errors(T_true, jaxlie.manifold.rplus(T_init, params['epsilon']))
        history['t error'].append(t_error)
        history['R error'].append(R_error)
        if i % FLAGS.print_every == 0:
            tqdm.write(f'|{i:5d}|{loss:7.4f}|{grads["epsilon"]}|{t_error:7.4f}|{R_error:7.4f}|')

    history['T_final'] = jaxlie.manifold.rplus(T_init, params['epsilon'])
    return history


def main(_):
    rng = init()
    rng, key_0, key_1 = jax.random.split(rng, 3)
    dataset = load_dataset()
    logging.info("Loading dataset finished")
    render_fn = load_model(key_0, dataset)
    renderer = Renderer(render_fn, dataset['img_shape'], dataset['focal'])
    logging.info("Loading model finished")

    rgbdm_img, T_matrix = get_frame(dataset, flags.FLAGS.frame_idx)
    T_true, T_init = twist_transformation(T_matrix, key_1)
    if flags.FLAGS.per_sample_gradient:
        history = fit2(T_init, rgbdm_img, renderer, rng, T_true)
    else:
        history = fit(T_init, rgbdm_img, renderer, rng, T_true)
    save_history(history, renderer, rng, rgbdm_img, "gl")


if __name__ == "__main__":
    app.run(main)
