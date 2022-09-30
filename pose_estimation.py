import functools
from os import path

import flax
import optax
from absl import app, flags, logging
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import jaxlie
from jax.experimental import host_callback as hcb

from idnerf.dataset import load_dataset
from idnerf.log import save_history, compute_errors
from idnerf.rendering import render_rays
from idnerf.sampling import sample_pixels, generate_rays
from jaxnerf.nerf import models, utils
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


def load_model(rng, ray_count=None):
    if FLAGS.pixel_sampling == "total":
        assert ray_count is not None
    elif ray_count is None:
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


def twist_transformation(T_true, rng):
    rng, subkey = jax.random.split(rng, 2)

    rotation_error = jaxlie.SO3.sample_uniform(rng).log() * FLAGS.perturbation_R
    translation_error = jax.random.uniform(key=subkey,
                                           shape=(3,),
                                           minval=-1.0,
                                           maxval=1.0) * FLAGS.perturbation_t
    error = jaxlie.SE3.from_rotation_and_translation(rotation=jaxlie.SO3.exp(rotation_error),
                                                     translation=translation_error)
    T_init = error @ T_true
    return T_init


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


def get_optimizer(scheduler):
    optimizer = flags.FLAGS.optimizer
    if optimizer == "sgd":
        return optax.sgd(scheduler, 0.4, False)
    elif optimizer == "adam":
        return optax.adam(scheduler)
    else:
        raise NotImplementedError()


def load_samples(rgbdm_imgs, T_init, rng, width, height, focal, T_rels=None):
    img_count = len(rgbdm_imgs)
    if img_count > 1:
        assert T_rels is not None
        assert img_count - 1 == len(T_rels)
    pixel_count = flags.FLAGS.pixel_count
    pixels_per_img = pixel_count // img_count
    rays = []
    rgbd_pixels = []
    T_img = T_init
    for img_idx in reversed(range(len(rgbdm_imgs))):
        rng, subkey = jax.random.split(rng, 2)
        _rgbd_pixels, pixel_coords = sample_pixels(rgbdm_imgs[img_idx], subkey, pixel_count=pixels_per_img)
        _rays = generate_rays(pixel_coords, T_img, width, height, focal)
        T_img = T_rels[img_idx-1] @ T_img
        rgbd_pixels.append(_rgbd_pixels)
        rays.append(_rays)
    rgbd_pixels = jnp.array(rgbd_pixels[:pixel_count])
    rays = jnp.array(rays[:pixel_count])
    return rgbd_pixels, rays


def fit(T_init, rgbd_pixels, rays, render_fun, rng, T_true):
    params = {'epsilon': jaxlie.manifold.zero_tangents(T_init)}
    scheduler = optax.exponential_decay(FLAGS.lr_init, FLAGS.decay_steps, FLAGS.decay_rate)
    optimizer = get_optimizer(scheduler)
    opt_state = optimizer.init(params)

    history = {'loss': [],
               't_error': [],
               'R_error': [],
               'grad': [],
               'epsilon': [],
               'sample_count': [],
               'T_init': T_init,
               'T_true': T_true}

    def compute_loss(params, rgbd_chunk, chunk_rays, rng):
        T = jaxlie.manifold.rplus(T_init, params['epsilon']).as_matrix()

        chunk_rays = utils.Rays(origins=chunk_rays[..., 0],
                                directions=chunk_rays[..., 1],
                                viewdirs=chunk_rays[..., 2])

        rgb, depth = render_rays(render_fun, chunk_rays, rng)
        mask = (depth > flags.FLAGS.near) * (depth < flags.FLAGS.far)
        sample_count = jnp.count_nonzero(mask)
        hcb.id_tap(lambda arg, transform: history['sample_count'].append(int(arg)), sample_count)
        loss_rgb = (optax.huber_loss(rgb, rgbd_chunk[:, :3], FLAGS.huber_delta) * mask[:, None]).mean()
        #loss_disp = (optax.huber_loss(depth, 1./rgbd_chunk[:, -1]*2., FLAGS.huber_delta) * mask[:, None]).mean()
        return loss_rgb# + loss_disp

    @jax.jit
    def step(_params, _opt_state, _rng):
        _loss = 0
        _grads = jnp.zeros_like(_params['epsilon'])
        for j in range(0, rays[0].shape[0], flags.FLAGS.chunk):
            rgbd_chunk = rgbd_pixels[j:j + flags.FLAGS.chunk]
            chunk_rays = rays[j:j + flags.FLAGS.chunk]

            chunk_loss, chunk_grads = jax.value_and_grad(compute_loss)(_params,
                                                           rgbd_chunk,
                                                           chunk_rays,
                                                           _rng)
            _loss += chunk_loss*rgbd_chunk.shape[0]
            _grads += chunk_grads*rgbd_chunk.shape[0]
        _loss = _loss / rays[0].shape[0]
        if flags.FLAGS.clip_grad != 0:
            grads['epsilon'] = jnp.clip(grads['epsilon'], -flags.FLAGS.clip_grad, flags.FLAGS.clip_grad)
        updates, _opt_state = optimizer.update(grads, _opt_state, _params)
        _params = optax.apply_updates(_params, updates)
        return _params, _opt_state, loss, grads

    print(f'|{"step":^5s}|{"loss":^7s}|{"t error":^7s}|{"R error":^7s}|{"Samples":^7s}|{"grads":^49s}|')
    for i in tqdm(range(FLAGS.max_steps)):
        rng, key = jax.random.split(rng, 2)
        params, opt_state, loss, grads = step(params, opt_state, key)

        history['loss'].append(loss.tolist())
        history['grad'].append(grads["epsilon"].tolist())
        history['epsilon'].append(params["epsilon"].tolist())
        t_error, R_error = compute_errors(T_true, jaxlie.manifold.rplus(T_init, params['epsilon']))
        history['t_error'].append(t_error)
        history['R_error'].append(R_error)
        if i % FLAGS.print_every == 0:
            tqdm.write(f'|{i:5d}|{loss:7.4f}|{t_error:7.4f}|{R_error:7.4f}|{history["sample_count"][-1]:7d}|{grads["epsilon"]}|')

    history['T_final'] = jaxlie.manifold.rplus(T_init, params['epsilon'])
    return history



def main(_):
    rng = init()
    rng_keys = jax.random.split(rng, 4)
    rgbdm_imgs, T_true, T_rels, width, height, focal = load_dataset()
    logging.info("Loading dataset finished")
    render_fn = load_model(rng_keys[0])
    logging.info("Loading model finished")

    T_init = twist_transformation(T_true, rng_keys[1])
    rgbd_pixels, rays = load_samples(rgbdm_imgs, T_init, rng, width, height, focal, T_rels)

    if flags.FLAGS.per_sample_gradient:
        raise NotImplementedError()
        #history = fit2(T_init, rgbdm_img, renderer, rng, T_true)
    else:
        history = fit(T_init, rgbd_pixels, rays, render_fn, rng_keys[2], T_true)
    save_history(history, rng_keys[3], rng, rgbdm_imgs[-1], "gl")


if __name__ == "__main__":
    app.run(main)
