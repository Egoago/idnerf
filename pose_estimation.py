import functools
from os import path

import flax
import matplotlib.pyplot as plt
import optax
from absl import app, flags, logging
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import jaxlie
from PIL import Image

from idnerf.renderer import Renderer
from jaxnerf.nerf import models, utils
from tqdm import tqdm

FLAGS = flags.FLAGS
utils.define_flags()


def init():
    import warnings
    warnings.filterwarnings("ignore")

    jnp.set_printoptions(precision=4)

    flags.DEFINE_enum("pixel_sampling", "total", ["total", "feature_points", "random", "patch"], "")
    flags.DEFINE_enum("subset", "lego", ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], "")
    flags.DEFINE_float("perturbation_R", 1.0, "", 0., 1e8)
    flags.DEFINE_float("perturbation_t", 1.0, "", 0., 1e8)
    flags.DEFINE_float("decay_rate", 0.6, "", 0., 1.)
    flags.DEFINE_float("huber_delta", 1.0, "", 0., 1e6)
    flags.DEFINE_integer("pixel_count", 128, "", 1, 16384)
    flags.DEFINE_integer("decay_steps", 100, "", 1, 16384)
    flags.DEFINE_bool("distributed_render", False, "")

    if FLAGS.config is not None:
        utils.update_flags(FLAGS)
    if FLAGS.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if FLAGS.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")

    rng = jax.random.PRNGKey(20220905)
    return rng


def load_img(name):
    img_path = path.join(FLAGS.train_dir, FLAGS.subset, "test_preds", name)
    img = Image.open(img_path)
    return jnp.asarray(img, jnp.float32) / 255.


def load_rgbd_img(idx):
    name = f'{idx:03}.png'
    rgb_img = load_img(name)
    disp_img = load_img("disp_" + name)
    rgbd_img = jnp.concatenate([rgb_img, disp_img[..., None]], axis=-1)
    return rgbd_img


def load_dataset():
    import json
    data_path = path.join(FLAGS.data_dir, FLAGS.subset, "transforms_test.json")
    with utils.open_file(data_path, "r") as fp:
        dataset = json.load(fp)
    camera_angle_x = float(dataset["camera_angle_x"])
    dataset['img_shape'] = load_img('000.png').shape
    dataset['focal'] = .5 * dataset['img_shape'][1] / jnp.tan(.5 * camera_angle_x)
    logging.info("Loading transformations finished")
    return dataset


def load_model(rng, dataset):
    rng, key = jax.random.split(rng)
    if FLAGS.pixel_sampling == "total":
        ray_count = dataset['img_shape'][0] * dataset['img_shape'][1]
    else:
        ray_count = FLAGS.pixel_count
    input_shape = (1, min(ray_count, FLAGS.chunk), 3)
    random_array = jax.random.uniform(rng, input_shape)
    sample_rays = utils.Rays(origins=random_array,
                             directions=random_array,
                             viewdirs=random_array)
    sample_rays = {"rays": utils.to_device(sample_rays)}
    model, variables = models.get_model(key, sample_rays, FLAGS)
    del variables
    variables = checkpoints.restore_checkpoint(path.join(FLAGS.train_dir, FLAGS.subset), None)['optimizer']['target']
    variables = jax.device_put(flax.core.freeze(variables))

    def _render_fn(variables, key_0, key_1, rays):
        return model.apply(variables, key_0, key_1, rays, FLAGS.randomized)

    render_fn = functools.partial(_render_fn, variables)
    logging.info("Loading model finished")
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


def compute_loss(params,
                 T_init,
                 pixels,
                 renderer):
    T = jaxlie.manifold.rplus(T_init, params['epsilon']).as_matrix()
    rgb, disp = renderer.render(T)
    mask = (disp < 1.) * (disp > 0.)
    loss_rgb = (optax.huber_loss(rgb, pixels[:, :3], FLAGS.huber_delta) * mask[:, None]).mean()
    # loss_disp = (optax.huber_loss(disp, pixels[:, -1], FLAGS.huber_delta) * mask[:, None]).mean()
    return loss_rgb  # + loss_disp


def compute_errors(T_true: jaxlie.SE3, T_pred: jaxlie.SE3):
    t_gt, t_pred = T_true.translation(), T_pred.translation()
    R_gt, R_pred = T_true.rotation(), T_pred.rotation()
    translation_error = jnp.linalg.norm(t_gt - t_pred)
    rotation_error = jnp.linalg.norm((R_gt @ R_pred.inverse()).log())
    return translation_error, rotation_error


def fit(T_init, rgbd_img, renderer, T_true=None):
    params = {'epsilon': jaxlie.manifold.zero_tangents(T_init)}
    scheduler = optax.exponential_decay(FLAGS.lr_init, FLAGS.decay_steps, FLAGS.decay_rate)
    optimizer = optax.adam(scheduler)
    opt_state = optimizer.init(params)
    history = {'loss': [],
               't error': [],
               'R error': [],
               'grad': []}

    pixels, pixel_coords = renderer.resample_pixels(rgbd_img)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(compute_loss)(params,
                                                       T_init,
                                                       pixels,
                                                       renderer)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    if T_true is not None:
        print(f'|{"step":^5s}|{"loss":^7s}|{"grads":^49s}|{"t error":^7s}|{"R error":^7s}|')
    else:
        print(f'|{"step":^5s}|{"loss":^7s}|{"grads":^49s}|')
    for i in tqdm(range(FLAGS.max_steps)):
        params, opt_state, loss, grads = step(params, opt_state)
        history['loss'].append(loss)
        history['grad'].append(grads["epsilon"])
        if T_true is not None:
            t_error, R_error = compute_errors(T_true, jaxlie.manifold.rplus(T_init, params['epsilon']))
            history['t error'].append(t_error)
            history['R error'].append(R_error)
        if i % FLAGS.print_every == 0:
            msg = f'|{i:5d}|{loss:7.4f}|{grads["epsilon"]}|'
            if T_true is not None:
                msg += f'{t_error:7.4f}|{R_error:7.4f}|'
            tqdm.write(msg)
    return jaxlie.manifold.rplus(T_init, params['epsilon']), history


def print_history(history):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(history['loss'], 'b-')
    ax2.plot(history['t error'], 'r-', label='t error')
    ax2.plot(history['R error'], 'r--', label='R error')

    ax1.set_xlabel('steps')
    ax1.set_ylabel('loss', color='b')
    ax2.set_ylabel('error', color='r')
    plt.title(f'lr: {FLAGS.lr_init} pixel count: {FLAGS.pixel_count}\n' +
              f'perturbation_R: {FLAGS.perturbation_R} perturbation_t: {FLAGS.perturbation_t}')
    plt.legend()
    plt.show()


def main(_):
    rng = init()
    dataset = load_dataset()
    render_fn = load_model(rng, dataset)

    rgbd_img = load_rgbd_img(0)
    renderer = Renderer(rng, render_fn, rgbd_img.shape, dataset['focal'])

    T_matrix = jnp.array(dataset["frames"][0]["transform_matrix"], jnp.float32)
    T_true, T_init = twist_transformation(T_matrix, rng)

    print('Initial errors:\t translation: %.3f\t rotation: %.3f' % compute_errors(T_true, T_init))

    T_final, history = fit(T_init, rgbd_img, renderer, T_true)

    print('\nFinal errors\t translation: %.3f\t rotation: %.3f' % compute_errors(T_true, T_final))
    print('T_true:', T_true)
    print('T_init:', T_init)
    print('T_final:', T_final)
    print_history(history)

    # for idx, frame in enumerate(tqdm(dataset["frames"], desc="Evaluating", unit="frame")):


if __name__ == "__main__":
    app.run(main)
