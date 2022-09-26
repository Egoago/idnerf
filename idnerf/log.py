import json
import os

import jax
import jax.numpy as jnp
import jaxlie
from absl import flags
from matplotlib import pyplot as plt

from idnerf.renderer import Renderer


def test_count(path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    return b


def save_graph(history, file_path):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(history['loss'], 'b-')
    ax2.plot(history['t error'], 'r-', label='t error')
    ax2.plot(history['R error'], 'r--', label='R error')

    ax1.set_xlabel('steps')
    ax1.set_ylabel('loss', color='b')
    ax2.set_ylabel('error', color='r')
    plt.legend()
    plt.grid()

    plt.savefig(file_path)


def save_render(history, file_path, renderer, rng):
    rng_keys = jax.random.split(rng, 3)
    img_true, disp_true = renderer.render_img(history['T_true'].as_matrix(), rng_keys[0])
    img_init, disp_init = renderer.render_img(history['T_init'].as_matrix(), rng_keys[1])
    img_final, disp_final = renderer.render_img(history['T_final'].as_matrix(), rng_keys[2])
    fig, axs = plt.subplots(2, 3, figsize=(24, 16), sharex=True, sharey=True)
    fig.tight_layout()
    axs[0, 0].imshow(img_true)
    axs[0, 1].imshow(img_init)
    axs[0, 2].imshow(img_final)
    axs[1, 0].imshow(disp_true)
    axs[1, 1].imshow(disp_init)
    axs[1, 2].imshow(disp_final)
    axs[0, 0].set_title('True')
    axs[0, 1].set_title('Init')
    axs[0, 2].set_title('Final')
    plt.savefig(file_path)


def compute_errors(T_true: jaxlie.SE3, T_pred: jaxlie.SE3):
    t_gt, t_pred = T_true.translation(), T_pred.translation()
    R_gt, R_pred = T_true.rotation(), T_pred.rotation()
    translation_error = jnp.linalg.norm(t_gt - t_pred).tolist()
    rotation_error = jnp.linalg.norm((R_gt @ R_pred.inverse()).log()).tolist()
    return translation_error, rotation_error


def save_log(history, file_name):
    params = ['dataset',
              'subset',
              'pixel_sampling',
              'perturbation_R',
              'perturbation_t',
              'pixel_count',
              'lr_init',
              'num_coarse_samples',
              'num_fine_samples',
              'use_pixel_centers',
              'max_steps',
              'decay_steps',
              'decay_rate',
              'huber_delta',
              'chunk']
    _flags = {}
    for param in params:
        _flags[param] = flags.FLAGS[param].value
    history['flags'] = _flags

    init_translation_error, init_rotation_error = compute_errors(history['T_true'], history['T_init'])
    final_translation_error, final_rotation_error = compute_errors(history['T_true'], history['T_final'])
    history['initial_error'] = {'translation_error': init_translation_error,
                                'rotation_error': init_rotation_error}
    history['final_error'] = {'translation_error': final_translation_error,
                              'rotation_error': final_rotation_error}
    for param in ['T_true', 'T_init', 'T_final']:
        history[param] = {'translation': history[param].translation().tolist(),
                          'rotation': history[param].rotation().parameters().tolist()}
    with open(file_name, "w") as fp:
        json.dump(history, fp)


def save_history(history, renderer: Renderer, rng):
    directory = flags.FLAGS.result_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_patterns = ['%06d-graph.png', '%06d-render.png', '%06d-log.json']
    test_id = test_count(os.path.join(directory, file_patterns[0]))

    save_graph(history, os.path.join(directory, file_patterns[0] % test_id))
    save_render(history, os.path.join(directory, file_patterns[1] % test_id), renderer, rng)
    save_log(history, os.path.join(directory, file_patterns[2] % test_id))
