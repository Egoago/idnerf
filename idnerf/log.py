import argparse
import json
import os
from copy import copy

import jax
import jax.numpy as jnp
import jaxlie
import yaml
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
    ax3 = ax1.twinx()
    fig.subplots_adjust(right=0.75)
    ax3.spines.right.set_position(("axes", 1.2))

    ax1.plot(history['loss'], 'b-')
    ax2.plot(history['t_error'], 'r-', label='t error')
    ax2.plot(history['R_error'], 'r--', label='R error')
    ax3.plot(history['sample_count'], 'g-', alpha=0.6)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Error', color='r')
    ax3.set_ylabel('Sample count', color='g')

    ax1.yaxis.label.set_color('b')
    ax2.yaxis.label.set_color('r')
    ax3.yaxis.label.set_color('g')

    ax2.legend()
    ax2.grid()

    plt.savefig(file_path)


def save_render(history, file_path, renderer, rng, rgbdm_img):
    rng_keys = jax.random.split(rng, 3)
    img_true, disp_true = rgbdm_img[:, :, :3], rgbdm_img[:, :, 3]
    img_init, disp_init = renderer.render_img(history['T_init'].as_matrix(), rng_keys[1])
    img_final, disp_final = renderer.render_img(history['T_final'].as_matrix(), rng_keys[2])
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
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
    history = copy(history)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(os.path.join('jaxnerf', args.config + '.yaml'), 'r') as config_file:
        configs = yaml.load(config_file, Loader=yaml.Loader)

    history['configs'] = configs

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


def save_history(history, renderer: Renderer, rng, rgbdm_img, mode="grl"):
    directory = flags.FLAGS.result_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_patterns = ['%06d-graph.png', '%06d-render.png', '%06d-log.json']
    test_id = max([test_count(os.path.join(directory, file_pattern)) for file_pattern in file_patterns])

    if 'g' in mode:
        save_graph(history, os.path.join(directory, file_patterns[0] % test_id))
    if 'l' in mode:
        save_log(history, os.path.join(directory, file_patterns[2] % test_id))
    if 'r' in mode:
        save_render(history, os.path.join(directory, file_patterns[1] % test_id), renderer, rng, rgbdm_img)
