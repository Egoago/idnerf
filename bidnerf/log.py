import argparse
import json
import os

import yaml
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from bidnerf import math, base, rendering


def __test_count(path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    return b


def save_graph(history: base.History, file_path, title):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    fig.subplots_adjust(right=0.75)
    ax3.spines.right.set_position(("axes", 1.2))

    ax1.plot(history.loss, 'b-')
    ax2.plot(history.t_error, 'r-', label='t error')
    ax2.plot(history.R_error, 'r--', label='R error')
    ax3.plot(history.sample_count, 'g-', alpha=0.6)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Error', color='r')
    ax3.set_ylabel('Sample count', color='g')

    ax1.yaxis.label.set_color('b')
    ax2.yaxis.label.set_color('r')
    ax3.yaxis.label.set_color('g')

    ax2.legend()
    ax2.grid()
    fig.suptitle(title, fontsize=16)

    plt.savefig(file_path)


def __save_render(data: base.Data, file_path, render_fn, rng):
    base_rgbdm_img = data.frames[-1].rgbdm_img
    img_true, depth_true = base_rgbdm_img[:, :, :3], base_rgbdm_img[:, :, 3]
    img_init, depth_init = rendering.render_img(render_fn, data.T_init, data.cam_params, rng)
    img_final, depth_final = rendering.render_img(render_fn, data.T_final, data.cam_params, rng)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex='all', sharey='all')
    fig.tight_layout()
    axs[0, 0].imshow(img_true)
    axs[0, 1].imshow(img_init)
    axs[0, 2].imshow(img_final)
    axs[1, 0].imshow(depth_true)
    axs[1, 1].imshow(depth_init)
    axs[1, 2].imshow(depth_final)
    axs[0, 0].set_title('True')
    axs[0, 1].set_title('Init')
    axs[0, 2].set_title('Final')
    plt.savefig(file_path)


def __read_configs() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(os.path.join('jaxnerf', args.config + '.yaml'), 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    return config


def save_data(data: base.Data, file_name):
    configs = __read_configs()

    data_dict = data.to_dict()
    data_dict['configs'] = {key:base.FLAGS.__getattr__(key) for key in configs.keys()}

    init_translation_error, init_rotation_error = math.compute_errors(data.T_true, data.T_init)
    final_translation_error, final_rotation_error = math.compute_errors(data.T_true, data.T_final)
    data_dict['initial_error'] = {'t_error': init_translation_error,
                                  'R_error': init_rotation_error}
    data_dict['final_error'] = {'t_error': final_translation_error,
                                'R_error': final_rotation_error}

    with open(file_name, "w") as fp:
        json.dump(data_dict, fp)


def save(data: base.Data, render_fn, rng, mode="grd") -> int:
    directory = base.FLAGS.result_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_patterns = ['%06d-graph.png', '%06d-render.png', '%06d-data.json']
    test_id = max([__test_count(os.path.join(directory, file_pattern)) for file_pattern in file_patterns])
    file_paths = [os.path.join(directory, pattern % test_id) for pattern in file_patterns]
    title = base.FLAGS.test_name
    if title is None:
        title = f"Test {test_id}"

    if 'g' in mode:
        save_graph(data.history, file_paths[0], title)
    if 'd' in mode:
        save_data(data, file_paths[2])
    if 'r' in mode:
        __save_render(data, file_paths[1], render_fn, rng)
    return test_id
