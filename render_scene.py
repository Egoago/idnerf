import json
import os

import jax
import jaxlie
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from tqdm import tqdm

import bidnerf


def render_all():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221026)

    data = bidnerf.load_data(rng, all_frames=True)
    render_fn = bidnerf.load_model(rng)
    dir_path = os.path.join(bidnerf.base.FLAGS.train_dir,
                            bidnerf.base.FLAGS.dataset,
                            bidnerf.base.FLAGS.subset, 'test_preds2')
    os.makedirs(dir_path, exist_ok=True)

    for frame in tqdm(data.frames, desc='Rendering frames', unit='frame'):
        rgb, depth = bidnerf.render_img(render_fn, data.T_true @ frame.T_cam2base, data.cam_params, rng)
        np.save(os.path.join(dir_path, f'{frame.id:03d}_rgb.npy'), np.array(rgb))
        np.save(os.path.join(dir_path, f'{frame.id:03d}_depth.npy'), np.array(depth))


def render_one():
    bidnerf.load_flags()
    bidnerf.base.FLAGS.dataset = "blender"
    bidnerf.base.FLAGS.subset = "lego"
    bidnerf.base.FLAGS.frame_ids = [32]
    rng = jax.random.PRNGKey(20221027)

    data = bidnerf.load_data(rng)
    render_fn = bidnerf.load_model(rng)

    fine_rgb, _ = bidnerf.render_img(render_fn, data.T_true @ data.frames[0].T_cam2base, data.cam_params, rng)
    fine_rgb = np.array(fine_rgb)
    coarse_rgb, _ = bidnerf.render_img(render_fn, data.T_true @ data.frames[0].T_cam2base, data.cam_params, rng, True)
    coarse_rgb = np.array(coarse_rgb)
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    axs[0].set_title('Fine model', fontsize=20)
    axs[0].imshow(fine_rgb)
    axs[0].set_axis_off()
    axs[1].set_title('Coarse model', fontsize=20)
    axs[1].imshow(coarse_rgb)
    axs[1].set_axis_off()
    plt.savefig('plots/fine-coarse.png', bbox_inches='tight', pad_inches=0)


def render_seq():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221031)

    rgbs = []
    depths = []
    test_id = 529
    step = 50

    file_name = os.path.join('results', f'{test_id:06d}-data.json')
    with open(file_name, "r") as fp:
        data = json.loads(fp.read())
    cam_params = bidnerf.CameraParameters(width=data['cam_params']['width'],
                                         height=data['cam_params']['height'],
                                         focal=data['cam_params']['focal'],)
    bidnerf.base.FLAGS.dataset = data['configs']['dataset']
    bidnerf.base.FLAGS.subset = data['configs']['subset']
    render_fn = bidnerf.load_model(rng)
    i = 0
    for log_T_pred in tqdm(data['history']['log_T_pred'][::step], desc='Rendering frames', unit='frame'):
        rgb, depth = bidnerf.render_img(render_fn, jaxlie.SE3.exp(np.array(log_T_pred)), cam_params, rng)

        np.save(os.path.join(f'diagrams/{i}_rgb.npy'), np.array(rgb))
        np.save(os.path.join(f'diagrams/{i}_depth.npy'), np.array(depth))
        i += step


def main(_):
    render_all()
    #render_seq()


if __name__ == "__main__":
    app.run(main)
