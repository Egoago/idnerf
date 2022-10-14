import os

import numpy as np
from absl import app, flags
from tqdm import tqdm

import idnerf


def main(_):
    rng = idnerf.init()
    
    data = idnerf.load_data(rng, all_frames=True)
    render_fn = idnerf.load_model(rng)
    dir_path = os.path.join(flags.FLAGS.train_dir, flags.FLAGS.subset, 'test_preds2')
    os.makedirs(dir_path, exist_ok=True)
    
    for frame in tqdm(data.frames, desc='Rendering frames', unit='frame'):
        rgb, depth = idnerf.render_img(render_fn, data.T_true @ frame.T_cam2base, data.cam_params, rng)
        np.save(os.path.join(dir_path, f'{frame.id:03d}_rgb.npy'), np.array(rgb))
        np.save(os.path.join(dir_path, f'{frame.id:03d}_depth.npy'), np.array(depth))


if __name__ == "__main__":
    app.run(main)
