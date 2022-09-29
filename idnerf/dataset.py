import glob
import os

from PIL import Image
from absl import flags
import jax.numpy as jnp


def load_img(img_path):
    img = Image.open(img_path)
    return jnp.asarray(img, jnp.float32) / 255.


def load_rgbdm_img(dataset, idx):
    if flags.FLAGS.use_original_img:
        path = os.path.join(flags.FLAGS.data_dir, flags.FLAGS.subset)
        img_name = dataset['frames'][idx]['file_path']
        rgb = load_img(os.path.join(path, img_name+'.png'))[:, :, :3]
        depth_path = glob.glob(os.path.join(path, img_name+'_depth_*.png'))
        assert len(depth_path) == 1
        depth = load_img(depth_path[0])[:, :, 0]
        mask = depth > 1e-4
    else:
        path = os.path.join(flags.FLAGS.train_dir, flags.FLAGS.subset, 'test_preds')
        img_name = f'{idx:03d}.png'
        rgb = load_img(os.path.join(path, img_name))
        disp = load_img(os.path.join(path, 'disp_' + img_name))
        mask = disp < (1. - 1e-4)
        depth = 1/disp
    rgbdm_img = jnp.concatenate([rgb, depth[..., None], mask[..., None]], axis=-1)
    return rgbdm_img


def get_frame(dataset, idx):
    rgbdm_img = load_rgbdm_img(dataset, idx)
    T_matrix = jnp.array(dataset["frames"][idx]["transform_matrix"], jnp.float32)
    return rgbdm_img, T_matrix


def load_dataset():
    import json
    data_path = os.path.join(flags.FLAGS.data_dir, flags.FLAGS.subset, "transforms_test.json")
    with open(data_path, "r") as fp:
        dataset = json.load(fp)
    camera_angle_x = float(dataset["camera_angle_x"])
    dataset['img_shape'] = load_rgbdm_img(dataset, 0).shape[:2]
    dataset['focal'] = .5 * dataset['img_shape'][1] / jnp.tan(.5 * camera_angle_x)
    return dataset
