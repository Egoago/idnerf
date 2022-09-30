import glob
import os

import jaxlie
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
    T = jaxlie.SE3.from_matrix(T_matrix)
    return rgbdm_img, T


def load_frames(dataset):
    rgbdm_imgs = []
    Ts = []
    T_rels = []
    for frame_id in flags.FLAGS.frame_ids:
        rgbdm_img, T = get_frame(dataset, frame_id)
        rgbdm_imgs.append(rgbdm_img)
        if len(Ts) > 0:
            T_prev = Ts[-1]
            T_rel = T_prev @ T.inverse()
            T_rels.append(T_rel)
        Ts.append(T)
    T_true = Ts[-1]
    return rgbdm_imgs, T_true, T_rels


def load_dataset():
    import json
    data_path = os.path.join(flags.FLAGS.data_dir, flags.FLAGS.subset, "transforms_test.json")
    with open(data_path, "r") as fp:
        dataset = json.load(fp)
    camera_angle_x = float(dataset["camera_angle_x"])
    height, width = load_rgbdm_img(dataset, 0).shape[:2]
    focal = .5 * dataset['img_shape'][1] / jnp.tan(.5 * camera_angle_x)
    rgbdm_imgs, T_true, T_rels = load_frames(dataset)
    return rgbdm_imgs, T_true, T_rels, width, height, focal
