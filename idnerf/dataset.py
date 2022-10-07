import glob
import os
from typing import List, Tuple

import jaxlie
from PIL import Image
from absl import logging
from absl.flags import FLAGS
import jax.numpy as jnp

from idnerf import base, math


def load_img(img_path):
    img = Image.open(img_path)
    return jnp.asarray(img, jnp.float32) / 255.


def load_rgbdm_img(dataset, idx) -> jnp.ndarray:
    if FLAGS.use_original_img:
        path = os.path.join(FLAGS.data_dir, FLAGS.subset)
        img_name = dataset['frames'][idx]['file_path']
        rgb = load_img(os.path.join(path, img_name + '.png'))[:, :, :3]
        depth_path = glob.glob(os.path.join(path, img_name + '_depth_*.png'))
        assert len(depth_path) == 1
        depth = load_img(depth_path[0])[:, :, 0]
        mask = depth > 1e-4
    else:
        path = os.path.join(FLAGS.train_dir, FLAGS.subset, 'test_preds')
        img_name = f'{idx:03d}.png'
        rgb = load_img(os.path.join(path, img_name))
        disp = load_img(os.path.join(path, 'disp_' + img_name))
        mask = disp < (1. - 1e-4)
        depth = 1 / disp
    rgbdm_img = jnp.concatenate([rgb, depth[..., None], mask[..., None]], axis=-1)
    return rgbdm_img


def get_frame(dataset, idx) -> Tuple[jnp.ndarray, jaxlie.SE3]:
    rgbdm_img = load_rgbdm_img(dataset, idx)
    T_matrix = jnp.array(dataset["frames"][idx]["transform_matrix"], jnp.float32)
    T = jaxlie.SE3.from_matrix(T_matrix)
    return rgbdm_img, T


def load_frames(dataset) -> Tuple[List[base.Frame], jaxlie.SE3]:
    assert len(FLAGS.frame_ids) > 0
    rgbdm_imgs = []
    Ts = []
    for frame_id in FLAGS.frame_ids:
        rgbdm_img, T = get_frame(dataset, frame_id)
        rgbdm_imgs.append(rgbdm_img)
        Ts.append(T)
    T_rels, T_true = math.calculate_T_rels(Ts)
    frames = [base.Frame(id=id,
                    rgbdm_img=rgbdm_img,
                    T_rel=T_rel) for T_rel, rgbdm_img, id in zip(T_rels, rgbdm_imgs, FLAGS.frame_ids)]
    return frames, T_true


def load_data(rng):
    import json
    data_path = os.path.join(FLAGS.data_dir, FLAGS.subset, "transforms_test.json")
    with open(data_path, "r") as fp:
        dataset = json.load(fp)
    assert dataset is not None

    camera_angle_x = float(dataset["camera_angle_x"])
    frames, T_true = load_frames(dataset)
    assert len(frames) > 0

    height, width = frames[0].rgbdm_img.shape[:2]
    focal = .5 * width / float(jnp.tan(.5 * camera_angle_x))
    cam_params = base.CameraParameters(height=height, width=width, focal=focal)
    T_init = math.twist_transformation(T_true, rng)
    data = base.Data(cam_params=cam_params, T_true=T_true, frames=frames, T_init=T_init)
    logging.info("Loading dataset finished")
    return data
