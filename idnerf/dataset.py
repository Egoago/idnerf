import glob
import os
from typing import List, Tuple

import jaxlie
from PIL import Image
from absl import logging, flags
from absl.flags import FLAGS
import jax.numpy as jnp

from idnerf import base, math


def load_img(img_path):
    img = Image.open(img_path)
    return jnp.asarray(img, jnp.float32) / 255.


def load_rgbdm_img(dataset, idx) -> jnp.ndarray:
    if flags.FLAGS.depth_param > 0.:
        dir_path = os.path.join(flags.FLAGS.train_dir, flags.FLAGS.subset, 'test_preds2')
        rgb = jnp.load(os.path.join(dir_path, f'{idx:03d}_rgb.npy'))
        depth = jnp.load(os.path.join(dir_path, f'{idx:03d}_depth.npy'))
        mask = (jnp.mean(rgb, axis=-1) < (1. - 1e-4)) * (depth > flags.FLAGS.near) * (depth < flags.FLAGS.far)
    elif FLAGS.use_original_img:
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
        mask = jnp.mean(rgb, axis=-1) < (1. - 1e-4)
        depth = 1 / disp
    rgbdm_img = jnp.concatenate([rgb, depth[..., None], mask[..., None]], axis=-1)
    return rgbdm_img


def get_T_cam2world(dataset, idx) -> jaxlie.SE3:
    T_matrix = jnp.array(dataset["frames"][idx]["transform_matrix"], jnp.float32)
    T_cam2world = jaxlie.SE3.from_matrix(T_matrix)
    # noinspection PyTypeChecker
    return T_cam2world


def load_frames(dataset, frame_ids, load_imgs=True) -> Tuple[List[base.Frame], jaxlie.SE3]:
    assert len(frame_ids) > 0
    T_cam2worlds = []
    for frame_id in frame_ids:
        T_cam2worlds.append(get_T_cam2world(dataset, frame_id))
    T_cam2lasts, T_true = math.relative_transformations(T_cam2worlds)
    frames = [base.Frame(T_cam2base=T_cam2last,
                         rgbdm_img=load_rgbdm_img(dataset, _id) if load_imgs else None,
                         id=_id) for T_cam2last, _id in zip(T_cam2lasts, frame_ids)]
    return frames, T_true


def get_cam_params(dataset, img_shape):
    camera_angle_x = float(dataset["camera_angle_x"])
    height, width = img_shape[:2]
    focal = .5 * width / float(jnp.tan(.5 * camera_angle_x))
    return base.CameraParameters(height=height, width=width, focal=focal)


def load_dataset():
    import json
    data_path = os.path.join(FLAGS.data_dir, FLAGS.subset, "transforms_test.json")
    with open(data_path, "r") as fp:
        dataset = json.load(fp)
    assert dataset is not None
    return dataset


def load_data(rng, all_frames=False):
    dataset = load_dataset()
    if all_frames:
        frames, T_true = load_frames(dataset, range(len(dataset['frames'])), load_imgs=False)
    else:
        frames, T_true = load_frames(dataset, flags.FLAGS.frame_ids, load_imgs=True)
    assert len(frames) > 0

    cam_params = get_cam_params(dataset, load_rgbdm_img(dataset, 0).shape)
    T_init = math.twist_transformation(T_true, rng)
    logging.info("Loading dataset finished")
    return base.Data(cam_params=cam_params, T_true=T_true, frames=frames, T_init=T_init)
