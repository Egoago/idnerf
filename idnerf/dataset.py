import os
from dataclasses import dataclass, field
from typing import List, Tuple

import jaxlie
from PIL import Image
from absl import logging
import jax.numpy as jnp

from idnerf import base, math


@dataclass
class Dataset:
    cam_params: base.CameraParameters
    cam2worlds: List[jaxlie.SE3] = field(default_factory=list)


def load_img(img_path):
    img = Image.open(img_path)
    return jnp.asarray(img, jnp.float32) / 255.


def load_rgbdm_img(idx) -> jnp.ndarray:
    dir_path = os.path.join(base.FLAGS.train_dir,
                            base.FLAGS.dataset,
                            base.FLAGS.subset, 'test_preds2')
    rgb = jnp.load(os.path.join(dir_path, f'{idx:03d}_rgb.npy'))
    depth = jnp.load(os.path.join(dir_path, f'{idx:03d}_depth.npy'))
    mask = (jnp.mean(rgb, axis=-1) < (1. - 1e-4)) * (depth > base.FLAGS.near) * (depth < base.FLAGS.far)
    rgbdm_img = jnp.concatenate([rgb, depth[..., None], mask[..., None]], axis=-1)
    return rgbdm_img


def load_frames(dataset: Dataset, frame_ids, load_imgs=True) -> Tuple[List[base.Frame], jaxlie.SE3]:
    assert len(frame_ids) > 0
    cam2worlds = [dataset.cam2worlds[i] for i in frame_ids]
    T_cam2lasts, T_true = math.relative_transformations(cam2worlds)
    frames = [base.Frame(T_cam2base=T_cam2last,
                         rgbdm_img=load_rgbdm_img(_id) if load_imgs else None,
                         id=_id) for T_cam2last, _id in zip(T_cam2lasts, frame_ids)]
    return frames, T_true


def _recenter_poses(poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = jnp.array([[0, 0, 0, 1.]])
    c2w = _poses_avg(poses)
    c2w = jnp.concatenate([c2w[:3, :4], bottom], -2)
    bottom = jnp.tile(jnp.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = jnp.concatenate([poses[:, :3, :4], bottom], -2)
    poses = jnp.linalg.inv(c2w) @ poses
    poses_.at[:, :3, :4].multiply(poses[:, :3, :4])
    poses = poses_
    return poses


def _poses_avg(poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = _normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = jnp.concatenate([_viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def _viewmatrix(z, up, pos):
    """Construct lookat view matrix."""
    vec2 = _normalize(z)
    vec1_avg = up
    vec0 = _normalize(jnp.cross(vec1_avg, vec2))
    vec1 = _normalize(jnp.cross(vec2, vec0))
    m = jnp.stack([vec0, vec1, vec2, pos], 1)
    return m


def _normalize(x):
    """Normalization helper function."""
    return x / jnp.linalg.norm(x)


def _load_dataset_llff() -> Dataset:
    factor = 4
    # Load poses and bds.
    dir_path = os.path.join(base.FLAGS.data_dir,
                            base.FLAGS.dataset,
                            base.FLAGS.subset)
    poses_arr = jnp.load(os.path.join(dir_path, "poses_bounds.npy"))

    imgs_path = os.path.join(dir_path, f'images_{factor}')
    sample_image = load_img(os.path.join(imgs_path, os.listdir(imgs_path)[0]))

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    # Update poses according to downsampling.
    poses.at[:2, 4, :].set(jnp.array(sample_image.shape[:2]).reshape([2, 1]))
    poses.at[2, 4, :].set(poses[2, 4, :] * 1. / factor)

    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = jnp.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = jnp.moveaxis(poses, -1, 0).astype(jnp.float32)
    bds = jnp.moveaxis(bds, -1, 0).astype(jnp.float32)

    # Rescale according to a default bd factor.
    scale = 1. / (bds.min() * .75)
    poses.at[:, :3, 3].multiply(scale)
    bds *= scale

    # Recenter poses.
    poses = _recenter_poses(poses)

    camtoworlds = poses[:, :3, :4]
    focal = poses[0, -1, -1]

    height, width = sample_image.shape[:2]
    cam2worlds = [jaxlie.SE3.from_matrix(jnp.concatenate([camtoworld,
                                                          jnp.float32([[0., 0., 0., 1.]])]))
                  for camtoworld in camtoworlds]

    cam_param = base.CameraParameters(height=height, width=width, focal=focal)
    return Dataset(cam_params=cam_param, cam2worlds=cam2worlds)


def _load_dataset_blender() -> Dataset:
    import json
    data_dir = os.path.join(base.FLAGS.data_dir,
                            base.FLAGS.dataset,
                            base.FLAGS.subset)
    with open(os.path.join(data_dir, "transforms_test.json"), 'r') as f:
        dataset = json.load(f)
    assert dataset is not None
    cam2worlds = [jaxlie.SE3.from_matrix(jnp.array(frame["transform_matrix"], jnp.float32))
                  for frame in dataset["frames"]]
    sample_image = load_img(os.path.join(data_dir, dataset["frames"][0]["file_path"] + '.png'))

    height, width = sample_image.shape[:2]
    camera_angle_x = float(dataset["camera_angle_x"])
    focal = .5 * width / float(jnp.tan(.5 * camera_angle_x))
    cam_param = base.CameraParameters(height=height, width=width, focal=focal)

    # noinspection PyTypeChecker
    return Dataset(cam_params=cam_param, cam2worlds=cam2worlds)


def load_dataset() -> Dataset:
    if base.FLAGS.dataset == "llff":
        return _load_dataset_llff()
    return _load_dataset_blender()


def load_data(rng, all_frames=False):
    dataset = load_dataset()
    if all_frames:
        frames, T_true = load_frames(dataset, range(len(dataset.cam2worlds)), load_imgs=False)
    else:
        frames, T_true = load_frames(dataset, base.FLAGS.frame_ids, load_imgs=True)
    assert len(frames) > 0
    T_init = math.twist_transformation(T_true, rng)
    logging.info("Loading dataset finished")
    return base.Data(cam_params=dataset.cam_params, T_true=T_true, frames=frames, T_init=T_init)
