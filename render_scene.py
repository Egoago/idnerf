import functools
from os import path

import numpy as np
from absl import app, flags, logging
import flax
from flax.training import checkpoints
import jax
from jax import random

from jaxnerf.nerf import models, utils
from tqdm import tqdm


FLAGS = flags.FLAGS

utils.define_flags()
flags.DEFINE_enum("pixel_sampling", "total", ["total", "feature_points", "random", "patch"], "")
flags.DEFINE_enum("split", "test", ["test", "train", "val", "all"], "")


def load_blender_transformations():
    import json
    from PIL import Image
    with utils.open_file(path.join(FLAGS.data_dir, "transforms_{}.json".format(FLAGS.split)), "r") as fp:
        dataset = json.load(fp)
    camera_angle_x = float(dataset["camera_angle_x"])
    img_path = path.join(FLAGS.data_dir, dataset["frames"][0]["file_path"] + ".png")
    with utils.open_file(img_path, "rb") as img_file:
        sample_img = np.asarray(Image.open(img_file), dtype=np.float32) / 255.
    focal = .5 * sample_img.shape[1] / np.tan(.5 * camera_angle_x)
    cam2worlds = [np.array(frame["transform_matrix"], dtype=np.float32) for frame in dataset["frames"]]
    sample_rays = generate_rays(focal, cam2worlds[0], sample_img)
    return cam2worlds, focal, sample_rays


def generate_rays(focal, cam2world, img):
    pixel_center = 0.5 if FLAGS.use_pixel_centers else 0.0
    if FLAGS.pixel_sampling == 'total':
        x, y = np.meshgrid(np.arange(img.shape[1], dtype=np.float32) + pixel_center,  # X-Axis (columns)
                           np.arange(img.shape[0], dtype=np.float32) + pixel_center,  # Y-Axis (rows)
                           indexing="xy")
    else:
        raise NotImplementedError(f"Sampling method not implemented: {FLAGS.sampling}")

    camera_dirs = np.stack([(x - img.shape[1] * 0.5) / focal,
                            -(y - img.shape[0] * 0.5) / focal,
                            -np.ones_like(x)],
                           axis=-1)
    directions = camera_dirs.dot(cam2world[:3, :3].T)
    origins = np.broadcast_to(cam2world[None, None, :3, -1], directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    return utils.Rays(origins=origins, directions=directions, viewdirs=viewdirs)


def init():
    import warnings
    warnings.filterwarnings("ignore")
    rng = random.PRNGKey(20220905)

    if FLAGS.config is not None:
        utils.update_flags(FLAGS)
    if FLAGS.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if FLAGS.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")

    cam2worlds, focal, sample_rays = load_blender_transformations()
    logging.info("Loading transformations finished")
    rng, key = random.split(rng)
    model, init_variables = models.get_model(key, {"rays": utils.to_device(sample_rays)}, FLAGS)
    optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables
    logging.info("Loading model finished")
    return rng, state, model, cam2worlds, focal, sample_rays


def main(_):
    rng, state, model, cam2worlds, focal, sample_rays = init()

    # pmap over only the data input.
    render_pfn = jax.pmap(
        lambda variables, key_0, key_1, rays:
        jax.lax.all_gather(model.apply(variables,
                                       key_0,
                                       key_1,
                                       rays,
                                       False),  # Rendering is forced to be deterministic even if training was
                           # randomized, as this eliminates "speckle" artifacts.
                           axis_name="batch"),
        in_axes=(None, None, None, 0),
        donate_argnums=3,
        axis_name="batch")

    out_dir = path.join(FLAGS.train_dir, "test_preds2")

    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    if FLAGS.save_output and (not utils.isdir(out_dir)):
        utils.makedirs(out_dir)
    for idx, cam2world in enumerate(tqdm(cam2worlds, desc="Rendering images", unit="image")):
        img_rays = generate_rays(focal, cam2world, sample_rays.origins)
        pred_color, pred_disp, pred_acc = utils.render_image(
            functools.partial(render_pfn, state.optimizer.target),
            img_rays,
            rng,
            FLAGS.dataset == "llff",
            chunk=FLAGS.chunk)
        if jax.process_index() == 0 and FLAGS.save_output:
            utils.save_img(pred_color, path.join(out_dir, "{:03d}.png".format(idx)))
            utils.save_img(pred_disp[Ellipsis, 0], path.join(out_dir, "disp_{:03d}.png".format(idx)))


if __name__ == "__main__":
    app.run(main)
