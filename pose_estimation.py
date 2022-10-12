from absl import app, flags
import jax

import idnerf


def main(_):
    rng = idnerf.init()
    rng_keys = jax.random.split(rng, 5)

    data = idnerf.load_data(rng_keys[0])

    render_fn = idnerf.load_model(rng_keys[2])

    # rgb, depth = idnerf.render_img(render_fn, data.T_true @ data.frames[0].T_cam2base, data.cam_params, rng)
    # img = Image.fromarray(np.uint8(rgb * 255))
    # img.save('test9.png')
    if flags.FLAGS.per_sample_gradient:
        raise NotImplementedError()  # TODO implement
    idnerf.fit(data, render_fn, rng_keys[3])

    idnerf.save(data, render_fn, rng_keys[4], "gdr")


if __name__ == "__main__":
    app.run(main)
