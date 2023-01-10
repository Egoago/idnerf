import numpy as np
from PIL import Image
from absl import app
import jax

import bidnerf


def main(_):
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221012)
    rng_keys = jax.random.split(rng, 4)

    data = bidnerf.load_data(rng_keys[0])

    render_fn = bidnerf.load_model(rng_keys[1])

    #rgb, depth = bidnerf.render_img(render_fn, data.T_true @ data.frames[0].T_cam2base, data.cam_params, rng)
    #img = Image.fromarray(np.uint8(rgb * 255))
    #img.save('test9.png')

    bidnerf.fit(data, render_fn, rng_keys[2])

    #bidnerf.save(data, render_fn, rng_keys[3], "gdr")


if __name__ == "__main__":
    app.run(main)
