import numpy as np
from PIL import Image
from absl import app
import jax
from matplotlib import pyplot as plt
from tqdm import tqdm

import bidnerf


def plot_graphs_in_grid(indices: np.ndarray, name: str):
    f, axarr = plt.subplots(indices.shape[0], indices.shape[1],
                            figsize=(indices.shape[1]*3, indices.shape[0]*3),
                            sharex=True, sharey=True, subplot_kw=dict(frameon=False))
    plt.subplots_adjust(hspace=.0, wspace=-0.2)
    for i, idx in enumerate(indices.flatten()):
        img = Image.open(f'results/{idx:06d}-graph.png')
        axarr.flatten()[i].imshow(img)
        axarr.flatten()[i].text(0, 0, f'Index: {idx}', fontsize=10)
        axarr.flatten()[i].tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False)
    plt.savefig(f'plots/{name}.png')


def depth_imgs_subsets_test():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221016)
    for subset in tqdm(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], desc='Subsets'):
        bidnerf.base.FLAGS.subset = subset
        render_fn = bidnerf.load_model(rng)
        data = bidnerf.load_data(rng)
        for use_depth in tqdm([False, True], desc='Use depth', leave=False):
            bidnerf.base.FLAGS.depth_param = 1.0 if use_depth else 0.0
            grid_name = f"{subset}_{'no_' if not use_depth else ''}depth"
            frames = data.frames
            test_ids = []
            for error_size in tqdm([0.1, 0.2, 0.3], desc='Error size', leave=False):
                bidnerf.base.FLAGS.perturbation_R = error_size
                bidnerf.base.FLAGS.perturbation_t = error_size+0.2
                data.T_init = bidnerf.twist_transformation(data.T_true, rng)
                for img_count in tqdm([1, 4, 8], desc='Image count', leave=False):
                    bidnerf.base.FLAGS.test_name = grid_name +\
                                                  f'_{int(error_size*10.001)}_error' \
                                                  f'_{img_count}_img'
                    data.frames = frames[-img_count:]
                    bidnerf.fit(data, render_fn, rng)
                    test_id = bidnerf.save(data, render_fn, rng, "gdr")
                    test_ids.append(test_id)
            plot_graphs_in_grid(np.array(test_ids).reshape(3, 3), grid_name)


def coarse_fine_test():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221017)
    bidnerf.base.FLAGS.frame_ids = [2, 3, 4, 5, 6, 7, 8, 9]
    #bidnerf.base.FLAGS.depth_param = 0.0
    for subset in tqdm(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], desc='Subsets'):
        bidnerf.base.FLAGS.subset = subset
        grid_name = f"{subset}_coarse-fine_no-depth"

        render_fn = bidnerf.load_model(rng)
        data = bidnerf.load_data(rng)
        test_ids = []
        for coarse_opt in tqdm([False, True], desc='Coarse-fine', leave=False):
            bidnerf.base.FLAGS.coarse_opt = coarse_opt
            for error_size in tqdm([0.1, 0.2, 0.3], desc='Error size', leave=False):
                bidnerf.base.FLAGS.perturbation_R = error_size
                bidnerf.base.FLAGS.perturbation_t = error_size+0.2
                bidnerf.base.FLAGS.test_name = f'{subset}_{"coarse" if coarse_opt else "fine"}' \
                                              f'_{int(error_size * 10.001)}_error'

                data.T_init = bidnerf.twist_transformation(data.T_true, rng)
                bidnerf.fit(data, render_fn, rng)

                test_id = bidnerf.save(data, render_fn, rng, "gdr")
                test_ids.append(test_id)
        plot_graphs_in_grid(np.array(test_ids).reshape(2, 3), grid_name)


def error_limit_test():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221018)
    bidnerf.base.FLAGS.frame_ids = [2, 3, 4, 5, 6, 7, 8, 9]
    bidnerf.base.FLAGS.depth_param = 1.0
    bidnerf.base.FLAGS.coarse_opt = True
    for subset in tqdm(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], desc='Subsets'):
        bidnerf.base.FLAGS.subset = subset
        grid_name = f"{subset}_coarse_deth_img_8_error-test"
        render_fn = bidnerf.load_model(rng)
        data = bidnerf.load_data(rng)
        test_ids = []
        for error_size in tqdm([0.4, 0.6, 0.8, 1.0, 1.2, 1.4], desc='Error size', leave=False):
            bidnerf.base.FLAGS.perturbation_R = error_size
            bidnerf.base.FLAGS.perturbation_t = error_size+0.2
            bidnerf.base.FLAGS.test_name = f'{subset}_coarse_deth_img_8_error_{int(error_size * 10.001)}'

            data.T_init = bidnerf.twist_transformation(data.T_true, rng)
            bidnerf.fit(data, render_fn, rng)

            test_id = bidnerf.save(data, render_fn, rng, "gdr")
            test_ids.append(test_id)
        plot_graphs_in_grid(np.array(test_ids).reshape(1, -1), grid_name)


def sparse_img_test():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221019)
    last_img = 199
    img_count = 8
    bidnerf.base.FLAGS.depth_param = 1.0
    bidnerf.base.FLAGS.coarse_opt = True
    for subset in tqdm(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], desc='Subsets'):
        bidnerf.base.FLAGS.subset = subset
        grid_name = f"{subset}_coarse_depth_8-img_sparse-img-test"
        render_fn = bidnerf.load_model(rng)
        test_ids = []
        for img_step in tqdm([1, 2, 4, 8, 16], desc='Img step'):
            bidnerf.base.FLAGS.frame_ids = list(range(last_img, 0, -img_step)[:img_count][::-1])
            data = bidnerf.load_data(rng)
            for error_size in tqdm([0.2, 0.4, 0.8], desc='Error size', leave=False):
                bidnerf.base.FLAGS.perturbation_R = error_size
                bidnerf.base.FLAGS.perturbation_t = error_size+0.2
                bidnerf.base.FLAGS.test_name = f'{subset}_coarse_depth_img-8_' \
                                              f'error-{int(error_size * 10.001)}_img-step-{img_step}'

                data.T_init = bidnerf.twist_transformation(data.T_true, rng)
                bidnerf.fit(data, render_fn, rng)

                test_id = bidnerf.save(data, render_fn, rng, "gdr")
                test_ids.append(test_id)
        plot_graphs_in_grid(np.array(test_ids).reshape(5, -1), grid_name)


def no_rgb():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221029)
    bidnerf.base.FLAGS.frame_ids = [2, 3, 4, 5, 6, 7, 8, 9]
    bidnerf.base.FLAGS.coarse_opt = True
    for subset in tqdm(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], desc='Subsets'):
        bidnerf.base.FLAGS.subset = subset
        grid_name = f"{subset}_coarse_rgb-no_rgb"
        render_fn = bidnerf.load_model(rng)
        data = bidnerf.load_data(rng)
        test_ids = []

        for use_depth, use_rgb in tqdm([(True, True),
                                        (False, True),
                                        (True, False)], desc='Error size', leave=False):
            bidnerf.base.FLAGS.depth_param = 1. if use_depth else 0.
            bidnerf.base.FLAGS.rgb_param = 1. if use_rgb else 0.
            for error_size in tqdm([0.2, 0.4, 0.8], desc='Error size', leave=False):
                bidnerf.base.FLAGS.perturbation_R = error_size
                bidnerf.base.FLAGS.perturbation_t = error_size + 0.2
                bidnerf.base.FLAGS.test_name = f'{subset}_'\
                                              f'{"_rgb" if use_rgb else ""}' \
                                              f'{"_depth" if use_depth else ""}'\
                                              f'_{int(error_size * 10.001)}_error'

                data.T_init = bidnerf.twist_transformation(data.T_true, rng)
                bidnerf.fit(data, render_fn, rng)

                test_id = bidnerf.save(data, render_fn, rng, "gdr")
                test_ids.append(test_id)
        plot_graphs_in_grid(np.array(test_ids).reshape(-1, 3), grid_name)


def multi_img():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221029)
    bidnerf.base.FLAGS.frame_ids = [2, 3, 4, 5, 6, 7, 8, 9]
    bidnerf.base.FLAGS.coarse_opt = True
    bidnerf.base.FLAGS.perturbation_R = 0.5
    bidnerf.base.FLAGS.perturbation_t = 0.8
    for subset in tqdm(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], desc='Subsets'):
        bidnerf.base.FLAGS.subset = subset
        grid_name = f"{subset}_multi_img"
        render_fn = bidnerf.load_model(rng)
        data = bidnerf.load_data(rng)
        frames = data.frames
        test_ids = []
        for img_count in tqdm([1, 4, 8], desc='Image count', leave=False):
            bidnerf.base.FLAGS.test_name = f'{grid_name}_{img_count}_img'
            data.frames = frames[-img_count:]
            bidnerf.fit(data, render_fn, rng)
            test_id = bidnerf.save(data, render_fn, rng, "gdr")
            test_ids.append(test_id)
        plot_graphs_in_grid(np.array(test_ids).reshape(-1, 3), grid_name)


def inerf_vs_idnerf():
    bidnerf.load_flags()
    rng = jax.random.PRNGKey(20221030)
    bidnerf.base.FLAGS.frame_ids = [2, 3, 4, 5, 6, 7, 8, 9]
    bidnerf.base.FLAGS.dataset = 'blender'
    for subset in tqdm(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"], desc='Subsets'):
        bidnerf.base.FLAGS.subset = subset
        grid_name = f"{subset}_inerf-bidnerf"
        render_fn = bidnerf.load_model(rng)
        data = bidnerf.load_data(rng)
        test_ids = []
        # IDNeRF
        bidnerf.base.FLAGS.coarse_opt = True
        bidnerf.base.FLAGS.depth_param = 1.0
        for error_size in tqdm([0.1, 0.2, 0.4], desc='Error size', leave=False):
            bidnerf.base.FLAGS.perturbation_R = error_size
            bidnerf.base.FLAGS.perturbation_t = error_size + 0.2
            bidnerf.base.FLAGS.test_name = f'{subset}_idnerf_{int(error_size * 10.001)}_error'
            data.T_init = bidnerf.twist_transformation(data.T_true, rng)
            bidnerf.fit(data, render_fn, rng)

            test_id = bidnerf.save(data, render_fn, rng, "gdr")
            test_ids.append(test_id)
        # INeRF
        bidnerf.base.FLAGS.coarse_opt = False
        bidnerf.base.FLAGS.depth_param = 0.0
        data.frames = [data.frames[-1]]
        for error_size in tqdm([0.1, 0.2, 0.4], desc='Error size', leave=False):
            bidnerf.base.FLAGS.perturbation_R = error_size
            bidnerf.base.FLAGS.perturbation_t = error_size + 0.2
            bidnerf.base.FLAGS.test_name = f'{subset}_inerf_{int(error_size * 10.001)}_error'
            data.T_init = bidnerf.twist_transformation(data.T_true, rng)
            bidnerf.fit(data, render_fn, rng)

            test_id = bidnerf.save(data, render_fn, rng, "gdr")
            test_ids.append(test_id)

        plot_graphs_in_grid(np.array(test_ids).reshape(-1, 3), grid_name)


def main(_):
    inerf_vs_idnerf()


if __name__ == "__main__":
    app.run(main)
