import functools
from copy import deepcopy

import jax
import jaxlie
import optax
from absl import flags
from tqdm import tqdm

from idnerf import base, math, rendering, sampling


def get_optimizer(scheduler):
    optimizer = flags.FLAGS.optimizer
    if optimizer == "sgd":
        return optax.sgd(scheduler, 0.4, False)
    elif optimizer == "adam":
        return optax.adam(scheduler)
    else:
        raise NotImplementedError()


@functools.partial(jax.jit, static_argnums=(3, 6))
def optimization_step(params, opt_state, rng, render_fn, rays_relative_to_base, rgbd_pixels, optimizer):
    def loss(_params):
        T_pred = _params['T_pred']
        rays = math.transform_rays(rays_relative_to_base, T_pred)
        rgb, depth = rendering.render_rays(render_fn, rays, rng)
        mask = (depth > flags.FLAGS.near) * (depth < flags.FLAGS.far)
        #   TODO add depth
        #   TODO per_sample gradient for clipping
        #   TODO add sample counting back

        loss_rgb = (optax.huber_loss(rgb, rgbd_pixels[:, :3], flags.FLAGS.huber_delta) * mask[:, None]).mean()
        return loss_rgb

    loss, grads = jaxlie.manifold.value_and_grad(loss)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = jaxlie.manifold.rplus(params, updates)
    return new_params, new_opt_state, loss, grads


def fit(data: base.Data, render_fn, rng):
    params = {'T_pred': deepcopy(data.T_init)}
    scheduler = optax.exponential_decay(flags.FLAGS.lr_init, flags.FLAGS.decay_steps, flags.FLAGS.decay_rate)
    optimizer = get_optimizer(scheduler)
    opt_state = optimizer.init(jaxlie.manifold.zero_tangents(params))

    history = base.History()
    sampler = sampling.Sampler(data)

    print(f'|{"step":^5s}|{"loss":^7s}|{"t error":^7s}|{"R error":^7s}|{"grads":^49s}|')
    for i in tqdm(range(flags.FLAGS.max_steps)):
        rng, subkey_1, subkey_2 = jax.random.split(rng, 3)
        rays_relative_to_base, rgbd_pixels = sampler.sample(subkey_1)
        params, opt_state, loss, grads = optimization_step(params, opt_state, subkey_2,
                                                           render_fn, rays_relative_to_base, rgbd_pixels, optimizer)

        T_pred = params['T_pred']
        t_error, R_error = math.compute_errors(data.T_true, T_pred)

        history.loss.append(loss.tolist())
        history.grads.append(grads["T_pred"].tolist())
        history.log_T_pred.append(params["T_pred"].log().tolist())
        history.t_error.append(t_error)
        history.R_error.append(R_error)
        if i % flags.FLAGS.print_every == 0:
            tqdm.write(f'|{i:5d}|{loss:7.4f}|{t_error:7.4f}|{R_error:7.4f}|{grads["T_pred"]}|')
    data.T_final = params['T_pred']
    data.history = history
