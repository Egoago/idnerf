import functools
from copy import deepcopy

import jax
import jax.numpy as jnp
import jaxlie
import optax
from tqdm import tqdm

from bidnerf import base, math, rendering, sampling


def get_optimizer(scheduler):
    optimizer = base.FLAGS.optimizer
    if optimizer == "sgd":
        return optax.sgd(scheduler, 0.4, False)
    elif optimizer == "adam":
        return optax.adam(scheduler)
    else:
        raise NotImplementedError()


@jax.custom_vjp
def clip_gradient(x, limit):
    return x


def clip_gradient_fwd(x, limit):
    return x, limit


def clip_gradient_bwd(g, limit):
    return jnp.clip(g, -limit, limit), None


clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)


@functools.partial(jax.jit, static_argnums=(3, 6))
def optimization_step(params, opt_state, rng, render_fn, rays_relative_to_base, rgbd_pixels, optimizer):
    def loss(_params):
        T_pred = _params['T_pred']
        rays = math.transform_rays(rays_relative_to_base, T_pred)
        #if base.FLAGS.dataset == 'llff':
         #   rays = math.convert_to_ndc(rays, cam_params)
        rgb, depth = rendering.render_rays(render_fn, rays, rng, base.FLAGS.coarse_opt)
        #   TODO add depth
        #   TODO per_sample gradient
        mask = (depth > base.FLAGS.near) * (depth < base.FLAGS.far)
        _sample_count = jnp.count_nonzero(mask)
        pixel_loss = optax.huber_loss(rgb,
                                      rgbd_pixels[:, :3],
                                      base.FLAGS.huber_delta).mean(axis=-1)
        if base.FLAGS.depth_param > 0.:
            pixel_loss = base.FLAGS.rgb_param*pixel_loss + base.FLAGS.depth_param * optax.huber_loss(depth,
                                                                                 rgbd_pixels[:, 3],
                                                                                 base.FLAGS.huber_delta)
        pixel_loss = pixel_loss * mask
        if base.FLAGS.clip_grad > 0.:
            pixel_loss = clip_gradient(pixel_loss, base.FLAGS.clip_grad)
        return pixel_loss.sum() / mask.sum(), _sample_count

    (loss, sample_count), grads = jaxlie.manifold.value_and_grad(loss, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = jaxlie.manifold.rplus(params, updates)
    return new_params, new_opt_state, loss, grads, sample_count


def fit(data: base.Data, render_fn, rng):
    params = {'T_pred': deepcopy(data.T_init)}
    scheduler = optax.exponential_decay(base.FLAGS.lr_init,
                                        base.FLAGS.decay_steps,
                                        base.FLAGS.decay_rate)
    optimizer = get_optimizer(scheduler)
    opt_state = optimizer.init(jaxlie.manifold.zero_tangents(params))

    history = base.History()
    sampler = sampling.Sampler(data)

    print(f'|{"step":^5s}|{"loss":^7s}|{"t error":^7s}|{"R error":^7s}|{"Samples":^7s}|{"grads":^49s}|')
    rays_relative_to_base, rgbd_pixels = sampler.sample(rng)
    for i in tqdm(range(base.FLAGS.max_steps), desc='Optimalization', unit='step', leave=False):
        rng, subkey_1, subkey_2 = jax.random.split(rng, 3)
        params, opt_state, loss, grads, sample_count = optimization_step(params, opt_state, subkey_2,
                                                                         render_fn, rays_relative_to_base, rgbd_pixels,
                                                                         optimizer)

        T_pred = params['T_pred']
        t_error, R_error = math.compute_errors(data.T_true, T_pred)
        sample_count = sample_count.tolist()

        history.loss.append(loss.tolist())
        history.grads.append(grads["T_pred"].tolist())
        history.log_T_pred.append(params["T_pred"].log().tolist())
        history.t_error.append(t_error)
        history.R_error.append(R_error)
        history.sample_count.append(sample_count)

        if base.FLAGS.resample_rays:
            rays_relative_to_base, rgbd_pixels = sampler.sample(rng)

        if i % base.FLAGS.print_every == 0:
            tqdm.write(f'|{i:5d}|{loss:7.4f}|{t_error:7.4f}|{R_error:7.4f}|{sample_count:7d}|{grads["T_pred"]}|')
    data.T_final = params['T_pred']
    data.history = history
