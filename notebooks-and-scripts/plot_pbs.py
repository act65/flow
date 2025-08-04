import jax.numpy as jnp
from jax import random, vmap, jit
from sinterp.exact_si_gmm import construct_p_b_s
from sinterp.interpolants import get_interp
from sinterp.utils import wrap_plot, plot_sf, plot_vf, plot_trajectories, get_v, plot_sf_and_mode_trajectory
from bayes.distribution import GaussianMixture

import matplotlib.pyplot as plt
import os
from functools import partial

import fire


def create_gmm(key, n_components, dim, std):
    keys = random.split(key, 3)
    means = random.uniform(keys[0], shape=(n_components, dim), minval=-5, maxval=5)
    weights = random.uniform(keys[1], shape=(n_components,))
    weights /= jnp.sum(weights)
    covs = jnp.array([jnp.eye(dim) * std**2 for _ in range(n_components)])
    return GaussianMixture(weights, means, covs)

def main(savedir, interp_name):
    m = 3  # m modes in dist 0
    n = 2  # n modes in dist 1
    key = random.PRNGKey(0)
    keys = random.split(key, 2)
    px = create_gmm(keys[0], m, 1, 0.1)
    py = create_gmm(keys[1], n, 1, 0.1)

    # Monkey-patch the b_p and sample methods
    px.b_p = lambda x: jnp.exp(px.b_log_prob(x))
    py.b_p = lambda x: jnp.exp(py.b_log_prob(x))

    mxs, cxs, wxs = list(px.means), list(px.covs), list(px.weights)
    mys, cys, wys = list(py.means), list(py.covs), list(py.weights)

    A = -5
    B = 5
    N = 200
    k = 30

    interp = get_interp(interp_name)
    
    p, b, s = construct_p_b_s(interp, mxs, mys, cxs, cys, wxs, wys)
    v = get_v(b, s, interp)

    p = jit(vmap(vmap(p, in_axes=(None, 0)), in_axes=(0, None)))
    b = jit(vmap(vmap(b, in_axes=(None, 0)), in_axes=(0, None)))
    s = jit(vmap(vmap(s, in_axes=(None, 0)), in_axes=(0, None)))
    v = jit(vmap(vmap(v, in_axes=(None, 0)), in_axes=(0, None)))

    plot_fn = partial(plot_sf, p=p)
    wrap_plot(plot_fn, A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'p_{interp_name}-{m}-{n}.png'))
    
    wrap_plot(partial(plot_vf, k=k, v=b), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'b_{interp_name}-{m}-{n}.png'))
    wrap_plot(partial(plot_vf, k=k, v=s), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f's_{interp_name}-{m}-{n}.png'))
    wrap_plot(partial(plot_vf, k=k, v=v), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'v_{interp_name}-{m}-{n}.png'))

    key = random.PRNGKey(0)
    wrap_plot(partial(plot_trajectories, n=k*10, px=px, py=py, key=key, interp=interp), A, B, N, px, py)
    plt.savefig(os.path.join(savedir, f'trajectories_{interp_name}-{m}-{n}.png'))

if __name__ == '__main__':
    fire.Fire(main)