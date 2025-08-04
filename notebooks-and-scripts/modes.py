import jax.numpy as jnp
from jax import random, vmap, jit
from sinterp.exact_si_gmm import construct_p_b_s
from sinterp.interpolants import get_interp
from sinterp.utils import wrap_plot, plot_sf, plot_vf, plot_trajectories, get_v
from bayes.distribution import GaussianMixture
import matplotlib.pyplot as plt
import os
from functools import partial
import fire

# --- Plotting Utilities (including the new one) ---

def plot_sf_and_mode_trajectory(z, t, a, b, p, v):
    """
    Plots the scalar field p(z, t) and overlays the trajectory of the mode.
    """
    # First, plot the background probability density heatmap
    plot_sf(z, t, a, b, p)

    # --- Calculate the mode trajectory ---
    # 1. Find the mode of the initial distribution p(z, t=0)
    z_fine_grid = jnp.linspace(a, b, 4000)
    p_at_t0 = p(z_fine_grid[:, None], jnp.array([0.0])).flatten()
    initial_mode_idx = jnp.argmax(p_at_t0)
    initial_mode = z_fine_grid[initial_mode_idx]

    # 2. Numerically integrate the velocity field v(z, t)
    dt = t[1] - t[0]
    mode_path = [initial_mode]
    current_z = initial_mode

    for i in range(len(t) - 1):
        current_t = t[i]
        # The vmapped 'v' expects inputs like (1, 1) and returns (1, 1, 1)
        velocity = v(jnp.array([[current_z]]), jnp.array([[current_t]]))[0, 0, 0]
        current_z = current_z + velocity * dt
        mode_path.append(current_z)

    # 3. Plot the calculated trajectory
    plt.plot(mode_path, t, color='white', linestyle='--', linewidth=2.0, label='Mode Trajectory')
    plt.legend()


def main(savedir=".", interp_name="LinearDeterministic"):
    """
    Runs an experiment to visualize the flow from a single Gaussian
    to a bimodal Gaussian Mixture Model.
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # --- Define p_0: A single Gaussian ---
    # CORRECTED: Parameters are now jax.numpy.ndarrays
    p0_means = jnp.array([[0.0]])      # Shape: (n_components, dim) -> (1, 1)
    p0_covs = jnp.array([[[1.0]]])     # Shape: (n_components, dim, dim) -> (1, 1, 1)
    p0_weights = jnp.array([1.0])      # Shape: (n_components,) -> (1,)
    py = GaussianMixture(p0_weights, p0_means, p0_covs)

    # --- Define p_1: A Gaussian Mixture Model ---
    # CORRECTED: Parameters are now jax.numpy.ndarrays
    p1_means = jnp.array([[-6.0], [4.0]])  # Shape: (2, 1)
    p1_covs = jnp.array([[[0.0001]], [[2.0]]]) # Shape: (2, 1, 1)
    p1_weights = jnp.array([0.2, 0.8])     # Shape: (2,)
    px = GaussianMixture(p1_weights, p1_means, p1_covs)


    # --- Monkey-patch helper methods for plotting ---
    px.b_p = lambda x: jnp.exp(px.b_log_prob(x))
    py.b_p = lambda x: jnp.exp(py.b_log_prob(x))
    px.sample = lambda key, n: vmap(px.sample)(random.split(key, n))
    py.sample = lambda key, n: vmap(py.sample)(random.split(key, n))

    # --- Extract parameters as lists for construct_p_b_s ---
    # This is necessary because construct_p_b_s uses list comprehensions
    mxs, cxs, wxs = list(px.means), list(px.covs), list(px.weights)
    mys, cys, wys = list(py.means), list(py.covs), list(py.weights)

    # --- Set up plot and simulation parameters ---
    A = -10
    B = 10
    N = 200

    # --- Get the interpolant and construct the flow functions ---
    interp = get_interp(interp_name)
    p, b, s = construct_p_b_s(interp, mxs, mys, cxs, cys, wxs, wys)
    v = get_v(b, s, interp)

    # --- JIT-compile the functions for performance ---
    p_jit = jit(vmap(vmap(p, in_axes=(None, 0)), in_axes=(0, None)))
    v_jit = jit(vmap(vmap(v, in_axes=(None, 0)), in_axes=(0, None)))

    # --- Generate and save the plot ---
    print(f"Generating plot for interpolant: {interp_name}...")
    
    plot_fn = partial(plot_sf_and_mode_trajectory, p=p_jit, v=v_jit)
    
    wrap_plot(plot_fn, A, B, N, px, py)
    
    filename = f'mode_trajectory_{interp_name}.png'
    filepath = os.path.join(savedir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Plot saved to {filepath}")
    plt.close()


if __name__ == '__main__':
    fire.Fire(main)