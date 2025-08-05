import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random
from functools import partial

import flax.linen as nn
import optax

from sinterp.interpolants import (
    get_interp,
    Interpolator,
    OneSidedLinear)
from sinterp.stochasticfield import OneSidedField
from flow import flow
from bayes.distribution import Gaussian, FlowDistribution
from bayes.posterior import FlowBasedPosterior, PRNGKeyManager

def find_map_from_samples(
    posterior: FlowBasedPosterior,
    key_manager: PRNGKeyManager,
    num_samples: int = 100,
    num_steps: int = 2000,
    learning_rate: float = 1e-3,
):
    """
    Finds the MAP estimate by starting from multiple samples from the posterior
    and performing gradient descent on each independently.
    """
    print("\n--- Finding MAP Estimate via Direct Search from Samples ---")

    # 1. Sample initial points from the posterior to start the search
    initial_xs = posterior.b_sample(key_manager.split(), num_samples)

    # 2. Set up the optimizer
    # We will use Adam, a standard gradient-based optimizer.
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_xs)

    # 3. Define the loss function for a single candidate 'x'.
    # We want to MAXIMIZE posterior.log_prob(x), which is equivalent to
    # MINIMIZING -posterior.log_prob(x).
    def single_loss_fn(x):
        return -posterior.log_prob(x)

    # 4. Vectorize the loss and gradient functions to handle a batch of 'x's.
    # jax.vmap allows us to apply the functions to each sample in the batch independently.
    loss_fn = vmap(single_loss_fn)
    grad_fn = vmap(grad(single_loss_fn))  # x_t+1 = x_t + -\nabla log_prob(x_t)

    # 5. JIT-compile the training step for efficiency
    @jit
    def step(xs, opt_state):
        """
        Performs one step of gradient descent on the batch of samples.
        """
        grads = grad_fn(xs)
        updates, opt_state = optimizer.update(grads, opt_state, xs)
        xs = optax.apply_updates(xs, updates)
        # We calculate the mean loss across the batch for monitoring purposes.
        mean_loss = jnp.mean(loss_fn(xs))
        return xs, opt_state, mean_loss

    # 6. Run the optimization loop
    xs = initial_xs
    for i in range(num_steps):
        xs, opt_state, loss = step(xs, opt_state)
        if i % 200 == 0:
            print(f"Step {i}, Mean Negative Log Posterior: {loss:.4f}")

    # 7. Identify the best sample after optimization.
    # We calculate the final log probability for all optimized samples and select the one
    # with the highest value (which corresponds to the minimum negative log probability).
    final_log_probs = vmap(posterior.log_prob)(xs)
    best_idx = jnp.argmax(final_log_probs)
    x_map = xs[best_idx]
    final_log_prob = final_log_probs[best_idx]

    print("--- MAP Finding Complete ---")
    print(f"Final Log Posterior: {final_log_prob:.4f}")
    print(f"Found MAP estimate x: {x_map}")

    return x_map, final_log_prob