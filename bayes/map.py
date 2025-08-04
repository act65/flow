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

class ParameterNet(nn.Module):
    """
    An MLP that starts from a trainable vector instead of an external input.
    
    This is useful when the goal is to optimize a fixed, learned starting point
    that gets transformed by a neural network.
    """
    # NOTE what's the other name for this? an implicit representation?
    # this isnt quite the same thing tho
    dim: int
    hidden_dim: int = 512
    depth: int = 6

    # The setup method is where we define our trainable parameters.
    def setup(self):
        # We declare a new trainable parameter using self.param().
        # The parameter is named 'trainable_z' and is initialized with a
        # normal distribution. Its shape is (self.hidden_dim,).
        self.trainable_z = self.param(
            'trainable_z',
            jax.nn.initializers.normal(stddev=1.0),
            (self.hidden_dim,)
        )

    # The __call__ method now takes no external input 'z'.
    # It directly uses the 'trainable_z' parameter defined in the setup method.
    @nn.compact
    def __call__(self):
        # The trainable vector 'trainable_z' becomes the input to the first layer.
        x = nn.Dense(features=self.hidden_dim)(self.trainable_z)
        x = nn.relu(x)
        for _ in range(self.depth):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.dim)(x)
        return x

def find_map_with_overparameterization(
    posterior: FlowBasedPosterior,
    key_manager: PRNGKeyManager,
    num_steps: int = 2000,
    learning_rate: float = 1e-3,
    hidden_dim: int = 256
):
    """
    Finds the MAP estimate of a posterior by overparameterizing the search
    variable x with a neural network and optimizing its parameters.
    """
    print("\n--- Finding MAP Estimate via Overparameterization ---")
    dim = posterior.dim

    # 1. Define the reparameterizing network and its parameters
    param_net = ParameterNet(dim=dim, hidden_dim=hidden_dim)
    param_net_params = param_net.init(key_manager.split())['params']

    # 2. Set up the optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )

    opt_state = optimizer.init(param_net_params)

    # 3. Define the loss function to be minimized
    # We want to MAXIMIZE posterior.log_prob(x), which is equivalent to
    # MINIMIZING -posterior.log_prob(x), where x = NN(theta).
    def loss_fn(params):
        # Generate the candidate x from the network
        x_candidate = param_net.apply({'params': params})
        x_candidate = jnp.squeeze(x_candidate)
        return -posterior.log_prob(x_candidate)

    # 4. JIT-compile the training step for efficiency
    @jit
    def step(params, opt_state):
        # NOTE do we need stochasticity in the optimisation!!?
        loss_value, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # 5. Run the optimization loop
    for i in range(num_steps):
        param_net_params, opt_state, loss = step(param_net_params, opt_state)
        if i % 200 == 0:
            print(f"Step {i}, Negative Log Posterior: {loss:.4f}")

    # 6. Get the final MAP estimate by applying the optimized parameters
    x_map = param_net.apply({'params': param_net_params})
    x_map = jnp.squeeze(x_map)
    final_log_prob = posterior.log_prob(x_map)

    print("--- MAP Finding Complete ---")
    print(f"Final Log Posterior: {final_log_prob:.4f}")
    print(f"Found MAP estimate x: {x_map}")

    return x_map, final_log_prob