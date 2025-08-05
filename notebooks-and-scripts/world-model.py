import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random
from functools import partial
from dataclasses import dataclass
import flax.linen as nn
from jax.tree_util import tree_flatten, tree_unflatten

from bayes.posterior import FlowBasedPosterior, PRNGKeyManager
from sinterp.interpolants import OneSidedLinear

# jax.config.update('jax_disable_jit', True)
jax.config.update("jax_debug_nans", True)

# 1. Define the Ground-Truth RL Environment (Continuous)
STATE_DIM = 2
ACTION_DIM = 1

# --- Define the Neural Network World Model using Flax ---
class WorldModel(nn.Module):
    """A neural network that takes (s, a) and predicts distributions for (s', r)."""
    @nn.compact
    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        
        # --- Output Head 1: Predicts next state distribution ---
        # Output: mean and log_variance for the next state
        s_prime_params = nn.Dense(features=STATE_DIM * 2, name="s_prime_head")(x)
        s_prime_mean, s_prime_log_var = jnp.split(s_prime_params, 2, axis=-1)
        
        # --- Output Head 2: Predicts reward distribution ---
        # Output: mean and log_variance for the reward
        r_params = nn.Dense(features=2, name="r_head")(x)
        r_mean, r_log_var = jnp.split(r_params, 2, axis=-1)
        
        return s_prime_mean, s_prime_log_var, r_mean, r_log_var

# --- Initialize the TRUE world model with fixed random weights ---
key = random.PRNGKey(0)
key, model_key = random.split(key)
true_model = WorldModel()
dummy_s = jnp.zeros((STATE_DIM,))
dummy_a = jnp.zeros((ACTION_DIM,))
true_model_params = true_model.init(model_key, dummy_s, dummy_a)['params']

# 2. Define Parameter Space and Likelihood Function

# --- Utility to handle conversion between flat vector and Flax params ---
flat_params, treedef = tree_flatten(true_model_params)
PARAM_DIM = sum(p.size for p in flat_params)
print(f"Total number of parameters to learn: {PARAM_DIM}")

def vec_to_params(h: jnp.ndarray) -> dict:
    leaves = []
    c = 0
    for param_template in flat_params:
        size = param_template.size
        leaves.append(h[c:c + size].reshape(param_template.shape))
        c += size
    return tree_unflatten(treedef, leaves)

def forward_model_log_likelihood(params: dict, observation: jnp.ndarray):
    """
    Calculates log p(s', r | s, a, params) for a single observation using a
    numerically stable MSE-based formulation.
    
    Note: We calculate the *negative* log-likelihood (a loss) and then return
    its negation, because the posterior object expects to *maximize* the log-likelihood.
    """
    s = observation[:STATE_DIM]
    a = observation[STATE_DIM : STATE_DIM + ACTION_DIM]
    r_obs = observation[STATE_DIM + ACTION_DIM]
    s_prime_obs = observation[STATE_DIM + ACTION_DIM + 1:]

    # Get the predicted distributions from the world model
    s_prime_mean, s_prime_log_var, r_mean, r_log_var = true_model.apply({'params': params}, s, a)
    
    # --- Calculate Negative Log-Likelihood for Transition (s') ---
    # Loss = 0.5 * (log(σ²) + (y - μ)² / σ²)
    #      = 0.5 * (log_var + (y - μ)² * exp(-log_var))
    s_prime_inv_var = jnp.exp(-s_prime_log_var)
    s_prime_mse = (s_prime_obs - s_prime_mean)**2
    s_prime_nll = 0.5 * (s_prime_log_var + s_prime_mse * s_prime_inv_var)
    
    # --- Calculate Negative Log-Likelihood for Reward (r) ---
    r_inv_var = jnp.exp(-r_log_var)
    r_mse = (r_obs - r_mean)**2
    r_nll = 0.5 * (r_log_var + r_mse * r_inv_var)
    
    # The total NLL is the sum over all independent dimensions
    total_nll = jnp.sum(s_prime_nll) + jnp.sum(r_nll)
    
    # Return the log-likelihood (the negative of the loss)
    return -total_nll

def build_total_log_likelihood_and_grad_rl(observations):
    y_data = jnp.array(observations)

    def total_log_likelihood(h, y_data_batch):
        params = vec_to_params(h)
        log_likelihoods = vmap(partial(forward_model_log_likelihood, params))(y_data_batch)
        return jnp.sum(log_likelihoods)

    total_log_likelihood_grad_fn = grad(total_log_likelihood, argnums=0)
    return total_log_likelihood_grad_fn, y_data

# 3. Initialize the Posterior Model
key_manager = PRNGKeyManager(seed=1)
posterior = FlowBasedPosterior(
    build_total_log_likelihood_and_grad=build_total_log_likelihood_and_grad_rl,
    dim=PARAM_DIM,
    key_manager=key_manager,
    interpolator=OneSidedLinear(),
    distillation_threshold=100
)

# 4. Simulate and add observations to the model
print("\n--- Generating and Adding RL Observations ---")
num_observations = 5000
for i in range(num_observations):
    if (i+1) % 1000 == 0:
        print(f"  ... generated {i+1}/{num_observations} observations")
    key, s_key, a_key, noise_key = random.split(key, 4)
    s = random.uniform(s_key, (STATE_DIM,), minval=-1, maxval=1)
    a = random.uniform(a_key, (ACTION_DIM,), minval=-1, maxval=1)
    
    # Get the true mean/log_var for s' and r from the ground-truth model
    s_prime_mean, s_prime_log_var, r_mean, r_log_var = true_model.apply({'params': true_model_params}, s, a)
    
    # Sample from the output distribution
    # std = sqrt(variance) = sqrt(exp(log_var)) = exp(0.5 * log_var)
    s_prime_std = jnp.exp(0.5 * s_prime_log_var)
    r_std = jnp.exp(0.5 * r_log_var)
    
    s_prime_noise, r_noise = random.normal(noise_key, (STATE_DIM,)), random.normal(noise_key)
    s_prime = s_prime_mean + s_prime_noise * s_prime_std
    r = r_mean + r_noise * r_std
    
    observation = jnp.concatenate([s, a, r.flatten(), s_prime])
    posterior.add_observation(observation)

# 5. Get a sample from the learned posterior distribution of parameters
print("\n--- Sampling from the posterior to get learned model parameters ---")
h_map = posterior.sample(key, (1,))[0]

# 6. Verify the result against the true RL model parameters
found_model_params = vec_to_params(h_map)

print("\n--- Verification of Learned World Model ---")
# We compare the predicted means of the true model and the found model.
key, test_s_key, test_a_key = random.split(key, 3)
test_s = random.uniform(test_s_key, (100, STATE_DIM))
test_a = random.uniform(test_a_key, (100, ACTION_DIM))

# Predict using the true model
true_s_prime_mean, _, true_r_mean, _ = vmap(true_model.apply, in_axes=({'params': None}, 0, 0))({'params': true_model_params}, test_s, test_a)

# Predict using the found model
found_s_prime_mean, _, found_r_mean, _ = vmap(true_model.apply, in_axes=({'params': None}, 0, 0))({'params': found_model_params}, test_s, test_a)

# Calculate Mean Squared Error
mse_s_prime = jnp.mean((true_s_prime_mean - found_s_prime_mean)**2)
mse_r = jnp.mean((true_r_mean - found_r_mean)**2)

print(f"Mean Squared Error on next state (s') prediction: {mse_s_prime:.6f}")
print(f"Mean Squared Error on reward (r) prediction:   {mse_r:.6f}")
print("\n(A low MSE indicates the learned model has functionally captured the dynamics of the true model.)")