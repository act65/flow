import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from functools import partial
# jax.config.update("jax_debug_nans", True)

import flax.linen as nn
import optax

from sinterp.interpolants import (
    get_interp, 
    Interpolator, 
    OneSidedLinear)
from sinterp.stochasticfield import OneSidedField
from flow import flow
from bayes.distribution import Gaussian, FlowDistribution

# A simple utility to manage JAX random keys
class PRNGKeyManager:
    def __init__(self, seed):
        self.key = jax.random.PRNGKey(seed)
    def split(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

class VelocityNet(nn.Module):
    """A simple MLP to model a velocity b(x, t)."""
    dim: int
    @nn.compact
    def __call__(self, x, t):
        # x will have shape [batch, dim]
        t = jnp.atleast_1d(t)
        # t will have shape [batch, 1]

        # No need for atleast_1d, t is already batched.
        t_emb = nn.Dense(features=32)(t)
        t_emb = nn.relu(t_emb)

        # Concatenate along the last axis (the feature axis)
        inputs = jnp.concatenate([x, t_emb], axis=-1)

        h = nn.Dense(features=128)(inputs)
        h = nn.relu(h)
        h = nn.Dense(features=128)(h)
        h = nn.relu(h)
        return nn.Dense(features=self.dim)(h)

class FlowBasedPosterior(FlowDistribution):
    def __init__(self, build_total_log_likelihood_and_grad, dim: int, key_manager: PRNGKeyManager, interpolator: Interpolator, learning_rate: float = 1e-4, distillation_threshold: int = 50, n_steps: int = 100):

        self.dim = dim
        self.build_total_log_likelihood_and_grad = build_total_log_likelihood_and_grad
        self.key_manager = key_manager
        self.distillation_threshold = distillation_threshold
        self.observation_buffer = []
        self.n_steps = n_steps
        self.interpolator = interpolator

        vel_nn = VelocityNet(dim=self.dim)

        self.model = OneSidedField(
            interpolator, 
            vel_nn.apply
        )

        self.base_distribution = Gaussian(dim)

        dummy_x = jnp.zeros((self.dim,))
        dummy_t = jnp.zeros((1,))
        self.vel_params = vel_nn.init(key_manager.split(), dummy_x, dummy_t)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate)
            # optax.adam(learning_rate)
        )
        self.opt_state = self.optimizer.init(self.vel_params)

        print(f"Initialized with a one-sided framework and {self.interpolator} interpolator.")

    def add_observation(self, observation):
        self.observation_buffer.append(observation)
        if len(self.observation_buffer) >= self.distillation_threshold:
            self._distill()

    def sample(self, num_samples: int):
        # TODO include non-parametric examples here.
        # use the obs to construct an augmented velocity_fn!
        velocity_fn = lambda x, t: self.model.get_velocity_b(self.vel_params, x, t)
        f = flow.Flow(velocity_fn, self.n_steps)

        x0_samples = self.base_distribution.sample(self.key_manager.split(), shape=(num_samples, self.dim))
        x1_samples = f.b_forward(x0_samples)
        return x1_samples
    
    def log_prob(self, x1):
        velocity_fn = lambda x, t: self.model.get_velocity_b(self.vel_params, x, t)
        f = flow.Flow(velocity_fn, self.n_steps)

        x0 = f.b_backward(x1)
        log_p_x0 = self.base_distribution.log_p(x0)
        log_p_x1 = f.b_push_forward_log_prob(log_p_x0, x0)

        return log_p_x1
    
    def entropy(self, num_samples: int):
        velocity_fn = lambda x, t: self.model.get_velocity_b(self.vel_params, x, t)
        f = flow.Flow(velocity_fn, self.n_steps)

        # MCMC estimate
        x0_samples = jax.random.normal(self.key_manager.split(), shape=(num_samples, self.dim))
        log_p_x0_samples = jax.scipy.stats.norm.logpdf(x0_samples)
        log_p_x1_samples =f.push_forward_logp(log_p_x0_samples)
        return -jnp.sum(log_p_x1_samples * jnp.exp(log_p_x1_samples))

    def _distill(self):
        print("\n--- Distilling Likelihood (Unified Stochastic Score-Matching) ---")
        
        if not self.observation_buffer:
            print("Observation buffer is empty. Skipping distillation.")
            return

        _, total_log_likelihood_grad_fn = self.build_total_log_likelihood_and_grad(self.observation_buffer)
        
        # --- Hyperparameters ---
        num_train_steps = 1000
        batch_size = 64
        epsilon = 1e-4
        clip_value = 1.0
        guidance_strength = 1.0

        # Freeze the model parameters at the start of distillation to define the "prior"
        frozen_params = jax.lax.stop_gradient(self.vel_params)

        def loss_fn(vel_params, x0_batch, x1_batch, z_batch, t_batch):
            xt_batch = vmap(self.interpolator)(x0_batch, x1_batch, z_batch, t_batch)
            
            prior_eta1 = vmap(self.model.get_denoiser_eta1, in_axes=(None, 0, 0))(frozen_params, xt_batch, t_batch)
            prior_score = vmap(self.model.get_score_s, in_axes=(None, 0, 0))(frozen_params, xt_batch, t_batch)

            pred_score = vmap(self.model.get_score_s, in_axes=(None, 0, 0))(vel_params, xt_batch, t_batch)

            # we dont have access to \nabla_xt log p(xt)
            # instead use \nabla_x1 log p(x1)
            likelihood_score = vmap(total_log_likelihood_grad_fn)(prior_eta1)
            
            # our Bayesian update.
            target_score = prior_score + guidance_strength * likelihood_score
            
            # sum not mean!
            # target_score = prior_score + sum_i d/dx_i logp(x_i | \theta)
            return jnp.sum(( pred_score - jax.lax.stop_gradient(target_score))**2)

        @jit
        def train_step(vel_params, opt, x0, x1, z, t):
            (loss, (g,)) = value_and_grad(loss_fn, argnums=(0,))(vel_params, x0, x1, z, t)

            g = jax.tree_util.tree_map(lambda x: jnp.clip(x, -clip_value, clip_value), g)

            up, opt = self.optimizer.update(g, opt)
            p = optax.apply_updates(vel_params, up)

            return p, opt, loss

        # Training loop
        x1_samples_for_training = self.sample(num_samples=1024)
        for step in range(num_train_steps):
            idx = jax.random.randint(self.key_manager.split(), (batch_size,), 0, x1_samples_for_training.shape[0])
            batch_x1 = x1_samples_for_training[idx]
            batch_x0 = jax.random.normal(self.key_manager.split(), shape=(batch_size, self.dim))
            batch_z = jax.random.normal(self.key_manager.split(), shape=(batch_size, self.dim))
            batch_t = jax.random.uniform(self.key_manager.split(), shape=(batch_size,), minval=epsilon, maxval=1.0 - epsilon)

            self.vel_params, self.opt_state, loss = train_step(
                self.vel_params, self.opt_state,
                batch_x0, batch_x1, batch_z, batch_t
            )
            if step % 500 == 0:
                print(f"Distillation Step {step}, Unified Loss: {loss:.4f}")

        # reset the momentum after each round. don't want it to cross over.
        # TODO momentum should work here!!
        self.opt_state = self.optimizer.init(self.vel_params)
        self.observation_buffer = []
        print("Distillation complete.")
        print("---\n")

if __name__ == '__main__':
    # TODO add more examples!
    """
    A test problem. We start with some prior over theta p(\theta).
    We recieve observations and want to update to form a posterior.

    p(\theta | D) = p(D | \theta)p(\theta) / P(D)

    In this case, p(D | \theta) is simply a gaussian.
    Data samples are generated from \mathcal N(\theta, 1).
    """

    DIM = 2
    TRUE_THETA = jnp.array([0.25, -0.43])

    def build_total_log_likelihood_and_grad(observations):
        """
        Builds functions that compute the total log-likelihood and its gradient.
        """
        y_data = jnp.array([obs[0] for obs in observations])

        def total_log_likelihood(theta):
            if y_data.shape[0] == 0:
                return 0.0
            else:
                # logpdf output shape: [num_obs, dim]
                log_likelihoods = jax.scipy.stats.norm.logpdf(y_data, loc=theta, scale=1.0)
                # Sum over dimensions for each observation, then mean over all observations.
                # TODO careful. actually want to sum!
                return jnp.sum(jnp.sum(log_likelihoods, axis=-1))

        return jit(total_log_likelihood), jit(grad(total_log_likelihood))

    # Setup
    key_manager = PRNGKeyManager(seed=42)
    
    # Let's choose a good stochastic one for this problem
    interpolator = OneSidedLinear()
    
    posterior = FlowBasedPosterior(
        dim=DIM, 
        key_manager=key_manager, 
        interpolator=interpolator,
        distillation_threshold=25
    )

    for i in range(1000+1):
        y_obs = TRUE_THETA.T + jax.random.normal(key_manager.split(), shape=(DIM,))
        posterior.add_observation((y_obs,))

        if i % 25 == 0:
            final_samples = posterior.sample(num_samples=2000)

            final_mean = jnp.mean(final_samples, axis=0)
            final_std = jnp.std(final_samples, axis=0)

            print("\n--- Final Result ---")
            print(f"True Theta:      {TRUE_THETA}")
            print(f"Final Post. Mean: {final_mean.round(3)}")
            print(f"Final Post. Std:  {final_std.round(3)}")