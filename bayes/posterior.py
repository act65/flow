import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from functools import partial
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_disable_jit', True)

import flax.linen as nn
import optax

from sinterp.interpolants import (
    get_interp, 
    Interpolator, 
    OneSidedLinear)
from sinterp.stochasticfield import OneSidedField
from sinterp.losses import make_loss_b
from flow import flow
from bayes.distribution import Gaussian, FlowDistribution
from sinterp.couplings import EMDCoupling

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
    depth: int = 4
    width: int = 256
    @nn.compact
    def __call__(self, x, t):
        t = jnp.atleast_1d(t)
        assert x.ndim == 1
        assert t.ndim == 1

        t_emb = nn.Dense(features=32)(t)
        t_emb = nn.relu(t_emb)

        inputs = jnp.concatenate([x, t_emb], axis=-1)

        h = nn.Dense(features=self.width)(inputs)
        for _ in range(self.depth):
            h = nn.relu(h)
            h = nn.Dense(features=self.width)(h)

        out = nn.Dense(features=self.dim)(h)

        return out

class FlowBasedPosterior(FlowDistribution):
    def __init__(self, build_total_log_likelihood_and_grad, dim: int, key_manager: PRNGKeyManager, interpolator: Interpolator, learning_rate: float = 1e-4, distillation_threshold: int = 50, n_steps: int = 50, num_train_steps: int = 1000):

        self.dim = dim
        self.build_total_log_likelihood_and_grad = build_total_log_likelihood_and_grad
        self.key_manager = key_manager
        self.distillation_threshold = distillation_threshold
        self.observation_buffer = []
        self.n_steps = n_steps
        self.interpolator = interpolator
        self.num_train_steps = num_train_steps

        self.coupling = EMDCoupling()

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
            # optax.sgd(learning_rate)
            optax.adam(learning_rate)
        )
        self.opt_state = self.optimizer.init(self.vel_params)

        print(f"Initialized with a one-sided framework and {self.interpolator} interpolator.")

    def add_observation(self, observation):
        self.observation_buffer.append(observation)
        if len(self.observation_buffer) >= self.distillation_threshold:
            self._distill()

    def _get_target_score(self, params, x, t, total_log_likelihood_grad_fn, y_data, guidance_strength=1.0):
        """
        Computes the target score by combining the prior score with likelihood guidance.

        Args:
            params: The model parameters to use (can be live or frozen).
            x: The input position x_t.
            t: The input time t.
            total_log_likelihood_grad_fn: The function to compute the likelihood gradient.
            y_data: The conditioning data for the likelihood.
            guidance_strength: A scalar to control the strength of the guidance.

        Returns:
            The target score vector.
        """
        # Get the score from the base model (the prior)
        s_prior_t = self.model.get_score_s(params, x, t)

        # Get the denoiser E[x₁|xₜ] to estimate the final sample
        eta_1 = self.model.get_denoiser_eta1(params, x, t)

        # Compute the likelihood score using the denoiser
        # we dont have access to likelihoodscore_t, rather we use at t=1
        s_likelihood_1 = total_log_likelihood_grad_fn(eta_1, y_data)

        max_guidance_norm = 10.0

        # Clip the norm of the likelihood score to prevent explosions
        likelihood_norm = jnp.linalg.norm(s_likelihood_1)
        clipped_s_likelihood_1 = s_likelihood_1 * jnp.minimum(1.0, max_guidance_norm / (likelihood_norm + 1e-6))

        # Return the combined, guided score
        # our bayesian update
        return s_prior_t + guidance_strength * clipped_s_likelihood_1

    def get_current_flow(self, params, use_prior=False):
        """
        Constructs the flow object with the appropriate velocity field.
        If observations are present, it uses a guided velocity field.
        """
        if (not self.observation_buffer) or use_prior:
            # Unconditional case: just use the model's learned velocity b
            b_velocity_fn = lambda x, t: self.model.get_velocity_b(params, x, t)
        else:
            # Guided case: construct the guided velocity b_guided
            total_log_likelihood_grad_fn, y_data = self.build_total_log_likelihood_and_grad(self.observation_buffer)

            def b_guided_velocity_fn(x, t):
                # For sampling, we always use the current, live parameters
                
                # Get the deterministic part of the velocity, v(t,x)
                v_velocity = self.model.get_velocity_v(params, x, t)
                
                # Get the full target score s_guided(t,x) using our new helper
                s_target = self._get_target_score(
                    params, x, t, total_log_likelihood_grad_fn, y_data
                )
                
                # Get the interpolator coefficients
                alpha_t = self.interpolator.alpha(t)
                dalpha_dt_t = self.interpolator.dalphadt(t)
                
                # Reconstruct the guided velocity using the fundamental equation:
                # b_guided = v - α(t)α'(t) * s_guided
                return v_velocity - (alpha_t * dalpha_dt_t * s_target)

            b_velocity_fn = b_guided_velocity_fn
            
        f = flow.Flow(b_velocity_fn, self.n_steps)
        return f

    def sample(self, key, x0_samples=None, params=None, use_prior=False):
        if params is None:
            params = self.vel_params
        f = self.get_current_flow(params, use_prior)
        if x0_samples is None:
            x0_samples = self.base_distribution.sample(key)
        x1_samples = f.forward(x0_samples)
        return x1_samples
    
    # @property
    def entropy(self, num_samples):
        # MCMC estimate
        x1_samples = self.b_sample(self.key_manager.split(), num_samples)
        log_p = self.b_log_prob(x1_samples)
        return -jnp.sum(log_p * jnp.exp(log_p))

    def log_prob(self, x1):
        f = self.get_current_flow(self.vel_params)
        x0 = f.backward(x1)
        log_p_x0 = self.base_distribution.log_prob(x0)
        log_p_x1, _ = f.push_forward_log_prob(log_p_x0, x0)
        return log_p_x1
    
    def _distill(self):
        print("\n--- Distilling Likelihood (Unified Stochastic Score-Matching) ---")
        
        if not self.observation_buffer:
            print("Observation buffer is empty. Skipping distillation.")
            return

        total_log_likelihood_grad_fn, y_data = self.build_total_log_likelihood_and_grad(self.observation_buffer)
        
        batch_size = 64
        epsilon = 1e-4
        guidance_strength = 1.0

        # Use frozen parameters for calculating the target
        frozen_params = jax.lax.stop_gradient(self.vel_params)

        # def loss_fn(vel_params, x0_batch, x1_batch, z_batch, t_batch, y_data_arg):
        #     """
        #     We are not doing stochastic interpolation training as usual.
        #     We simply want to force the current score to include the observed data.
        #     To Bayesian update via posterior_score = prior_score + likelihood_score.  
        #     """

        #     # alternative ways to do this.
        #     # step in direction of likelihood. but also say within a trust region?
            
        #     xt_batch = vmap(self.interpolator)(x0_batch, x1_batch, z_batch, t_batch)
            
        #     # The score predicted by the current trainable parameters
        #     pred_score = vmap(self.model.get_score_s, in_axes=(None, 0, 0))(vel_params, xt_batch, t_batch)

        #     # The target score is now calculated cleanly using our helper method
        #     # We use the frozen_params here to define a stable target.
        #     target_score = vmap(self._get_target_score, in_axes=(None, 0, 0, None, None, None))(
        #         frozen_params, xt_batch, t_batch, total_log_likelihood_grad_fn, y_data_arg, guidance_strength
        #     )
            
        #     # The loss is the difference between the predicted score and the (fixed) target score
        #     return jnp.mean((pred_score - jax.lax.stop_gradient(target_score))**2)

        loss_b = make_loss_b(self.interpolator, self.model.get_velocity_v)
        def loss_fn(vel_params, x0_batch, x1_batch, z_batch, t_batch, y_data_arg):
            """
            We can sample from the posterior using our non-parametric estimate.
            We want to match our current prior flow to this non-parametric posterior.
            """
            return jnp.mean(vmap(loss_b, in_axes=(None, 0, 0, 0, 0))(vel_params, x0_batch, x1_batch, z_batch, t_batch))

        @jit
        def train_step(vel_params, opt, x0, x1, z, t, y_data_arg):
            (loss, (g,)) = value_and_grad(loss_fn, argnums=(0,))(vel_params, x0, x1, z, t, y_data_arg)
            g = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1.0, 1.0), g)
            up, opt = self.optimizer.update(g, opt)
            p = optax.apply_updates(vel_params, up)
            return p, opt, loss

        for step in range(self.num_train_steps):
            # ... (batch creation logic) ...
            batch_x0 = jax.random.normal(self.key_manager.split(), shape=(batch_size, self.dim))
            
            # BUG for loss_b we should be sampling from the frozen prior + parametric score
            batch_x1 = vmap(self.sample, in_axes=(None, 0, None, None))(self.key_manager.split(), batch_x0, frozen_params, False)            
            # for score_matching_loss_fn we should be sampling from
            # batch_x1 = vmap(self.sample, in_axes=(None, 0, None, None))(self.key_manager.split(), batch_x0, frozen_params, True)
            # coupling
            # use our own flow to couple the distributions p(x0, x1) = p(x1 | x0)p(x0)!
            # could use OT coupling 
            # batch_x0, batch_x1 = self.coupling(self.key_manager.split(), batch_x0, batch_x1)

            batch_z = jax.random.normal(self.key_manager.split(), shape=(batch_size, self.dim))
            batch_t = jax.random.uniform(self.key_manager.split(), shape=(batch_size,), minval=epsilon, maxval=1.0 - epsilon)

            self.vel_params, self.opt_state, loss = train_step(
                self.vel_params, self.opt_state,
                batch_x0, batch_x1, batch_z, batch_t,
                y_data
            )
            if step % 500 == 0:
                print(f"Distillation Step {step}, Unified Loss: {loss:.4f}")

        self.opt_state = self.optimizer.init(self.vel_params)
        self.observation_buffer = []
        print("Distillation complete.\n---")