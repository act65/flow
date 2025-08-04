import pytest
import jax
import jax.numpy as jnp
from bayes.posterior import FlowBasedPosterior, VelocityNet, PRNGKeyManager
from sinterp.interpolants import OneSidedLinear

def test_velocity_net():
    dim = 2
    key = jax.random.PRNGKey(0)
    net = VelocityNet(dim)

    # Test with batch dimension
    x_batch = jnp.ones((10, dim))
    t_batch = jnp.ones((10, 1))
    params = net.init(key, x_batch, t_batch)
    out_batch = net.apply(params, x_batch, t_batch)
    assert out_batch.shape == (10, dim)

    # Test without batch dimension
    x_single = jnp.ones((dim,))
    t_single = jnp.ones((1,))
    out_single = net.apply(params, x_single, t_single)
    assert out_single.shape == (dim,)

def test_flow_based_posterior_end_to_end():
    DIM = 2
    TRUE_THETA = jnp.array([0.25, -0.43]).T

    def build_total_log_likelihood_and_grad(observations):
        """
        Builds a function that computes the gradient of the total log-likelihood
        and returns the data required for that function.
        """
        y_data = jnp.array([obs[0] for obs in observations])

        # 1. Define log-likelihood for a SINGLE observation y_i and a single theta.
        def single_log_likelihood(theta, y_i):
            # theta shape: (2,), y_i shape: (2,)
            return jnp.sum(jax.scipy.stats.norm.logpdf(y_i, loc=theta, scale=1.0))

        # 2. Create a function that computes the gradient of the total log likelihood.
        def total_log_likelihood_grad(theta, y_data_arg):
            # To get the total gradient, we sum the gradients from each data point.
            # We can get the gradients for all data points by vmapping the gradient
            # of the single_log_likelihood function.

            # Get the gradient function for a single observation.
            grad_fn_single = jax.grad(single_log_likelihood, argnums=0)

            # Map this gradient function over all the data points in y_data_arg.
            # in_axes=(None, 0) means:
            #   - Don't map over theta (it's treated as constant for this vmap).
            #   - Map over the first axis of y_data_arg.
            all_grads = jax.vmap(grad_fn_single, in_axes=(None, 0))(theta, y_data_arg)

            # all_grads has shape (num_observations, dim). Sum them to get the total gradient.
            return jnp.sum(all_grads, axis=0)

        # 3. JIT the final gradient function and return it along with the data.
        return jax.jit(total_log_likelihood_grad), y_data
    
    key_manager = PRNGKeyManager(seed=42)
    interpolator = OneSidedLinear()

    posterior = FlowBasedPosterior(
        dim=DIM,
        key_manager=key_manager,
        interpolator=interpolator,
        build_total_log_likelihood_and_grad=build_total_log_likelihood_and_grad,
        distillation_threshold=10 # Use a smaller threshold for testing
    )

    for i in range(21):
        y_obs = TRUE_THETA + jax.random.normal(key_manager.split(), shape=(DIM,))
        posterior.add_observation((y_obs,))

    final_samples = posterior.sample(num_samples=100)
    assert final_samples.shape == (100, DIM)

    final_mean = jnp.mean(final_samples, axis=0)
    # Check that the posterior mean is closer to the true theta than the prior mean (0.0)
    assert jnp.linalg.norm(final_mean - TRUE_THETA) < jnp.linalg.norm(TRUE_THETA)
