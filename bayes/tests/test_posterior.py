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
    TRUE_THETA = jnp.array([0.25, -0.43])

    def build_total_log_likelihood_and_grad(observations):
        y_data = jnp.array([obs[0] for obs in observations])
        def total_log_likelihood(theta):
            if y_data.shape[0] == 0:
                return 0.0
            else:
                log_likelihoods = jax.scipy.stats.norm.logpdf(y_data, loc=theta, scale=1.0)
                return jnp.sum(jnp.sum(log_likelihoods, axis=-1))
        return jax.jit(total_log_likelihood), jax.jit(jax.grad(total_log_likelihood))

    key_manager = PRNGKeyManager(seed=42)
    interpolator = OneSidedLinear()

    posterior = FlowBasedPosterior(
        dim=DIM,
        key_manager=key_manager,
        interpolator=interpolator,
        build_total_log_likelihood_and_grad=build_total_log_likelihood_and_grad,
        distillation_threshold=10 # Use a smaller threshold for testing
    )

    for i in range(11):
        y_obs = TRUE_THETA.T + jax.random.normal(key_manager.split(), shape=(DIM,))
        posterior.add_observation((y_obs,))

    final_samples = posterior.sample(num_samples=100)
    assert final_samples.shape == (100, DIM)

    final_mean = jnp.mean(final_samples, axis=0)
    # Check that the posterior mean is closer to the true theta than the prior mean (0.0)
    assert jnp.linalg.norm(final_mean - TRUE_THETA) < jnp.linalg.norm(TRUE_THETA)
