import pytest
import jax
import jax.numpy as jnp
from bayes.distribution import Gaussian, FlowDistribution, ProcessDistribution
from flow.flow import Flow

key = jax.random.PRNGKey(0)

def test_gaussian():
    dim = 2
    dist = Gaussian(dim)
    samples = dist.sample(key, shape=(10,))
    assert samples.shape == (10, dim)
    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (10,)
    entropy = dist.entropy
    assert isinstance(entropy.item(), float)

def test_flow_distribution():
    dim = 2
    base_dist = Gaussian(dim)

    # A simple identity flow
    def v(x, t, *args):
        return jnp.zeros_like(x)

    flow = Flow(v, n_steps=10)

    dist = FlowDistribution(flow, base_dist)

    samples = dist.sample(key, shape=(10,))
    assert samples.shape == (10, dim)

    log_probs = dist.log_prob(samples)
    assert log_probs.shape == (10,)

    # For an identity flow, the log_prob should be the same as the base distribution's
    base_log_probs = base_dist.log_prob(samples)
    assert jnp.allclose(log_probs, base_log_probs, atol=1e-2)

def test_process_distribution():
    dim = 2
    base_dist = Gaussian(dim)

    # Dummy process
    class MockProcess:
        def b_forward(self, x0, keys):
            return x0, None

    process = MockProcess()
    dist = ProcessDistribution(process, base_dist)

    samples = dist.sample(key, shape=(10,))
    assert samples.shape == (10, dim)

    with pytest.raises(NotImplementedError):
        dist.log_prob(samples)

    with pytest.raises(NotImplementedError):
        dist.entropy(key)
