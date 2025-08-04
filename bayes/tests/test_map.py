import jax
import jax.numpy as jnp
from bayes.map import ParameterNet, find_map_with_overparameterization
from bayes.posterior import PRNGKeyManager
from bayes.distribution import GaussianMixture

def test_parameter_net():
    dim = 2
    key = jax.random.PRNGKey(0)
    net = ParameterNet(dim=dim)
    params = net.init(key)['params']

    # Check that the output is of the correct shape
    x_candidate = net.apply({'params': params})
    assert x_candidate.shape == (dim,)

def test_find_map_with_gmm():
    dim = 2
    key_manager = PRNGKeyManager(seed=42)

    # 1. Create a GMM with two modes, one with a higher weight
    means = jnp.array([
        [-3.0, -3.0],
        [3.0, 3.0]
    ])
    covs = jnp.array([
        jnp.eye(dim) * 0.1,
        jnp.eye(dim) * 0.1
    ])
    weights = jnp.array([0.3, 0.7])  # Second component has higher weight

    gmm = GaussianMixture(weights=weights, means=means, covs=covs)

    # The true MAP is the mean of the second component
    true_map = means[1]

    # 2. Create a mock posterior whose log_prob is the GMM's log_prob
    class MockPosterior:
        def __init__(self):
            self.dim = dim

        def log_prob(self, x):
            return gmm.log_prob(x)

    mock_posterior = MockPosterior()

    # 3. Run the MAP finding algorithm
    x_map, final_log_prob = find_map_with_overparameterization(
        posterior=mock_posterior,
        key_manager=key_manager,
        num_steps=3000,
        learning_rate=1e-3,
        hidden_dim=128
    )

    # 4. Check that the found MAP is close to the true MAP
    assert jnp.allclose(x_map, true_map, atol=1e-1)
