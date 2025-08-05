import jax
import jax.numpy as jnp
from bayes.map import find_map_from_samples # Updated import
from bayes.posterior import PRNGKeyManager
from bayes.distribution import GaussianMixture

def test_find_map_from_samples_with_gmm():
    """
    Tests if the direct search method can find the correct mode of a
    Gaussian Mixture Model.
    """
    dim = 2
    key_manager = PRNGKeyManager(seed=0)

    # 1. Create a GMM with two modes. The second mode has a higher weight,
    # making its mean the true Maximum a Posteriori (MAP) estimate.
    means = jnp.array([
        [-3.0, -3.0],
        [3.0, 3.0]
    ])
    covs = jnp.array([
        jnp.eye(dim) * 0.2,
        jnp.eye(dim) * 0.2
    ])
    weights = jnp.array([0.3, 0.7])  # Second component has a higher probability

    gmm = GaussianMixture(weights=weights, means=means, covs=covs)

    # The true MAP is the mean of the component with the highest weight.
    true_map = means[1]

    # 3. Run the MAP finding algorithm. We provide multiple starting points
    #    by sampling from the posterior.
    x_map, final_log_prob = find_map_from_samples(
        posterior=gmm,
        key_manager=key_manager,
        num_samples=100,  # Start from 100 different points
        num_steps=1000,
        learning_rate=1e-2,
    )

    # 4. Check that the found MAP is close to the true MAP.
    #    With enough samples, at least one should start in the correct
    #    basin of attraction and converge to the global optimum.
    print(f"True MAP: {true_map}")
    print(f"Found MAP: {x_map}")
    assert jnp.allclose(x_map, true_map, atol=1e-1)