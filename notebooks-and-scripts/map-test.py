import jax
import jax.numpy as jnp
import fire
from bayes.map import find_map_from_samples
from bayes.posterior import PRNGKeyManager
from bayes.distribution import GaussianMixture

import itertools

def generate_random_gmm(key_manager: PRNGKeyManager, dim: int, num_modes: int):
    """
    Generates a random Gaussian Mixture Model (GMM) with a single dominant mode.

    Args:
        key_manager: A PRNGKeyManager to handle JAX random keys.
        dim: The dimensionality of the GMM.
        num_modes: The number of mixture components.

    Returns:
        A tuple containing:
        - gmm (GaussianMixture): The generated GMM instance.
        - true_map (jnp.ndarray): The mean of the dominant mode, which is the true MAP.
    """
    print(f"\n--- Generating Random GMM (dim={dim}, modes={num_modes}) ---")
    # 1. Generate random means for each mode, spread out in the space
    means = jax.random.normal(key_manager.split(), (num_modes, dim)) * 5

    # 2. Generate random covariance matrices (diagonal for simplicity)
    # We ensure the diagonal elements are positive.
    # cov_diagonals = jax.random.uniform(key_manager.split(), (num_modes, dim)) * 0.4 + 0.1
    # covs = jnp.array([jnp.diag(d) for d in cov_diagonals])
    covs = jnp.array([jnp.eye(dim) for m in range(num_modes)])

    # 3. Generate weights with one dominant mode
    # This ensures a clear, single MAP for testing purposes.
    dominant_mode_idx = jax.random.randint(key_manager.split(), (), 0, num_modes)
    # Start with small random weights
    weights = jax.random.uniform(key_manager.split(), (num_modes,))
    # Make one weight significantly larger
    # weights = weights.at[dominant_mode_idx].add(num_modes * 2.0)
    # Normalize to sum to 1
    weights = weights / weights.sum()

    print(f"Dominant mode is at index: {dominant_mode_idx}")
    print(f"GMM Weights: {weights}")

    # 4. The true MAP is the mean of the component with the highest weight
    true_map = means[dominant_mode_idx]

    # 5. Create the GMM instance
    gmm = GaussianMixture(weights=weights, means=means, covs=covs)

    return gmm, true_map


def rum_experiment(seed, dim, num_modes, num_samples, num_steps, learning_rate):
    """
    Main function to run the MAP finding experiment.
    """
    key_manager = PRNGKeyManager(seed=seed)

    # 1. Generate a random GMM to serve as the posterior distribution
    # The GMM object itself has the required .log_prob() and .sample() methods.
    gmm_posterior, true_map = generate_random_gmm(
        key_manager=key_manager,
        dim=dim,
        num_modes=num_modes
    )

    # 2. Run the MAP finding algorithm
    # We start from multiple points sampled from the GMM posterior.
    x_map, final_log_prob = find_map_from_samples(
        posterior=gmm_posterior,
        key_manager=key_manager,
        num_samples=num_samples,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )

    # 3. Check if the found MAP is close to the true MAP and report results
    print("\n--- Verification ---")
    print(f"True MAP:      {true_map}")
    print(f"Found MAP:     {x_map}")
    print(f"Final Log Prob:  {final_log_prob:.4f}")

    # With enough samples, one should start in the correct basin of attraction
    # and converge to the global optimum.
    if jnp.allclose(x_map, true_map, atol=1e-1):
        print("\n✅ SUCCESS: Found MAP is close to the true MAP.")
    else:
        print("\n❌ FAILURE: Found MAP is NOT close to the true MAP.")

    if dim == 1:
        import matplotlib.pyplot as plt

        x = jnp.linspace(-10, 10, 200)
        px = jnp.exp(gmm_posterior.b_log_prob(x))
        plt.plot(x, px)
        plt.scatter(x_map, jnp.exp(gmm_posterior.log_prob(x_map)))
        plt.show()


def main():
    modes = [5, 10, 20]
    seeds = [0, 1, 2]
    dims = [1]
    for m, s, d in itertools.product(modes, seeds, dims):
        rum_experiment(
            seed=s, 
            dim=d, 
            num_modes=m, 
            num_samples=64, 
            num_steps=1000, 
            learning_rate=1e-4
        )


if __name__ == "__main__":
    fire.Fire(main)