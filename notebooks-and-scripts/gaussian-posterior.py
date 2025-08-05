import fire
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from bayes.posterior import FlowBasedPosterior, VelocityNet, PRNGKeyManager
from sinterp.interpolants import OneSidedLinear
from bayes.distribution import Gaussian, FlowDistribution
from flow import flow

def run_simulation(dim, distillation_threshold):
    """
    Runs the Bayesian inference simulation, and plots the MSE and posterior variance.
    """
    key_manager = PRNGKeyManager(seed=0)

    DIM = dim
    TRUE_THETA = jax.random.normal(key_manager.split(), (dim,)).T #jnp.array([10.2, -6.7]).T

    def build_total_log_likelihood_and_grad(observations):
        """
        Builds a function that computes the gradient of the total log-likelihood
        and returns the data required for that function.
        """
        y_data = jnp.array([obs[0] for obs in observations])

        def single_log_likelihood(theta, y_i):
            return jnp.sum(jax.scipy.stats.norm.logpdf(y_i, loc=theta, scale=1.0))

        def total_log_likelihood_grad(theta, y_data_arg):
            grad_fn_single = jax.grad(single_log_likelihood, argnums=0)
            all_grads = jax.vmap(grad_fn_single, in_axes=(None, 0))(theta, y_data_arg)
            return jnp.sum(all_grads, axis=0)

        return jax.jit(total_log_likelihood_grad), y_data


    interpolator = OneSidedLinear()

    posterior = FlowBasedPosterior(
        dim=DIM,
        key_manager=key_manager,
        interpolator=interpolator,
        build_total_log_likelihood_and_grad=build_total_log_likelihood_and_grad,
        distillation_threshold=distillation_threshold,
        num_train_steps=200
    )

    mses = []
    variances = []
    observations_count = []

    for i in range(21):
        y_obs = TRUE_THETA + jax.random.normal(key_manager.split(), shape=(DIM,))
        posterior.add_observation((y_obs,))

        # Sample from the posterior to calculate metrics
        samples = posterior.b_sample(key_manager.split(), 1000)
        
        # Calculate posterior mean and variance
        posterior_mean = jnp.mean(samples, axis=0)
        posterior_variance = jnp.var(samples, axis=0)

        # Calculate MSE
        mse = jnp.mean((posterior_mean - TRUE_THETA)**2)
        
        mses.append(mse)
        variances.append(jnp.mean(posterior_variance)) #-V-
        observations_count.append(i + 1)

    return mses, variances, observations_count

def plot(mses, variances, observations_count):
    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.plot(observations_count, mses, marker='o')
    plt.xlabel("Number of Observations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE of Posterior Mean vs. Number of Observations")
    plt.grid(True)

    # Plot Variance
    plt.subplot(1, 2, 2)
    plt.plot(observations_count, variances, marker='o')
    plt.xlabel("Number of Observations")
    plt.ylabel("Mean Posterior Variance")
    plt.title("Posterior Variance vs. Number of Observations")
    plt.grid(True)

    plt.tight_layout()
    

def main(dim, distillation_threshold):
    plt.figure(figsize=(12, 5))
    plot(*run_simulation(dim, 5))
    plot(*run_simulation(dim, 30))

    # plt.show()
    plt.savefig('test.png')

if __name__ == '__main__':
    fire.Fire(main)