import abc
import jax
from jax import vmap, random
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp

class Distribution(abc.ABC):
    """Abstract base class for probability distributions."""

    @abc.abstractmethod
    def sample(self, key):
        """Samples from the distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, x):
        """Computes the log probability of a sample."""
        raise NotImplementedError

    def prob(self, x):
        """Computes the probability of a sample."""
        return jnp.exp(self.log_prob(x))

    def entropy(self, key, n_samples=1000):
        """
        Estimates the entropy of the distribution via Monte Carlo.
        H(p) = E_p[-log p(x)]
        This can be overridden by subclasses with an analytical solution.
        """
        samples = self.sample(key, shape=(n_samples,))
        log_probs = self.log_prob(samples)
        return -jnp.mean(log_probs)
    
    def b_sample(self, key, n_samples):
        keys = random.split(key, n_samples)
        return vmap(self.sample)(keys)
    
    def b_log_prob(self, x):
        return vmap(self.log_prob)(x)

class Gaussian(Distribution):
    """A standard multivariate Gaussian distribution N(0, I)."""

    def __init__(self, dim):
        """
        Args:
            dim (int): The dimensionality of the distribution.
        """
        self.dim = dim
        self.mean = jnp.zeros(dim)
        self.cov = jnp.eye(dim)

    def sample(self, key):
        """Samples from the Gaussian distribution."""
        return jax.random.multivariate_normal(key, self.mean, self.cov)

    def log_prob(self, x):
        """Computes the log probability of a sample."""
        return multivariate_normal.logpdf(x, self.mean, self.cov)

    @property
    def entropy(self):
        """Computes the analytical entropy of the Gaussian distribution."""
        return 0.5 * self.dim * (1.0 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.linalg.slogdet(self.cov)[1]

def gmm_log_p(means, covs, log_weights, x):
    """
    Computes the log probability of samples using the log-sum-exp trick for stability.
    log p(x) = log(sum_k [w_k * N(x|mu_k, cov_k)])
                = logsumexp(log(w_k) + log(N(x|mu_k, cov_k)))
    """
    # Ensure x is at least 2D for consistent processing
    x = jnp.atleast_2d(x)

    # Calculate the log probability of each point x under each Gaussian component.
    # vmap over the points in x. For each point, compute its log_prob under all components.
    # The inner logpdf call broadcasts the point over all component means and covs.
    log_probs_all_components = jax.vmap(
        lambda point: multivariate_normal.logpdf(point, means, covs)
    )(x)
    
    # Add the log weights and compute the log-sum-exp over the components axis
    weighted_log_probs = log_probs_all_components + log_weights
    log_p = logsumexp(weighted_log_probs, axis=1)
    
    # Squeeze the result if the original input was a single vector
    return jnp.squeeze(log_p)

class GaussianMixture(Distribution):
    """A Gaussian Mixture Model (GMM) distribution."""

    def __init__(self, weights, means, covs):
        """
        Args:
            weights (jnp.ndarray): A 1D array of shape (n_components,) with the weight of each Gaussian component. Must sum to 1.
            means (jnp.ndarray): A 2D array of shape (n_components, dim) with the mean of each component.
            covs (jnp.ndarray): A 3D array of shape (n_components, dim, dim) with the covariance matrix of each component.
        """
        self.n_components, self.dim = means.shape
        
        # --- Input validation ---
        assert self.n_components == len(weights), "Number of weights must match number of means."
        assert self.n_components == len(covs), "Number of covariance matrices must match number of means."
        assert self.dim == covs.shape[1] and self.dim == covs.shape[2], "Dimensions of covariance matrices are incorrect."
        assert jnp.isclose(jnp.sum(weights), 1.0), "Component weights must sum to 1."

        self.weights = weights
        self.means = means
        self.covs = covs
        self.log_weights = jnp.log(weights)

    def sample(self, key):
        """
        Samples a single example from the Gaussian mixture model.
        This is a two-step process:
        1. Choose a component based on the mixture weights.
        2. Sample from the chosen Gaussian component.
        """
        key_choice, key_sample = jax.random.split(key)

        # 1. Choose a single component index based on the weights
        component_index = jax.random.choice(
            key_choice, self.n_components, p=self.weights
        )

        # 2. Get the parameters for the chosen component
        chosen_mean = self.means[component_index]
        chosen_cov = self.covs[component_index]

        # 3. Sample from the chosen Gaussian component
        sample = jax.random.multivariate_normal(key_sample, chosen_mean, chosen_cov)
        
        return sample

    def log_prob(self, x):
        return gmm_log_p(self.means, self.covs, self.log_weights, x)

class FlowDistribution(Distribution):
    """A distribution defined by a deterministic Flow transformation."""

    def __init__(self, flow, base_distribution):
        """
        Args:
            flow (Flow): The deterministic flow mapping the base to the target.
            base_distribution (Distribution): The base distribution (e.g., Gaussian).
        """
        self.flow = flow
        self.base = base_distribution

    def sample(self, key):
        """
        Generates samples by transforming samples from the base distribution.
        Assumes the flow maps from the base (Gaussian) to the target distribution.
        """
        x0 = self.base.sample(key)
        y = self.flow.forward(x0)
        return y

    def log_prob(self, x1):
        x0 = self.flow.backward(x1)
        log_p_x0 = self.base.log_prob(x0)
        return self.flow.push_forward_log_prob(log_p_x0, x0)

class ProcessDistribution(Distribution):
    """
    A distribution defined by a stochastic Process (SDE).
    While this class can generate samples, it does not support exact
    likelihood evaluation, which is a key feature of the deterministic
    probability flow counterpart (FlowDistribution).
    """

    def __init__(self, process, base_distribution):
        """
        Args:
            process (Process): The stochastic process mapping the base to the target.
            base_distribution (Distribution): The base distribution (e.g., Gaussian).
        """
        self.process = process
        self.base = base_distribution

    def sample(self, key):
        """
        Generates samples by solving the forward SDE from base samples.
        """
        key1, key2 = jax.random.split(key)
        x0 = self.base.sample(key1)
        y = self.process.forward(x0, key2)[0]
        return y

    def log_prob(self, x):
        """
        The log probability for a distribution defined by a stochastic process
        is generally intractable to compute for a single point. This is a
        distinguishing feature from the deterministic FlowDistribution, which
        allows for exact likelihood calculation.
        """
        raise NotImplementedError(
            "Exact likelihood is not tractable for a distribution defined by a stochastic process. "
            "Use the FlowDistribution for likelihood-based tasks."
        )

    def entropy(self, key, n_samples=1000):
        """
        Entropy estimation requires log_prob, which is not available.
        """
        raise NotImplementedError(
            "Cannot compute entropy without a tractable log_prob method."
        )