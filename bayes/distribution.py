import abc
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

class Distribution(abc.ABC):
    """Abstract base class for probability distributions."""

    @abc.abstractmethod
    def sample(self, key, shape=(1,)):
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

    def sample(self, key, shape=(1,)):
        """Samples from the Gaussian distribution."""
        return jax.random.multivariate_normal(key, self.mean, self.cov, shape)

    def log_prob(self, x):
        """Computes the log probability of a sample."""
        return multivariate_normal.logpdf(x, self.mean, self.cov)

    @property
    def entropy(self):
        """Computes the analytical entropy of the Gaussian distribution."""
        return 0.5 * self.dim * (1.0 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.linalg.slogdet(self.cov)[1]


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

    def sample(self, key, shape=(1,)):
        """
        Generates samples by transforming samples from the base distribution.
        Assumes the flow maps from the base (Gaussian) to the target distribution.
        """
        x0 = self.base.sample(key, shape)
        y, _ = self.flow.b_forward(x0)
        return y

    def log_prob(self, y):
        """
        Computes the log probability of a sample using the change of variables formula.
        This is achieved by pulling the sample back to the base distribution.
        log p_Y(y) = log p_X(f^{-1}(y)) + log |det(d f^{-1}(y) / dy)|
        """
        # The push_backward_log_prob method computes exactly this.
        # It returns the log_prob on the base distribution and the final state,
        # which is the point on the base distribution.
        logp_y, _ = self.flow.b_push_backward_log_prob(jnp.log(1.0), y) # Start with log(1.0) = 0
        
        # We need to evaluate the log_prob of the base distribution at the
        # point where y is mapped to by the backward flow.
        # Let's get the backward trajectory to find the point in the base.
        _, x0 = self.flow.b_backward(y)
        logp_base = self.base.log_prob(x0)

        return logp_base + logp_y


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

    def sample(self, key, shape=(1,)):
        """
        Generates samples by solving the forward SDE from base samples.
        """
        keys = jax.random.split(key, shape[0])
        x0 = self.base.sample(key, shape)
        y, _ = self.process.b_forward(x0, keys)
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