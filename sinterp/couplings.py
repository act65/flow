import jax.numpy as jnp
from jax import random, jit
import jax

import ot

class Coupling():
    def __call__(self, x, y=None, unused_key=None):
        """
        Return a coupling between two distributions.

        Returns:
            x, y: the coupled samples
        """
        raise NotImplementedError
    
class IndependentCoupling(Coupling):
    """
    p(x, y) = p(x) p(y)
    """
    def __call__(self, x, y, unused_key=None):
        return x, y
    
class ConditionalCoupling(Coupling):
    """
    p(x, y) = p(x | y) p(y)
    """
    def __init__(self, cond_fn):
        self.cond_fn = cond_fn

    def __call__(self, x, y=None, unused_key=None):
        y = self.cond_fn(x)
        return x, y

class EMDCoupling(Coupling):
    def __init__(self, reg=1e-1, max_iter=30):
        """
        Initializes the EMD-based coupling.

        Args:
            reg (float): The regularization parameter for the Sinkhorn algorithm.
            max_iter (int): The maximum number of Sinkhorn iterations.
        """
        self.reg = reg
        self.max_iter = max_iter
        self.__call__ = jit(self.__call__)

    def __call__(self, key, x, y):
        """
        Computes the optimal transport coupling between x and y and samples from it.

        Args:
            key (jax.random.PRNGKey): The random key for sampling.
            x (jnp.ndarray): The first batch of samples (n, d).
            y (jnp.ndarray): The second batch of samples (m, d).

        Returns:
            A tuple (x_hat, y_hat) representing the coupled samples.
        """
        n = x.shape[0]
        m = y.shape[0]

        # Define the marginal distributions (uniform in this case)
        a = jnp.ones(n) / n
        b = jnp.ones(m) / m

        # 1. Compute the cost matrix M (squared Euclidean distance)
        # ot.dist uses numpy by default, so we convert JAX arrays.
        M = ot.dist(x, y)
        # It's good practice to normalize the cost matrix to prevent numerical issues.
        M /= M.max()

        # 2. Solve the regularized optimal transport problem (Sinkhorn)
        # This function returns the transport plan Gs.
        Gs = ot.sinkhorn(a, b, M, self.reg, numItermax=self.max_iter)

        # 3. Sample from the transport plan Gs to get the coupled pairs.
        # We want to sample n pairs.
        
        # Flatten the transport matrix to a 1D probability distribution
        p = Gs.flatten()
        
        # Generate n samples of indices from the flattened distribution
        # The indices range from 0 to n*m - 1.
        all_indices = jnp.arange(n * m)
        sampled_indices = random.choice(key, a=all_indices, shape=(n,), p=p)
        
        # Convert the 1D indices back to 2D indices (i, j)
        i = sampled_indices // m  # Row indices (for x)
        j = sampled_indices % m   # Column indices (for y)
        
        # Return the new coupled samples
        return x[i], y[j]

class Rectification(Coupling):
    def __init__(self, flow):
        self.flow = flow

    def __call__(self, x, y, t, unused_key=None):
        yield x, self.flow.forward(x, t)
        yield self.flow.backward(y, t), y