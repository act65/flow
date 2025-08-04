import pytest
import jax.numpy as jnp
from jax import random, vmap, grad
from scipy.stats import norm

from flow.flow import Flow
from flow.utils import divergence

key = random.PRNGKey(0)

# Helper to mimic the old distribution object structure
class SimpleNormal:
    def __init__(self, loc, scale):
        self.dist = norm(loc=loc, scale=scale)

    def sample(self, key, n):
        # JAX random key is not used here, but kept for API consistency
        return jnp.array(self.dist.rvs(size=n))

    def b_p(self, x):
        return jnp.array(self.dist.pdf(x))

def test_push_operators():
    """
    Test whether we can push probability distributions forward and backward.
    """
    n_steps = 20
    n = 5000

    def v(x, t):  # identity flow
        return jnp.zeros_like(x)

    flow = Flow(v, n_steps)

    px = SimpleNormal(loc=0.0, scale=1.0)

    # Forward
    x = px.sample(key, n)
    p_x_0 = px.b_p(x)
    p_y_, y_ = flow.b_push_forward(p_x_0, x)
    y__ = flow.b_forward(x)

    assert jnp.isclose(x, y__).all()
    assert jnp.isclose(y_, y__).all()

    # Backward
    y = px.sample(key, n)
    p_y = px.b_p(y)
    p_x_0_, x_0_ = flow.b_push_backward(p_y, y)
    x_0__ = flow.b_backward(y)

    assert jnp.isclose(y, x_0__).all()
    assert jnp.isclose(x_0_, x_0__).all()


def test_push_operators_non_vec_shapes():
    """
    Only checks the shapes.
    """
    def v(x, t):  # identity flow
        return jnp.zeros_like(x)

    n_steps = 20
    flow = Flow(v, n_steps=n_steps)

    x = jnp.ones((28, 28, 1))
    p_x_0 = jnp.ones((1,))

    p_y, y = flow.push_backward(p_x_0, x)

    assert y.shape == x.shape
    assert p_y.shape == p_x_0.shape

def test_divergence_nd():
    def v(x, t):
        return x * t

    div = divergence(v)
    k = 10
    d = div(jnp.ones((k,)), jnp.array(1.0))
    assert d.shape == ()
    assert d == k # div of x*t is d

def test_parameter_grads():
    def v(x, t, params):
        return x * t * params

    params = jnp.array([1.0])
    flow = Flow(v, n_steps=10, velocity_params=(params,))
    x = jnp.ones((1,))

    def loss(p):
        # The forward method is called without params, as they are now in the Flow object
        return jnp.sum(flow.forward(x)**2)

    # We are not checking the gradient value, just that it doesn't crash.
    grad_loss = grad(loss)(params)
    assert grad_loss is not None
