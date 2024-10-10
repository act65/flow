import unittest
import parameterized
import jax.numpy as jnp

from jax import random, lax, vmap, grad, jit

from flow.flow import Flow

key = random.PRNGKey(0)

def v1(x, t):
    return jnp.sin(x) + jnp.cos(10*t)
def v2(x, t):
    return x * t + 5.0 * (1.0 - t)

n_steps = 200
d = 2

class TestFlow(unittest.TestCase):
    @parameterized.parameterized.expand([
        [v1],
        [v2]
    ])
    def test_forward_and_backward(self, v):
        """
        test that the forward and backward passes are consistent
        """
        x = random.normal(key, (d,))
        f = Flow(v, n_steps=n_steps, integrator_name='rk4')

        y = f.forward(x)
        x_recon = f.backward(y)

        self.assertTrue(jnp.isclose(x, x_recon, 1e-6).all())

    @parameterized.parameterized.expand([
        [v1],
        [v2]
    ])
    def test_forward_and_backward_trajectory(self, v):
        """
        test that the forward and backward passes are consistent
        """
        x = random.normal(key, (d,))
        f = Flow(v, n_steps=n_steps, integrator_name='rk4')

        xts = f.forward_trajectory(x)
        xts_rev = f.backward_trajectory(xts[-1])

        xts_ = jnp.flip(xts_rev, axis=0)

        self.assertTrue(jnp.isclose(xts, xts_, 1e-3).all())

    @parameterized.parameterized.expand([
        [v1],
        [v2]
    ])
    def test_forward_and_forward_trajectory(self, v):
        """
        test that the forward and forward trajectory are consistent
        """
        x = random.normal(key, (d,))
        f = Flow(v, n_steps=n_steps, integrator_name='rk4')

        y = f.forward(x)
        yts = f.forward_trajectory(x)

        self.assertTrue(jnp.isclose(y, yts[-1], 1e-4).all())