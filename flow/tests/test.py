import unittest
import jax.numpy as jnp
from jax import random, lax, vmap, grad, jit

from flow.flow import Flow
from flow.integrators import Euler, RK4, BDIA, EDICT, Coupled
from flow.utils import divergence
from flow.eval import kl_divergence, build_log_likelihood_fn

import matplotlib.pyplot as plt


from exp_utils.data import gaussians

key = random.PRNGKey(0)

class TestFlow(unittest.TestCase):

    def test_push_operators(self, plot=True):
        """
        test whether we can push probability distributions forward and backward.
        tests using the identity flow between two gaussians.

        acts as an integration test for kl_divergence, forward, backward, divergence, flow, 
        """
        n_steps = 20
        n = 5000
        b_kl = lambda x, y: jnp.mean(vmap(kl_divergence, in_axes=(0, 0))(x, y))

        def v(x, t):  # identity flow
            return jnp.zeros_like(x)

        flow = Flow(v, n_steps)

        px, py = gaussians.gg()
        
        # forward
        x = px.sample(key, n)
        p_x_0 = px.b_p(x)
        p_y_, y_ = flow.b_push_forward(p_x_0, x)

        # check that push forward gives the same results as forward
        y__ = flow.b_forward(x)
        self.assertTrue(jnp.isclose(x, y__).all())  # because we are using the identity flow
        self.assertTrue(jnp.isclose(y_, y__).all())

        # check whether distributions match
        p_y_true = py.b_p(y_)
        kl = b_kl(p_y_true, p_y_)

        self.assertTrue(jnp.isclose(kl, 0.0, atol=1e-3))
        if plot:
            plt.hist(y_.flatten(), bins=100, density=True)
            plt.scatter(y_, p_y_true, label='true',s=3, c='r', alpha=0.5)
            plt.scatter(y_, p_y_, label='approx',s=1, c='g', alpha=0.5)
            plt.title(f"Forward {kl}")
            plt.legend()
            plt.show()

        # backward
        y = py.sample(key, n)
        p_y = py.b_p(y)
        p_x_0_, x_0_ = flow.b_push_backward(p_y, y)

        # check that push backward gives the same results as backward
        x_0__ = flow.b_backward(y)
        self.assertTrue(jnp.isclose(y, x_0__).all())  # because we are using the identity flow
        self.assertTrue(jnp.isclose(x_0_, x_0__).all())

        # check whether distributions match
        p_x_true = px.b_p(x_0_)
        kl = b_kl(p_x_true, p_x_0_)
        self.assertTrue(jnp.isclose(kl, 0.0, atol=1e-3))

        if plot:
            plt.hist(x_0_.flatten(), bins=100, density=True)
            plt.scatter(x_0_, p_x_true, label='true',s=1, c='r', alpha=0.5)
            plt.scatter(x_0_, p_x_0_, label='approx',s=1, c='g', alpha=0.5)
            plt.title(f"Backward {kl}")
            plt.legend()
            plt.show()

    def test_push_operators_nd(self):
        def v(x, t):  # identity flow
            return jnp.zeros_like(x)
        
        n_steps = 20
        flow = Flow(v, n_steps=n_steps)

        px, py = gaussians.gg_nd(64)

        x = px.sample(key, 1)
        p_x_0 = px.b_p(x)

        print(flow.ds.div_v(x[0], jnp.array([1.0])).shape)

        print(p_x_0.shape, x.shape)

        y, p_y = flow.push_backward(p_x_0, x[0])

        print(y.shape, p_y.shape)


    def test_push_operators_non_vec_shapes(self):
        """
        only checks the shapes
        """
        def v(x, t):  # identity flow
            return jnp.zeros_like(x)
        
        n_steps = 20
        flow = Flow(v, n_steps=n_steps)

        x = jnp.ones((28, 28, 1))
        p_x_0 = jnp.ones((1,))

        p_y, y = flow.push_backward(p_x_0, x)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(p_y.shape, p_x_0.shape)

    def test_divergence_nd(self):
        def v(x, t):
            return x * t

        div = divergence(v)

        k = 10
        d = div(jnp.ones((k, )), jnp.ones((1, )))
        print(d.shape)

        # div = divergence(v)
        # d = div(jnp.ones((28, 28, 1)), jnp.ones((1, )))
        # print(d.shape)

    def test_parameter_grads(self):
        def v(x, t, params):
            return x * t * params

        flow = Flow(v, n_steps=10)
        params = jnp.array([1.0])

        x = jnp.ones((1, ))
        t = jnp.ones((1, ))

        def loss(params):
            return jnp.sum(flow.forward(x, t, params)**2)
        
        grad_loss = grad(loss)

        print(grad_loss(params))

    def test_likelihood(self):
        def v(x, t):
            return jnp.zeros_like(x)
        

        flow = Flow(v, n_steps=50)

        log_likelihood = build_log_likelihood_fn(flow)

        x = jnp.zeros((1, ))
        t = jnp.ones((1, ))

        log_px = log_likelihood(x)
        print(log_px)

        # TODO test with actual values.

# TODO test that forward and forward trajectory give the same results
