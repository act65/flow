import unittest
import jax.numpy as jnp
from jax import random, lax, vmap, grad, jit

from flow.flow import Flow
from flow.integrators import Euler, RK4, EulerMaruyama, RK4Maruyama
from flow.utils import divergence
# from flow.eval import kl_divergence, build_log_likelihood_fn

from flow.sdes import VP, VE

import matplotlib.pyplot as plt


from exp_utils.data import gaussians

key = random.PRNGKey(0)

class TestIntegrators(unittest.TestCase):
    def test_lk_ode(self, plot=True):
        """
        test with lotka volterra
        """
        def lotka_volterra(x, t):
            return jnp.array([x[0] * (1 - x[1]), -x[1] * (1 - x[0])])

        n = 1000
        T = 20
        x = jnp.array([0.5, 0.5])

        integrators = [Euler, RK4]
        k = len(integrators)

        plt.figure()
        for i, integrator in enumerate(integrators):
            integ = integrator(lotka_volterra)

            t = jnp.linspace(0.0, T, n+1)

            traj = integ.trajectory(x, t0=0.0, t1=T, n_steps=n, key=key)

            # self.assert

            if plot:
                plt.subplot(k, 2, 2*i+1)
                plt.plot(t, traj[:, 0], label='x')
                plt.plot(t, traj[:, 1], label='y')

                plt.subplot(k, 2, 2*i+2)
                plt.scatter(traj[:, 0], traj[:, 1], c=jnp.arange(n+1))
                # plt.legend()
        plt.show()

    def test_vp_sde(self):
        """
        simple test of the variance preserving sde
        """

        integ = RK4Maruyama(VP())
        # integ = EulerMaruyama(VP())

        key = random.PRNGKey(0)
        n = 100

        B = 500
        xs = random.normal(key, (B,))
        t = jnp.linspace(0.0, 1.0, n+1)

        keys = random.split(key, B)

        trajs = integ.b_trajectory(xs, 0.0, 1.0, n, keys)
        m = jnp.mean(trajs, axis=0)
        std = jnp.std(trajs, axis=0)

        self.assertTrue(jnp.all(jnp.isclose(m, 0.0, atol=0.1)))
        self.assertTrue(jnp.all(jnp.isclose(std, 1.0, atol=0.1)))

        plt.plot(t, m)
        plt.fill_between(t, m-std, m+std, alpha=0.5)
        trajs = trajs[::10, :]
        plt.plot(jnp.repeat(t[None, :], B//10, axis=0).T, trajs.T, color='gray', alpha=0.1)
        plt.title('VE SDE')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.show()

    def test_ve_sde(self):
        """
        simple test of the variance preserving sde
        """

        integ = EulerMaruyama(VE())

        key = random.PRNGKey(0)
        n = 100

        B = 20
        xs = random.normal(key, (B,))
        t = jnp.linspace(0.0, 1.0, n+1)

        keys = random.split(key, B)

        trajs = integ.b_trajectory(xs, 0.0, 1.0, n, keys)

        m = jnp.mean(trajs, axis=0)
        std = jnp.std(trajs, axis=0)
        plt.plot(t, m)
        plt.fill_between(t, m-std, m+std, alpha=0.5)
        plt.plot(jnp.repeat(t[None, :], B, axis=0).T, trajs.T, color='gray', alpha=0.1)
        plt.title('VE SDE')
        plt.show()


