import unittest

from flow.sdes import VP
from exp_utils.models.ddim import DiscreteVP, DiscreteVE

import jax.numpy as jnp 

import matplotlib.pyplot as plt

class TestSDEs(unittest.TestCase):
    def test_vp(self):
        ts = jnp.linspace(0., 1., 100)
        vp = DiscreteVP(ts)

        plt.plot(ts, vp.discrete_betas, label='betas')
        plt.plot(ts, jnp.exp(vp.log_mean_coeff(ts)), label='log_mean_coeff')
        plt.plot(ts, vp.alphas, label='alphas')
        plt.plot(ts, vp.alphas_cumprod, label='alphas_cumprod')
        plt.plot(ts, vp.sqrt_alphas_cumprod, label='sqrt_alphas_cumprod')
        plt.plot(ts, vp.sqrt_1m_alphas_cumprod, label='sqrt_1m_alphas_cumprod')
        plt.plot(ts, vp.alphas_cumprod_prev, label='alphas_cumprod_prev')
        plt.plot(ts, vp.sqrt_alphas_cumprod_prev, label='sqrt_alphas_cumprod_prev')
        plt.plot(ts, vp.sqrt_1m_alphas_cumprod_prev, label='sqrt_1m_alphas_cumprod_prev')
        plt.title('VP')
        plt.legend()
        plt.xlabel('t')
        plt.show()

    def test_ve(self):
        ts = jnp.linspace(0., 1., 100)
        ve = DiscreteVE(ts)

        plt.plot(ts, ve.discrete_sigmas, label='sigmas')
        plt.title('VE')
        plt.legend()
        plt.show()

    def test_variance(self):
        ts = jnp.linspace(0., 1., 100)
        ve = DiscreteVE(ts)
        vp = DiscreteVP(ts)

        # plt.plot(ts, ve.variance(ts), label='VE')
        plt.plot(ts, vp.variance(ts), label='VP')
        # plt.plot(ts, vp.mean_coeff(ts)**2 + vp.variance(ts), label='VP')

        plt.title('Variance')
        plt.legend()
        plt.show()