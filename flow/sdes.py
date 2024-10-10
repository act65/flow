"""
https://github.com/bb515/diffusionjax/blob/aeb5de94d5a55d780b3a0c30fd6b6a17f6586e56/diffusionjax/sde.py
"""

import jax.numpy as jnp
from jax import random

def get_linear_beta_function(beta_min, beta_max):
	"""Returns:
		Linear beta (cooling rate parameter) as a function of time,
		It's integral multiplied by -0.5, which is the log mean coefficient of the VP SDE.
	"""
	def beta(t):
		return beta_min + t * (beta_max - beta_min)

	def log_mean_coeff(t):
		"""..math: -0.5 * \int_{0}^{t} \beta(t) dt"""
		return -0.5 * t * beta_min - 0.25 * t**2 * (beta_max - beta_min)

	return beta, log_mean_coeff

def get_sigma_function(sigma_min, sigma_max):
	log_sigma_min = jnp.log(sigma_min)
	log_sigma_max = jnp.log(sigma_max)
	def sigma(t):
		# return sigma_min * (sigma_max / sigma_min)**t  # Has large relative error close to zero compared to alternative, below
		return jnp.exp(log_sigma_min + t * (log_sigma_max - log_sigma_min))
	return sigma

class SDE:
	"""Stochastic differential equation class."""
	def __init__(self, *args, **kwargs):
		pass
	
	def __call__(self, x, t):
		"""Returns the drift and diffusion values."""
		raise NotImplementedError
	
	def mean_coeff(self, t):
		"""Mean coefficient of the marginal distribution."""
		raise NotImplementedError
	
	def std(self, t):
		"""Standard deviation of the marginal distribution."""
		raise NotImplementedError
	
	def variance(self, t):
		"""Variance of the marginal distribution."""
		raise NotImplementedError
	
	def marginal_prob(self, x, t):
		"""Mean and standard deviation of the marginal distribution."""
		raise NotImplementedError
	
	def prior(self, rng, shape):
		"""Sample from the prior distribution."""
		raise NotImplementedError
	
	def reverse(self, score):
		"""Reverse SDE."""
		raise NotImplementedError
	
	def r2(self, t, data_variance):
		"""Analytic variance of the distribution at time zero conditioned on x_t."""
		raise NotImplementedError
	
	def ratio(self, t):
		"""Ratio of marginal variance and mean coeff."""
		raise NotImplementedError

class RSDE(SDE):
	"""Reverse SDE class."""
	def __init__(self, score, forward_sde, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.score = score
		self.forward_sde = forward_sde

	def __call__(self, x, t):
		tau = 1.0 - t
		drift, diffusion = self.forward_sde(x, tau)
		drift = -drift + diffusion**2 * self.score(x, tau)
		return drift, diffusion

class VP(SDE):
	"""Variance preserving (VP) SDE, a.k.a. time rescaled Ornstein Uhlenbeck (OU) SDE."""
	def __init__(self, beta=None, log_mean_coeff=None):
		super().__init__()
		if beta is None:
			self.beta, self.log_mean_coeff = get_linear_beta_function(
				beta_min=0.1, beta_max=20.)
		else:
			self.beta = beta
			self.log_mean_coeff = log_mean_coeff

		self.beta_min = self.beta(0.)
		self.beta_max = self.beta(1.)

	def __call__(self, x, t):
		beta_t = self.beta(t)
		drift = -0.5 * beta_t * x
		diffusion = jnp.sqrt(beta_t)
		return drift, diffusion

	def mean_coeff(self, t):
		return jnp.exp(self.log_mean_coeff(t))

	def std(self, t):
		return jnp.sqrt(self.variance(t))

	def variance(self, t):
		return 1.0 - jnp.exp(2 * self.log_mean_coeff(t))

	def marginal_prob(self, x, t):
		return self.mean_coeff(t) * x, jnp.sqrt(self.variance(t))

	def prior(self, rng, shape):
		return random.normal(rng, shape)

	# def reverse(self, score):
	# 	fwd_sde = self.sde
	# 	beta = self.beta
	# 	log_mean_coeff = self.log_mean_coeff
	# 	return RVP(score, fwd_sde, beta, log_mean_coeff)

	def r2(self, t, data_variance):
		r"""Analytic variance of the distribution at time zero conditioned on x_t, given crude assumption that
		the data distribution is isotropic-Gaussian.

		.. math::
			\text{Variance of }p_{0}(x_{0}|x_{t}) \text{ if } p_{0}(x_{0}) = \mathcal{N}(0, \text{data_variance}I)
		"""
		alpha = jnp.exp(2 * self.log_mean_coeff(t))
		return (1 - alpha) * data_variance / (1 - alpha + alpha * data_variance)

	def ratio(self, t):
		"""Ratio of marginal variance and mean coeff."""
		return self.variance(t) / self.mean_coeff(t)


class VE(SDE):
	"""Variance exploding (VE) SDE, a.k.a. diffusion process with a time dependent diffusion coefficient."""
	def __init__(self, sigma=None):
		super().__init__()
		if sigma is None:
			self.sigma = get_sigma_function(sigma_min=0.01, sigma_max=378.)
		else:
			self.sigma = sigma
		self.sigma_min = self.sigma(0.)
		self.sigma_max = self.sigma(1.)
		self.std = self.sigma

	def __call__(self, x, t):
		sigma_t = self.sigma(t)
		drift = jnp.zeros_like(x)
		diffusion = sigma_t * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
		return drift, diffusion

	def log_mean_coeff(self, t):
		return jnp.zeros_like(t)

	def mean_coeff(self, t):
		return jnp.ones_like(t)

	def variance(self, t):
		return self.std(t)**2

	def prior(self, rng, shape):
		return random.normal(rng, shape) * self.sigma_max

	# def reverse(self, score):
	#     forward_sde = self.sde
	#     sigma = self.sigma

	#     return RVE(score, forward_sde, sigma)

	def r2(self, t, data_variance):
		r"""Analytic variance of the distribution at time zero conditioned on x_t, given crude assumption that
		the data distribution is isotropic-Gaussian.

		.. math::
		\text{Variance of }p_{0}(x_{0}|x_{t}) \text{ if } p_{0}(x_{0}) = \mathcal{N}(0, \text{data_variance}I)
		"""
		variance = self.variance(t)
		return variance * data_variance / (variance + data_variance)

	def ratio(self, t):
		"""Ratio of marginal variance and mean coeff."""
		return self.variance(t)