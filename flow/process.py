from jax import random, lax, vmap, jit
from flow.sdes import RSDE, SDE
from flow.integrators import EulerMaruyama
from typing import Callable

class Process():
    # a stochastic process
    def __init__(self, score_fn: Callable, sde: SDE, n_steps: int=50):
        self.score_fn = score_fn
        self.sde = sde
        self.n_steps = n_steps
        self.rsde = RSDE(score_fn, sde)

        self.integrator_f = EulerMaruyama(self.sde)
        self.integrator_b = EulerMaruyama(self.rsde)

        self.forward = jit(self.forward)
        self.backward = jit(self.backward)

        self.b_forward = jit(vmap(self.forward, in_axes=(0, 0)))
        self.b_backward = jit(vmap(self.backward, in_axes=(0, 0)))

    def forward(self, x_0, key):
        return self.integrator_f.solve(x_0, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)
    def backward(self, y_T, key):
        return self.integrator_b.solve(y_T, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)
    def forward_trajectory(self, x_0, key):
        return self.integrator_f.trajectory(x_0, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)
    def backward_trajectory(self, y_T, key):
        return self.integrator_b.trajectory(y_T, t0=0.0, t1=1.0, n_steps=self.n_steps, key=key)
    










    # TODO could also support a push_forward method?
    # but it would work differently.
    # would assume p(x_tp1 | x_t) is available
    def push_forward(self, x_0, p_x_0, key):
        """
        Can compute p(x_T | x_0, x_1, ...) = \prod_{t=0}^T p(x_tp1 | x_t).
        But this isnt what we really care about.
        Want to compute p(x_T | x_0) = \int \prod_{t=0}^T p(x_t | x_{t-1}) dx_{1:T-1}
        """
        pass