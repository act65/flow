import jax.numpy as jnp
from jax import lax, jit, vmap

from flow.utils import divergence
from flow.integrators import get_integrator

class Flow():
    # a deterministic flow
    def __init__(self, velocity, n_steps, integrator_name='euler'):
        """
        Given an integrator, construct functions to simulate;
            - forward and backward trajectories
            - push forward and pull backward probabilities

        Args:
            - v (callable): a velocity field v: X x T -> X
            - n_steps (int): number of steps to take
        """
        self.n_steps = n_steps
        self.velocity = velocity
        integrator = get_integrator(integrator_name)

        self.integrator_f = integrator(velocity)
        self.integrator_b = integrator(lambda x, t: -velocity(x, 1.0-t))

        self.div_v = divergence(velocity, argnums=1)

        self.push_forward = jit(self.push_forward)
        self.push_backward = jit(self.push_backward)

        # the batch versions
        self.b_push_forward = jit(vmap(self.push_forward, in_axes=(0, 0)))
        self.b_push_backward = jit(vmap(self.push_backward, in_axes=(0, 0)))
        self.b_push_forward_log_prob = jit(vmap(self.push_forward_log_prob, in_axes=(0, 0)))
        self.b_push_backward_log_prob = jit(vmap(self.push_backward_log_prob, in_axes=(0, 0)))

        self.forward = jit(self.forward)
        self.backward = jit(self.backward)

        self.b_forward = jit(vmap(self.forward))
        self.b_backward = jit(vmap(self.backward))

        # TODO cant jit these for some reason
        # self.forward_trajectory = jit(self.forward_trajectory)
        # self.backward_trajectory = jit(self.backward_trajectory)

        self.b_forward_trajectory = vmap(self.forward_trajectory, in_axes=0)
        self.b_backward_trajectory = vmap(self.backward_trajectory, in_axes=0)
    
    def forward(self, x_0, *args):
        return self.integrator_f.solve(x_0, t0=0.0, t1=1.0, n_steps=self.n_steps, key=None)
    def backward(self, y_T, *args):
        return self.integrator_b.solve(y_T, t0=0.0, t1=1.0, n_steps=self.n_steps, key=None)
    def forward_trajectory(self, x_0, *args):
        return self.integrator_f.trajectory(x_0, t0=0.0, t1=1.0, n_steps=self.n_steps, key=None)
    def backward_trajectory(self, y_T, *args):
        return self.integrator_b.trajectory(y_T, t0=0.0, t1=1.0, n_steps=self.n_steps, key=None)

    def accumulate(self, xts, ts):
        dt = ts[1] - ts[0]
        def _body(n, acc_p_t):
            acc_p_tp1 = acc_p_t + dt * self.div_v(xts[n], ts[n])
            return acc_p_tp1
        
        acc_p_t = lax.fori_loop(0, self.n_steps, _body, 0.0)

        return -acc_p_t
    
    def push_forward(self, p_x_0, x_0):
        """
        p(x, t) = exp(- \int_0^t \nabla \cdot v(\tau) X(x, t) d\tau) p(x_0))
        """
        # NOTE it is possible to compute the forward traj and accumulate in parallel.
        # but to support arbitrary integrators (which may require carrying extra state)
        # we hide all that in forward_trajectory
        xts = self.forward_trajectory(x_0)
        ts = jnp.linspace(0, 1, self.n_steps+1)
        acc = self.accumulate(xts, ts)
        return jnp.exp(acc)*p_x_0, xts[-1]

    def push_forward_log_prob(self, logp_x_0, x_0):
        xts = self.forward_trajectory(x_0)
        ts = jnp.linspace(0, 1, self.n_steps+1)
        acc = self.accumulate(xts, ts)
        return acc + logp_x_0, xts[-1]
    
    def push_backward(self, p_y, y):
        """
        p(x, t) = exp(\int_t^1 \nabla \cdot v(\tau) X(x, t) d\tau) p(y))
        """
        xts = self.backward_trajectory(y)
        ts = jnp.linspace(1, 0, self.n_steps+1)
        acc = self.accumulate(xts, ts)
        return jnp.exp(acc)*p_y, xts[-1]
    
    def push_backward_log_prob(self, logp_y, y):
        xts = self.backward_trajectory(y)
        ts = jnp.linspace(1, 0, self.n_steps+1)
        acc = self.accumulate(xts, ts)
        return acc + logp_y, xts[-1]