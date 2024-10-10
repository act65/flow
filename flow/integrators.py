from flow.utils import scan_with_init_state, scan_wrapper

import jax.numpy as jnp
from jax import lax, random, vmap, jit

class Integrator():
    """
    Assumes ts are uniformly spaced.
    """
    def __init__(self, ode):
        self.ode = ode

        self.b_solve = jit(vmap(self.solve, in_axes=(0, None, None, None, 0)), static_argnums=(1, 2, 3))
        self.b_trajectory = jit(vmap(self.trajectory, in_axes=(0, None, None, None, 0)), static_argnums=(1, 2, 3))
        
    def step(self, x, *args, **kwargs):
        raise NotImplementedError

    def solve(self, x_0, t0, t1, n_steps, key, *args):
        ts = jnp.linspace(t0, t1, n_steps+1)
        dt = ts[1] - ts[0]
        if key is not None:
            keys = random.split(key, n_steps)
        else:
            keys = jnp.zeros((n_steps, 2), dtype=jnp.int32)

        fn = lambda n, x: self.step(x, ts[n], dt, keys[n], *args)
        return lax.fori_loop(0, n_steps, fn, x_0)
    
    def trajectory(self, x_0, t0, t1, n_steps, key, *args):
        """
        Returns sequence of states from x_0 to x_N (of length n_steps+1)
        """
        ts = jnp.linspace(t0, t1, n_steps+1)
        dt = ts[1] - ts[0]
        if key is not None:
            keys = random.split(key, n_steps)
        else:
            keys = jnp.zeros((n_steps, 2), dtype=jnp.int32)
        fn = lambda n, x: self.step(x, ts[n], dt, keys[n], *args)
        x_ts = scan_with_init_state(scan_wrapper(fn), x_0, jnp.arange(n_steps+1), length=n_steps+1)
        return x_ts

def get_integrator(name):
    if name == 'euler':
        return Euler
    elif name == 'rk4':
        return RK4
    elif name == 'euler-maruyama':
        return EulerMaruyama
    else:
        raise ValueError(f'Unknown integrator {name}')

class Euler(Integrator):
    def __init__(self, ode):
        super().__init__(ode)

    def step(self, x, t, dt, unused_key, *args):
        return x+self.ode(x, t, *args) * dt

class RK4(Integrator):
    def __init__(self, ode):
        super().__init__(ode)

    def step(self, x, t, dt, unused_key, *args):
        k1 = self.ode(x, t, *args)
        k2 = self.ode(x + dt/2 * k1, t + dt/2, *args)
        k3 = self.ode(x + dt/2 * k2, t + dt/2, *args)
        k4 = self.ode(x + dt * k3, t + dt, *args)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)