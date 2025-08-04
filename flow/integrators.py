import jax.numpy as jnp
from jax import lax, random, jit

from flow.utils import scan_with_init_state, scan_wrapper

class Integrator():
    def __init__(self, ode, diffusion=None):
        self.ode = ode
        self.diffusion = diffusion

    def step(self, x, *args, **kwargs):
        raise NotImplementedError

    def solve(self, x_0, t0, t1, n_steps, key, *args):
        ts = jnp.linspace(t0, t1, n_steps + 1)
        dt = ts[1] - ts[0]
        if key is not None:
            keys = random.split(key, n_steps)
        else:
            keys = jnp.zeros((n_steps, 2), dtype=jnp.uint32)

        def step_fn(i, x):
            return self.step(x, ts[i], dt, keys[i], *args)

        return lax.fori_loop(0, n_steps, step_fn, x_0)

    def trajectory(self, x_0, t0, t1, n_steps, key, *args):
        ts = jnp.linspace(t0, t1, n_steps + 1)
        dt = ts[1] - ts[0]
        if key is not None:
            keys = random.split(key, n_steps)
        else:
            keys = jnp.zeros((n_steps, 2), dtype=jnp.uint32)

        def step_fn(x, i):
            return self.step(x, ts[i], dt, keys[i], *args)

        def scan_fn(acc, i):
            y = step_fn(acc, i)
            return y, y

        return scan_with_init_state(scan_fn, x_0, jnp.arange(n_steps + 1), length=n_steps + 1)

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
    def step(self, x, t, dt, unused_key, *args):
        return x + self.ode(x, t, *args) * dt

class RK4(Integrator):
    def step(self, x, t, dt, unused_key, *args):
        k1 = self.ode(x, t, *args)
        k2 = self.ode(x + dt / 2 * k1, t + dt / 2, *args)
        k3 = self.ode(x + dt / 2 * k2, t + dt / 2, *args)
        k4 = self.ode(x + dt * k3, t + dt, *args)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

class EulerMaruyama(Integrator):
    def __init__(self, ode, diffusion):
        if diffusion is None:
            raise ValueError("EulerMaruyama integrator requires a diffusion function.")
        super().__init__(ode, diffusion)

    def step(self, x, t, dt, key, *args):
        drift = self.ode(x, t, *args)
        diffusion_term = self.diffusion(t)
        noise = random.normal(key, shape=x.shape)
        return x + drift * dt + diffusion_term * jnp.sqrt(dt) * noise