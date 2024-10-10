
def sde_2_ode(sde, score_fn):
    """
    Given an SDE, return the corresponding ODE.
    """
    def ode(x, t):
        drift, diffusion = sde(x, t)
        return drift + 0.5 * diffusion**2 * score_fn(x, t)
    return ode

def ode_2_sde():
    """
    Assumes ???
    """
    pass



class EulerMaruyama(Integrator):
    def __init__(self, sde):
        super().__init__(None)
        self.sde = sde

    def step(self, x, t, dt, key, *args):
        drift, diffusion = self.sde(x, t)
        noise = random.normal(key, x.shape) * jnp.sqrt(dt)
        return x + drift * dt + diffusion * noise
    
class RK4Maruyama(Integrator):
    def __init__(self, sde):
        super().__init__(None)
        self.sde = sde

    def step(self, x, t, dt, key, *args):
        noise = random.normal(key, x.shape) * jnp.sqrt(dt)
        dr_1, dif_1 = self.sde(x, t, *args)
        dr_2, dif_2 = self.sde(x + dt/2 * dr_1 + 0.5 * dif_1 * noise, t + dt/2, *args)
        dr_3, dif_3 = self.sde(x + dt/2 * dr_2 + 0.5 * dif_2 * noise, t + dt/2, *args)
        dr_4, dif_4 = self.sde(x + dt * dr_3 + 0.5 * dif_3 * noise, t + dt, *args)
        return x + dt/6 * (dr_1 + 2*dr_2 + 2*dr_3 + dr_4) + 1/6 * (dif_1 + 2*dif_2 + 2*dif_3 + dif_4) * noise

# TODO heun
# https://arxiv.org/pdf/2206.00364.pdf alg 2.